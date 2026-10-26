"""
g0_recovery.py
==============

Original-G0 hidden-geometry recovery for single-view Stage-1 refinement.

Why this module exists
----------------------
A single RGB view cannot prove that an occluded/back-side Gaussian is useless.
If Stage-1 pruning removes such Gaussians, the defect may only become visible
later when the object moves/rotates.

This module treats the ORIGINAL G0 as a conservative donor bank:

    original G0
        -> keep only HIDDEN (optionally UNCERTAIN) donor candidates
        -> compare candidates against the CURRENT corrected G0 in 3D
        -> recover only donor Gaussians whose local 3D coverage is missing

Important safety properties
---------------------------
1. No RGB hallucination is used for the hidden side.
2. The front side is not restored wholesale.
3. Recovered Gaussians copy the original G0 attributes exactly.
4. The original donor bank is stored on CPU, so it does not permanently
   consume additional GPU memory during Stage-1 optimization.
5. Recovery is performed at the END of Stage-1.

Expected import
---------------
from gaussian_utils.g0_recovery import (
    build_g0_donor_bank,
    recover_missing_hidden_gaussians,
    recovery_summary,
)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from simple_knn._C import distCUDA2


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _cfg_bool(cfg: Any, name: str, default: bool) -> bool:
    value = getattr(cfg, name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


@torch.no_grad()
def build_g0_donor_bank(
    gaussians: Any,
    visibility_info: Optional[Dict[str, Any]],
    cfg: Any,
) -> Dict[str, Any]:
    """
    Snapshot the ORIGINAL G0 before Stage-1 changes topology.

    All heavy tensors are moved to CPU immediately.

    `visibility_info` should come from visibility.py and should describe the
    original G0 under the fixed Stage-1 camera.
    """
    xyz_gpu = gaussians.get_xyz.detach().contiguous()
    n = int(xyz_gpu.shape[0])

    # Characteristic raw-G0 point spacing.  We compute this once on GPU while
    # the original G0 is already resident there.
    if n >= 2:
        spacing = torch.sqrt(
            torch.clamp_min(distCUDA2(xyz_gpu), 1e-12)
        )
        finite = torch.isfinite(spacing) & (spacing > 0)
        if bool(finite.any()):
            spacing_median = float(
                torch.quantile(spacing[finite], 0.50).item()
            )
        else:
            spacing_median = 0.0
    else:
        spacing_median = 0.0

    hidden = torch.zeros(n, dtype=torch.bool)
    uncertain = torch.zeros(n, dtype=torch.bool)
    outside = torch.zeros(n, dtype=torch.bool)
    visible = torch.zeros(n, dtype=torch.bool)

    if isinstance(visibility_info, dict):
        for key, dst in (
            ("hidden_mask", hidden),
            ("uncertain_mask", uncertain),
            ("outside_mask", outside),
            ("visible_mask", visible),
        ):
            value = visibility_info.get(key, None)
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == n:
                dst.copy_(value.detach().bool().cpu())

    # Raw parameter tensors are copied rather than activated values because
    # densification_postfix() expects the model's internal parameterization.
    bank = {
        "xyz": gaussians._xyz.detach().float().cpu().clone(),
        "features_dc": gaussians._features_dc.detach().float().cpu().clone(),
        "features_rest": gaussians._features_rest.detach().float().cpu().clone(),
        "opacity": gaussians._opacity.detach().float().cpu().clone(),
        "scaling": gaussians._scaling.detach().float().cpu().clone(),
        "rotation": gaussians._rotation.detach().float().cpu().clone(),
        "visible_mask": visible,
        "hidden_mask": hidden,
        "uncertain_mask": uncertain,
        "outside_mask": outside,
        "count": n,
        "spacing_median": spacing_median,
    }

    return bank


def donor_bank_summary(bank: Dict[str, Any]) -> str:
    n = int(bank.get("count", 0))
    hidden = int(bank.get("hidden_mask", torch.zeros(0)).sum().item())
    uncertain = int(bank.get("uncertain_mask", torch.zeros(0)).sum().item())
    visible = int(bank.get("visible_mask", torch.zeros(0)).sum().item())
    outside = int(bank.get("outside_mask", torch.zeros(0)).sum().item())
    spacing = float(bank.get("spacing_median", 0.0))
    return (
        f"G0 donor bank: total={n:,} | visible={visible:,} | "
        f"hidden={hidden:,} | uncertain={uncertain:,} | "
        f"outside={outside:,} | spacing_median={spacing:.8g}"
    )


def _query_nearest_distance(
    reference_xyz: np.ndarray,
    query_xyz: np.ndarray,
) -> np.ndarray:
    """
    Query nearest current-G0 distance for each raw donor candidate.

    scipy.cKDTree is O(N log N) and is the practical path for hundreds of
    thousands of Gaussians.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError(
            "G0 recovery requires scipy for large-cloud 3D nearest-neighbour "
            "queries. Install it in the ed3dgs environment: pip install scipy"
        ) from exc

    if reference_xyz.shape[0] == 0 or query_xyz.shape[0] == 0:
        return np.empty((query_xyz.shape[0],), dtype=np.float32)

    tree = cKDTree(reference_xyz)

    try:
        distance, _ = tree.query(query_xyz, k=1, workers=-1)
    except TypeError:
        # Older scipy versions do not expose `workers`.
        distance, _ = tree.query(query_xyz, k=1)

    return np.asarray(distance, dtype=np.float32)


@torch.no_grad()
def recover_missing_hidden_gaussians(
    gaussians: Any,
    donor_bank: Dict[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    """
    Recover missing hidden geometry from the original G0 donor bank.

    Recovery rule
    -------------
    A raw donor Gaussian is eligible when:
      1) it was HIDDEN in the original fixed Stage-1 view;
      2) optionally it was UNCERTAIN (off by default);
      3) its original opacity is above a small confidence floor;
      4) no current Gaussian lies within the adaptive 3D coverage threshold.

    Adaptive threshold
    ------------------
        threshold = max(
            stage1_g0_recovery_min_distance,
            raw_spacing_median * stage1_g0_recovery_spacing_mult
        )

    This prevents restoration of raw points that are already represented by a
    slightly moved/refined Gaussian.
    """
    result = {
        "enabled": _cfg_bool(cfg, "stage1_enable_g0_recovery", True),
        "donor_total": int(donor_bank.get("count", 0)),
        "eligible_hidden": 0,
        "eligible_uncertain": 0,
        "candidate_count": 0,
        "recovered_count": 0,
        "distance_threshold": 0.0,
        "nearest_distance_mean": 0.0,
        "nearest_distance_median": 0.0,
        "nearest_distance_max": 0.0,
        "spacing_median": float(donor_bank.get("spacing_median", 0.0)),
    }

    if not result["enabled"]:
        return result

    required = (
        "xyz",
        "features_dc",
        "features_rest",
        "opacity",
        "scaling",
        "rotation",
        "hidden_mask",
        "uncertain_mask",
    )
    for key in required:
        if key not in donor_bank:
            raise KeyError(f"G0 donor bank is missing required field: {key}")

    raw_xyz = donor_bank["xyz"]
    raw_n = int(raw_xyz.shape[0])
    if raw_n == 0:
        return result

    hidden = donor_bank["hidden_mask"].bool()
    uncertain = donor_bank["uncertain_mask"].bool()

    eligible = hidden.clone()
    result["eligible_hidden"] = int(hidden.sum().item())

    include_uncertain = _cfg_bool(
        cfg,
        "stage1_g0_recovery_include_uncertain",
        False,
    )
    if include_uncertain:
        eligible |= uncertain
        result["eligible_uncertain"] = int(uncertain.sum().item())

    if _cfg_bool(cfg, "stage1_g0_recovery_include_outside", False):
        outside = donor_bank.get(
            "outside_mask",
            torch.zeros(raw_n, dtype=torch.bool),
        ).bool()
        eligible |= outside

    # Ignore extremely transparent raw donor splats.
    raw_opacity_actual = torch.sigmoid(
        donor_bank["opacity"].reshape(raw_n, -1)[:, 0]
    )
    min_opacity = max(
        0.0,
        _cfg_float(cfg, "stage1_g0_recovery_min_opacity", 0.03),
    )
    eligible &= raw_opacity_actual >= min_opacity

    eligible_idx = torch.nonzero(
        eligible,
        as_tuple=False,
    ).squeeze(-1)

    if eligible_idx.numel() == 0:
        return result

    current_xyz = (
        gaussians.get_xyz.detach().float().cpu().numpy()
    )
    donor_xyz = raw_xyz[eligible_idx].numpy()

    nearest = _query_nearest_distance(
        reference_xyz=current_xyz,
        query_xyz=donor_xyz,
    )

    spacing_median = float(
        donor_bank.get("spacing_median", 0.0)
    )
    spacing_mult = max(
        0.1,
        _cfg_float(
            cfg,
            "stage1_g0_recovery_spacing_mult",
            1.75,
        ),
    )
    min_distance = max(
        0.0,
        _cfg_float(
            cfg,
            "stage1_g0_recovery_min_distance",
            0.0,
        ),
    )

    threshold = max(
        min_distance,
        spacing_median * spacing_mult,
    )

    # Degenerate scale fallback.
    if threshold <= 1e-12:
        raw_scale = torch.exp(donor_bank["scaling"])
        effective = torch.topk(
            raw_scale,
            k=min(2, raw_scale.shape[1]),
            dim=1,
            largest=True,
        ).values.mean(dim=1)
        finite = torch.isfinite(effective) & (effective > 0)
        if bool(finite.any()):
            threshold = float(
                torch.quantile(effective[finite], 0.50).item()
            )
        threshold = max(threshold, 1e-8)

    result["distance_threshold"] = float(threshold)

    nearest_t = torch.from_numpy(nearest)
    missing_local = nearest_t > float(threshold)

    candidate_local_idx = torch.nonzero(
        missing_local,
        as_tuple=False,
    ).squeeze(-1)

    result["candidate_count"] = int(
        candidate_local_idx.numel()
    )

    if candidate_local_idx.numel() == 0:
        return result

    candidate_raw_idx = eligible_idx[candidate_local_idx]
    candidate_distance = nearest_t[candidate_local_idx]

    # Independent structural-recovery cap.  This deliberately does not use the
    # ordinary Stage-1 point budget: hidden geometry preservation has priority
    # over a cosmetic training budget.
    max_points = max(
        0,
        _cfg_int(
            cfg,
            "stage1_g0_recovery_max_points",
            100000,
        ),
    )

    if max_points == 0:
        return result

    if candidate_raw_idx.numel() > max_points:
        # Prioritize the clearest 3D holes first.
        order = torch.argsort(
            candidate_distance,
            descending=True,
        )[:max_points]
        candidate_raw_idx = candidate_raw_idx[order]
        candidate_distance = candidate_distance[order]

    if candidate_distance.numel() > 0:
        result["nearest_distance_mean"] = float(
            candidate_distance.float().mean().item()
        )
        result["nearest_distance_median"] = float(
            torch.quantile(
                candidate_distance.float(),
                0.50,
            ).item()
        )
        result["nearest_distance_max"] = float(
            candidate_distance.max().item()
        )

    if not hasattr(gaussians, "densification_postfix"):
        raise RuntimeError(
            "GaussianModel has no densification_postfix(); "
            "cannot append recovered G0 Gaussians."
        )

    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype

    chunk_size = max(
        1,
        _cfg_int(
            cfg,
            "stage1_g0_recovery_append_chunk",
            25000,
        ),
    )

    recovered = 0

    for start in range(
        0,
        int(candidate_raw_idx.numel()),
        chunk_size,
    ):
        idx = candidate_raw_idx[
            start:start + chunk_size
        ].long()

        new_xyz = donor_bank["xyz"][idx].to(
            device=device,
            dtype=dtype,
        )
        new_features_dc = donor_bank["features_dc"][idx].to(
            device=device,
            dtype=gaussians._features_dc.dtype,
        )
        new_features_rest = donor_bank["features_rest"][idx].to(
            device=device,
            dtype=gaussians._features_rest.dtype,
        )
        new_opacity = donor_bank["opacity"][idx].to(
            device=device,
            dtype=gaussians._opacity.dtype,
        )
        new_scaling = donor_bank["scaling"][idx].to(
            device=device,
            dtype=gaussians._scaling.dtype,
        )
        new_rotation = donor_bank["rotation"][idx].to(
            device=device,
            dtype=gaussians._rotation.dtype,
        )

        gaussians.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )
        recovered += int(idx.numel())

    result["recovered_count"] = int(recovered)
    return result


def recovery_summary(info: Dict[str, Any]) -> str:
    if not info.get("enabled", False):
        return "G0 recovery: disabled"

    return (
        "G0 recovery: "
        f"hidden_eligible={int(info.get('eligible_hidden', 0)):,} | "
        f"candidates={int(info.get('candidate_count', 0)):,} | "
        f"recovered={int(info.get('recovered_count', 0)):,} | "
        f"threshold={float(info.get('distance_threshold', 0.0)):.8g} | "
        f"raw_spacing={float(info.get('spacing_median', 0.0)):.8g}"
    )
