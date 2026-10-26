"""
surface_prior.py
================

Raw-G0 local surface prior for single-view Stage-1 Gaussian refinement.

Purpose
-------
The existing Stage-1 Anchor/Smooth regularizer stabilizes the *current*
topology, but after prune/reindex it may rebuild its reference from the current
surviving surface.  This module is different:

    RAW original G0  -> permanent surface teacher

It never changes its teacher during Stage-1.

The prior uses:
    1) nearest raw-G0 surface association in 3D;
    2) local PCA around the associated raw point;
    3) point-to-plane loss to keep the refined Gaussian on the original local
       surface;
    4) a weaker tangential-drift penalty;
    5) stronger weights for raw-G0 HIDDEN / UNCERTAIN regions.

To remain practical for 500k-1M Gaussians, the loss is evaluated on a sampled
subset every N iterations.  Correspondences are computed on detached points
with scipy.cKDTree; gradients flow only through the current sampled Gaussian
positions.

This module does NOT add/prune Gaussians.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _cfg_bool(cfg: Any, name: str, default: bool) -> bool:
    value = getattr(cfg, name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _extract_raw_xyz_cpu(source: Any) -> torch.Tensor:
    """
    Accept either:
      - the CPU donor-bank dict from g0_recovery.py; or
      - a GaussianModel / Nx3 tensor.
    """
    if isinstance(source, dict):
        xyz = source.get("xyz", None)
        if isinstance(xyz, torch.Tensor):
            return xyz.detach().float().cpu().contiguous()

    if torch.is_tensor(source):
        if source.ndim == 2 and source.shape[-1] == 3:
            return source.detach().float().cpu().contiguous()

    if hasattr(source, "get_xyz"):
        xyz = source.get_xyz
        xyz = xyz() if callable(xyz) else xyz
        if isinstance(xyz, torch.Tensor):
            return xyz.detach().float().cpu().contiguous()

    raise TypeError(
        "surface_prior.py cannot extract raw G0 xyz from the supplied source."
    )


def _extract_visibility_cpu(
    source: Any,
    visibility_info: Optional[Dict[str, Any]],
    n: int,
) -> Dict[str, torch.Tensor]:
    """
    Prefer visibility masks already stored in donor_bank, otherwise use the
    initial visibility result.
    """
    out = {
        "visible_mask": torch.zeros(n, dtype=torch.bool),
        "hidden_mask": torch.zeros(n, dtype=torch.bool),
        "uncertain_mask": torch.zeros(n, dtype=torch.bool),
        "outside_mask": torch.zeros(n, dtype=torch.bool),
    }

    candidates = []
    if isinstance(source, dict):
        candidates.append(source)
    if isinstance(visibility_info, dict):
        candidates.append(visibility_info)

    for container in candidates:
        for key in out:
            value = container.get(key, None)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim == 1
                and int(value.shape[0]) == n
            ):
                out[key] = value.detach().bool().cpu().clone()

    return out


def _build_tree(raw_xyz_np: np.ndarray):
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError(
            "surface_prior.py requires scipy.cKDTree. "
            "Install scipy in the ed3dgs environment: pip install scipy"
        ) from exc

    return cKDTree(raw_xyz_np)


def _estimate_spacing(tree, raw_xyz_np: np.ndarray, max_samples: int = 200000) -> float:
    n = int(raw_xyz_np.shape[0])
    if n < 2:
        return 1.0

    if n <= max_samples:
        sample = raw_xyz_np
    else:
        ids = np.linspace(0, n - 1, num=max_samples, dtype=np.int64)
        sample = raw_xyz_np[ids]

    try:
        dist, _ = tree.query(sample, k=2, workers=-1)
    except TypeError:
        dist, _ = tree.query(sample, k=2)

    nearest = np.asarray(dist[:, 1], dtype=np.float32)
    finite = np.isfinite(nearest) & (nearest > 1e-12)
    if not finite.any():
        return 1.0

    return float(np.median(nearest[finite]))


def build_raw_g0_surface_prior(
    raw_g0_source: Any,
    initial_visibility: Optional[Dict[str, Any]],
    cfg: Any,
) -> Dict[str, Any]:
    """
    Build the permanent raw-G0 surface teacher.

    Heavy data remain on CPU.
    """
    raw_xyz_cpu = _extract_raw_xyz_cpu(raw_g0_source)
    raw_xyz_np = raw_xyz_cpu.numpy()
    n = int(raw_xyz_np.shape[0])

    masks = _extract_visibility_cpu(
        raw_g0_source,
        initial_visibility,
        n,
    )

    tree = _build_tree(raw_xyz_np)
    spacing = _estimate_spacing(
        tree,
        raw_xyz_np,
        max_samples=max(
            10000,
            _cfg_int(cfg, "stage1_surface_prior_spacing_samples", 200000),
        ),
    )

    return {
        "xyz": raw_xyz_cpu,
        "xyz_np": raw_xyz_np,
        "tree": tree,
        "count": n,
        "spacing": float(max(spacing, 1e-8)),
        **masks,
    }


def surface_prior_summary(prior: Dict[str, Any]) -> str:
    n = int(prior.get("count", 0))
    vis = int(prior.get("visible_mask", torch.zeros(0)).sum().item())
    hid = int(prior.get("hidden_mask", torch.zeros(0)).sum().item())
    unc = int(prior.get("uncertain_mask", torch.zeros(0)).sum().item())
    out = int(prior.get("outside_mask", torch.zeros(0)).sum().item())
    spacing = float(prior.get("spacing", 0.0))
    return (
        f"Raw-G0 surface prior: total={n:,} | visible={vis:,} | "
        f"hidden={hid:,} | uncertain={unc:,} | outside={out:,} | "
        f"spacing={spacing:.8g}"
    )


def _choose_sample_indices(
    n: int,
    max_points: int,
    device: torch.device,
    iteration: int,
) -> torch.Tensor:
    """
    Deterministic-ish low-overhead sampling without allocating randperm(N)
    for million-point clouds.
    """
    if n <= max_points:
        return torch.arange(n, device=device, dtype=torch.long)

    # Evenly spaced coverage plus an iteration-dependent cyclic offset.
    step = float(n) / float(max_points)
    base = torch.arange(max_points, device=device, dtype=torch.float32)
    offset = int((iteration * 15485863) % max(n, 1))
    ids = torch.floor(base * step).long()
    ids = (ids + offset) % n
    return ids


def _query_nearest(tree, query_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        distance, index = tree.query(query_np, k=1, workers=-1)
    except TypeError:
        distance, index = tree.query(query_np, k=1)

    return (
        np.asarray(distance, dtype=np.float32),
        np.asarray(index, dtype=np.int64),
    )


def _local_pca_normals(
    tree,
    raw_xyz_np: np.ndarray,
    center_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    PCA normal = eigenvector corresponding to the smallest covariance eigenvalue.
    Normals are sign-ambiguous, which is fine for point-to-plane distance.
    """
    if center_indices.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    k = max(3, min(int(k), int(raw_xyz_np.shape[0])))

    centers = raw_xyz_np[center_indices]

    try:
        _, nbr_idx = tree.query(centers, k=k, workers=-1)
    except TypeError:
        _, nbr_idx = tree.query(centers, k=k)

    nbr_idx = np.asarray(nbr_idx, dtype=np.int64)
    if nbr_idx.ndim == 1:
        nbr_idx = nbr_idx[:, None]

    neighborhoods = raw_xyz_np[nbr_idx]
    mean = neighborhoods.mean(axis=1, keepdims=True)
    centered = neighborhoods - mean

    cov = np.einsum(
        "nki,nkj->nij",
        centered,
        centered,
        optimize=True,
    )
    denom = max(int(neighborhoods.shape[1]) - 1, 1)
    cov = cov / float(denom)

    _, eigenvectors = np.linalg.eigh(cov)
    normals = eigenvectors[:, :, 0].astype(np.float32)

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    return normals


def _raw_visibility_weights(
    prior: Dict[str, Any],
    raw_indices: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    device = raw_indices.device
    m = int(raw_indices.numel())

    weights = torch.full(
        (m,),
        _cfg_float(cfg, "stage1_surface_prior_visible_weight", 1.0),
        device=device,
        dtype=torch.float32,
    )

    raw_idx_cpu = raw_indices.detach().cpu()

    hidden = prior["hidden_mask"][raw_idx_cpu].to(device=device)
    uncertain = prior["uncertain_mask"][raw_idx_cpu].to(device=device)
    outside = prior["outside_mask"][raw_idx_cpu].to(device=device)

    weights = torch.where(
        hidden,
        torch.full_like(
            weights,
            _cfg_float(cfg, "stage1_surface_prior_hidden_weight", 2.0),
        ),
        weights,
    )
    weights = torch.where(
        uncertain,
        torch.full_like(
            weights,
            _cfg_float(cfg, "stage1_surface_prior_uncertain_weight", 1.5),
        ),
        weights,
    )
    weights = torch.where(
        outside,
        torch.full_like(
            weights,
            _cfg_float(cfg, "stage1_surface_prior_outside_weight", 0.5),
        ),
        weights,
    )

    return weights.clamp_min(0.0)


def raw_g0_surface_prior_loss(
    gaussians: Any,
    prior: Optional[Dict[str, Any]],
    cfg: Any,
    iteration: int,
    *,
    force: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Sample current Gaussian centres and pull them toward the permanent raw-G0
    local tangent surface.

    The nearest-neighbour / PCA association is detached.  The final loss is
    differentiable w.r.t. the current Gaussian xyz.
    """
    xyz = gaussians.get_xyz
    zero = xyz.sum() * 0.0

    diag = {
        "active": False,
        "sample_count": 0,
        "valid_count": 0,
        "valid_ratio": 0.0,
        "plane": 0.0,
        "tangent": 0.0,
        "mean_nn_distance": 0.0,
        "max_allowed_distance": 0.0,
    }

    if prior is None:
        return zero, diag

    if not _cfg_bool(cfg, "stage1_enable_surface_prior", True):
        return zero, diag

    every = max(
        1,
        _cfg_int(cfg, "stage1_surface_prior_every", 10),
    )
    if (not force) and (int(iteration) % every != 0):
        return zero, diag

    n = int(xyz.shape[0])
    if n == 0:
        return zero, diag

    max_points = max(
        128,
        _cfg_int(cfg, "stage1_surface_prior_sample_points", 8192),
    )
    sample_idx = _choose_sample_indices(
        n,
        min(max_points, n),
        xyz.device,
        int(iteration),
    )

    current_sample = xyz[sample_idx]
    query_np = (
        current_sample.detach()
        .float()
        .cpu()
        .numpy()
    )

    nearest_distance_np, raw_index_np = _query_nearest(
        prior["tree"],
        query_np,
    )

    raw_index_cpu = torch.from_numpy(raw_index_np).long()
    raw_ref = prior["xyz"][raw_index_cpu].to(
        device=xyz.device,
        dtype=xyz.dtype,
    )

    pca_k = max(
        3,
        _cfg_int(cfg, "stage1_surface_prior_knn", 12),
    )
    normal_np = _local_pca_normals(
        prior["tree"],
        prior["xyz_np"],
        raw_index_np,
        pca_k,
    )
    normals = torch.from_numpy(normal_np).to(
        device=xyz.device,
        dtype=xyz.dtype,
    )

    nearest_distance = torch.from_numpy(
        nearest_distance_np
    ).to(
        device=xyz.device,
        dtype=xyz.dtype,
    )

    spacing = max(
        float(prior.get("spacing", 1.0)),
        1e-8,
    )

    max_mult = max(
        1.0,
        _cfg_float(
            cfg,
            "stage1_surface_prior_max_distance_mult",
            6.0,
        ),
    )
    max_allowed = spacing * max_mult

    valid = (
        torch.isfinite(nearest_distance)
        & (nearest_distance <= max_allowed)
    )

    diag["active"] = True
    diag["sample_count"] = int(sample_idx.numel())
    diag["valid_count"] = int(valid.sum().item())
    diag["valid_ratio"] = float(valid.float().mean().item())
    diag["mean_nn_distance"] = float(
        nearest_distance[valid].mean().item()
        if bool(valid.any())
        else 0.0
    )
    diag["max_allowed_distance"] = float(max_allowed)

    if not bool(valid.any()):
        return zero, diag

    delta = current_sample - raw_ref
    plane_signed = (delta * normals).sum(dim=-1)
    plane_abs = plane_signed.abs()

    delta_sq = delta.square().sum(dim=-1)
    tangent_sq = (delta_sq - plane_signed.square()).clamp_min(0.0)
    tangent = torch.sqrt(tangent_sq + 1e-12)

    plane_rel = plane_abs / spacing
    tangent_rel = tangent / spacing

    beta = max(
        1e-5,
        _cfg_float(cfg, "stage1_surface_prior_smooth_l1_beta", 0.10),
    )

    plane_each = F.smooth_l1_loss(
        plane_rel,
        torch.zeros_like(plane_rel),
        beta=beta,
        reduction="none",
    )

    tangent_tolerance = max(
        0.0,
        _cfg_float(
            cfg,
            "stage1_surface_prior_tangent_tolerance_mult",
            1.50,
        ),
    )
    tangent_excess = F.relu(
        tangent_rel - tangent_tolerance
    )
    tangent_each = tangent_excess.square()

    raw_idx_device = torch.from_numpy(
        raw_index_np
    ).to(device=xyz.device, dtype=torch.long)

    weights = _raw_visibility_weights(
        prior,
        raw_idx_device,
        cfg,
    ).to(dtype=xyz.dtype)

    valid_w = weights[valid]
    denom = valid_w.sum().clamp_min(1e-8)

    plane_loss = (
        plane_each[valid] * valid_w
    ).sum() / denom

    tangent_loss = (
        tangent_each[valid] * valid_w
    ).sum() / denom

    tangent_lambda = max(
        0.0,
        _cfg_float(
            cfg,
            "stage1_surface_prior_tangent_fraction",
            0.15,
        ),
    )

    total = plane_loss + tangent_lambda * tangent_loss

    diag["plane"] = float(plane_loss.detach().item())
    diag["tangent"] = float(tangent_loss.detach().item())

    return total, diag
