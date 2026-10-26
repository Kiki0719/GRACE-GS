"""
visibility.py
=============

Visibility-aware Gaussian classification for single-view Stage1 refinement.

Purpose
-------
Given a fixed Stage1 camera and Gaussian centers, classify each Gaussian as:

    VISIBLE   : confidently visible from the current camera
    HIDDEN    : projected inside the image but occluded by a closer Gaussian
    UNCERTAIN : near the front surface / depth ambiguity / border ambiguity
    OUTSIDE   : behind camera or outside the image

This module is intentionally conservative:
- HIDDEN Gaussians should normally be protected from 2D-driven pruning.
- UNCERTAIN Gaussians should also normally be protected.
- Only confidently VISIBLE Gaussians should be aggressively pruned according
  to RGB / mask / silhouette evidence.

This file does NOT modify Gaussians.
It only returns masks and diagnostics.

Designed for E-D3DGS / 3DGS-style Camera objects.

Typical usage
-------------
from Stage1.refinement_utils.visibility import classify_gaussian_visibility

vis = classify_gaussian_visibility(
    gaussians=gaussians,
    camera=camera,
    depth_abs_margin=0.01,
    depth_rel_margin=0.01,
    pixel_radius=1,
)

visible_mask   = vis["visible_mask"]
hidden_mask    = vis["hidden_mask"]
uncertain_mask = vis["uncertain_mask"]
outside_mask   = vis["outside_mask"]

# Conservative pruning:
safe_prune_mask = original_bad_mask & visible_mask
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Public result container
# -------------------------------------------------------------------------

@dataclass
class VisibilityResult:
    visible_mask: torch.Tensor
    hidden_mask: torch.Tensor
    uncertain_mask: torch.Tensor
    outside_mask: torch.Tensor

    uv: torch.Tensor
    depth: torch.Tensor
    front_depth: torch.Tensor
    depth_gap: torch.Tensor

    width: int
    height: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "visible_mask": self.visible_mask,
            "hidden_mask": self.hidden_mask,
            "uncertain_mask": self.uncertain_mask,
            "outside_mask": self.outside_mask,
            "uv": self.uv,
            "depth": self.depth,
            "front_depth": self.front_depth,
            "depth_gap": self.depth_gap,
            "width": self.width,
            "height": self.height,
        }


# -------------------------------------------------------------------------
# Camera helpers
# -------------------------------------------------------------------------

def _get_hw(camera: Any) -> Tuple[int, int]:
    """
    Extract image width/height from common 3DGS Camera attributes.
    """
    width_candidates = (
        "image_width",
        "width",
        "W",
    )
    height_candidates = (
        "image_height",
        "height",
        "H",
    )

    width = None
    height = None

    for name in width_candidates:
        if hasattr(camera, name):
            width = int(getattr(camera, name))
            break

    for name in height_candidates:
        if hasattr(camera, name):
            height = int(getattr(camera, name))
            break

    if width is None or height is None:
        if hasattr(camera, "original_image"):
            img = camera.original_image
            if torch.is_tensor(img) and img.ndim >= 2:
                height = int(img.shape[-2])
                width = int(img.shape[-1])

    if width is None or height is None:
        raise AttributeError(
            "Cannot infer camera image size. Expected camera.image_width/"
            "image_height, width/height, W/H, or original_image."
        )

    return width, height


def _camera_center(camera: Any, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    Try to obtain camera center in world coordinates.
    """
    if hasattr(camera, "camera_center"):
        c = getattr(camera, "camera_center")
        if torch.is_tensor(c):
            return c.to(device=device, dtype=dtype).reshape(3)

    # Fallback from R/T if available.
    if hasattr(camera, "R") and hasattr(camera, "T"):
        R = torch.as_tensor(getattr(camera, "R"), device=device, dtype=dtype)
        T = torch.as_tensor(getattr(camera, "T"), device=device, dtype=dtype).reshape(3)

        # In standard 3DGS, R is stored transposed relative to the usual
        # world-to-camera rotation convention. We cannot safely assume one
        # convention here, so this helper is only used as a fallback.
        try:
            return -(R @ T)
        except Exception:
            return None

    return None


def _extract_full_proj_transform(
    camera: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return the full world -> clip transform.

    Standard graphdeco/3DGS Camera provides:
        camera.full_proj_transform

    In the original implementation this matrix is stored in a transposed
    convention and points are multiplied as homogeneous row-vectors.
    """
    if not hasattr(camera, "full_proj_transform"):
        raise AttributeError(
            "Camera must expose `full_proj_transform` for visibility.py. "
            "This is available in the standard 3DGS / E-D3DGS Camera class."
        )

    M = getattr(camera, "full_proj_transform")
    if not torch.is_tensor(M):
        M = torch.as_tensor(M)

    M = M.to(device=device, dtype=dtype)

    if M.shape != (4, 4):
        raise ValueError(
            f"camera.full_proj_transform must have shape (4,4), got {tuple(M.shape)}"
        )

    return M


# -------------------------------------------------------------------------
# Gaussian helpers
# -------------------------------------------------------------------------

def _extract_xyz(gaussians: Any) -> torch.Tensor:
    """
    Extract Nx3 Gaussian centers from common GaussianModel interfaces.
    """
    candidates = (
        "get_xyz",
        "_xyz",
        "xyz",
    )

    for name in candidates:
        if hasattr(gaussians, name):
            value = getattr(gaussians, name)
            value = value() if callable(value) else value

            if torch.is_tensor(value):
                if value.ndim != 2 or value.shape[-1] != 3:
                    raise ValueError(
                        f"Gaussian xyz must be Nx3, got {tuple(value.shape)} from `{name}`"
                    )
                return value

    if torch.is_tensor(gaussians):
        if gaussians.ndim == 2 and gaussians.shape[-1] == 3:
            return gaussians

    raise AttributeError(
        "Cannot extract Gaussian xyz. Expected `get_xyz`, `_xyz`, `xyz`, "
        "or an Nx3 Tensor."
    )


# -------------------------------------------------------------------------
# Projection
# -------------------------------------------------------------------------

@torch.no_grad()
def project_gaussian_centers(
    xyz: torch.Tensor,
    camera: Any,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Project Gaussian centers using camera.full_proj_transform.

    Returns
    -------
    uv:
        Nx2 floating-point pixel coordinates.
    ndc:
        Nx3 normalized device coordinates.
    clip_w:
        Nx1 clip-space w.
    valid_w:
        Boolean mask where clip-space projection is numerically valid.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be Nx3, got {tuple(xyz.shape)}")

    device = xyz.device
    dtype = xyz.dtype
    width, height = _get_hw(camera)
    M = _extract_full_proj_transform(camera, device, dtype)

    ones = torch.ones((xyz.shape[0], 1), device=device, dtype=dtype)
    xyz_h = torch.cat([xyz, ones], dim=-1)

    # Standard 3DGS uses row-vector convention with the transposed matrix.
    clip = xyz_h @ M

    w = clip[:, 3:4]
    valid_w = torch.isfinite(w[:, 0]) & (w[:, 0].abs() > eps)

    safe_w = torch.where(
        w.abs() > eps,
        w,
        torch.full_like(w, eps),
    )

    ndc = clip[:, :3] / safe_w

    x = ((ndc[:, 0] + 1.0) * 0.5) * float(width - 1)
    y = ((ndc[:, 1] + 1.0) * 0.5) * float(height - 1)

    uv = torch.stack([x, y], dim=-1)

    return {
        "uv": uv,
        "ndc": ndc,
        "clip_w": w[:, 0],
        "valid_w": valid_w,
        "width": torch.tensor(width, device=device),
        "height": torch.tensor(height, device=device),
    }


@torch.no_grad()
def compute_camera_depth(
    xyz: torch.Tensor,
    camera: Any,
) -> torch.Tensor:
    """
    Compute a monotonic camera-distance depth.

    We use Euclidean distance to camera center when available. This is robust
    for visibility ordering in a fixed perspective camera: along the same
    image ray, a closer point has smaller distance.

    If camera_center is unavailable, falls back to abs(clip_w), which is less
    geometrically clean but still useful as an ordering cue.
    """
    device = xyz.device
    dtype = xyz.dtype

    center = _camera_center(camera, device, dtype)
    if center is not None:
        depth = torch.linalg.norm(xyz - center[None, :], dim=-1)
        return depth

    proj = project_gaussian_centers(xyz, camera)
    return proj["clip_w"].abs()


# -------------------------------------------------------------------------
# Raster-style front-depth construction
# -------------------------------------------------------------------------

def _scatter_amin(
    index: torch.Tensor,
    src: torch.Tensor,
    size: int,
    fill_value: float,
) -> torch.Tensor:
    """
    Torch-compatible scatter amin helper.
    """
    out = torch.full(
        (size,),
        float(fill_value),
        dtype=src.dtype,
        device=src.device,
    )

    if index.numel() == 0:
        return out

    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(
            0,
            index,
            src,
            reduce="amin",
            include_self=True,
        )
        return out

    # Very old PyTorch fallback.
    unique_idx = torch.unique(index)
    for idx in unique_idx.tolist():
        mask = index == idx
        out[idx] = src[mask].min()

    return out


@torch.no_grad()
def build_front_depth_map(
    uv: torch.Tensor,
    depth: torch.Tensor,
    inside_mask: torch.Tensor,
    width: int,
    height: int,
    pixel_radius: int = 1,
) -> torch.Tensor:
    """
    Build an approximate front-surface depth map from projected Gaussian centers.

    Each Gaussian center contributes to a small (2r+1)x(2r+1) pixel window.
    Using r=1 or 2 makes the visibility classifier less brittle than a strict
    single-pixel test.

    Returns
    -------
    front_depth_map:
        HxW. Pixels without any projected Gaussian contain +inf.
    """
    device = uv.device
    dtype = depth.dtype

    if pixel_radius < 0:
        raise ValueError("pixel_radius must be >= 0")

    ids = torch.nonzero(inside_mask, as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
        return torch.full(
            (height, width),
            float("inf"),
            dtype=dtype,
            device=device,
        )

    u0 = torch.round(uv[ids, 0]).long()
    v0 = torch.round(uv[ids, 1]).long()
    d0 = depth[ids]

    offsets = []
    for dy in range(-pixel_radius, pixel_radius + 1):
        for dx in range(-pixel_radius, pixel_radius + 1):
            offsets.append((dx, dy))

    all_lin = []
    all_depth = []

    for dx, dy in offsets:
        u = u0 + dx
        v = v0 + dy

        valid = (
            (u >= 0) &
            (u < width) &
            (v >= 0) &
            (v < height)
        )

        if not valid.any():
            continue

        lin = v[valid] * width + u[valid]
        all_lin.append(lin)
        all_depth.append(d0[valid])

    if not all_lin:
        return torch.full(
            (height, width),
            float("inf"),
            dtype=dtype,
            device=device,
        )

    lin = torch.cat(all_lin, dim=0)
    dep = torch.cat(all_depth, dim=0)

    flat = _scatter_amin(
        index=lin,
        src=dep,
        size=width * height,
        fill_value=float("inf"),
    )

    return flat.view(height, width)


@torch.no_grad()
def _local_min_depth_map(
    depth_map: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """
    Local minimum of the finite front-depth map.

    Useful near Gaussian splat boundaries and small holes.
    """
    if radius <= 0:
        return depth_map

    finite = torch.isfinite(depth_map)

    safe = torch.where(
        finite,
        depth_map,
        torch.full_like(depth_map, 1e10),
    )

    x = -safe[None, None, :, :]
    k = 2 * radius + 1
    pooled = -F.max_pool2d(
        x,
        kernel_size=k,
        stride=1,
        padding=radius,
    )[0, 0]

    pooled = torch.where(
        pooled >= 1e9,
        torch.full_like(pooled, float("inf")),
        pooled,
    )

    return pooled


# -------------------------------------------------------------------------
# Main classifier
# -------------------------------------------------------------------------

@torch.no_grad()
def classify_gaussian_visibility(
    gaussians: Any,
    camera: Any,
    *,
    depth_abs_margin: float = 0.01,
    depth_rel_margin: float = 0.01,
    uncertain_multiplier: float = 2.5,
    pixel_radius: int = 1,
    local_depth_radius: int = 1,
    border_margin_px: int = 2,
    protect_border: bool = True,
) -> Dict[str, Any]:
    """
    Classify Gaussian centers as visible / hidden / uncertain / outside.

    Parameters
    ----------
    gaussians:
        GaussianModel or Nx3 Tensor.
    camera:
        Standard 3DGS/E-D3DGS Camera with `full_proj_transform` and image size.

    depth_abs_margin:
        Absolute depth tolerance used to regard a Gaussian as front-visible.

    depth_rel_margin:
        Relative tolerance multiplied by the local front depth.

    uncertain_multiplier:
        Defines the intermediate ambiguity region:
            visible if gap <= margin
            uncertain if margin < gap <= uncertain_multiplier * margin
            hidden if gap > uncertain_multiplier * margin

    pixel_radius:
        Radius used when splatting Gaussian centers into the approximate
        front-depth map.

    local_depth_radius:
        Radius used to locally smooth front-depth by minimum pooling.

    border_margin_px:
        Pixels near the image border can be marked uncertain.

    protect_border:
        If True, border Gaussians become UNCERTAIN instead of VISIBLE so
        pruning remains conservative.

    Returns
    -------
    Dictionary containing masks and diagnostics.
    """
    xyz = _extract_xyz(gaussians)
    device = xyz.device
    dtype = xyz.dtype

    proj = project_gaussian_centers(xyz, camera)
    uv = proj["uv"]
    ndc = proj["ndc"]
    valid_w = proj["valid_w"]

    width, height = _get_hw(camera)

    depth = compute_camera_depth(xyz, camera)

    finite = (
        torch.isfinite(uv).all(dim=-1) &
        torch.isfinite(depth) &
        torch.isfinite(ndc).all(dim=-1)
    )

    # clip_w > 0 is normally in front for standard 3DGS perspective projection.
    # To avoid over-trusting conventions, we also require a valid projection and
    # NDC z in a broad range.
    in_front = valid_w & finite & (proj["clip_w"] > 0)

    inside_image = (
        in_front &
        (uv[:, 0] >= 0.0) &
        (uv[:, 0] <= float(width - 1)) &
        (uv[:, 1] >= 0.0) &
        (uv[:, 1] <= float(height - 1))
    )

    outside_mask = ~inside_image

    front_depth_map = build_front_depth_map(
        uv=uv,
        depth=depth,
        inside_mask=inside_image,
        width=width,
        height=height,
        pixel_radius=pixel_radius,
    )

    if local_depth_radius > 0:
        front_depth_map = _local_min_depth_map(
            front_depth_map,
            radius=local_depth_radius,
        )

    u_idx = torch.round(uv[:, 0]).long().clamp(0, width - 1)
    v_idx = torch.round(uv[:, 1]).long().clamp(0, height - 1)

    front_depth = front_depth_map[v_idx, u_idx]

    no_reference = ~torch.isfinite(front_depth)

    # Positive = Gaussian is behind the estimated front surface.
    depth_gap = depth - front_depth

    margin = (
        float(depth_abs_margin) +
        float(depth_rel_margin) * torch.clamp(front_depth, min=0.0)
    )

    visible_mask = (
        inside_image &
        (~no_reference) &
        (depth_gap <= margin)
    )

    uncertain_mask = (
        inside_image &
        (
            no_reference |
            (
                (depth_gap > margin) &
                (depth_gap <= float(uncertain_multiplier) * margin)
            )
        )
    )

    hidden_mask = (
        inside_image &
        (~no_reference) &
        (depth_gap > float(uncertain_multiplier) * margin)
    )

    # Conservative border protection.
    if protect_border and border_margin_px > 0:
        b = int(border_margin_px)

        border = (
            inside_image &
            (
                (uv[:, 0] < b) |
                (uv[:, 0] > (width - 1 - b)) |
                (uv[:, 1] < b) |
                (uv[:, 1] > (height - 1 - b))
            )
        )

        uncertain_mask = uncertain_mask | (visible_mask & border)
        visible_mask = visible_mask & (~border)

    # Enforce disjoint masks.
    visible_mask = visible_mask & (~outside_mask)
    uncertain_mask = uncertain_mask & (~outside_mask) & (~visible_mask)
    hidden_mask = hidden_mask & (~outside_mask) & (~visible_mask) & (~uncertain_mask)

    # Safety: any inside Gaussian not assigned due to numerical edge cases
    # becomes uncertain, never silently prunable.
    assigned_inside = visible_mask | hidden_mask | uncertain_mask
    unresolved_inside = inside_image & (~assigned_inside)
    uncertain_mask = uncertain_mask | unresolved_inside

    result = VisibilityResult(
        visible_mask=visible_mask,
        hidden_mask=hidden_mask,
        uncertain_mask=uncertain_mask,
        outside_mask=outside_mask,
        uv=uv,
        depth=depth,
        front_depth=front_depth,
        depth_gap=depth_gap,
        width=width,
        height=height,
    )

    out = result.as_dict()

    n = int(xyz.shape[0])
    out["num_gaussians"] = n
    out["num_visible"] = int(visible_mask.sum().item())
    out["num_hidden"] = int(hidden_mask.sum().item())
    out["num_uncertain"] = int(uncertain_mask.sum().item())
    out["num_outside"] = int(outside_mask.sum().item())

    out["ratio_visible"] = out["num_visible"] / max(n, 1)
    out["ratio_hidden"] = out["num_hidden"] / max(n, 1)
    out["ratio_uncertain"] = out["num_uncertain"] / max(n, 1)
    out["ratio_outside"] = out["num_outside"] / max(n, 1)

    return out


# -------------------------------------------------------------------------
# Conservative pruning helper
# -------------------------------------------------------------------------

@torch.no_grad()
def gate_prune_mask_with_visibility(
    proposed_prune_mask: torch.Tensor,
    visibility: Dict[str, Any],
    *,
    allow_uncertain_prune: bool = False,
    allow_hidden_prune: bool = False,
    allow_outside_prune: bool = False,
) -> torch.Tensor:
    """
    Gate an existing prune mask using the visibility classification.

    Default behavior:
        ONLY confidently visible Gaussians can be pruned.

    This is the recommended mode for single-view Stage1 refinement.

    Example
    -------
    bad_mask = opacity_mask | outside_silhouette_mask

    prune_mask = gate_prune_mask_with_visibility(
        bad_mask,
        visibility,
    )
    """
    if proposed_prune_mask.dtype != torch.bool:
        proposed_prune_mask = proposed_prune_mask.bool()

    keep_allowed = visibility["visible_mask"].clone()

    if allow_uncertain_prune:
        keep_allowed |= visibility["uncertain_mask"]

    if allow_hidden_prune:
        keep_allowed |= visibility["hidden_mask"]

    if allow_outside_prune:
        keep_allowed |= visibility["outside_mask"]

    if keep_allowed.shape != proposed_prune_mask.shape:
        raise ValueError(
            "Visibility mask shape does not match proposed prune mask: "
            f"{tuple(keep_allowed.shape)} vs {tuple(proposed_prune_mask.shape)}"
        )

    return proposed_prune_mask & keep_allowed


# -------------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------------

def visibility_summary(visibility: Dict[str, Any]) -> str:
    """
    Human-readable summary for Stage1 logs.
    """
    n = int(visibility["num_gaussians"])
    nv = int(visibility["num_visible"])
    nh = int(visibility["num_hidden"])
    nu = int(visibility["num_uncertain"])
    no = int(visibility["num_outside"])

    return (
        f"Visibility: total={n:,} | "
        f"visible={nv:,} ({100.0 * nv / max(n,1):.2f}%) | "
        f"hidden={nh:,} ({100.0 * nh / max(n,1):.2f}%) | "
        f"uncertain={nu:,} ({100.0 * nu / max(n,1):.2f}%) | "
        f"outside={no:,} ({100.0 * no / max(n,1):.2f}%)"
    )
