"""
Shared helpers for Gaussian refinement.

Contains:
- config helpers
- projection / depth / G0 regularization
- RGB / mask / losses / error maps
- silhouette constraints
- diagnostics

The training loop stays in ../gaussian_refinement.py.
Topology and densification stay in densification.py.
"""

from pathlib import Path
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gaussian_renderer import render
from utils.loss_utils import ssim
from gaussian_utils.visibility import classify_gaussian_visibility


# ============================================================================
# CONFIG
# ============================================================================

def cfg_float(cfg, name, default):
    return float(getattr(cfg, name, default))

def cfg_int(cfg, name, default):
    return int(getattr(cfg, name, default))

def cfg_bool(cfg, name, default):
    value = getattr(cfg, name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)

def make_stage1_cfg(cfg, output_dir, g0_path, iterations, camera_file=None):
    values = cfg._asdict().copy()
    values["model_path"] = str(Path(output_dir).expanduser().resolve())
    values["initial_gaussian_ply"] = str(Path(g0_path).expanduser().resolve())
    if camera_file is not None:
        values["camera_file"] = str(Path(camera_file).expanduser().resolve())
    values["train_select"] = [0]
    values["test_select"] = []
    values["event_assist"] = False
    # IMPORTANT:
    # This does NOT mean the real target background is assumed black.
    # It only makes Scene / renderer use the same neutral background that we
    # explicitly composite the target object onto.
    values["white_background"] = False
    values["position_lr_init"] = cfg_float(cfg, "stage1_position_lr_init", 2e-5)
    values["position_lr_final"] = cfg_float(cfg, "stage1_position_lr_final", 2e-6)
    values["position_lr_delay_mult"] = cfg_float(cfg, "stage1_position_lr_delay_mult", 0.01)
    values["position_lr_max_steps"] = int(iterations)
    # Fixed single view: optimize only SH DC color; higher-order SH is frozen.
    values["feature_lr"] = cfg_float(cfg, "stage1_color_lr", 0.0025)
    values["opacity_lr"] = cfg_float(cfg, "stage1_opacity_refine_lr", 0.005)
    values["scaling_lr"] = cfg_float(cfg, "stage1_scaling_lr", 5e-5)
    values["rotation_lr"] = cfg_float(cfg, "stage1_rotation_lr", 5e-5)
    Config = namedtuple("Stage1Config", values.keys())
    return Config(**values)

# ============================================================================
# GEOMETRY / DEPTH / G0 REGULARIZATION
# ============================================================================

@torch.no_grad()

def project_gaussians(gaussians, camera):
    """
    Projection follows the row-vector convention used by standard 3DGS camera
    matrices in this repository.
    """
    xyz = gaussians.get_xyz
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=-1)
    proj = camera.full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
    clip = xyz_h @ proj
    clip_w = clip[:, 3]
    safe_w = torch.where(clip_w.abs() < 1e-8, torch.full_like(clip_w, 1e-8), clip_w)
    ndc = clip[:, :3] / safe_w[:, None]
    width = int(camera.image_width)
    height = int(camera.image_height)
    x = (ndc[:, 0] + 1.0) * 0.5 * (width - 1)
    y = (ndc[:, 1] + 1.0) * 0.5 * (height - 1)
    valid = (
        torch.isfinite(x)
        & torch.isfinite(y)
        & torch.isfinite(ndc[:, 2])
        & torch.isfinite(clip_w)
        & (clip_w.abs() > 1e-8)
        & (x >= 0.0)
        & (x <= width - 1)
        & (y >= 0.0)
        & (y <= height - 1)
    )
    px = x.round().long().clamp(0, width - 1)
    py = y.round().long().clamp(0, height - 1)
    return x, y, px, py, ndc[:, 2], valid

@torch.no_grad()

def unproject_pixels_at_ndc_depth(camera, pixel_xy, ndc_z, dtype, device):
    """
    Back-project a 2D target pixel at an EXISTING nearby parent's NDC depth.
    This does not claim that monocular RGB provides true metric depth.
    It deliberately inherits depth from G0 as a conservative 3D prior.
    """
    width = int(camera.image_width)
    height = int(camera.image_height)
    x_ndc = (pixel_xy[:, 0] / max(width - 1, 1) * 2.0 - 1.0)
    y_ndc = (pixel_xy[:, 1] / max(height - 1, 1) * 2.0 - 1.0)
    ones = torch.ones_like(x_ndc)
    ndc_h = torch.stack([x_ndc, y_ndc, ndc_z, ones], dim=-1).to(device=device, dtype=dtype)
    proj = camera.full_proj_transform.to(device=device, dtype=dtype)
    inv_proj = torch.linalg.inv(proj)
    world_h = ndc_h @ inv_proj
    w = world_h[:, 3:4]
    safe_w = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8), w)
    return world_h[:, :3] / safe_w

def _package_alpha(package, height, width):
    """
    Return differentiable alpha directly from the renderer if this fork exposes
    it. Do NOT detach here.
    """
    for key in ("alpha", "rendered_alpha", "accumulation", "opacity"):
        value = package.get(key, None)
        if not torch.is_tensor(value):
            continue
        v = value
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]
        if v.ndim == 2 and tuple(v.shape) == (height, width):
            return v.clamp(0.0, 1.0)
    return None

def differentiable_render_alpha(camera, gaussians, args, black_package):
    """
    Prefer direct renderer alpha.
    Fallback:
        main render uses black background:
            C_black = C_object_premultiplied
        render once more with white:
            C_white = C_object_premultiplied + (1-alpha)
        Therefore:
            alpha = 1 - mean(C_white - C_black)
    This fallback remains differentiable.
    """
    height = int(camera.image_height)
    width = int(camera.image_width)
    direct = _package_alpha(black_package, height, width)
    if direct is not None:
        return direct, "renderer"
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    white_background = torch.ones(3, dtype=dtype, device=device)
    white_package = render(camera, gaussians, args, white_background, 0.0, 0.0, 0.0, args.is_6dof)
    black_rgb = black_package["render"]
    white_rgb = white_package["render"]
    alpha = (1.0 - (white_rgb - black_rgb).mean(dim=0)).clamp(0.0, 1.0)
    return alpha, "black-white-pair"

@torch.no_grad()
def _zero_optimizer_rows(gaussians, param, row_mask):
    """Clear Adam moments for rows changed by a hard geometric constraint."""
    if getattr(gaussians, "optimizer", None) is None:
        return
    state = gaussians.optimizer.state.get(param, None)
    if not isinstance(state, dict):
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        buf = state.get(key, None)
        if isinstance(buf, torch.Tensor) and buf.shape == param.shape:
            buf[row_mask] = 0

def _extract_depth_map(package, height, width):
    """
    Try to obtain a differentiable per-pixel depth map from this repository's
    renderer output without assuming one exact key name.

    Returns:
        depth: HW tensor or None
        source: string
    """
    if not isinstance(package, dict):
        return None, "none"

    for key in (
        "depth",
        "rendered_depth",
        "render_depth",
        "depth_map",
        "expected_depth",
        "median_depth",
    ):
        value = package.get(key, None)
        if not torch.is_tensor(value):
            continue

        d = value
        if d.ndim == 3 and d.shape[0] == 1:
            d = d[0]
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]

        if d.ndim == 2 and tuple(d.shape) == (height, width):
            return d, key

    return None, "none"

def _differentiable_ndc_depth(xyz, camera):
    """
    Differentiable fallback depth for Gaussian centres.

    This is used only when the renderer does not expose a differentiable
    per-pixel depth map. It preserves G0 depth along the fixed camera view
    direction, but is a per-Gaussian fallback rather than a rasterized depth map.
    """
    ones = torch.ones(
        (xyz.shape[0], 1),
        dtype=xyz.dtype,
        device=xyz.device,
    )
    xyz_h = torch.cat([xyz, ones], dim=-1)
    proj = camera.full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
    clip = xyz_h @ proj
    w = clip[:, 3]
    safe_w = torch.where(
        w.abs() < 1e-8,
        torch.full_like(w, 1e-8),
        w,
    )
    return clip[:, 2] / safe_w

@torch.no_grad()
def build_g0_depth_reference(
    initial_package,
    initial_alpha,
    target_mask,
    gaussians,
    camera,
    height,
    width,
    cfg,
):
    """
    Freeze depth from the ORIGINAL loaded G0 before Stage-1 optimization.

    Preferred reference:
        rasterized G0 depth map returned by the renderer.

    Fallback:
        per-Gaussian NDC depth of the original G0 centres.
    """
    depth_map, depth_source = _extract_depth_map(
        initial_package,
        height,
        width,
    )

    alpha_min = cfg_float(cfg, "stage1_depth_alpha_min", 0.10)

    if depth_map is not None:
        ref = depth_map.detach().float()
        valid = (
            torch.isfinite(ref)
            & (initial_alpha.detach() >= alpha_min)
            & (target_mask.detach() > 0.5)
        )

        # Reject degenerate reference maps.
        if bool(valid.any()):
            vals = ref[valid]
            spread = (vals.max() - vals.min()).abs()
            if torch.isfinite(spread) and float(spread.item()) > 1e-8:
                scale = vals.abs().median().clamp_min(1e-6)
                return {
                    "mode": "rendered_depth",
                    "depth": ref,
                    "mask": valid,
                    "scale": scale,
                    "source": depth_source,
                    "xyz_count": int(gaussians.get_xyz.shape[0]),
                }

    # Robust fallback if this renderer fork does not emit a depth image.
    xyz0 = gaussians.get_xyz.detach().clone()
    z0 = _differentiable_ndc_depth(xyz0, camera).detach()
    finite = torch.isfinite(z0)
    if bool(finite.any()):
        z_scale = z0[finite].abs().median().clamp_min(1e-6)
    else:
        z_scale = xyz0.new_tensor(1.0)

    return {
        "mode": "gaussian_ndc",
        "depth": z0,
        "mask": finite,
        "scale": z_scale,
        "source": "gaussian_ndc_fallback",
        "xyz_count": int(xyz0.shape[0]),
    }

def g0_depth_preservation_loss(
    gaussians,
    camera,
    current_package,
    current_alpha,
    depth_reference,
    target_mask,
    cfg,
):
    """
    Preserve depth of the original G0 while still allowing small corrections.

    Preferred pixel-space loss:
        SmoothL1((D_cur - D_G0) / s) on pixels where the original G0 was visible.

    Fallback:
        SmoothL1((z_cur - z_G0) / s) over Gaussian centres if the renderer
        exposes no depth map.

    The reference is detached; gradients flow only into the current GS.
    """
    zero = gaussians.get_xyz.sum() * 0.0
    if depth_reference is None:
        return zero, "none", 0.0

    mode = depth_reference.get("mode", "none")
    scale = depth_reference["scale"].to(
        device=gaussians.get_xyz.device,
        dtype=gaussians.get_xyz.dtype,
    ).clamp_min(1e-6)

    if mode == "rendered_depth":
        h, w = target_mask.shape
        current_depth, current_source = _extract_depth_map(
            current_package,
            h,
            w,
        )
        if current_depth is None:
            return zero, "rendered_depth_missing_current", 0.0

        ref = depth_reference["depth"].to(
            device=current_depth.device,
            dtype=current_depth.dtype,
        )
        valid = depth_reference["mask"].to(device=current_depth.device).bool()
        valid = (
            valid
            & torch.isfinite(current_depth)
            & torch.isfinite(ref)
            & (target_mask > 0.5)
            & (
                current_alpha
                >= cfg_float(cfg, "stage1_depth_current_alpha_min", 0.05)
            )
        )

        if not bool(valid.any()):
            return zero, current_source, 0.0

        residual = (current_depth[valid] - ref[valid]) / scale.to(current_depth.dtype)
        beta = max(1e-5, cfg_float(cfg, "stage1_depth_smooth_l1_beta", 0.02))
        loss = F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            beta=beta,
            reduction="mean",
        )
        valid_ratio = float(valid.float().mean().detach().item())
        return loss, current_source, valid_ratio

    if mode == "gaussian_ndc":
        # This fallback requires unchanged row correspondence with the original G0.
        n0 = int(depth_reference.get("xyz_count", -1))
        n = int(gaussians.get_xyz.shape[0])
        if n != n0:
            return zero, "gaussian_ndc_topology_changed", 0.0

        z_cur = _differentiable_ndc_depth(gaussians.get_xyz, camera)
        z_ref = depth_reference["depth"].to(
            device=z_cur.device,
            dtype=z_cur.dtype,
        )
        valid = depth_reference["mask"].to(device=z_cur.device).bool()
        valid &= torch.isfinite(z_cur) & torch.isfinite(z_ref)

        if not bool(valid.any()):
            return zero, "gaussian_ndc", 0.0

        residual = (z_cur[valid] - z_ref[valid]) / scale.to(z_cur.dtype)
        beta = max(1e-5, cfg_float(cfg, "stage1_depth_smooth_l1_beta", 0.02))
        loss = F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            beta=beta,
            reduction="mean",
        )
        valid_ratio = float(valid.float().mean().detach().item())
        return loss, "gaussian_ndc", valid_ratio

    return zero, "none", 0.0

@torch.no_grad()
def normalized_depth_for_save(depth, mask):
    """Robust [0,1] visualization for diagnostics only."""
    if depth is None:
        return None
    valid = mask.bool() & torch.isfinite(depth)
    if not bool(valid.any()):
        return torch.zeros_like(depth)
    vals = depth[valid]
    lo = torch.quantile(vals, 0.02)
    hi = torch.quantile(vals, 0.98)
    denom = (hi - lo).clamp_min(1e-6)
    out = ((depth - lo) / denom).clamp(0.0, 1.0)
    out = torch.where(valid, out, torch.zeros_like(out))
    return out

@torch.no_grad()
def build_surface_regularizer_reference(xyz_reference, cfg, previous_reference=None):
    """
    Build a fixed spatial reference and a KNN graph for Anchor/Smooth losses.

    - At initialization, xyz_reference is the original G0, so the anchor is a
      true G0 anchor.
    - If topology only APPENDS new Gaussians, previous reference positions are
      retained exactly and only the newly appended points use their birth
      positions as reference.
    - If pruning/reindexing reduces point count, the correspondence is no longer
      guaranteed, so the current surface is safely adopted as a new reference.
      This avoids applying an anchor to the wrong Gaussian after prune_points().
    """
    xyz = xyz_reference.detach().contiguous()
    n = int(xyz.shape[0])
    if n == 0:
        return {
            "xyz": xyz.clone(),
            "edge_src": torch.empty(0, dtype=torch.long, device=xyz.device),
            "edge_dst": torch.empty(0, dtype=torch.long, device=xyz.device),
            "spacing": xyz.new_tensor(1.0),
            "count": 0,
        }

    if previous_reference is not None:
        prev_xyz = previous_reference.get("xyz", None)
        prev_count = int(previous_reference.get("count", 0))
        # Appending preserves the old anchors exactly.
        if isinstance(prev_xyz, torch.Tensor) and n >= prev_count and prev_count > 0:
            ref_xyz = torch.cat(
                [
                    prev_xyz.detach().to(device=xyz.device, dtype=xyz.dtype),
                    xyz[prev_count:].detach().clone(),
                ],
                dim=0,
            )
        else:
            ref_xyz = xyz.detach().clone()
    else:
        ref_xyz = xyz.detach().clone()

    if n < 2:
        return {
            "xyz": ref_xyz,
            "edge_src": torch.empty(0, dtype=torch.long, device=xyz.device),
            "edge_dst": torch.empty(0, dtype=torch.long, device=xyz.device),
            "spacing": xyz.new_tensor(1.0),
            "count": n,
        }

    k = max(1, min(cfg_int(cfg, "stage1_smooth_knn", 6), n - 1))
    ref_cpu = ref_xyz.float().cpu().numpy()

    nn_index = None
    nn_distance = None

    # Preferred path: O(N log N) CPU KNN construction.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_cpu)
        dist, idx = tree.query(ref_cpu, k=k + 1, workers=-1)
        if k == 1:
            dist = dist[:, None] if dist.ndim == 1 else dist
            idx = idx[:, None] if idx.ndim == 1 else idx
        nn_distance = dist[:, 1:k + 1]
        nn_index = idx[:, 1:k + 1]
    except Exception:
        # Dependency-free fallback for smaller clouds. This is only performed
        # when the reference topology is built/refreshed, not every iteration.
        max_bruteforce = cfg_int(cfg, "stage1_smooth_bruteforce_max_points", 20000)
        if n > max_bruteforce:
            raise RuntimeError(
                "Anchor/Smooth KNN graph requires scipy for large Gaussian clouds. "
                "Install scipy in the ed3dgs environment: pip install scipy"
            )
        ref_t = ref_xyz.detach()
        chunk = max(128, cfg_int(cfg, "stage1_smooth_knn_chunk", 1024))
        idx_parts = []
        dist_parts = []
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            d = torch.cdist(ref_t[start:end], ref_t)
            row = torch.arange(end - start, device=d.device)
            col = torch.arange(start, end, device=d.device)
            d[row, col] = float("inf")
            vals, inds = torch.topk(d, k=k, dim=1, largest=False, sorted=False)
            idx_parts.append(inds.cpu())
            dist_parts.append(vals.cpu())
        nn_index = torch.cat(idx_parts, dim=0).numpy()
        nn_distance = torch.cat(dist_parts, dim=0).numpy()

    edge_src = (
        torch.arange(n, dtype=torch.long)
        .view(-1, 1)
        .expand(n, k)
        .reshape(-1)
    )
    edge_dst = torch.from_numpy(nn_index.reshape(-1)).long()

    # Remove duplicate undirected edges to keep the regularizer compact.
    lo = torch.minimum(edge_src, edge_dst)
    hi = torch.maximum(edge_src, edge_dst)
    packed = lo.to(torch.int64) * int(n) + hi.to(torch.int64)
    unique_packed = torch.unique(packed)
    edge_src = torch.div(unique_packed, int(n), rounding_mode="floor")
    edge_dst = unique_packed % int(n)

    # Characteristic scale makes the two lambda values largely independent of
    # the object's world-coordinate scale.
    nearest = torch.from_numpy(nn_distance[:, 0]).float()
    finite = torch.isfinite(nearest) & (nearest > 1e-12)
    if bool(finite.any()):
        spacing = nearest[finite].median().to(device=xyz.device, dtype=xyz.dtype)
    else:
        spacing = xyz.new_tensor(1.0)
    spacing = spacing.clamp_min(1e-8)

    return {
        "xyz": ref_xyz,
        "edge_src": edge_src.to(device=xyz.device),
        "edge_dst": edge_dst.to(device=xyz.device),
        "spacing": spacing,
        "count": n,
    }

def g0_anchor_and_smooth_loss(gaussians, reference):
    """
    L_anchor:
        mean(||x_i - x_i_ref||^2) / spacing^2

    L_smooth:
        mean(||(x_i-x_i_ref) - (x_j-x_j_ref)||^2) / spacing^2
        over fixed KNN edges (i,j).

    Both losses act on XYZ only; they do not directly modify color, opacity,
    rotation, or scale.
    """
    xyz = gaussians.get_xyz
    zero = xyz.sum() * 0.0

    if reference is None or int(reference.get("count", -1)) != int(xyz.shape[0]):
        return zero, zero

    ref_xyz = reference["xyz"].to(device=xyz.device, dtype=xyz.dtype)
    if ref_xyz.shape != xyz.shape:
        return zero, zero

    spacing2 = reference["spacing"].to(device=xyz.device, dtype=xyz.dtype).square().clamp_min(1e-12)
    displacement = xyz - ref_xyz

    anchor_loss = displacement.square().sum(dim=-1).mean() / spacing2

    edge_src = reference["edge_src"]
    edge_dst = reference["edge_dst"]
    if edge_src.numel() == 0:
        smooth_loss = zero
    else:
        delta = displacement[edge_src] - displacement[edge_dst]
        smooth_loss = delta.square().sum(dim=-1).mean() / spacing2

    return anchor_loss, smooth_loss

@torch.no_grad()
def restore_original_g0_footprint(
    gaussians,
    g0_scaling_raw,
    g0_rotation_raw,
    protected_count,
    cfg,
):
    """
    Keep the original G0 Gaussian footprint intact.

    The user's G0 is already a dense, smooth splat surface. Stage-1 is allowed
    to move its centres, but by default it must not turn the original splats
    into visibly disconnected ellipses by changing scale/rotation.
    """
    # Strict contour mode is incompatible with restoring the original footprint:
    # restoring scale after a hard shrink would re-create silhouette violations.
    if cfg_bool(cfg, "stage1_strict_contour_from_start", True):
        return
    if not cfg_bool(cfg, "stage1_lock_original_g0_footprint", True):
        return

    n = int(gaussians.get_xyz.shape[0])
    pc = max(0, min(int(protected_count), n, int(g0_scaling_raw.shape[0])))
    if pc <= 0:
        return

    gaussians._scaling.data[:pc].copy_(
        g0_scaling_raw[:pc].to(
            device=gaussians._scaling.device,
            dtype=gaussians._scaling.dtype,
        )
    )
    gaussians._rotation.data[:pc].copy_(
        g0_rotation_raw[:pc].to(
            device=gaussians._rotation.device,
            dtype=gaussians._rotation.dtype,
        )
    )

    mask = torch.zeros(n, dtype=torch.bool, device=gaussians.get_xyz.device)
    mask[:pc] = True
    _zero_optimizer_rows(gaussians, gaussians._scaling, mask)
    _zero_optimizer_rows(gaussians, gaussians._rotation, mask)

def g0_scale_anchor_and_overlap_loss(
    gaussians,
    surface_reference,
    g0_scaling_raw,
    protected_count,
    cfg,
):
    """
    L_scale_anchor:
        keep original G0 log-scales close to their initial values.

    L_overlap:
        on the fixed KNN graph, penalize neighbouring splats whose centre
        distance becomes too large for their current tangential footprint.

    This explicitly preserves the continuous-surface character of the original
    G0 instead of only optimizing its 2D render.
    """
    xyz = gaussians.get_xyz
    zero = xyz.sum() * 0.0
    n = int(xyz.shape[0])
    if n == 0:
        return zero, zero, 0.0

    # ----- scale anchor on original G0 prefix -----
    # Once strict mode is allowed to prune original G0 points, prune_points()
    # may reindex rows and the old prefix identity is no longer valid.
    if cfg_bool(cfg, "stage1_strict_contour_from_start", True):
        pc = 0
    else:
        pc = max(0, min(int(protected_count), n, int(g0_scaling_raw.shape[0])))
    if pc > 0:
        ref_raw = g0_scaling_raw[:pc].to(
            device=gaussians._scaling.device,
            dtype=gaussians._scaling.dtype,
        )
        scale_anchor = (gaussians._scaling[:pc] - ref_raw).square().mean()
    else:
        scale_anchor = zero

    # ----- overlap continuity on the fixed/reference KNN graph -----
    if surface_reference is None or int(surface_reference.get("count", -1)) != n:
        return scale_anchor, zero, 0.0

    edge_src = surface_reference["edge_src"]
    edge_dst = surface_reference["edge_dst"]
    if edge_src.numel() == 0:
        return scale_anchor, zero, 0.0

    scale = gaussians.get_scaling
    # For a surface Gaussian, the two largest axes approximate the tangential footprint.
    tangential = torch.topk(
        scale,
        k=min(2, scale.shape[1]),
        dim=1,
        largest=True,
    ).values.mean(dim=1)

    center_dist = torch.linalg.norm(xyz[edge_src] - xyz[edge_dst], dim=-1)
    sigma_mult = max(0.1, cfg_float(cfg, "stage1_overlap_sigma_mult", 2.5))
    allowed = sigma_mult * (tangential[edge_src] + tangential[edge_dst])

    spacing = surface_reference["spacing"].to(
        device=xyz.device,
        dtype=xyz.dtype,
    ).clamp_min(1e-8)

    gap = F.relu(center_dist - allowed)
    overlap_loss = (gap / spacing).square().mean()

    with torch.no_grad():
        gap_ratio = float((gap > 0).float().mean().item())

    return scale_anchor, overlap_loss, gap_ratio

# ============================================================================
# IMAGE / MASK / LOSS
# ============================================================================

def load_target_image(path, device):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"t0 image does not exist: {path}")
    pil = Image.open(path)
    alpha = None
    if "A" in pil.getbands():
        rgba = np.asarray(pil.convert("RGBA"), dtype=np.float32) / 255.0
        rgb_arr = rgba[..., :3]
        alpha_arr = rgba[..., 3]
        # A fully opaque alpha plane is not a useful foreground mask.
        if float(alpha_arr.min()) < 0.999:
            alpha = (torch.from_numpy(alpha_arr).to(device=device, dtype=torch.float32).clamp(0.0, 1.0))
    else:
        rgb_arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    rgb = (torch.from_numpy(rgb_arr).permute(2, 0, 1).contiguous().to(device=device, dtype=torch.float32))
    return rgb, alpha, pil.width, pil.height

def load_mask(path, width, height, device):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"mask does not exist: {path}")
    image = Image.open(path).convert("L")
    if image.size != (width, height):
        raise ValueError(f"Mask size {image.size} does not match target image " f"{(width, height)}")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return (torch.from_numpy(arr).to(device=device, dtype=torch.float32).clamp(0.0, 1.0))

def save_rgb(image, path):
    image = image.detach().float().clamp(0.0, 1.0)
    arr = (image.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def save_gray(value, path):
    value = value.detach().float().cpu()
    if value.ndim == 3:
        value = value.squeeze(0)
    value = value.clamp(0.0, 1.0)
    arr = (value.numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def dilate_mask(mask, pixels):
    pixels = max(0, int(pixels))
    if pixels == 0:
        return mask
    kernel = 2 * pixels + 1
    return F.max_pool2d(mask[None, None], kernel_size=kernel, stride=1, padding=pixels)[0, 0]

def erode_mask(mask, pixels):
    pixels = max(0, int(pixels))
    if pixels == 0:
        return mask
    kernel = 2 * pixels + 1
    return 1.0 - F.max_pool2d((1.0 - mask)[None, None], kernel_size=kernel, stride=1, padding=pixels)[0, 0]

def clean_binary_mask(mask, radius=1, occupancy=0.35):
    radius = max(0, int(radius))
    if radius == 0:
        return (mask > 0.5).float()
    kernel = 2 * radius + 1
    density = F.avg_pool2d(mask.float()[None, None], kernel_size=kernel, stride=1, padding=radius)[0, 0]
    return (density >= float(occupancy)).float()

def sobel_edges(image_2d):
    if image_2d.ndim == 2:
        x = image_2d[None, None]
    elif image_2d.ndim == 3:
        x = image_2d[None]
    else:
        raise ValueError(f"sobel_edges expected HW or 1HW, got {tuple(image_2d.shape)}")
    dtype = x.dtype
    device = x.device
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=dtype, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8)
    # Normalize to approximately [0, 1].
    edge = edge / 4.0
    return edge[0, 0].clamp(0.0, 1.0)

@torch.no_grad()
def rgb_non_background_mask(target, cfg):
    """
    Build the target object mask ONLY from the target RGB image.

    Definition used by this Stage-1:
        foreground = every RGB pixel that is NOT classified as background.

    IMPORTANT:
      - G0 / initial_alpha are NEVER used to decide the foreground mask.
      - GrabCut is NOT used.
      - PNG alpha is ignored unless the user explicitly supplies t0_mask_path.

    Background colour is estimated from the image border in CIE-Lab space,
    because Lab distance is substantially more stable than raw RGB distance
    under small illumination changes.
    """
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "RGB background removal requires OpenCV (cv2). "
            "Install opencv-python or provide an explicit t0_mask_path."
        ) from exc

    rgb = target.detach().permute(1, 2, 0).float().cpu().numpy()
    rgb_u8 = (rgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    h, w = rgb_u8.shape[:2]

    border_px = cfg_int(cfg, "stage1_bg_border_px", 12)
    border_px = max(1, min(border_px, max(1, min(h, w) // 4)))

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    top = lab[:border_px, :, :].reshape(-1, 3)
    bottom = lab[h - border_px:, :, :].reshape(-1, 3)
    if h > 2 * border_px:
        left = lab[border_px:h - border_px, :border_px, :].reshape(-1, 3)
        right = lab[border_px:h - border_px, w - border_px:, :].reshape(-1, 3)
        border = np.concatenate([top, bottom, left, right], axis=0)
    else:
        border = np.concatenate([top, bottom], axis=0)

    # Robust background prototype from border pixels.
    bg_lab = np.median(border, axis=0)
    border_dist = np.linalg.norm(border - bg_lab[None, :], axis=1)

    q = cfg_float(cfg, "stage1_bg_distance_quantile", 0.995)
    q = min(max(q, 0.50), 0.9999)
    border_q = float(np.quantile(border_dist, q))

    min_dist = cfg_float(cfg, "stage1_bg_min_lab_distance", 12.0)
    margin = cfg_float(cfg, "stage1_bg_distance_margin", 5.0)
    threshold = max(float(min_dist), border_q + float(margin))

    distance = np.linalg.norm(lab - bg_lab[None, None, :], axis=2)

    # EXACT semantic rule requested by the user:
    # background -> 0; every remaining RGB pixel -> 1 (object).
    foreground = (distance > threshold).astype(np.uint8)

    # Optional conservative morphology. Defaults are zero here because the user
    # explicitly wants every surviving non-background RGB pixel to remain object.
    close_radius = max(0, cfg_int(cfg, "stage1_mask_close_radius", 0))
    open_radius = max(0, cfg_int(cfg, "stage1_mask_open_radius", 0))
    if close_radius > 0:
        k = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    if open_radius > 0:
        k = 2 * open_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    mask = torch.from_numpy(foreground.astype(np.float32)).to(
        device=target.device, dtype=target.dtype
    )
    source = (
        f"RGB-background-removal(Lab; threshold={threshold:.3f}; "
        f"bg_lab={bg_lab.tolist()})"
    )
    return mask.clamp(0.0, 1.0), source

@torch.no_grad()
def clean_target_subject_mask(mask, initial_alpha, cfg):
    """
    Remove distant RGB foreground satellites before the mask becomes the hard
    Stage-1 silhouette.

    The original G0 render is used only as a STRUCTURAL SUPPORT cue here:
      - the dominant connected component is always kept;
      - a second component is also kept if it is large, very close to the
        dominant component, or substantially supported by the original G0;
      - small, distant, weakly-G0-supported fragments are removed.

    This prevents a tiny RGB extraction artifact from becoming a protected
    target island for Gaussian refinement.
    """
    if not cfg_bool(cfg, 'stage1_clean_main_subject_mask', True):
        return (mask > 0.5).float(), {
            'components': 1,
            'kept_components': 1,
            'removed_components': 0,
            'removed_pixels': 0,
        }

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            'Main-subject mask cleanup requires OpenCV (cv2).'
        ) from exc

    fg = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    h, w = fg.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg,
        connectivity=8,
    )

    if num_labels <= 2:
        return (mask > 0.5).float(), {
            'components': max(0, int(num_labels) - 1),
            'kept_components': max(0, int(num_labels) - 1),
            'removed_components': 0,
            'removed_pixels': 0,
        }

    g0_thr = cfg_float(cfg, 'stage1_mask_g0_alpha_threshold', 0.05)
    g0_support = (
        initial_alpha.detach().cpu().numpy() >= g0_thr
    ).astype(np.uint8)

    component_ids = list(range(1, int(num_labels)))
    areas = {
        i: int(stats[i, cv2.CC_STAT_AREA])
        for i in component_ids
    }
    overlap_px = {
        i: int(np.logical_and(labels == i, g0_support > 0).sum())
        for i in component_ids
    }

    # Structural main component: area dominates, G0 overlap breaks ties and
    # strongly favors the component supported by the loaded canonical model.
    main_id = max(
        component_ids,
        key=lambda i: float(areas[i]) + 2.0 * float(overlap_px[i]),
    )
    main_area = max(1, areas[main_id])

    keep = {main_id}
    keep_large_fraction = max(
        0.0,
        cfg_float(cfg, 'stage1_mask_keep_large_component_fraction', 0.12),
    )
    g0_overlap_keep_ratio = min(
        max(cfg_float(cfg, 'stage1_mask_g0_overlap_keep_ratio', 0.25), 0.0),
        1.0,
    )
    g0_overlap_keep_pixels = max(
        1,
        cfg_int(cfg, 'stage1_mask_g0_overlap_keep_pixels', 16),
    )
    join_distance = max(
        0,
        cfg_int(cfg, 'stage1_mask_component_join_distance_px', 3),
    )
    tiny_area = max(
        1,
        cfg_int(cfg, 'stage1_mask_satellite_min_area_px', 32),
    )

    # First keep obviously meaningful components by size / G0 support.
    for i in component_ids:
        if i == main_id:
            continue
        area = areas[i]
        ov = overlap_px[i]
        ov_ratio = float(ov) / float(max(area, 1))
        large = float(area) >= keep_large_fraction * float(main_area)
        g0_supported = (
            ov >= g0_overlap_keep_pixels
            and ov_ratio >= g0_overlap_keep_ratio
        )
        if large or g0_supported:
            keep.add(i)

    # Then preserve small pieces that are spatially connected to what we kept
    # up to a very small image-space gap (anti-alias / thin-part tolerance).
    if join_distance > 0:
        k = 2 * join_distance + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        changed = True
        while changed:
            changed = False
            kept_mask = np.isin(labels, np.array(sorted(keep), dtype=np.int32)).astype(np.uint8)
            expanded = cv2.dilate(kept_mask, kernel, iterations=1)
            for i in component_ids:
                if i in keep:
                    continue
                comp = labels == i
                if np.any(expanded[comp] > 0):
                    keep.add(i)
                    changed = True

    cleaned = np.zeros_like(fg, dtype=np.uint8)
    for i in keep:
        cleaned[labels == i] = 1

    # Extra safeguard: a tiny remote component must not survive merely because
    # of a handful of noisy G0-alpha pixels.
    for i in list(keep):
        if i == main_id:
            continue
        area = areas[i]
        if area >= tiny_area:
            continue
        ov_ratio = float(overlap_px[i]) / float(max(area, 1))
        if ov_ratio < g0_overlap_keep_ratio:
            cleaned[labels == i] = 0
            keep.discard(i)

    removed = (fg > 0) & (cleaned == 0)
    cleaned_t = torch.from_numpy(cleaned.astype(np.float32)).to(
        device=mask.device,
        dtype=mask.dtype,
    )
    return cleaned_t, {
        'components': int(num_labels) - 1,
        'kept_components': len(keep),
        'removed_components': (int(num_labels) - 1) - len(keep),
        'removed_pixels': int(removed.sum()),
    }

@torch.no_grad()
def build_target_mask(target, png_alpha, initial_alpha, cfg):
    """
    Build the RGB target mask, then clean disconnected foreground satellites.

    Priority:
      1) explicit t0_mask_path if supplied;
      2) otherwise RGB Lab background removal.

    G0 alpha does NOT create new foreground. It is used only to decide whether
    a disconnected RGB component is structurally plausible.
    """
    mask_value = getattr(cfg, 't0_mask_path', None)
    if mask_value is not None and str(mask_value).strip().lower() not in ('', 'none', 'null'):
        h, w = target.shape[-2:]
        raw_mask = load_mask(mask_value, w, h, target.device)
        raw_mask = (
            raw_mask >= cfg_float(cfg, 'stage1_explicit_mask_threshold', 0.5)
        ).float()
        source = str(mask_value)
    else:
        raw_mask, source = rgb_non_background_mask(target, cfg)

    cleaned, info = clean_target_subject_mask(
        raw_mask,
        initial_alpha,
        cfg,
    )
    source = (
        f'{source}; connected-clean '
        f"components={info['components']} kept={info['kept_components']} "
        f"removed={info['removed_components']} removed_px={info['removed_pixels']}"
    )
    return cleaned.clamp(0.0, 1.0), source

@torch.no_grad()
def build_rgb_inside_distance_map(target_mask):
    """Distance in pixels from each foreground pixel centre to the nearest background.

    This is used as a hard screen-space support budget for every visible Gaussian.
    """
    fg_np = (target_mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    try:
        import cv2
        dist = cv2.distanceTransform(fg_np, cv2.DIST_L2, 5).astype(np.float32)
        return torch.from_numpy(dist).to(device=target_mask.device, dtype=target_mask.dtype)
    except Exception:
        # Dependency-free fallback: iterative binary erosion gives a conservative
        # Chebyshev-distance approximation. It is built once, not every iteration.
        current = (target_mask > 0.5).float()
        dist = torch.zeros_like(target_mask)
        max_steps = int(max(target_mask.shape))
        for _ in range(max_steps):
            if not bool((current > 0.5).any()):
                break
            dist = dist + (current > 0.5).to(dist.dtype)
            current = erode_mask(current, 1)
        return dist

def masked_mean(value, weight, eps=1e-6):
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(0)
    weighted = value * weight
    channels = (value.shape[0] if value.ndim == 3 else 1)
    denom = (weight.sum() * channels).clamp_min(eps)
    return weighted.sum() / denom

def bbox_from_mask(mask, padding=16):
    """
    Returns y0, y1, x0, x1 with y1/x1 exclusive.
    """
    mask_bool = mask > 0.01
    h, w = mask.shape
    ys, xs = torch.nonzero(mask_bool, as_tuple=True)
    if ys.numel() == 0:
        return 0, h, 0, w
    y0 = max(0, int(ys.min().item()) - int(padding))
    y1 = min(h, int(ys.max().item()) + int(padding) + 1)
    x0 = max(0, int(xs.min().item()) - int(padding))
    x1 = min(w, int(xs.max().item()) + int(padding) + 1)
    # SSIM implementations usually expect at least 11x11.
    if y1 - y0 < 16:
        center = (y0 + y1) // 2
        y0 = max(0, center - 8)
        y1 = min(h, y0 + 16)
    if x1 - x0 < 16:
        center = (x0 + x1) // 2
        x0 = max(0, center - 8)
        x1 = min(w, x0 + 16)
    return y0, y1, x0, x1

def silhouette_dice_loss(alpha, target_mask, eps=1e-6):
    intersection = (alpha * target_mask).sum()
    denom = (alpha.sum() + target_mask.sum())
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice

def object_aware_loss(rendered_black, render_alpha, target, target_mask, cfg):
    """Foreground-only appearance + opacity + structure objective.

    RGB / color / DSSIM are evaluated only on the extracted RGB foreground.
    Background pixels never act as a black RGB target. Outside the object, only
    alpha / extra / edge terms supervise the model.
    """
    target_mask_3 = target_mask[None]
    target_object = target * target_mask_3
    photo_erode = max(0, cfg_int(cfg, "stage1_photometric_mask_erode", 1))
    photo_mask = erode_mask(target_mask, photo_erode).detach()
    if float(photo_mask.sum().item()) < 16.0:
        photo_mask = target_mask.detach()
    photo_mask_3 = photo_mask[None]
    rgb_l1 = masked_mean((rendered_black - target).abs(), photo_mask)
    color_alpha_min = cfg_float(cfg, "stage1_color_alpha_min", 0.15)
    render_color = (rendered_black / render_alpha.clamp_min(color_alpha_min)[None]).clamp(0.0, 1.0)
    color_support = ((photo_mask > 0.5) & (render_alpha.detach() >= color_alpha_min)).float()
    color_support = erode_mask(color_support, cfg_int(cfg, "stage1_color_support_erode", 1)).detach()
    if float(color_support.sum().item()) < 16.0:
        color_support = ((photo_mask > 0.5) & (render_alpha.detach() >= color_alpha_min)).float().detach()
    color_l1 = masked_mean(F.smooth_l1_loss(render_color, target, reduction="none", beta=0.03), color_support)
    render_ssim = rendered_black * photo_mask_3
    target_ssim = target * photo_mask_3
    y0, y1, x0, x1 = bbox_from_mask(photo_mask, padding=cfg_int(cfg, "stage1_ssim_crop_padding", 16))
    dssim = 1.0 - ssim(render_ssim[:, y0:y1, x0:x1], target_ssim[:, y0:y1, x0:x1])
    alpha_safe = render_alpha.clamp(1e-5, 1.0 - 1e-5)
    alpha_bce = F.binary_cross_entropy(alpha_safe, target_mask)
    alpha_dice = silhouette_dice_loss(render_alpha, target_mask)
    alpha_loss = cfg_float(cfg, "stage1_alpha_bce_fraction", 0.50) * alpha_bce + cfg_float(cfg, "stage1_alpha_dice_fraction", 0.50) * alpha_dice
    interior = erode_mask(target_mask, cfg_int(cfg, "stage1_opacity_interior_erode", 2)).detach()
    exterior = 1.0 - dilate_mask(target_mask, cfg_int(cfg, "stage1_opacity_exterior_dilate", 2)).detach()
    opacity_inside = ((1.0 - render_alpha).square() * interior).sum() / interior.sum().clamp_min(1.0)
    opacity_outside = (render_alpha.square() * exterior).sum() / exterior.sum().clamp_min(1.0)
    opacity_loss = opacity_inside + cfg_float(cfg, "stage1_opacity_outside_fraction", 0.50) * opacity_outside
    target_edge = sobel_edges(target_mask)
    render_edge = sobel_edges(render_alpha)
    boundary_band = dilate_mask((target_edge > cfg_float(cfg, "stage1_edge_target_threshold", 0.05)).float(), cfg_int(cfg, "stage1_edge_band_dilation", 6))
    boundary_band = torch.maximum(boundary_band, dilate_mask((render_edge.detach() > cfg_float(cfg, "stage1_edge_render_threshold", 0.05)).float(), cfg_int(cfg, "stage1_edge_band_dilation", 6))).detach()
    edge_loss = masked_mean((render_edge - target_edge).abs(), boundary_band)
    missing_loss = (target_mask * (1.0 - render_alpha)).sum() / target_mask.sum().clamp_min(1.0)
    extra_mass = ((1.0 - target_mask) * render_alpha).sum()
    extra_loss = extra_mass / render_alpha.sum().clamp_min(1.0)
    total = (
        cfg_float(cfg, "stage1_lambda_rgb", 0.75) * rgb_l1
        + cfg_float(cfg, "stage1_lambda_color", 1.50) * color_l1
        + cfg_float(cfg, "stage1_lambda_ssim", 0.25) * dssim
        + cfg_float(cfg, "stage1_lambda_alpha", 2.0) * alpha_loss
        + cfg_float(cfg, "stage1_lambda_opacity", 2.5) * opacity_loss
        + cfg_float(cfg, "stage1_lambda_edge", 1.0) * edge_loss
        + cfg_float(cfg, "stage1_lambda_missing", 4.0) * missing_loss
        + cfg_float(cfg, "stage1_lambda_extra", 2.0) * extra_loss
    )
    return total, {
        "rgb": rgb_l1,
        "color": color_l1,
        "dssim": dssim,
        "alpha": alpha_loss,
        "alpha_bce": alpha_bce,
        "alpha_dice": alpha_dice,
        "opacity": opacity_loss,
        "opacity_inside": opacity_inside,
        "opacity_outside": opacity_outside,
        "edge": edge_loss,
        "missing": missing_loss,
        "extra": extra_loss,
        "target_object": target_object,
        "target_edge": target_edge,
        "render_edge": render_edge,
    }

@torch.no_grad()

def structure_masks(target_mask, render_alpha, cfg):
    missing_thr = cfg_float(cfg, "stage1_missing_alpha_threshold", 0.15)
    extra_thr = cfg_float(cfg, "stage1_extra_alpha_threshold", 0.25)
    target_fg = target_mask > 0.5
    missing = (target_fg & (render_alpha <= missing_thr))
    extra = ((~target_fg) & (render_alpha >= extra_thr))
    radius = cfg_int(cfg, "stage1_structure_cleanup_radius", 1)
    occupancy = cfg_float(cfg, "stage1_structure_cleanup_occupancy", 0.40)
    if cfg_bool(cfg, "stage1_strict_surface_fill", True):
        # Do not erase one-pixel / thin missing regions in strict coverage mode.
        missing = missing.float()
    else:
        missing = (clean_binary_mask(missing.float(), radius, occupancy) > 0.5)
    extra = (clean_binary_mask(extra.float(), radius, occupancy) > 0.5)
    return missing.float(), extra.float()

@torch.no_grad()

def build_error_map(rendered_black, target_object, render_alpha, target_mask, missing_mask, extra_mask, cfg):
    rgb_error = (rendered_black - target_object).abs().mean(dim=0)
    alpha_error = (render_alpha - target_mask).abs()
    error = (
        cfg_float(cfg, "stage1_error_map_rgb_weight", 1.0)
        * rgb_error
        + cfg_float(cfg, "stage1_error_map_alpha_weight", 1.5)
        * alpha_error
        + cfg_float(cfg, "stage1_error_map_missing_boost", 1.0)
        * missing_mask
        + cfg_float(cfg, "stage1_error_map_extra_boost", 0.5)
        * extra_mask
    )
    scale = cfg_float(cfg, "stage1_error_map_normalizer", 1.0)
    return (error / max(scale, 1e-6)).clamp(0.0, 1.0)

@torch.no_grad()

def gaussian_update_weights(gaussians, camera, error_map, extra_mask, cfg):
    _, _, px, py, _, valid = project_gaussians(gaussians, camera)
    n = int(gaussians.get_xyz.shape[0])
    weights = torch.zeros(n, dtype=error_map.dtype, device=error_map.device)
    if bool(valid.any()):
        e = error_map[py[valid], px[valid]]
        low = cfg_float(cfg, "stage1_residual_low", 0.02)
        high = cfg_float(cfg, "stage1_residual_high", 0.15)
        gamma = cfg_float(cfg, "stage1_residual_gamma", 1.0)
        floor = cfg_float(cfg, "stage1_gradient_floor", 0.01)
        w = ((e - low) / max(high - low, 1e-6)).clamp(0.0, 1.0)
        if gamma != 1.0:
            w = w.pow(gamma)
        w = (floor + (1.0 - floor) * w)
        extra_here = (extra_mask[py[valid], px[valid]] > 0.5)
        extra_floor = cfg_float(cfg, "stage1_extra_gradient_floor", 0.65)
        w = torch.where(extra_here, torch.maximum(w, torch.full_like(w, extra_floor)), w)
        weights[valid] = w
    return weights, valid

def apply_gaussian_gradient_gating(gaussians, weights):
    for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc"):
        value = getattr(gaussians, name, None)
        if not isinstance(value, torch.Tensor) or value.grad is None:
            continue
        shape = [weights.shape[0]] + [1] * (value.grad.ndim - 1)
        value.grad.mul_(weights.view(*shape))

# ============================================================================
# CONSTRAINTS
# ============================================================================

@torch.no_grad()
def compute_stage1_visibility(gaussians, camera, cfg):
    """Classify current Gaussians as visible/hidden/uncertain/outside.

    IMPORTANT:
        This classifier is used only as a conservative deletion gate.
        A Gaussian that is HIDDEN or UNCERTAIN is never deleted merely because
        the current single RGB view cannot explain it.

    Set stage1_enable_visibility_protection = False only for ablation/debugging.
    """
    if not cfg_bool(cfg, "stage1_enable_visibility_protection", True):
        return None

    try:
        return classify_gaussian_visibility(
            gaussians=gaussians,
            camera=camera,
            depth_abs_margin=max(
                0.0,
                cfg_float(cfg, "stage1_visibility_depth_abs_margin", 0.01),
            ),
            depth_rel_margin=max(
                0.0,
                cfg_float(cfg, "stage1_visibility_depth_rel_margin", 0.01),
            ),
            uncertain_multiplier=max(
                1.0,
                cfg_float(cfg, "stage1_visibility_uncertain_multiplier", 2.5),
            ),
            # 0 is intentionally the default for large G0 clouds: it avoids
            # expanding N Gaussian centres to 9N samples every iteration.
            pixel_radius=max(
                0,
                cfg_int(cfg, "stage1_visibility_pixel_radius", 0),
            ),
            local_depth_radius=max(
                0,
                cfg_int(cfg, "stage1_visibility_local_depth_radius", 1),
            ),
            border_margin_px=max(
                0,
                cfg_int(cfg, "stage1_visibility_border_margin_px", 2),
            ),
            protect_border=cfg_bool(
                cfg,
                "stage1_visibility_protect_border",
                True,
            ),
        )
    except Exception as exc:
        # Never silently fall back to destructive single-view pruning.
        raise RuntimeError(
            "Stage1 visibility protection failed. Refusing to continue because "
            "falling back to ordinary single-view pruning could delete hidden "
            "G0 geometry. Check Stage1/gaussian_utils/visibility.py and camera "
            f"matrices. Original error: {exc}"
        ) from exc

def _geometry_visible_mask(visibility_info, n, device):
    """Return an Nx bool mask of confidently visible Gaussians, or None."""
    if visibility_info is None:
        return None
    if not isinstance(visibility_info, dict):
        return None
    mask = visibility_info.get("visible_mask", None)
    if not isinstance(mask, torch.Tensor):
        return None
    if int(mask.shape[0]) != int(n):
        return None
    return mask.to(device=device).bool()

def set_appearance_trainability(gaussians, color_dc=True, color_rest=False, opacity=True):
    value = getattr(gaussians, "_features_dc", None)
    if isinstance(value, torch.Tensor):
        value.requires_grad_(bool(color_dc))
    value = getattr(gaussians, "_features_rest", None)
    if isinstance(value, torch.Tensor):
        value.requires_grad_(bool(color_rest))
    value = getattr(gaussians, "_opacity", None)
    if isinstance(value, torch.Tensor):
        value.requires_grad_(bool(opacity))

@torch.no_grad()
def enforce_hard_rgb_support(
    gaussians,
    camera,
    target_mask,
    inside_distance,
    visibility=None,
    radii=None,
    cfg=None,
    protected_count=0,
    geometry_visibility=None,
):
    """
    Unified silhouette constraint.

    1) PROJECTED CENTRE outside target mask -> prune.
       Off-screen and invalid projections are also outside and are pruned.

    2) Centre inside target but finite splat support crosses the contour:
       - tiny crossing is ignored;
       - moderate crossing shrinks the Gaussian;
       - severe crossing prunes it.

    The footprint decision uses an inexpensive proxy:
        overflow = 1 - inside_distance(center) / raster_radius
    which avoids sampling millions of Gaussian ellipses every iteration.
    """
    n = int(gaussians.get_xyz.shape[0])
    info = {
        'hard_center_pruned': 0,
        'hard_support_pruned': 0,
        'hard_scale_shrunk': 0,
    }
    if n == 0:
        return info

    _, _, px, py, _, valid = project_gaussians(
        gaussians,
        camera,
    )

    center_inside = torch.zeros(
        n,
        dtype=torch.bool,
        device=gaussians.get_xyz.device,
    )
    if bool(valid.any()):
        center_inside[valid] = (
            target_mask[py[valid], px[valid]] > 0.5
        )

    # Absolute rule requested by the user: frame-background, off-screen and
    # invalid projected centres do not survive.
    center_bad = ~center_inside

    support_prune = torch.zeros_like(center_bad)
    shrink_mask = torch.zeros_like(center_bad)

    if radii is not None and int(radii.shape[0]) == n:
        rr = radii.detach().to(
            device=gaussians.get_xyz.device,
            dtype=gaussians.get_xyz.dtype,
        ).clamp_min(0.0)
        allowed = torch.zeros_like(rr)
        if bool(valid.any()):
            allowed[valid] = inside_distance[
                py[valid],
                px[valid],
            ].to(rr.dtype)

        # Allow a small contour overhang. This is deliberately NOT zero.
        tolerance_px = max(
            0.0,
            cfg_float(cfg, 'stage1_footprint_tolerance_px', 1.5)
            if cfg is not None else 1.5,
        )
        effective_allowed = allowed + tolerance_px
        crossing = center_inside & (rr > effective_allowed + 1e-6)

        overflow = torch.zeros_like(rr)
        overflow[crossing] = (
            1.0
            - effective_allowed[crossing]
            / rr[crossing].clamp_min(1e-6)
        ).clamp(0.0, 1.0)

        shrink_ratio = min(
            max(
                cfg_float(cfg, 'stage1_footprint_shrink_overflow_ratio', 0.18)
                if cfg is not None else 0.18,
                0.0,
            ),
            1.0,
        )
        prune_ratio = min(
            max(
                cfg_float(cfg, 'stage1_footprint_prune_overflow_ratio', 0.68)
                if cfg is not None else 0.68,
                shrink_ratio,
            ),
            1.0,
        )

        support_prune = crossing & (overflow >= prune_ratio)
        shrink_mask = crossing & (~support_prune) & (overflow >= shrink_ratio)

        if bool(shrink_mask.any()):
            current_scale = gaussians.get_scaling.detach()
            # Fit the raster support approximately into the available contour
            # budget, but cap the per-update shrink to avoid destroying edge
            # coverage in a single iteration.
            desired_factor = (
                effective_allowed[shrink_mask]
                / rr[shrink_mask].clamp_min(1e-6)
            )
            min_factor = min(
                max(
                    cfg_float(cfg, 'stage1_footprint_min_shrink_factor', 0.70)
                    if cfg is not None else 0.70,
                    0.20,
                ),
                1.0,
            )
            factor = desired_factor.clamp(min=min_factor, max=1.0)
            new_scale = current_scale[shrink_mask] * factor[:, None]
            gaussians._scaling.data[shrink_mask] = torch.log(
                new_scale.clamp_min(1e-8)
            )
            _zero_optimizer_rows(
                gaussians,
                gaussians._scaling,
                shrink_mask,
            )
            info['hard_scale_shrunk'] = int(shrink_mask.sum().item())

    prune_mask = center_bad | support_prune
    info['hard_center_pruned'] = int(center_bad.sum().item())
    info['hard_support_pruned'] = int(support_prune.sum().item())

    if bool(prune_mask.any()):
        if not hasattr(gaussians, 'prune_points'):
            raise RuntimeError(
                'GaussianModel has no prune_points(), required by RGB support.'
            )
        gaussians.prune_points(prune_mask)
        set_appearance_trainability(
            gaussians,
            color_dc=True,
            color_rest=False,
            opacity=True,
        )

    return info

@torch.no_grad()
def prune_extra_gaussians(
    gaussians,
    camera,
    extra_mask,
    visibility,
    cfg,
    protected_count=0,
    geometry_visibility=None,
):
    """Disabled: deletion is exclusively outside-contour centre pruning."""
    return 0

@torch.no_grad()
def prune_boundary_stray_gaussians(
    gaussians,
    camera,
    target_mask,
    visibility,
    radii,
    cfg,
    protected_count=0,
    geometry_visibility=None,
):
    """
    Disabled: do not delete an in-contour Gaussian because of isolation,
    opacity, radius, or boundary heuristics.
    """
    return {
        "stray_candidates": 0,
        "stray_pruned": 0,
        "stray_hard": 0,
        "stray_soft": 0,
        "stray_isolated": 0,
        "stray_edge_radius": 0,
    }

@torch.no_grad()
def enforce_point_budget(
    gaussians,
    max_total_points,
    protected_count=0,
    geometry_visibility=None,
):
    """
    Disabled: never satisfy the point budget by deleting a valid in-contour
    Gaussian. New seeding already obeys remaining_budget.
    """
    return 0

# ============================================================================
# DIAGNOSTICS
# ============================================================================

@torch.no_grad()

def make_overlay_50_50(target_object, rendered):
    return (0.5 * target_object + 0.5 * rendered).clamp(0.0, 1.0)

@torch.no_grad()

def make_structure_overlay(target_object, rendered, missing_mask, extra_mask):
    overlay = make_overlay_50_50(target_object, rendered).clone()
    missing = (missing_mask > 0.5)
    extra = (extra_mask > 0.5)
    if bool(missing.any()):
        overlay[:, missing] = torch.tensor([1.0, 0.0, 0.0], device=overlay.device, dtype=overlay.dtype)[:, None]
    if bool(extra.any()):
        overlay[:, extra] = torch.tensor([0.0, 0.0, 1.0], device=overlay.device, dtype=overlay.dtype)[:, None]
    return overlay

@torch.no_grad()

def gaussian_weights_to_image(gaussians, camera, weights):
    height = int(camera.image_height)
    width = int(camera.image_width)
    result = torch.zeros((height, width), dtype=weights.dtype, device=weights.device)
    (_, _, px, py, _, valid) = project_gaussians(gaussians, camera)
    if bool(valid.any()):
        flat_index = (py[valid] * width + px[valid])
        values = weights[valid]
        flat = result.reshape(-1)
        # PyTorch 2.1+ in this repository supports scatter_reduce_.
        flat.scatter_reduce_(0, flat_index, values, reduce="amax", include_self=True)
    return result

@torch.no_grad()

def image_metrics(rendered, target_object):
    residual = (rendered - target_object)
    mae = residual.abs().mean()
    mse = residual.square().mean()
    rmse = torch.sqrt(mse.clamp_min(1e-12))
    psnr = (-10.0 * torch.log10(mse.clamp_min(1e-12)))
    ssim_value = ssim(rendered, target_object)
    return {"mae": float(mae.item()), "rmse": float(rmse.item()), "psnr": float(psnr.item()), "ssim": float(ssim_value.item())}

@torch.no_grad()

def save_diagnostics(
    output_dir,
    iteration,
    gaussians,
    camera,
    target,
    rendered,
    target_mask,
    render_alpha,
    missing_mask,
    extra_mask,
    error_map,
    gaussian_weights,
    loss,
    loss_diag,
    alpha_source,
    mask_source,
):
    diag_dir = (Path(output_dir) / "diagnostics" / f"iteration_{iteration:04d}")
    diag_dir.mkdir(parents=True, exist_ok=True)
    target_object = (target * target_mask[None])
    save_rgb(target, diag_dir / "target.png")
    save_rgb(target_object, diag_dir / "target_object.png")
    save_rgb(rendered, diag_dir / "render.png")
    save_rgb(make_overlay_50_50(target_object, rendered), diag_dir / "overlay_50_50.png")
    save_rgb(make_structure_overlay(target_object, rendered, missing_mask, extra_mask), diag_dir / "overlay_structure.png")
    save_gray(target_mask, diag_dir / "target_mask.png")
    save_gray(render_alpha, diag_dir / "render_alpha.png")
    save_gray(missing_mask, diag_dir / "missing_gaussian_mask.png")
    save_gray(extra_mask, diag_dir / "extra_gaussian_mask.png")
    save_gray(error_map, diag_dir / "residual_map.png")
    save_gray(loss_diag["target_edge"], diag_dir / "edge_target.png")
    save_gray(loss_diag["render_edge"], diag_dir / "edge_render.png")
    weight_map = gaussian_weights_to_image(gaussians, camera, gaussian_weights)
    save_gray(weight_map, diag_dir / "gaussian_update_weight.png")
    metrics = image_metrics(rendered, target_object)
    with (diag_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"iteration = {iteration}\n")
        f.write(f"gaussian_count = " f"{int(gaussians.get_xyz.shape[0])}\n")
        f.write(f"mask_source = {mask_source}\n")
        f.write(f"alpha_source = {alpha_source}\n")
        f.write(f"total_loss = " f"{float(loss):.9f}\n")
        for key in ("rgb", "color", "dssim", "alpha", "alpha_bce", "alpha_dice", "opacity", "opacity_inside", "opacity_outside", "edge", "missing", "extra", "coverage", "anchor", "smooth", "surface_prior", "depth", "scale_anchor", "overlap"):
            if key in loss_diag:
                f.write(f"{key} = " f"{float(loss_diag[key]):.9f}\n")
        f.write(f"psnr_neutralized = " f"{metrics['psnr']:.6f}\n")
        f.write(f"ssim_neutralized = " f"{metrics['ssim']:.6f}\n")
        f.write(f"mae_neutralized = " f"{metrics['mae']:.9f}\n")
        f.write(f"rmse_neutralized = " f"{metrics['rmse']:.9f}\n")
        f.write(f"target_foreground_ratio = " f"{float(target_mask.mean().item()):.9f}\n")
        f.write(f"render_alpha_mean = " f"{float(render_alpha.mean().item()):.9f}\n")
        f.write(f"missing_ratio = " f"{float(missing_mask.mean().item()):.9f}\n")
        f.write(f"extra_ratio = " f"{float(extra_mask.mean().item()):.9f}\n")
