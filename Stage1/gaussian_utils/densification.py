"""

Topology and local detail refinement for Gaussian refinement.

Contains:

- densification statistics

- local detail target

- residual/detail/absolute-gradient driven adaptive split

- G0 birth-structure gate

- coverage regularization

- scale floor

- native densification wrapper

- missing-region seeding/refill

"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from gaussian_renderer import render
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from gaussian_utils.refinement_utils import (

    cfg_bool,
    cfg_float,
    cfg_int,
    erode_mask,
    gaussian_update_weights,
    sobel_edges,
    structure_masks,
    _package_alpha,
    _zero_optimizer_rows,
    project_gaussians,
    unproject_pixels_at_ndc_depth,
    set_appearance_trainability,
)

# ---------------------------------------------------------------------
# Residual / absolute-gradient densification state

# ---------------------------------------------------------------------
#
# This is intentionally "AbsGS-inspired", not a claim of exact AbsGS CUDA
# reproduction.  The renderer already returns the aggregated view-space point
# gradient.  We keep the absolute x/y components over iterations so a point is
# not selected solely from one noisy backward pass.
_ABS_GRAD_STATE = {"n": 0, "sum_xy": None, "count": None,}


@torch.no_grad()
def _reset_abs_grad_state(n, device, dtype):
    _ABS_GRAD_STATE["n"] = int(n)
    _ABS_GRAD_STATE["sum_xy"] = torch.zeros((int(n), 2), device=device, dtype=dtype,)
    _ABS_GRAD_STATE["count"] = torch.zeros(int(n), device=device, dtype=dtype,)


@torch.no_grad()
def _accumulate_abs_grad(points_grad, active_mask):
    n = int(points_grad.shape[0])
    device = points_grad.device
    dtype = points_grad.dtype
    if (_ABS_GRAD_STATE["sum_xy"] is None or int(_ABS_GRAD_STATE["n"]) != n or _ABS_GRAD_STATE["sum_xy"].device != device):
        _reset_abs_grad_state(n, device, dtype)
    gxy = points_grad[:, :2].detach().abs()
    m = active_mask.bool()
    if bool(m.any()):
        _ABS_GRAD_STATE["sum_xy"][m] += gxy[m]
        _ABS_GRAD_STATE["count"][m] += 1.0


@torch.no_grad()
def _abs_grad_signal(n, device, dtype):
    if (_ABS_GRAD_STATE["sum_xy"] is None or int(_ABS_GRAD_STATE["n"]) != int(n)):
        return torch.zeros(int(n), device=device, dtype=dtype)
    count = _ABS_GRAD_STATE["count"].to(device=device, dtype=dtype,).clamp_min(1.0)
    mean_xy = _ABS_GRAD_STATE["sum_xy"].to(device=device, dtype=dtype,) / count[:, None]
    # L1 over the two screen-space directions.  Keeping x/y absolute before
    # summation gives a homodirectional-style proxy across iterations.
    return mean_xy.abs().sum(dim=-1)


@torch.no_grad()
def _decay_abs_grad_state(factor=0.20):
    if _ABS_GRAD_STATE["sum_xy"] is None:
        return
    factor = float(min(max(factor, 0.0), 1.0))
    _ABS_GRAD_STATE["sum_xy"].mul_(factor)
    _ABS_GRAD_STATE["count"].mul_(factor)


@torch.no_grad()
def _g0_tangent_basis(parent_xyz, raw_g0_surface_prior, cfg):
    """

    Estimate two local tangent directions from the ORIGINAL G0 using PCA.

    Returns:

        tangent1, tangent2, normal   [M,3] tensors

        or (None, None, None) when the raw-G0 teacher is unavailable.

    """
    if (
        raw_g0_surface_prior is None
        or parent_xyz.numel() == 0
        or raw_g0_surface_prior.get("tree", None) is None
        or raw_g0_surface_prior.get("xyz_np", None) is None
    ):
        return None, None, None
    tree = raw_g0_surface_prior["tree"]
    raw_xyz_np = raw_g0_surface_prior["xyz_np"]
    k = max(6, cfg_int(cfg, "stage1_detail_g0_tangent_knn", 16),)
    k = min(k, int(raw_xyz_np.shape[0]))
    q = parent_xyz.detach().float().cpu().numpy()
    try:
        _, nn_idx = tree.query(q, k=k, workers=-1)
    except TypeError:
        _, nn_idx = tree.query(q, k=k)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]
    nbr = raw_xyz_np[nn_idx]
    mean = nbr.mean(axis=1, keepdims=True)
    centered = nbr - mean
    cov = np.einsum("nki,nkj->nij", centered, centered, optimize=True,)
    cov /= float(max(int(nbr.shape[1]) - 1, 1))
    try:
        _, eigvec = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None, None, None
    # eigh ascending eigenvalues:
    #   col 0 = surface normal
    #   col 1/2 = tangent directions
    normal_np = eigvec[:, :, 0].astype(np.float32)
    tangent1_np = eigvec[:, :, 2].astype(np.float32)
    tangent2_np = eigvec[:, :, 1].astype(np.float32)
    device = parent_xyz.device
    dtype = parent_xyz.dtype
    tangent1 = torch.from_numpy(tangent1_np).to(device=device, dtype=dtype)
    tangent2 = torch.from_numpy(tangent2_np).to(device=device, dtype=dtype)
    normal = torch.from_numpy(normal_np).to(device=device, dtype=dtype)
    tangent1 = F.normalize(tangent1, dim=-1)
    tangent2 = F.normalize(tangent2, dim=-1)
    normal = F.normalize(normal, dim=-1)
    return tangent1, tangent2, normal


@torch.no_grad()
def _snap_children_toward_raw_g0(child_xyz, raw_g0_surface_prior, cfg):
    """

    Gentle birth-time snap toward the nearest raw-G0 point.

    This does not freeze the child on G0; it only prevents split children from

    being born as detached layers.  Later RGB / geometry optimization may still

    move them.

    """
    if (
        child_xyz.numel() == 0
        or raw_g0_surface_prior is None
        or raw_g0_surface_prior.get("tree", None) is None
        or raw_g0_surface_prior.get("xyz_np", None) is None
    ):
        return child_xyz
    fraction = min(max(cfg_float(cfg, "stage1_detail_g0_snap_fraction", 0.20), 0.0,), 0.75,)
    if fraction <= 0.0:
        return child_xyz
    tree = raw_g0_surface_prior["tree"]
    raw_xyz_np = raw_g0_surface_prior["xyz_np"]
    q = child_xyz.detach().float().cpu().numpy()
    try:
        dist, idx = tree.query(q, k=1, workers=-1)
    except TypeError:
        dist, idx = tree.query(q, k=1)
    idx = np.asarray(idx, dtype=np.int64)
    dist = np.asarray(dist, dtype=np.float32)
    nearest_np = raw_xyz_np[idx]
    nearest = torch.from_numpy(nearest_np.astype(np.float32)).to(device=child_xyz.device, dtype=child_xyz.dtype,)
    dist_t = torch.from_numpy(dist).to(device=child_xyz.device, dtype=child_xyz.dtype,)
    spacing = max(float(raw_g0_surface_prior.get("spacing", 1.0)), 1e-8,)
    activate_mult = max(0.0, cfg_float(cfg, "stage1_detail_g0_snap_start_mult", 0.75),)
    active = dist_t > spacing * activate_mult
    out = child_xyz.clone()
    if bool(active.any()):
        out[active] = ((1.0 - fraction) * child_xyz[active] + fraction * nearest[active])
    return out


def get_viewspace_points(package):
    points = package.get("viewspace_points_densify", None)
    if points is None:
        points = package.get("viewspace_points", None)
    return points


@torch.no_grad()
def update_max_radii(gaussians, visibility, radii):
    if (visibility is None or radii is None):
        return
    if (radii.shape[0] != gaussians.max_radii2D.shape[0]):
        return
    gaussians.max_radii2D[visibility] = torch.maximum(gaussians.max_radii2D[visibility], radii[visibility])


def accumulate_local_densification_stats(gaussians, package, camera, error_map, extra_mask, cfg,):
    """

    Accumulate both ordinary 3DGS statistics and an absolute screen-gradient

    proxy used by detail_aware_local_split().

    The API is unchanged, so gaussian_refinement.py needs no modification.

    """
    points = get_viewspace_points(package)
    visibility = package.get("visibility_filter", None)
    radii = package.get("radii", None)
    if (points is None or visibility is None or points.grad is None):
        return 0
    weights, valid = gaussian_update_weights(gaussians, camera, error_map, extra_mask, cfg,)
    # Lower than the final split threshold: collect evidence from moderately
    # wrong regions too, then let the split scorer rank them later.
    threshold = cfg_float(cfg, "stage1_densify_weight_threshold", 0.45,)
    local_visibility = (visibility.bool() & valid.bool() & (weights >= threshold))
    with torch.no_grad():
        update_max_radii(gaussians, visibility, radii,)
        _accumulate_abs_grad(points.grad, local_visibility,)
        if bool(local_visibility.any()):
            if hasattr(gaussians, "add_densification_stats"):
                gaussians.add_densification_stats(points, local_visibility,)
            elif hasattr(gaussians, "add_densification_stats_grad"):
                gaussians.add_densification_stats_grad(points.grad, local_visibility,)
            else:
                raise RuntimeError("GaussianModel has neither " "add_densification_stats() nor " "add_densification_stats_grad().")
    return int(local_visibility.sum().item())


def _quat_to_rotmat_local(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            w*w + x*x - y*y - z*z,
            2.0*(x*y - w*z),
            2.0*(x*z + w*y),
            2.0*(x*y + w*z),
            w*w - x*x + y*y - z*z,
            2.0*(y*z - w*x),
            2.0*(x*z - w*y),
            2.0*(y*z + w*x),
            w*w - x*x - y*y + z*z,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def _inverse_sigmoid_local(x):
    x = x.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(x / (1.0 - x))


@torch.no_grad()
def prepare_local_detail_target(target_rgb, target_mask, cfg):
    """Target-side detail map used to decide where large splats need splitting."""
    gray = (0.299 * target_rgb[0] + 0.587 * target_rgb[1] + 0.114 * target_rgb[2])
    rgb_edge = sobel_edges(gray)
    mask_edge = sobel_edges(target_mask.float())
    r = max(1, cfg_int(cfg, 'stage1_detail_highfreq_radius', 2))
    k = 2 * r + 1
    smooth = F.avg_pool2d(target_rgb[None], kernel_size=k, stride=1, padding=r,)[0]
    highfreq = (target_rgb - smooth).abs().mean(dim=0)
    fg = target_mask > 0.5
    if bool(fg.any()):
        scale = torch.quantile(highfreq[fg], 0.95).clamp_min(1e-6)
    else:
        scale = highfreq.new_tensor(1.0)
    highfreq = (highfreq / scale).clamp(0.0, 1.0)
    edge_w = max(0.0, cfg_float(cfg, 'stage1_detail_target_edge_weight', 1.0))
    hf_w = max(0.0, cfg_float(cfg, 'stage1_detail_target_highfreq_weight', 1.0))
    sil_w = max(0.0, cfg_float(cfg, 'stage1_detail_target_silhouette_weight', 0.35))
    denom = max(edge_w + hf_w + sil_w, 1e-8)
    detail = (edge_w * rgb_edge + hf_w * highfreq + sil_w * mask_edge) / denom
    detail = (detail * target_mask).clamp(0.0, 1.0)
    return {
        'detail_map': detail.detach(),
        'rgb_edge': rgb_edge.detach(),
        'highfreq': highfreq.detach(),
        'mask_edge': mask_edge.detach(),
    }


@torch.no_grad()
def raw_g0_support_mask_for_xyz(xyz, raw_g0_surface_prior, cfg):
    """

    Keep new topology close to the ORIGINAL canonical G0 surface.

    This is a hard birth-time structural gate, not a permanent clamp: children

    and missing-region seeds may correct the surface, but they cannot appear as

    detached 3D islands far from the original object block.

    """
    if xyz.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=xyz.device), torch.zeros(0, device=xyz.device)
    if not cfg_bool(cfg, 'stage1_enable_g0_birth_structure_gate', True):
        return torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device), torch.zeros(xyz.shape[0], device=xyz.device)
    tree = raw_g0_surface_prior.get('tree', None)
    spacing = float(raw_g0_surface_prior.get('spacing', 0.0))
    if tree is None or spacing <= 0.0:
        return torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device), torch.zeros(xyz.shape[0], device=xyz.device)
    query = xyz.detach().float().cpu().numpy()
    try:
        dist, _ = tree.query(query, k=1, workers=-1)
    except TypeError:
        dist, _ = tree.query(query, k=1)
    dist = np.asarray(dist, dtype=np.float32)
    max_mult = max(1.0, cfg_float(cfg, 'stage1_g0_birth_max_distance_mult', 8.0),)
    max_abs = max(0.0, cfg_float(cfg, 'stage1_g0_birth_max_distance_abs', 0.0),)
    threshold = max(spacing * max_mult, max_abs)
    keep_np = np.isfinite(dist) & (dist <= threshold)
    keep = torch.from_numpy(keep_np).to(device=xyz.device, dtype=torch.bool)
    dist_t = torch.from_numpy(dist).to(device=xyz.device, dtype=xyz.dtype)
    return keep, dist_t


@torch.no_grad()
def detail_aware_local_split(
    gaussians,
    camera,
    target_rgb,
    target_mask,
    rendered_rgb,
    package,
    detail_target,
    raw_g0_surface_prior,
    cfg,
    remaining_budget,
    iteration,
):
    """

    Strong residual-aware local split.

    Compared with the previous implementation:

      1) RGB residual + edge error + target detail + screen radius;

      2) absolute view-space gradient proxy accumulated over backward passes;

      3) adaptive score quantile so the threshold follows the current error;

      4) very large / very wrong splats use a 4-way G0-tangent split;

      5) ordinary coarse splats use a 2-way G0-tangent split;

      6) hard screen-footprint replacement: oversized visible splats are split

         even when fixed-view RGB/SSIM residual is already very small;

      7) child scale/opacity are initialized from the ORIGINAL parent state

         (no double attenuation), with coverage-conserving child alpha;

      8) coarse parents are strongly suppressed during training. Parent pruning

         is deferred by default to avoid Gaussian row reindexing while the caller

         still holds Anchor/Smooth/KNN references; explicit training-time prune

         requires stage1_detail_allow_parent_reindex=True;

      9) child SH-DC colors use sub-pixel bilinear target sampling;

     10) all children pass the permanent raw-G0 structure gate.

    The function keeps the same signature and return keys expected by

    gaussian_refinement.py.

    """
    info = {
        "detail_candidates": 0,
        "detail_split_parents": 0,
        "detail_added": 0,
        "detail_g0_rejected": 0,
        "detail_mean_score": 0.0,
        "detail_mean_residual": 0.0,
        "detail_mean_radius": 0.0,
        "detail_mean_abs_grad": 0.0,
        "detail_cross_parents": 0,
        # Anti-watercolor diagnostics.
        "detail_hard_oversized": 0,
        "detail_pruned_parents": 0,
        "detail_parent_reindexed": 0,
        "detail_deferred_parent_prune": 0,
        "detail_soft_suppressed_parents": 0,
        "detail_mean_child_scale": 0.0,
        "detail_mean_child_opacity": 0.0,
    }
    if not cfg_bool(cfg, "stage1_enable_detail_densification", True,):
        return info
    if remaining_budget <= 0:
        return info
    n = int(gaussians.get_xyz.shape[0])
    if n == 0:
        return info
    radii = package.get("radii", None)
    vis = package.get("visibility_filter", None)
    if (not isinstance(radii, torch.Tensor) or int(radii.shape[0]) != n):
        return info
    _, _, px, py, _, valid = project_gaussians(gaussians, camera,)
    active = valid.clone()
    if (isinstance(vis, torch.Tensor) and int(vis.shape[0]) == n):
        active &= vis.bool()
    inside = torch.zeros(n, dtype=torch.bool, device=gaussians.get_xyz.device,)
    if bool(valid.any()):
        inside[valid] = (target_mask[py[valid], px[valid],] > 0.5)
    active &= inside
    if not bool(active.any()):
        return info

    # ------------------------------------------------------------
    # Per-pixel local evidence

    # ------------------------------------------------------------
    residual_map = (rendered_rgb.detach() - target_rgb).abs().mean(dim=0)
    edge_render = sobel_edges(
        0.299 * rendered_rgb.detach()[0]
        + 0.587 * rendered_rgb.detach()[1]
        + 0.114 * rendered_rgb.detach()[2]
    )
    edge_target = detail_target["rgb_edge"].to(device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype,)
    edge_error_map = (edge_render - edge_target).abs().clamp(0.0, 1.0)
    dtype = gaussians.get_xyz.dtype
    device = gaussians.get_xyz.device
    residual = torch.zeros(n, device=device, dtype=dtype,)
    edge_error = torch.zeros_like(residual)
    detail_value = torch.zeros_like(residual)
    if bool(valid.any()):
        residual[valid] = residual_map[py[valid], px[valid],].to(dtype)
        edge_error[valid] = edge_error_map[py[valid], px[valid],].to(dtype)
        detail_value[valid] = detail_target["detail_map"][py[valid], px[valid],].to(device=device, dtype=dtype,)
    rr = radii.detach().to(device=device, dtype=dtype,).clamp_min(0.0)

    # ------------------------------------------------------------
    # AbsGS-inspired absolute-gradient evidence

    # ------------------------------------------------------------
    abs_grad = _abs_grad_signal(n, device, dtype,)
    grad_active = abs_grad[active]
    if grad_active.numel() > 0 and bool((grad_active > 0).any()):
        grad_scale = torch.quantile(
            grad_active[grad_active > 0],
            min(max(cfg_float(cfg, "stage1_detail_abs_grad_norm_quantile", 0.90,), 0.50,), 0.995,),
        ).clamp_min(1e-12)
        grad_norm = (abs_grad / grad_scale).clamp(0.0, 4.0)
    else:
        grad_norm = torch.zeros_like(abs_grad)

    # ------------------------------------------------------------
    # Normalize each evidence source

    # ------------------------------------------------------------
    min_radius = max(0.75, cfg_float(cfg, "stage1_detail_min_screen_radius_px", 2.0,),)
    min_residual = max(1e-6, cfg_float(cfg, "stage1_detail_min_residual", 0.025,),)
    min_edge_error = max(1e-6, cfg_float(cfg, "stage1_detail_min_edge_error", 0.045,),)
    min_detail = max(1e-6, cfg_float(cfg, "stage1_detail_min_target_detail", 0.06,),)
    residual_score = (residual / min_residual).clamp(0.0, 4.0)
    edge_score = (edge_error / min_edge_error).clamp(0.0, 4.0)
    target_detail_score = (detail_value / min_detail).clamp(0.0, 4.0)
    radius_score = (rr / min_radius).clamp(0.0, 4.0)
    score = (
        cfg_float(cfg, "stage1_detail_score_residual_weight", 1.20,)
        * residual_score
        + cfg_float(cfg, "stage1_detail_score_edge_weight", 1.15,)
        * edge_score
        + cfg_float(cfg, "stage1_detail_score_detail_weight", 0.90,)
        * target_detail_score
        + cfg_float(cfg, "stage1_detail_score_radius_weight", 0.75,)
        * radius_score
        + cfg_float(cfg, "stage1_detail_score_abs_grad_weight", 1.00,)
        * grad_norm
    )
    coarse = active & (rr >= min_radius)

    # ------------------------------------------------------------------
    # Anti-watercolor hard footprint rule.
    #
    # A broad Gaussian can obtain an excellent RGB/SSIM score while still
    # looking like a watercolor brush when the result is magnified.  Therefore
    # oversized visible foreground splats are eligible for replacement even
    # when their current photometric residual is already small.

    # ------------------------------------------------------------------
    anti_watercolor = cfg_bool(cfg, "stage1_detail_anti_watercolor_mode", True,)
    hard_split_radius = max(min_radius, cfg_float(cfg, "stage1_detail_hard_split_radius_px", 4.0,),)
    hard_oversized = (active & (rr >= hard_split_radius) if anti_watercolor else torch.zeros_like(active))
    info["detail_hard_oversized"] = int(hard_oversized.sum().item())
    # Require at least one meaningful local signal; target detail alone may
    # trigger a split even if current RGB residual is already modest.
    evidence = (
        (residual >= 0.60 * min_residual)
        | (edge_error >= 0.60 * min_edge_error)
        | (detail_value >= 0.60 * min_detail)
        | (grad_norm >= cfg_float(cfg, "stage1_detail_min_abs_grad_norm", 0.65,))
    )
    eligible = coarse & (evidence | hard_oversized)
    if not bool(eligible.any()):
        _decay_abs_grad_state(cfg_float(cfg, "stage1_detail_abs_grad_decay", 0.35,))
        return info
    # Adaptive quantile: on each topology tick select the worst fraction of
    # currently coarse / wrong visible splats instead of relying only on a
    # single fixed threshold.
    q = min(max(cfg_float(cfg, "stage1_detail_candidate_quantile", 0.82,), 0.50,), 0.995,)
    eligible_scores = score[eligible]
    adaptive_cut = torch.quantile(eligible_scores, q,)
    absolute_cut = max(0.0, cfg_float(cfg, "stage1_detail_min_combined_score", 2.25,),)
    cut = torch.maximum(adaptive_cut, score.new_tensor(absolute_cut),)
    candidate = eligible & (score >= cut)
    # Hard oversized splats bypass the photometric-score cut.  They are exactly
    # the splats that can remain visually smooth at 640x640 yet become obvious
    # watercolor blobs when the viewer zooms in.
    candidate |= hard_oversized
    # Fallback: if the quantile cut is too strict on a nearly-flat iteration,
    # still permit clearly large/high-residual splats.
    fallback = (
        coarse
        & ((rr >= min_radius * 1.8) & ((residual >= min_residual) | (edge_error >= min_edge_error) | (grad_norm >= 1.0)))
    )
    candidate |= fallback
    info["detail_candidates"] = int(candidate.sum().item())
    if info["detail_candidates"] <= 0:
        _decay_abs_grad_state(cfg_float(cfg, "stage1_detail_abs_grad_decay", 0.35,))
        return info
    candidate_idx = torch.nonzero(candidate, as_tuple=False,).squeeze(-1)
    # 4 children is the maximum possible per selected parent in the adaptive
    # cross split. Reserve budget conservatively.
    max_children_per_parent = 4
    max_parents_cfg = max(1, cfg_int(cfg, "stage1_detail_max_parents_per_update", 1024,),)
    max_parents = min(max_parents_cfg, int(candidate_idx.numel()), int(remaining_budget) // max_children_per_parent,)
    if max_parents <= 0:
        return info
    # Prioritize hard oversized splats before ordinary residual candidates.
    # The additive bonus is relative to the current score range, so this remains
    # stable if the individual loss weights are changed in the INI.
    candidate_score = score[candidate_idx]
    priority_bonus = (
        candidate_score.detach().max().clamp_min(1.0)
        * cfg_float(cfg, "stage1_detail_hard_split_priority_bonus", 4.0,)
    )
    candidate_priority = (candidate_score + hard_oversized[candidate_idx].to(candidate_score.dtype) * priority_bonus)
    selected_local = torch.topk(candidate_priority, k=max_parents, largest=True, sorted=False,).indices
    parent_idx = candidate_idx[selected_local]
    parent_xyz = gaussians.get_xyz[parent_idx].detach()
    parent_scale = gaussians.get_scaling[parent_idx].detach()

    # ------------------------------------------------------------
    # Split directions: prefer original-G0 tangent plane

    # ------------------------------------------------------------
    tangent1, tangent2, _ = _g0_tangent_basis(parent_xyz, raw_g0_surface_prior, cfg,)
    if tangent1 is None or tangent2 is None:
        # Fallback to the two largest Gaussian local axes.
        parent_rot_raw = gaussians._rotation[parent_idx].detach().clone()
        rot = _quat_to_rotmat_local(parent_rot_raw)
        axes = torch.argsort(parent_scale, dim=1, descending=True,)
        row = torch.arange(parent_idx.shape[0], device=parent_idx.device,)
        axis0 = axes[:, 0]
        axis1 = axes[:, min(1, axes.shape[1] - 1),]
        tangent1 = rot[row, :, axis0,]
        tangent2 = rot[row, :, axis1,]
        tangent1 = F.normalize(tangent1, dim=-1,)
        tangent2 = F.normalize(tangent2, dim=-1,)
    # Tangential spatial step uses the two largest actual Gaussian scales.
    top2_scale = torch.topk(parent_scale, k=min(2, parent_scale.shape[1]), dim=1, largest=True,).values
    tangential_scale = top2_scale.mean(dim=1)
    offset_factor = max(0.05, cfg_float(cfg, "stage1_detail_split_offset_factor", 0.40,),)
    step = tangential_scale * offset_factor
    # Strong parents get a 4-way cross split. Moderate parents use only ±t1.
    selected_score = score[parent_idx]
    selected_radius = rr[parent_idx]
    strong_radius = max(min_radius * 1.8, cfg_float(cfg, "stage1_detail_cross_split_radius_px", 5.0,),)
    strong_score_quantile = min(max(cfg_float(cfg, "stage1_detail_cross_score_quantile", 0.70,), 0.0,), 1.0,)
    strong_score_cut = torch.quantile(selected_score, strong_score_quantile,)
    strong_parent = (
        cfg_bool(cfg, "stage1_detail_enable_cross_split", True,)
        & ((selected_radius >= strong_radius) | (selected_score >= strong_score_cut))
    )
    info["detail_cross_parents"] = int(strong_parent.sum().item())
    # Base two children for every selected parent.
    child_xyz_parts = [parent_xyz + tangent1 * step[:, None], parent_xyz - tangent1 * step[:, None],]
    child_parent_parts = [parent_idx, parent_idx,]
    # Extra ±t2 children only for strong parents.
    strong_idx_local = torch.nonzero(strong_parent, as_tuple=False,).squeeze(-1)
    if strong_idx_local.numel() > 0:
        pxyz_s = parent_xyz[strong_idx_local]
        t2_s = tangent2[strong_idx_local]
        step_s = step[strong_idx_local]
        child_xyz_parts.extend([pxyz_s + t2_s * step_s[:, None], pxyz_s - t2_s * step_s[:, None],])
        strong_parent_ids = parent_idx[strong_idx_local]
        child_parent_parts.extend([strong_parent_ids, strong_parent_ids,])
    child_xyz = torch.cat(child_xyz_parts, dim=0,)
    parent_ids_for_children = torch.cat(child_parent_parts, dim=0,)
    # Gentle G0 snap before the hard birth gate.
    child_xyz = _snap_children_toward_raw_g0(child_xyz, raw_g0_surface_prior, cfg,)

    # ------------------------------------------------------------
    # G0 structural birth gate

    # ------------------------------------------------------------
    keep_child, _ = raw_g0_support_mask_for_xyz(child_xyz, raw_g0_surface_prior, cfg,)
    rejected = int((~keep_child).sum().item())
    info["detail_g0_rejected"] = rejected
    if not bool(keep_child.any()):
        _decay_abs_grad_state(cfg_float(cfg, "stage1_detail_abs_grad_decay", 0.35,))
        return info
    # Never exceed the remaining point budget, even if the 2/4-way adaptive
    # split produced more accepted children than estimated.
    accepted_idx = torch.nonzero(keep_child, as_tuple=False,).squeeze(-1)
    if int(accepted_idx.numel()) > int(remaining_budget):
        # Keep children of higher-score parents first.
        child_parent_score = score[parent_ids_for_children[accepted_idx]]
        keep_top = torch.topk(child_parent_score, k=int(remaining_budget), largest=True, sorted=False,).indices
        accepted_idx = accepted_idx[keep_top]
    if accepted_idx.numel() <= 0:
        return info
    accepted_xyz = child_xyz[accepted_idx]
    child_parent_idx = parent_ids_for_children[accepted_idx]
    # Map each accepted child to a compact accepted-parent row and count how
    # many children survived the G0 gate for each parent.
    accepted_parent_ids, child_parent_local, child_counts = torch.unique(
        child_parent_idx,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    # ------------------------------------------------------------
    # Cache ORIGINAL parent attributes BEFORE touching the parent.

    # ------------------------------------------------------------
    # This fixes the previous double attenuation bug:
    #   parent scale/opacity was reduced first
    #       -> child then inherited that reduced parent
    #       -> child was reduced a second time
    #
    # The result was tiny, transparent children sitting under a still-visible
    # coarse parent, which is exactly the "watercolor underlayer" failure mode.
    source_parent_scale = gaussians.get_scaling[accepted_parent_ids].detach().clone()
    source_parent_opacity = gaussians.get_opacity[accepted_parent_ids].detach().reshape(-1).clone()
    child_source_scale = source_parent_scale[child_parent_local]
    child_source_opacity = source_parent_opacity[child_parent_local]
    child_count_per_child = child_counts[child_parent_local].to(device=device, dtype=child_source_opacity.dtype,)

    # ------------------------------------------------------------
    # Child scale: actual replacement splats, not tiny transparent dust.

    # ------------------------------------------------------------
    anti_watercolor = cfg_bool(cfg, "stage1_detail_anti_watercolor_mode", True,)
    if anti_watercolor:
        # 2-way and 4-way replacement factors are intentionally different.
        # 2-way children need more overlap to avoid a gap through the centre;
        # 4-way cross children can be smaller because they cover two tangent
        # directions.
        factor_single = min(max(cfg_float(cfg, "stage1_detail_child_scale_factor_single", 0.72,), 0.35,), 0.90,)
        factor_2way = min(max(cfg_float(cfg, "stage1_detail_child_scale_factor_2way", 0.62,), 0.35,), 0.80,)
        factor_4way = min(max(cfg_float(cfg, "stage1_detail_child_scale_factor_4way", 0.50,), 0.25,), 0.70,)
        child_factor = torch.full_like(child_count_per_child, factor_single,)
        child_factor = torch.where(child_count_per_child >= 2.0, child_factor.new_tensor(factor_2way), child_factor,)
        child_factor = torch.where(child_count_per_child >= 4.0, child_factor.new_tensor(factor_4way), child_factor,)
        # Extra shrink for truly oversized screen footprints.  This prevents a
        # 20-40 px parent from merely becoming several still-large splats.
        parent_radius_per_child = rr[child_parent_idx].detach().to(device=device, dtype=child_factor.dtype,)
        adapt_strength = min(max(cfg_float(cfg, "stage1_detail_child_radius_adapt_strength", 0.50,), 0.0,), 1.0,)
        radius_ratio = (
            hard_split_radius
            / parent_radius_per_child.clamp_min(hard_split_radius)
        ).clamp(
            cfg_float(cfg, "stage1_detail_child_radius_adapt_min", 0.60,),
            1.0,
        )
        radius_adapt = radius_ratio.pow(adapt_strength)
        child_factor = child_factor * radius_adapt
        child_scale_actual = (child_source_scale * child_factor[:, None])
    else:
        legacy_child_scale_factor = min(
            max(cfg_float(cfg, "stage1_detail_child_scale_factor", 0.48,), 0.20,),
            cfg_float(cfg, "stage1_detail_child_scale_cap", 0.52,),
        )
        child_scale_actual = (child_source_scale * legacy_child_scale_factor)
    new_scaling = torch.log(child_scale_actual.clamp_min(1e-8))
    new_rotation = gaussians._rotation[child_parent_idx].detach().clone()

    # ------------------------------------------------------------
    # Child opacity: coverage-conserving initialization.

    # ------------------------------------------------------------
    if anti_watercolor:
        # If n children overlap exactly, choosing
        #   a_child = 1 - (1-a_parent)^(1/n)
        # preserves the parent's accumulated alpha.  The children are offset,
        # so a small coverage gain compensates for the reduced overlap.
        n_child = child_count_per_child.clamp_min(1.0)
        coverage_child_opacity = (1.0 - torch.pow((1.0 - child_source_opacity).clamp(1e-5, 1.0,), 1.0 / n_child,))
        coverage_gain = max(0.50, cfg_float(cfg, "stage1_detail_child_opacity_coverage_gain", 1.08,),)
        single_factor = min(max(cfg_float(cfg, "stage1_detail_single_child_opacity_factor", 0.60,), 0.10,), 1.0,)
        child_opacity = torch.where(n_child >= 2.0, coverage_child_opacity * coverage_gain, child_source_opacity * single_factor,)
        child_opacity = child_opacity.clamp(
            cfg_float(cfg, "stage1_detail_child_opacity_min", 0.02,),
            cfg_float(cfg, "stage1_detail_child_opacity_max", 0.95,),
        )
    else:
        child_opacity_factor = min(max(cfg_float(cfg, "stage1_detail_child_opacity_factor", 0.50,), 0.05,), 1.0,)
        child_opacity = (child_source_opacity * child_opacity_factor).clamp(1e-4, 1.0 - 1e-4,)
    new_opacity = _inverse_sigmoid_local(child_opacity)[:, None]

    # ------------------------------------------------------------
    # Parent fallback suppression.

    # ------------------------------------------------------------
    # Parents that are successfully represented by >=2 children will normally
    # be deleted after densification_postfix.  We still attenuate all accepted
    # parents first as a safe fallback for repositories without prune_points()
    # and for parents with only one surviving child.
    parent_scale_factor = min(
        max(cfg_float(cfg, "stage1_detail_parent_scale_factor", 0.50 if anti_watercolor else 0.62,), 0.25,),
        cfg_float(cfg, "stage1_detail_parent_scale_cap", 0.60 if anti_watercolor else 0.68,),
    )
    parent_opacity_factor = min(
        max(cfg_float(cfg, "stage1_detail_parent_opacity_factor", 0.18 if anti_watercolor else 0.22,), 0.03,),
        cfg_float(cfg, "stage1_detail_parent_opacity_cap", 0.24 if anti_watercolor else 0.28,),
    )
    gaussians._scaling.data[accepted_parent_ids] = torch.log((source_parent_scale * parent_scale_factor).clamp_min(1e-8))
    new_parent_opacity = (source_parent_opacity * parent_opacity_factor).clamp(1e-4, 1.0 - 1e-4,)
    gaussians._opacity.data[accepted_parent_ids, 0,] = _inverse_sigmoid_local(new_parent_opacity)
    parent_mask = torch.zeros(int(gaussians.get_xyz.shape[0]), dtype=torch.bool, device=device,)
    parent_mask[accepted_parent_ids] = True
    _zero_optimizer_rows(gaussians, gaussians._scaling, parent_mask,)
    _zero_optimizer_rows(gaussians, gaussians._opacity, parent_mask,)
    # Project accepted children and initialize SH-DC directly from the target.
    ones = torch.ones((accepted_xyz.shape[0], 1), device=accepted_xyz.device, dtype=accepted_xyz.dtype,)
    clip = (
        torch.cat([accepted_xyz, ones], dim=-1,)

        @ camera.full_proj_transform.to(device=accepted_xyz.device, dtype=accepted_xyz.dtype,)
    )
    w = clip[:, 3]
    safe_w = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8,), w,)
    ndc = clip[:, :3] / safe_w[:, None]
    cw = int(camera.image_width)
    ch = int(camera.image_height)
    cx = (ndc[:, 0] + 1.0) * 0.5 * (cw - 1)
    cy = (ndc[:, 1] + 1.0) * 0.5 * (ch - 1)
    cvalid = (
        torch.isfinite(cx)
        & torch.isfinite(cy)
        & torch.isfinite(w)
        & (w.abs() > 1e-8)
        & (cx >= 0.0)
        & (cx <= cw - 1)
        & (cy >= 0.0)
        & (cy <= ch - 1)
    )
    cpx = cx.round().long().clamp(0, cw - 1,)
    cpy = cy.round().long().clamp(0, ch - 1,)
    child_rgb = torch.zeros((accepted_xyz.shape[0], 3), device=device, dtype=target_rgb.dtype,)
    use_bilinear_child_color = cfg_bool(cfg, "stage1_detail_bilinear_child_color", True,)
    if bool(cvalid.any()) and use_bilinear_child_color:
        # Sub-pixel color initialization avoids nearest-pixel blockiness after
        # aggressive splitting and is especially useful at 640x640.
        gx = (2.0 * cx / max(cw - 1, 1) - 1.0)
        gy = (2.0 * cy / max(ch - 1, 1) - 1.0)
        sample_grid = torch.stack([gx, gy], dim=-1,).view(1, -1, 1, 2,)
        sampled_rgb = F.grid_sample(
            target_rgb[None],
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[
            0,
            :,
            :,
            0,
        ].transpose(
            0,
            1,
        )
        sampled_mask = F.grid_sample(
            target_mask[None, None].to(dtype=target_rgb.dtype),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[
            0,
            0,
            :,
            0,
        ]
        color_valid = (cvalid & (sampled_mask > 0.50))
        if bool(color_valid.any()):
            child_rgb[color_valid] = sampled_rgb[color_valid]
        cvalid = color_valid
    elif bool(cvalid.any()):
        child_rgb[cvalid] = target_rgb[:, cpy[cvalid], cpx[cvalid],].transpose(0, 1)
    parent_dc = gaussians._features_dc[child_parent_idx].detach().clone()
    new_features_dc = RGB2SH(child_rgb.clamp(0.0, 1.0))[:, None, :]
    if bool((~cvalid).any()):
        new_features_dc[~cvalid] = parent_dc[~cvalid]
    new_features_rest = torch.zeros_like(gaussians._features_rest[child_parent_idx].detach())
    # Identify parents fully represented by surviving children.
    prune_min_children = max(2, cfg_int(cfg, "stage1_detail_parent_prune_min_children", 2,))
    parent_prune_ids = accepted_parent_ids[child_counts >= prune_min_children]
    soft_parent_ids = accepted_parent_ids[child_counts < prune_min_children]

    # SAFETY: densification_postfix only appends children and preserves old row
    # indices. Training-time parent pruning reindexes Gaussian rows, which is
    # unsafe while gaussian_refinement.py still holds row-indexed
    # Anchor/Smooth/KNN references. Therefore parent pruning is disabled unless
    # the caller explicitly opts in after implementing immediate reference rebuilds.
    allow_parent_reindex = cfg_bool(cfg, "stage1_detail_allow_parent_reindex", False)
    prune_split_parents = (
        anti_watercolor
        and cfg_bool(cfg, "stage1_detail_prune_replaced_parents", True)
        and allow_parent_reindex
        and hasattr(gaussians, "prune_points")
    )

    gaussians.densification_postfix(
        accepted_xyz, new_features_dc, new_features_rest,
        new_opacity, new_scaling, new_rotation,
    )

    pruned_parent_count = 0
    if prune_split_parents and parent_prune_ids.numel() > 0:
        prune_mask = torch.zeros(
            int(gaussians.get_xyz.shape[0]),
            dtype=torch.bool,
            device=device,
        )
        prune_mask[parent_prune_ids] = True
        gaussians.prune_points(prune_mask)
        pruned_parent_count = int(parent_prune_ids.numel())

    set_appearance_trainability(
        gaussians, color_dc=True, color_rest=False, opacity=True,
    )
    info["detail_pruned_parents"] = pruned_parent_count
    info["detail_parent_reindexed"] = int(pruned_parent_count > 0)
    info["detail_deferred_parent_prune"] = (
        int(parent_prune_ids.numel()) if not prune_split_parents else 0
    )
    info["detail_soft_suppressed_parents"] = (
        int(soft_parent_ids.numel())
        + int(parent_prune_ids.numel() if not prune_split_parents else 0)
    )
    chosen = score[parent_idx]
    info["detail_split_parents"] = int(accepted_parent_ids.numel())
    info["detail_added"] = int(accepted_xyz.shape[0])
    info["detail_mean_score"] = float(chosen.mean().item())
    info["detail_mean_residual"] = float(residual[parent_idx].mean().item())
    info["detail_mean_radius"] = float(rr[parent_idx].mean().item())
    info["detail_mean_abs_grad"] = float(abs_grad[parent_idx].mean().item())
    info["detail_mean_child_scale"] = float(child_scale_actual.mean().item() if child_scale_actual.numel() > 0 else 0.0)
    info["detail_mean_child_opacity"] = float(child_opacity.mean().item() if child_opacity.numel() > 0 else 0.0)
    # Keep some memory of persistent difficult regions, but do not repeatedly
    # select exactly the same parents forever.
    _decay_abs_grad_state(cfg_float(cfg, "stage1_detail_abs_grad_decay", 0.20,))
    return info


@torch.no_grad()
def compute_coverage_target(gaussians, cfg):
    """

    Build a detached local 3D spacing target for Gaussian surface coverage.

    """
    n = int(gaussians.get_xyz.shape[0])
    if n < 2:
        target = torch.zeros(n, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype,)
        return target, {"spacing_median": 0.0, "target_median": 0.0,}
    xyz = gaussians.get_xyz.detach().contiguous()
    spacing = torch.sqrt(torch.clamp_min(distCUDA2(xyz), 1e-12))
    finite = torch.isfinite(spacing) & (spacing > 0)
    if not bool(finite.any()):
        target = torch.zeros_like(spacing)
        return target, {"spacing_median": 0.0, "target_median": 0.0,}
    spacing_median = torch.quantile(spacing[finite], 0.50)
    cap_mult = max(1.0, cfg_float(cfg, "stage1_coverage_spacing_cap_mult", 3.0),)
    spacing = torch.minimum(spacing, spacing_median * cap_mult,)
    beta = min(max(cfg_float(cfg, "stage1_coverage_beta", 0.55), 0.0), 2.0,)
    target = (spacing * beta).detach()
    target_median = (torch.quantile(target[finite], 0.50) if bool(finite.any()) else target.new_tensor(0.0))
    return target, {"spacing_median": float(spacing_median.item()), "target_median": float(target_median.item()),}


def local_coverage_loss(gaussians, coverage_target, visibility, cfg):
    """

    Penalize only visible Gaussians whose in-surface scale is too small

    for their local nearest-neighbour spacing.

    """
    n = int(gaussians.get_xyz.shape[0])
    if (n == 0 or coverage_target is None or coverage_target.shape[0] != n):
        zero = gaussians.get_xyz.sum() * 0.0
        return zero, 0.0
    scale = gaussians.get_scaling
    # For surface Gaussians, use the two largest axes as tangential footprint.
    k = min(2, scale.shape[1])
    effective_scale = torch.topk(scale, k=k, dim=1, largest=True,).values.mean(dim=1)
    target = coverage_target.to(device=effective_scale.device, dtype=effective_scale.dtype,)
    valid = torch.isfinite(target) & (target > 1e-8)
    if visibility is not None and visibility.shape[0] == n:
        valid &= visibility.detach().bool()
    min_opacity = cfg_float(cfg, "stage1_coverage_min_opacity", 0.05,)
    valid &= (gaussians.get_opacity.detach().squeeze(-1) >= min_opacity)
    if not bool(valid.any()):
        zero = effective_scale.sum() * 0.0
        return zero, 0.0
    deficit = F.relu(target - effective_scale)
    relative_deficit = deficit / target.clamp_min(1e-8)
    loss = relative_deficit[valid].square().mean()
    with torch.no_grad():
        under_ratio = float((effective_scale[valid] < target[valid]).float().mean().item())
    return loss, under_ratio


@torch.no_grad()
def foreground_safe_color_map(target, target_mask, radius):
    """Local foreground-only RGB average used for newly seeded Gaussians."""
    radius = max(0, int(radius))
    if radius == 0:
        return target * target_mask[None]
    kernel = 2 * radius + 1
    area = float(kernel * kernel)
    rgb_sum = F.avg_pool2d((target * target_mask[None])[None], kernel_size=kernel, stride=1, padding=radius)[0] * area
    mask_sum = F.avg_pool2d(target_mask[None, None], kernel_size=kernel, stride=1, padding=radius)[0, 0] * area
    return (rgb_sum / mask_sum.clamp_min(1e-6)[None]).clamp(0.0, 1.0)


@torch.no_grad()
def enforce_minimum_gaussian_scale(gaussians, visibility, cfg, protected_count=0):
    """Immediately enlarge clearly under-sized visible NEW splats after topology updates."""
    if not cfg_bool(cfg, "stage1_enable_scale_floor", False):
        return {"scale_floor_candidates": 0, "scale_floor_adjusted": 0, "scale_floor_mean_boost": 1.0}
    n = int(gaussians.get_xyz.shape[0])
    if n < 2:
        return {"scale_floor_candidates": 0, "scale_floor_adjusted": 0, "scale_floor_mean_boost": 1.0}
    xyz = gaussians.get_xyz.detach().contiguous()
    spacing = torch.sqrt(torch.clamp_min(distCUDA2(xyz), 1e-12))
    finite = torch.isfinite(spacing) & (spacing > 0)
    if not bool(finite.any()):
        return {"scale_floor_candidates": 0, "scale_floor_adjusted": 0, "scale_floor_mean_boost": 1.0}
    median = torch.quantile(spacing[finite], 0.50)
    spacing = torch.minimum(spacing, median * cfg_float(cfg, "stage1_scale_floor_spacing_cap_mult", 3.0))
    beta = min(max(cfg_float(cfg, "stage1_scale_floor_beta", 0.68), 0.0), 2.0)
    desired = spacing * beta
    scale = gaussians.get_scaling.detach()
    effective = torch.topk(scale, k=min(2, scale.shape[1]), dim=1, largest=True).values.mean(dim=1)
    candidates = finite & (effective < desired)
    pc = max(0, min(int(protected_count), n))
    if pc > 0:
        candidates[:pc] = False
    if visibility is not None and visibility.shape[0] == n:
        candidates &= visibility.bool()
    candidates &= gaussians.get_opacity.detach().squeeze(-1) >= cfg_float(cfg, "stage1_scale_floor_min_opacity", 0.03)
    count = int(candidates.sum().item())
    if count == 0:
        return {"scale_floor_candidates": 0, "scale_floor_adjusted": 0, "scale_floor_mean_boost": 1.0}
    factor = (desired / effective.clamp_min(1e-8)).clamp_min(1.0)
    factor = factor.clamp_max(cfg_float(cfg, "stage1_scale_floor_max_boost", 1.25))
    factor = torch.where(candidates, factor, torch.ones_like(factor))
    new_scale = scale * factor[:, None]
    gaussians._scaling.data.copy_(torch.log(new_scale.clamp_min(1e-8)))
    return {
        "scale_floor_candidates": count,
        "scale_floor_adjusted": int((factor > 1.0001).sum().item()),
        "scale_floor_mean_boost": float(factor[candidates].mean().item()),
    }


@torch.no_grad()
def native_densify_with_budget(gaussians, scene, cfg, max_total_points):
    if not hasattr(gaussians, "densify_and_prune"):
        raise RuntimeError("GaussianModel has no densify_and_prune()")
    before = int(gaussians.get_xyz.shape[0])
    budget = max(0, int(max_total_points) - before)
    threshold = cfg_float(cfg, "stage1_densify_grad_threshold", 0.0007)
    grads = (gaussians.xyz_gradient_accum / gaussians.denom)
    grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    candidate_idx = torch.nonzero(grads >= threshold, as_tuple=False).squeeze(-1)
    candidate_count = int(candidate_idx.numel())
    if budget <= 0:
        gaussians.xyz_gradient_accum.zero_()
        gaussians.denom.fill_(1.0)
    elif candidate_count > budget:
        candidate_grads = grads[candidate_idx]
        selected_local = torch.topk(candidate_grads, k=budget, largest=True, sorted=False).indices
        selected_idx = candidate_idx[selected_local]
        keep = torch.zeros_like(grads, dtype=torch.bool)
        keep[selected_idx] = True
        gaussians.xyz_gradient_accum[~keep] = 0.0
        gaussians.denom[~keep] = 1.0
    # IMPORTANT:
    # min_opacity=0 and max_screen_size=None intentionally disable global
    # pruning inside this call. Explicit extra-region pruning is handled below.
    gaussians.densify_and_prune(threshold, 0.0, scene.cameras_extent, None)
    after = int(gaussians.get_xyz.shape[0])
    set_appearance_trainability(gaussians, color_dc=True, color_rest=False, opacity=True)
    return {
        "native_before": before,
        "native_candidates": candidate_count,
        "native_budget": budget,
        "native_after": after,
        "native_added": max(0, after - before),
    }


@torch.no_grad()
def select_missing_seed_pixels(missing_mask, cfg):
    if cfg_bool(cfg, "stage1_strict_surface_fill", True):
        stride = max(1, cfg_int(cfg, "stage1_strict_missing_seed_stride", 1))
    else:
        stride = max(1, cfg_int(cfg, "stage1_missing_seed_stride", 4))
    h, w = missing_mask.shape
    yy = torch.arange(0, h, stride, device=missing_mask.device)
    xx = torch.arange(0, w, stride, device=missing_mask.device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    keep = (missing_mask[gy, gx] > 0.5)
    coords = torch.stack([gx[keep].float(), gy[keep].float()], dim=-1)
    if cfg_bool(cfg, "stage1_strict_surface_fill", True):
        max_seeds = max(0, cfg_int(cfg, "stage1_strict_missing_max_seeds_per_update", 4096))
    else:
        max_seeds = max(0, cfg_int(cfg, "stage1_missing_max_seeds_per_update", 1024))
    if (max_seeds == 0 or coords.shape[0] == 0):
        return coords[:0]
    if coords.shape[0] > max_seeds:
        ids = torch.linspace(0, coords.shape[0] - 1, steps=max_seeds, device=coords.device).round().long()
        coords = coords[ids]
    return coords


@torch.no_grad()
def nearest_parent_indices(seed_pixels, projected_xy, candidate_indices, max_distance_px, chunk=128):
    if (seed_pixels.numel() == 0 or candidate_indices.numel() == 0):
        empty_long = torch.empty(0, dtype=torch.long, device=seed_pixels.device)
        empty_float = torch.empty(0, dtype=seed_pixels.dtype, device=seed_pixels.device)
        return (empty_long, empty_long, empty_float)
    candidate_xy = projected_xy[candidate_indices]
    seed_result = []
    parent_result = []
    distance_result = []
    for start in range(0, seed_pixels.shape[0], chunk):
        seeds = seed_pixels[start:start + chunk]
        distance = torch.cdist(seeds, candidate_xy, p=2.0)
        min_distance, local_idx = (distance.min(dim=1))
        ok = (min_distance <= float(max_distance_px))
        if bool(ok.any()):
            seed_ids = torch.arange(start, start + seeds.shape[0], device=seed_pixels.device)[ok]
            parent_ids = (candidate_indices[local_idx[ok]])
            seed_result.append(seed_ids)
            parent_result.append(parent_ids)
            distance_result.append(min_distance[ok])
    if not seed_result:
        empty_long = torch.empty(0, dtype=torch.long, device=seed_pixels.device)
        empty_float = torch.empty(0, dtype=seed_pixels.dtype, device=seed_pixels.device)
        return (empty_long, empty_long, empty_float)
    return (torch.cat(seed_result), torch.cat(parent_result), torch.cat(distance_result))


@torch.no_grad()
def seed_missing_gaussians(
    gaussians,
    camera,
    target,
    target_mask,
    missing_mask,
    visibility,
    cfg,
    remaining_budget,
    seed_erode_override=None,
    raw_g0_surface_prior=None,
):
    if remaining_budget <= 0:
        return {"seed_requested": 0, "seed_added": 0, "seed_mean_parent_distance_px": 0.0}
    if not hasattr(gaussians, "densification_postfix"):
        raise RuntimeError("GaussianModel has no densification_postfix().")
    if seed_erode_override is None:
        seed_erode = max(0, cfg_int(cfg, "stage1_seed_foreground_erode", 2),)
    else:
        seed_erode = max(0, int(seed_erode_override))
    safe_foreground = erode_mask(target_mask, seed_erode)
    safe_missing = missing_mask * (safe_foreground > 0.5).float()
    seed_pixels = select_missing_seed_pixels(safe_missing, cfg)
    requested = int(seed_pixels.shape[0])
    if requested == 0:
        return {"seed_requested": 0, "seed_added": 0, "seed_mean_parent_distance_px": 0.0}
    (x, y, _, _, ndc_z, valid) = project_gaussians(gaussians, camera)
    projected_xy = torch.stack([x, y], dim=-1)
    candidate_mask = valid.clone()
    if (visibility is not None and visibility.shape[0] == candidate_mask.shape[0]):
        candidate_mask &= (visibility.bool())
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
    if candidate_indices.numel() == 0:
        candidate_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    max_parent_distance = cfg_float(cfg, "stage1_missing_parent_max_distance_px", 56.0)
    (
        seed_ids,
        parent_idx,
        parent_distance,
    ) = nearest_parent_indices(
        seed_pixels,
        projected_xy,
        candidate_indices,
        max_parent_distance,
        chunk=cfg_int(cfg, "stage1_missing_nn_chunk", 128),
    )
    if seed_ids.numel() == 0:
        return {"seed_requested": requested, "seed_added": 0, "seed_mean_parent_distance_px": 0.0}
    if (seed_ids.numel() > remaining_budget):
        # Closer parent -> more conservative 3D hypothesis.
        order = torch.argsort(parent_distance)
        keep = order[:remaining_budget]
        seed_ids = seed_ids[keep]
        parent_idx = parent_idx[keep]
        parent_distance = (parent_distance[keep])
    chosen_pixels = seed_pixels[seed_ids]
    target_world = (
        unproject_pixels_at_ndc_depth(
            camera,
            chosen_pixels,
            ndc_z[parent_idx],
            dtype=gaussians.get_xyz.dtype,
            device=gaussians.get_xyz.device,
        )
    )
    parent_xyz = gaussians.get_xyz[parent_idx]
    step = cfg_float(cfg, "stage1_missing_seed_step", 1.0)
    step = min(max(step, 0.05), 1.0)
    new_xyz = (parent_xyz + step * (target_world - parent_xyz))
    # Birth-time G0 structure gate. Newly seeded points must remain near the
    # original canonical surface so they form a coherent object block instead
    # of detached sheets / floaters.
    if raw_g0_surface_prior is not None:
        keep_birth, birth_distance = raw_g0_support_mask_for_xyz(new_xyz, raw_g0_surface_prior, cfg,)
        if not bool(keep_birth.any()):
            return {
                "seed_requested": requested,
                "seed_added": 0,
                "seed_g0_rejected": int(new_xyz.shape[0]),
                "seed_mean_parent_distance_px": (float(parent_distance.mean().item()) if parent_distance.numel() > 0 else 0.0),
            }
        rejected_birth = int((~keep_birth).sum().item())
        new_xyz = new_xyz[keep_birth]
        chosen_pixels = chosen_pixels[keep_birth]
        parent_idx = parent_idx[keep_birth]
        parent_distance = parent_distance[keep_birth]
    else:
        rejected_birth = 0
    color_map = foreground_safe_color_map(target, target_mask, cfg_int(cfg, "stage1_seed_color_radius", 2))
    target_rgb = color_map[:, chosen_pixels[:, 1].round().long(), chosen_pixels[:, 0].round().long()].transpose(0, 1).contiguous()
    new_features_dc = RGB2SH(target_rgb.clamp(0.0, 1.0))[:, None, :]
    new_features_rest = torch.zeros_like(gaussians._features_rest[parent_idx])
    seed_opacity = cfg_float(cfg, "stage1_seed_surface_opacity", 0.18)
    seed_opacity = min(max(seed_opacity, 1e-4), 1.0 - 1e-4)
    seed_logit = math.log(seed_opacity / (1.0 - seed_opacity))
    new_opacity = torch.full_like(gaussians._opacity[parent_idx], seed_logit)
    # Adaptive seed scale: preserve parent anisotropy, but do not create tiny
    # splats that leave 3D gaps. The local target comes from the parent's 3D
    # nearest-neighbor spacing.
    parent_scale = gaussians.get_scaling[parent_idx].detach()
    base_factor = cfg_float(cfg, "stage1_seed_surface_scale_factor", 1.00)
    base_factor = min(max(base_factor, 0.90), 1.25)
    base_scale = parent_scale * base_factor
    if gaussians.get_xyz.shape[0] >= 2:
        parent_nn = torch.sqrt(torch.clamp_min(distCUDA2(gaussians.get_xyz.detach().contiguous()), 1e-12))[parent_idx]
        seed_coverage_beta = min(max(cfg_float(cfg, "stage1_seed_surface_coverage_beta", 0.75), 0.0), 2.0)
        desired_effective = parent_nn * seed_coverage_beta
        base_effective = torch.topk(base_scale, k=min(2, base_scale.shape[1]), dim=1, largest=True).values.mean(dim=1)
        boost = (desired_effective / base_effective.clamp_min(1e-8)).clamp_min(1.0)
        boost = boost.clamp_max(cfg_float(cfg, "stage1_seed_surface_scale_max_boost", 1.35))
        new_scale_actual = base_scale * boost[:, None]
    else:
        new_scale_actual = base_scale
    new_scaling = torch.log(new_scale_actual.clamp_min(1e-8))
    new_rotation = (gaussians._rotation[parent_idx].detach().clone())
    gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
    set_appearance_trainability(gaussians, color_dc=True, color_rest=False, opacity=True)
    return {
        "seed_requested": requested,
        "seed_added": int(new_xyz.shape[0]),
        "seed_g0_rejected": int(rejected_birth),
        "seed_mean_parent_distance_px": (float(parent_distance.mean().item()) if parent_distance.numel() > 0 else 0.0),
        "seed_scale_mean": float(new_scale_actual.mean().item()),
    }


@torch.no_grad()
def refill_missing_after_absolute_prune(
    gaussians,
    camera,
    target,
    target_mask,
    cfg,
    black_background,
    max_total_points,
    raw_g0_surface_prior=None,
):
    """

    After outside-contour pruning, immediately re-render and refill any missing

    pixels inside the RGB subject mask.

    This repair reaches all the way to the true silhouette boundary.

    """
    info = {"refill_passes": 0, "refill_added": 0, "refill_missing_px": 0,}
    max_passes = max(1, cfg_int(cfg, "stage1_continuous_refill_passes", 2),)
    height = int(camera.image_height)
    width = int(camera.image_width)
    for _ in range(max_passes):
        info["refill_passes"] += 1
        package = render(camera, gaussians, cfg, black_background, 0.0, 0.0, 0.0, cfg.is_6dof,)
        alpha_direct = _package_alpha(package, height, width,)
        if alpha_direct is not None:
            alpha = alpha_direct.detach().clamp(0.0, 1.0)
        else:
            white_background = torch.ones(3, dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device,)
            white_package = render(camera, gaussians, cfg, white_background, 0.0, 0.0, 0.0, cfg.is_6dof,)
            alpha = (1.0 - (white_package["render"] - package["render"]).mean(dim=0)).clamp(0.0, 1.0)
        missing, _ = structure_masks(target_mask, alpha, cfg,)
        missing_count = int((missing > 0.5).sum().item())
        info["refill_missing_px"] = missing_count
        if missing_count <= 0:
            break
        remaining_budget = max(0, int(max_total_points) - int(gaussians.get_xyz.shape[0]),)
        if remaining_budget <= 0:
            break
        seed_info = seed_missing_gaussians(
            gaussians,
            camera,
            target,
            target_mask,
            missing,
            package.get("visibility_filter", None),
            cfg,
            remaining_budget,
            seed_erode_override=0,
            raw_g0_surface_prior=raw_g0_surface_prior,
        )
        added = int(seed_info.get("seed_added", 0))
        info["refill_added"] += added
        if added <= 0:
            break
    return info
