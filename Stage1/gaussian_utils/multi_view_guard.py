"""Visibility-aware multi-view pruning protection for Stage-1 Gaussian refinement.
This V2 module never adds a multi-view optimization loss and never hallucinates RGB.
It uses the untouched raw G0 only to discover side/back-supporting Gaussians, then
vetoes destructive single-view pruning/shrinking when a current Gaussian remains
close to one of those protected raw-G0 anchors. The real RGB view remains the only
appearance/geometry supervision source.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
def _cfg_bool(cfg, name: str, default: bool) -> bool:
    value = getattr(cfg, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")
def _cfg_float(cfg, name: str, default: float) -> float:
    try:
        return float(getattr(cfg, name, default))
    except (TypeError, ValueError):
        return float(default)
def _cfg_text(cfg, name: str, default: str) -> str:
    return str(getattr(cfg, name, default)).strip()
def _parse_views(text: str) -> List[Tuple[float, float]]:
    views: List[Tuple[float, float]] = []
    for token in str(text).replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            yaw_s, pitch_s = token.split(":", 1)
        elif "/" in token:
            yaw_s, pitch_s = token.split("/", 1)
        else:
            yaw_s, pitch_s = token, "0"
        try:
            yaw = float(yaw_s.strip())
            pitch = float(pitch_s.strip())
        except ValueError:
            continue
        if abs(yaw) < 1e-8 and abs(pitch) < 1e-8:
            continue
        views.append((yaw, pitch))
    return views
def _axis_angle(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-12)
    x, y, z = axis.unbind()
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    oc = 1.0 - c
    return torch.stack([
        torch.stack([c + x*x*oc, x*y*oc - z*s, x*z*oc + y*s]),
        torch.stack([y*x*oc + z*s, c + y*y*oc, y*z*oc - x*s]),
        torch.stack([z*x*oc - y*s, z*y*oc + x*s, c + z*z*oc]),
    ])
def _robust_object_center(xyz: torch.Tensor) -> torch.Tensor:
    xyz = xyz.detach()
    if xyz.shape[0] > 200_000:
        step = max(1, int(xyz.shape[0] // 200_000))
        xyz = xyz[::step][:200_000]
    xyz64 = xyz.to(dtype=torch.float64)
    try:
        lo = torch.quantile(xyz64, 0.05, dim=0)
        hi = torch.quantile(xyz64, 0.95, dim=0)
        return 0.5 * (lo + hi)
    except Exception:
        return xyz64.median(dim=0).values
def _make_orbit_camera(base_camera, center: torch.Tensor, yaw_deg: float, pitch_deg: float):
    if not hasattr(base_camera, "world_view_transform") or not hasattr(base_camera, "projection_matrix"):
        raise RuntimeError("Camera is missing world_view_transform/projection_matrix")
    wvt = base_camera.world_view_transform.detach()
    device = wvt.device
    out_dtype = wvt.dtype
    w2c = wvt.transpose(0, 1).to(dtype=torch.float64)
    c2w = torch.linalg.inv(w2c)
    rot_c2w = c2w[:3, :3]
    cam_center = c2w[:3, 3]
    center = center.to(device=device, dtype=torch.float64)
    right_axis = rot_c2w[:, 0]
    up_axis = rot_c2w[:, 1]
    r_yaw = _axis_angle(up_axis, math.radians(float(yaw_deg)))
    rotated_right = r_yaw @ right_axis
    r_pitch = _axis_angle(rotated_right, math.radians(float(pitch_deg)))
    r_orbit = r_pitch @ r_yaw
    new_rot_c2w = r_orbit @ rot_c2w
    new_center = center + r_orbit @ (cam_center - center)
    new_c2w = torch.eye(4, dtype=torch.float64, device=device)
    new_c2w[:3, :3] = new_rot_c2w
    new_c2w[:3, 3] = new_center
    new_w2c = torch.linalg.inv(new_c2w)
    cam = copy.copy(base_camera)
    cam.world_view_transform = new_w2c.transpose(0, 1).to(dtype=out_dtype)
    cam.projection_matrix = base_camera.projection_matrix
    cam.full_proj_transform = cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0)).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    try:
        cam.R = new_w2c[:3, :3].transpose(0, 1).detach().cpu().numpy().astype(np.float32)
        cam.T = new_w2c[:3, 3].detach().cpu().numpy().astype(np.float32)
        cam.image_name = f"mv_y{yaw_deg:+.0f}_p{pitch_deg:+.0f}"
    except Exception:
        pass
    return cam
def _save_gray(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().float().cpu().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
@dataclass
class ProtectionView:
    yaw_deg: float
    pitch_deg: float
    camera: object
    teacher_alpha: Optional[torch.Tensor]
    raw_visible_count: int
@dataclass
class MultiViewGuardState:
    enabled: bool
    views: List[ProtectionView]
    raw_protected_mask: torch.Tensor
    protected_xyz: np.ndarray
    match_radius: float
    teacher_alpha_min: float
    diagnostics_dir: Optional[Path]
    tree: object = None
    calls: int = 0
    veto_center: int = 0
    veto_support: int = 0
    veto_shrink: int = 0
    nonfront_exempt: int = 0
    raw_side_visible: int = 0
    raw_protected: int = 0
    category_counts: dict = field(default_factory=dict)
    raw_anchor_xyz: Optional[np.ndarray] = None
    anchor_thresholds: Optional[np.ndarray] = None
    spacing_median: float = 0.0
    gap_veto: int = 0
    gap_checks: int = 0
def _empty_state() -> MultiViewGuardState:
    return MultiViewGuardState(False, [], torch.zeros(0, dtype=torch.bool), np.empty((0,3), dtype=np.float32), 0.0, 0.05, None)
def _visibility_mask_from_package(pkg: dict, n: int, min_radius: float = 0.0) -> torch.Tensor:
    vis = pkg.get("visibility_filter", None)
    if isinstance(vis, torch.Tensor) and vis.ndim == 1 and int(vis.shape[0]) == n:
        return vis.detach().bool()
    radii = pkg.get("radii", None)
    if isinstance(radii, torch.Tensor) and radii.ndim == 1 and int(radii.shape[0]) == n:
        return radii.detach() > float(min_radius)
    return torch.zeros(n, dtype=torch.bool, device=next(v.device for v in pkg.values() if isinstance(v, torch.Tensor)))
def prepare_multi_view_guard(gaussians, base_camera, initial_visibility, donor_bank, cfg, render_fn: Callable, alpha_fn: Callable, background: torch.Tensor, output_dir: Path) -> MultiViewGuardState:
    if not _cfg_bool(cfg, "stage1_enable_multiview_guard", True):
        return _empty_state()
    view_specs = _parse_views(_cfg_text(cfg, "stage1_mv_views", "-60:0;-30:0;30:0;60:0;0:-25;0:25"))
    if not view_specs:
        raise ValueError("stage1_enable_multiview_guard=True but stage1_mv_views is empty")
    n = int(gaussians.get_xyz.shape[0])
    if n == 0:
        return _empty_state()
    center = _robust_object_center(gaussians.get_xyz)
    min_radius = max(0.0, _cfg_float(cfg, "stage1_mv_min_projected_radius", 0.0))
    teacher_alpha_min = max(0.0, _cfg_float(cfg, "stage1_mv_teacher_alpha_min", 0.05))
    diag_dir = Path(output_dir) / "multiview_guard"
    diag_dir.mkdir(parents=True, exist_ok=True)
    side_visible = torch.zeros(n, dtype=torch.bool, device=gaussians.get_xyz.device)
    views: List[ProtectionView] = []
    with torch.no_grad():
        for idx, (yaw, pitch) in enumerate(view_specs):
            cam = _make_orbit_camera(base_camera, center, yaw, pitch)
            pkg = render_fn(cam, gaussians, cfg, background, 0.0, 0.0, 0.0, cfg.is_6dof)
            view_vis = _visibility_mask_from_package(pkg, n, min_radius)
            side_visible |= view_vis.to(side_visible.device)
            alpha = None
            try:
                alpha_result = alpha_fn(cam, gaussians, cfg, pkg)
                alpha = alpha_result[0] if isinstance(alpha_result, tuple) else alpha_result
                if isinstance(alpha, torch.Tensor):
                    alpha = alpha.detach().clamp(0.0, 1.0)
                    _save_gray(alpha, diag_dir / f"view_{idx:02d}_yaw_{yaw:+.0f}_pitch_{pitch:+.0f}_raw_g0_alpha.png")
            except Exception:
                alpha = None
            views.append(ProtectionView(float(yaw), float(pitch), cam, alpha, int(view_vis.sum().item())))
    visible = torch.zeros(n, dtype=torch.bool)
    hidden = torch.zeros(n, dtype=torch.bool)
    uncertain = torch.zeros(n, dtype=torch.bool)
    outside = torch.zeros(n, dtype=torch.bool)
    if isinstance(initial_visibility, dict):
        for key, dst in (("visible_mask", visible),("hidden_mask", hidden),("uncertain_mask", uncertain),("outside_mask", outside)):
            src = initial_visibility.get(key, None)
            if isinstance(src, torch.Tensor) and int(src.shape[0]) == n:
                dst.copy_(src.detach().bool().cpu())
    side_cpu = side_visible.detach().bool().cpu()
    allowed = torch.zeros(n, dtype=torch.bool)
    if _cfg_bool(cfg, "stage1_mv_protect_hidden", True):
        allowed |= hidden
    if _cfg_bool(cfg, "stage1_mv_protect_uncertain", True):
        allowed |= uncertain
    if _cfg_bool(cfg, "stage1_mv_protect_outside", True):
        allowed |= outside
    if _cfg_bool(cfg, "stage1_mv_protect_visible_front", False):
        allowed |= visible
    if not bool(allowed.any()):
        allowed = ~visible
    protected = side_cpu & allowed
    min_opacity = max(0.0, _cfg_float(cfg, "stage1_mv_min_raw_opacity", 0.03))
    if isinstance(donor_bank, dict) and isinstance(donor_bank.get("opacity", None), torch.Tensor):
        op = torch.sigmoid(donor_bank["opacity"].reshape(n, -1)[:, 0])
        protected &= op >= min_opacity
    raw_xyz = donor_bank.get("xyz", None) if isinstance(donor_bank, dict) else None
    if not isinstance(raw_xyz, torch.Tensor) or int(raw_xyz.shape[0]) != n:
        raw_xyz = gaussians.get_xyz.detach().float().cpu()
    protected_xyz = raw_xyz[protected].detach().float().cpu().numpy().astype(np.float32, copy=False)
    spacing = float(donor_bank.get("spacing_median", 0.0)) if isinstance(donor_bank, dict) else 0.0
    radius = max(_cfg_float(cfg, "stage1_mv_match_min_distance", 0.0), spacing * max(0.1, _cfg_float(cfg, "stage1_mv_match_spacing_mult", 1.75)))
    if radius <= 1e-12 and protected_xyz.shape[0] > 0:
        scale = gaussians.get_scaling.detach().float().cpu()
        eff = torch.topk(scale, k=min(2, scale.shape[1]), dim=1, largest=True).values.mean(dim=1)
        finite = torch.isfinite(eff) & (eff > 0)
        if bool(finite.any()):
            radius = max(float(torch.quantile(eff[finite], 0.50).item()), 1e-8)
    # Local G0 scale rather than a single global Gaussian-gap threshold.
    # kNN includes self as index 0. The 2nd neighbor serves as a local scale;
    # local threshold is limited by robust global quantiles to avoid connecting
    # unrelated parts across large geometric discontinuities.
    from scipy.spatial import cKDTree
    raw_np = raw_xyz.detach().float().cpu().numpy().astype(np.float32, copy=False)
    raw_tree = cKDTree(raw_np)
    nearest, _ = raw_tree.query(raw_np, k=min(3, n), workers=-1)
    local_spacing = nearest[:, 1] if n >= 2 else np.full(n, 1e-6, dtype=np.float32)
    positive = local_spacing[np.isfinite(local_spacing) & (local_spacing > 1e-10)]
    median_spacing = float(np.median(positive)) if positive.size else 1e-6
    spacing_cap = float(np.quantile(positive, 0.95)) if positive.size else median_spacing
    local_spacing = np.where(local_spacing > 1e-10, local_spacing, median_spacing)
    spacing_mult = max(1.01, _cfg_float(cfg, "stage1_mv_gap_spacing_mult", 2.25))
    # Raw G0 itself may contain separate components, so do not enforce bridges
    # over distances substantially larger than the 95th-percentile spacing.
    cap_mult = max(1.0, _cfg_float(cfg, "stage1_mv_gap_cap_mult", 1.5))
    anchor_thresholds = np.maximum(1e-8, np.minimum(local_spacing * spacing_mult, spacing_cap * cap_mult))
    tree = None
    if protected_xyz.shape[0] > 0:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(protected_xyz)
        except Exception as exc:
            raise RuntimeError("Multi-view pruning protection requires scipy.cKDTree. Install scipy in ed3dgs.") from exc
    counts = {
        "visible": int((protected & visible).sum().item()),
        "hidden": int((protected & hidden).sum().item()),
        "uncertain": int((protected & uncertain).sum().item()),
        "outside": int((protected & outside).sum().item()),
    }
    state = MultiViewGuardState(True, views, protected, protected_xyz, float(radius), teacher_alpha_min, diag_dir, tree=tree, raw_side_visible=int(side_cpu.sum().item()), raw_protected=int(protected.sum().item()), category_counts=counts, raw_anchor_xyz=raw_np[protected.cpu().numpy()], anchor_thresholds=anchor_thresholds[protected.cpu().numpy()], spacing_median=median_spacing)
    with (diag_dir / "protection_summary.txt").open("w", encoding="utf-8") as f:
        f.write(multi_view_guard_summary(state) + "\n")
    return state
def multi_view_guard_summary(state: MultiViewGuardState) -> str:
    if not state.enabled:
        return "enabled=False"
    view_text = ";".join(f"{v.yaw_deg:+.0f}:{v.pitch_deg:+.0f}" for v in state.views)
    c = state.category_counts
    return (f"enabled=True mode=pruning_veto views={len(state.views)} [{view_text}] raw_side_visible={state.raw_side_visible:,} raw_protected={state.raw_protected:,} hidden={c.get('hidden',0):,} uncertain={c.get('uncertain',0):,} outside={c.get('outside',0):,} front={c.get('visible',0):,} match_radius={state.match_radius:.8g} raw_spacing_median={state.spacing_median:.8g} gap_veto={state.gap_veto:,}")
def _query_veto_mask(gaussians, candidate_mask: torch.Tensor, state: MultiViewGuardState) -> torch.Tensor:
    """Protect only the UNIQUE current contributor to a G0-backed local surface.

    A current Gaussian is protected iff it is the closest contributor to at
    least one protected raw G0 anchor and removing it would leave that anchor
    farther than its LOCAL raw-G0 threshold from every surviving Gaussian.
    """
    out = torch.zeros_like(candidate_mask)
    if not state.enabled or state.tree is None or not bool(candidate_mask.any()):
        return out
    from scipy.spatial import cKDTree
    xyz = gaussians.get_xyz.detach().float().cpu().numpy()
    if xyz.shape[0] < 2:
        return out
    candidates = candidate_mask.detach().cpu().numpy().astype(bool)
    anchor_xyz = state.raw_anchor_xyz
    thresholds = state.anchor_thresholds
    if anchor_xyz is None or len(anchor_xyz) == 0:
        return out
    current_tree = cKDTree(xyz)
    # Query in blocks to control host memory on large point clouds.
    protected_ids = set()
    for start in range(0, len(anchor_xyz), 50000):
        pts = anchor_xyz[start:start+50000]
        thr = thresholds[start:start+50000]
        distances, inds = current_tree.query(pts, k=2, workers=-1)
        first, second = distances[:, 0], distances[:, 1]
        first_idx = inds[:, 0]
        # If already missing, refusing deletion cannot fix it; recovery owns it.
        at_risk = (first <= thr) & (second > thr) & candidates[first_idx]
        protected_ids.update(first_idx[at_risk].tolist())
    state.gap_checks += 1
    state.gap_veto += len(protected_ids)
    if protected_ids:
        ids = torch.as_tensor(sorted(protected_ids), device=out.device, dtype=torch.long)
        out[ids] = True
    return out

@torch.no_grad()
def validate_g0_surface_distances(gaussians, state: MultiViewGuardState, strict: bool = False) -> dict:
    """Audit protected raw-G0 surface occupancy against local adaptive distances.

    This checks anchor-to-current coverage, not global connectivity or mesh topology.
    It cannot certify genuine surface continuity from centers alone.
    """
    result = {"checked": 0, "violations": 0, "violation_ratio": 0.0,
              "max_distance_ratio": 0.0, "raw_spacing_median": state.spacing_median}
    if not state.enabled or state.raw_anchor_xyz is None or len(state.raw_anchor_xyz) == 0:
        return result
    from scipy.spatial import cKDTree
    xyz = gaussians.get_xyz.detach().float().cpu().numpy()
    if len(xyz) == 0:
        raise RuntimeError("Selective guard: all Gaussian points have been removed")
    tree = cKDTree(xyz)
    distances, _ = tree.query(state.raw_anchor_xyz, k=1, workers=-1)
    thresholds = np.maximum(state.anchor_thresholds, 1e-8)
    ratio = distances / thresholds
    violations = ratio > (1.0 + 1e-5)
    result.update(checked=int(len(ratio)), violations=int(violations.sum()),
                  violation_ratio=float(violations.mean()),
                  max_distance_ratio=float(np.max(ratio)))
    if state.diagnostics_dir is not None:
        with (state.diagnostics_dir / "surface_gap_audit.txt").open("w", encoding="utf-8") as f:
            for k, v in result.items():
                f.write(f"{k} = {v}\n")
    if strict and result["violations"] > 0:
        raise RuntimeError("Adaptive G0 surface-gap constraint violated: "
                           f"{result['violations']}/{result['checked']} raw anchors "
                           f"exceed their local thresholds; see surface_gap_audit.txt. "
                           "Existing G0 recovery must fill these gaps before this can pass.")
    return result

@torch.no_grad()
def enforce_hard_rgb_support_with_multiview_veto(gaussians, camera, target_mask, inside_distance, visibility=None, radii=None, cfg=None, protected_count=0, geometry_visibility=None, multiview_state: Optional[MultiViewGuardState]=None):
    """Strict RGB contour on current front-visible Gaussians, with multi-view veto.
    Hidden/uncertain/outside Gaussians are never deleted or shrunk solely because
    they are unsupported by the single observed RGB view. Current front-visible
    candidates can additionally be vetoed when they remain close to a raw-G0
    side/back anchor identified by the virtual views.
    """
    from gaussian_utils.refinement_utils import cfg_float, project_gaussians, set_appearance_trainability, _zero_optimizer_rows
    n = int(gaussians.get_xyz.shape[0])
    info = {"hard_center_pruned":0,"hard_support_pruned":0,"hard_scale_shrunk":0}
    if n == 0:
        return info
    _, _, px, py, _, valid = project_gaussians(gaussians, camera)
    center_inside = torch.zeros(n, dtype=torch.bool, device=gaussians.get_xyz.device)
    if bool(valid.any()):
        center_inside[valid] = target_mask[py[valid], px[valid]] > 0.5
    front_eligible = torch.ones(n, dtype=torch.bool, device=gaussians.get_xyz.device)
    if isinstance(geometry_visibility, dict):
        visible_mask = geometry_visibility.get("visible_mask", None)
        if isinstance(visible_mask, torch.Tensor) and int(visible_mask.shape[0]) == n:
            front_eligible = visible_mask.to(device=front_eligible.device).bool()
            if multiview_state is not None and multiview_state.enabled:
                multiview_state.nonfront_exempt += int((~front_eligible).sum().item())
    center_bad = (~center_inside) & front_eligible
    support_prune = torch.zeros_like(center_bad)
    shrink_mask = torch.zeros_like(center_bad)
    rr = None
    effective_allowed = None
    if radii is not None and int(radii.shape[0]) == n:
        rr = radii.detach().to(device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype).clamp_min(0.0)
        allowed = torch.zeros_like(rr)
        if bool(valid.any()):
            allowed[valid] = inside_distance[py[valid], px[valid]].to(rr.dtype)
        tolerance_px = max(0.0, cfg_float(cfg, "stage1_footprint_tolerance_px", 1.5) if cfg is not None else 1.5)
        effective_allowed = allowed + tolerance_px
        crossing = center_inside & front_eligible & (rr > effective_allowed + 1e-6)
        overflow = torch.zeros_like(rr)
        overflow[crossing] = (1.0 - effective_allowed[crossing] / rr[crossing].clamp_min(1e-6)).clamp(0.0, 1.0)
        shrink_ratio = min(max(cfg_float(cfg, "stage1_footprint_shrink_overflow_ratio", 0.18) if cfg is not None else 0.18, 0.0), 1.0)
        prune_ratio = min(max(cfg_float(cfg, "stage1_footprint_prune_overflow_ratio", 0.68) if cfg is not None else 0.68, shrink_ratio), 1.0)
        support_prune = crossing & (overflow >= prune_ratio)
        shrink_mask = crossing & (~support_prune) & (overflow >= shrink_ratio)
    if multiview_state is not None and multiview_state.enabled:
        multiview_state.calls += 1
        # Outside-mask centers are never kept solely because of a virtual view.
        # Only switch high-overflow pruning to contour-safe shrink when deletion
        # destroys unique G0-supported local surface occupancy.
        veto = _query_veto_mask(gaussians, support_prune, multiview_state)
        rescued = veto & support_prune & center_inside
        if bool(rescued.any()):
            multiview_state.veto_support += int(rescued.sum().item())
            support_prune &= ~rescued
            shrink_mask |= rescued
    if rr is not None and effective_allowed is not None and bool(shrink_mask.any()):
        current_scale = gaussians.get_scaling.detach()
        desired_factor = effective_allowed[shrink_mask] / rr[shrink_mask].clamp_min(1e-6)
        min_factor = min(max(cfg_float(cfg, "stage1_footprint_min_shrink_factor", 0.70) if cfg is not None else 0.70, 0.20), 1.0)
        factor = desired_factor.clamp(min=min_factor, max=1.0)
        # For protected support-prune rescues, obey target contour over min shrink.
        if multiview_state is not None and multiview_state.enabled:
            rescue_rows = (veto & shrink_mask)[shrink_mask]
            factor[rescue_rows] = desired_factor[rescue_rows].clamp(min=1e-5, max=1.0)
        new_scale = current_scale[shrink_mask] * factor[:, None]
        gaussians._scaling.data[shrink_mask] = torch.log(new_scale.clamp_min(1e-8))
        _zero_optimizer_rows(gaussians, gaussians._scaling, shrink_mask)
        info["hard_scale_shrunk"] = int(shrink_mask.sum().item())
    prune_mask = center_bad | support_prune
    info["hard_center_pruned"] = int(center_bad.sum().item())
    info["hard_support_pruned"] = int(support_prune.sum().item())
    if bool(prune_mask.any()):
        if not hasattr(gaussians, "prune_points"):
            raise RuntimeError("GaussianModel has no prune_points(), required by RGB support")
        gaussians.prune_points(prune_mask)
        set_appearance_trainability(gaussians, color_dc=True, color_rest=False, opacity=True)
    return info
@torch.no_grad()
def save_multi_view_guard_diagnostics(gaussians, cfg, state: MultiViewGuardState, render_fn: Callable, alpha_fn: Callable, background: torch.Tensor) -> dict:
    result = {"views":0,"iou":0.0,"missing":0.0,"extra":0.0}
    if not state.enabled or state.diagnostics_dir is None or not state.views:
        return result
    ious=[]; miss=[]; extras=[]
    for idx, teacher in enumerate(state.views):
        pkg = render_fn(teacher.camera, gaussians, cfg, background, 0.0,0.0,0.0,cfg.is_6dof)
        alpha_result = alpha_fn(teacher.camera, gaussians, cfg, pkg)
        current = alpha_result[0] if isinstance(alpha_result, tuple) else alpha_result
        current = current.detach().clamp(0.0,1.0)
        if teacher.teacher_alpha is None:
            continue
        raw = teacher.teacher_alpha.to(current.device, current.dtype)
        raw_m = raw >= state.teacher_alpha_min
        cur_m = current >= state.teacher_alpha_min
        inter = (raw_m & cur_m).sum().float()
        union = (raw_m | cur_m).sum().float().clamp_min(1.0)
        denom = raw_m.sum().float().clamp_min(1.0)
        iou=float((inter/union).item())
        missing=float(((raw_m & ~cur_m).sum().float()/denom).item())
        extra=float(((cur_m & ~raw_m).sum().float()/denom).item())
        ious.append(iou); miss.append(missing); extras.append(extra)
        stem=f"view_{idx:02d}_yaw_{teacher.yaw_deg:+.0f}_pitch_{teacher.pitch_deg:+.0f}"
        _save_gray(current, state.diagnostics_dir/f"{stem}_final_alpha.png")
        _save_gray((raw-current).clamp_min(0.0), state.diagnostics_dir/f"{stem}_final_missing.png")
        _save_gray((current-raw).clamp_min(0.0), state.diagnostics_dir/f"{stem}_final_extra.png")
    if ious:
        result={"views":len(ious),"iou":sum(ious)/len(ious),"missing":sum(miss)/len(miss),"extra":sum(extras)/len(extras)}
    with (state.diagnostics_dir/"final_report.txt").open("w",encoding="utf-8") as f:
        f.write(multi_view_guard_summary(state)+"\n")
        f.write(f"hard_guard_calls = {state.calls}\n")
        f.write(f"veto_center = {state.veto_center}\n")
        f.write(f"veto_support = {state.veto_support}\n")
        f.write(f"veto_shrink = {state.veto_shrink}\n")
        f.write(f"gap_checks = {state.gap_checks}\n")
        f.write(f"gap_veto = {state.gap_veto}\n")
        f.write(f"raw_spacing_median = {state.spacing_median:.9g}\n")
        f.write(f"final_iou = {result['iou']:.9f}\n")
        f.write(f"final_missing = {result['missing']:.9f}\n")
        f.write(f"final_extra = {result['extra']:.9f}\n")
    return result
