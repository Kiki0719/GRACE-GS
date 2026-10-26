import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import math
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils import parse_utils
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2

# Stage-1 single-view visibility protection.
# File location on the project server:
#   /E-D3DGS/Stage1/gaussian_utils/visibility.py
from gaussian_utils.visibility import (
    classify_gaussian_visibility,
    visibility_summary,
)
from gaussian_utils.g0_recovery import (
    build_g0_donor_bank,
    donor_bank_summary,
    recover_missing_hidden_gaussians,
    recovery_summary,
)
from gaussian_utils.surface_prior import (
    build_raw_g0_surface_prior,
    raw_g0_surface_prior_loss,
    surface_prior_summary,
)
from gaussian_utils.feature_prior import (
    feature_prior_loss,
    feature_prior_summary,
    prepare_feature_prior,
)

DEFAULT_OUTPUT = "/E-D3DGS/Stage1/round1/refinement"

from gaussian_utils.refinement_utils import (
    cfg_bool,
    cfg_float,
    cfg_int,
    make_stage1_cfg,
    apply_gaussian_gradient_gating,
    build_error_map,
    build_rgb_inside_distance_map,
    build_target_mask,
    gaussian_update_weights,
    load_target_image,
    object_aware_loss,
    save_gray,
    save_rgb,
    structure_masks,
    _package_alpha,
    build_g0_depth_reference,
    build_surface_regularizer_reference,
    differentiable_render_alpha,
    g0_anchor_and_smooth_loss,
    g0_depth_preservation_loss,
    g0_scale_anchor_and_overlap_loss,
    normalized_depth_for_save,
    restore_original_g0_footprint,
    compute_stage1_visibility,
    enforce_hard_rgb_support,
    enforce_point_budget,
    prune_boundary_stray_gaussians,
    prune_extra_gaussians,
    set_appearance_trainability,
    save_diagnostics,
)
from gaussian_utils.densification import (
    accumulate_local_densification_stats,
    compute_coverage_target,
    detail_aware_local_split,
    enforce_minimum_gaussian_scale,
    local_coverage_loss,
    native_densify_with_budget,
    prepare_local_detail_target,
    refill_missing_after_absolute_prune,
    seed_missing_gaussians,
)


# =========================================================================
# STRICT SINGLE-VIEW PROTECTION
# =========================================================================
# A single RGB view must NEVER be allowed to rewrite the unseen/back-side G0.
# Two protections are used:
#   1) only confidently front-visible Gaussians receive optimizer updates;
#   2) at save time, ONLY non-front Gaussians (hidden / uncertain / outside)
#      recover ORIGINAL-G0 SH appearance. Confidently visible front Gaussians
#      keep the RGB-refined color learned from the target image.


def _gate_single_view_updates(gaussians, camera, cfg, base_weights):
    """Return visibility-gated per-Gaussian optimizer weights.

    Visible:   full update.
    Uncertain: default 0 (strict protection; configurable).
    Hidden:    0.
    Outside:   0.

    This is intentionally stricter than the existing deletion protection:
    it protects xyz / scale / rotation / opacity / SH-DC from single-view
    gradients, not only from pruning.
    """
    if not cfg_bool(cfg, "stage1_front_surface_only_updates", True):
        return base_weights, None, None

    vis = compute_stage1_visibility(gaussians, camera, cfg)
    if vis is None:
        return base_weights, None, None

    n = int(base_weights.shape[0])
    visible = vis.get("visible_mask", None)
    uncertain = vis.get("uncertain_mask", None)
    hidden = vis.get("hidden_mask", None)
    outside = vis.get("outside_mask", None)
    masks = (visible, uncertain, hidden, outside)
    if any((not isinstance(m, torch.Tensor) or int(m.shape[0]) != n) for m in masks):
        raise RuntimeError("Visibility masks do not match the current Gaussian count")

    visible = visible.to(device=base_weights.device).bool()
    uncertain = uncertain.to(device=base_weights.device).bool()
    hidden = hidden.to(device=base_weights.device).bool()
    outside = outside.to(device=base_weights.device).bool()

    gate = torch.zeros_like(base_weights)
    gate[visible] = 1.0

    # Strict default: ambiguous/front-border splats are frozen too.
    uncertain_weight = min(
        1.0,
        max(0.0, cfg_float(cfg, "stage1_uncertain_update_weight", 0.0)),
    )
    if uncertain_weight > 0.0:
        gate[uncertain] = uncertain_weight

    gated = base_weights * gate
    locked = hidden | outside | (uncertain & (uncertain_weight <= 0.0))
    return gated, locked, vis


@torch.no_grad()
def _zero_optimizer_state_for_locked_gaussians(gaussians, locked_mask):
    """Clear Adam momentum on frozen rows so old momentum cannot move them."""
    if locked_mask is None or not bool(locked_mask.any()):
        return
    optimizer = getattr(gaussians, "optimizer", None)
    if optimizer is None:
        return

    n = int(locked_mask.shape[0])
    for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc"):
        param = getattr(gaussians, name, None)
        if not isinstance(param, torch.Tensor) or int(param.shape[0]) != n:
            continue
        state = optimizer.state.get(param, None)
        if not isinstance(state, dict):
            continue
        local_mask = locked_mask.to(device=param.device)
        for value in state.values():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and int(value.shape[0]) == n
            ):
                value[local_mask] = 0


@torch.no_grad()
def _restore_original_g0_appearance(gaussians, donor_bank, camera, cfg):
    """Restore ORIGINAL-G0 SH color only on non-front Gaussians.

    Confidently front-visible Gaussians KEEP their RGB-refined SH-DC color.
    Hidden / uncertain / outside Gaussians are restored from the nearest raw-G0
    donor so a single front image can never repaint the back or side surfaces.

    Geometry, topology, scale, rotation and opacity behavior are unchanged.
    New non-front Gaussians are also protected by inheriting the nearest raw-G0
    appearance at save time.
    """
    # Hard requirement for single-view refinement: non-front color protection
    # is always enabled. This does not depend on an INI switch, so an old config
    # cannot accidentally allow the front image to repaint the back side.

    raw_xyz = donor_bank.get("xyz", None)
    if not isinstance(raw_xyz, torch.Tensor) or raw_xyz.ndim != 2 or raw_xyz.shape[-1] != 3:
        raise RuntimeError("Raw G0 donor bank has no valid xyz tensor")

    n = int(gaussians._xyz.shape[0])
    if n == 0 or int(raw_xyz.shape[0]) == 0:
        return {
            "enabled": True,
            "restored": 0,
            "front_kept": n,
            "copy_opacity": False,
        }

    # IMPORTANT: classify the FINAL cloud, after all move/add/prune/recovery.
    # Only confidently visible front-surface splats are allowed to keep the
    # single-view RGB-refined color.
    vis = compute_stage1_visibility(gaussians, camera, cfg)
    if vis is None:
        raise RuntimeError(
            "Cannot protect non-front appearance because final visibility "
            "classification is unavailable"
        )

    visible = vis.get("visible_mask", None)
    uncertain = vis.get("uncertain_mask", None)
    hidden = vis.get("hidden_mask", None)
    outside = vis.get("outside_mask", None)
    masks = (visible, uncertain, hidden, outside)
    if any((not isinstance(m, torch.Tensor) or int(m.shape[0]) != n) for m in masks):
        raise RuntimeError("Final visibility masks do not match Gaussian count")

    device = gaussians._xyz.device
    visible = visible.to(device=device).bool()
    uncertain = uncertain.to(device=device).bool()
    hidden = hidden.to(device=device).bool()
    outside = outside.to(device=device).bool()

    # EXACT policy requested:
    #   front visible  -> KEEP refined color
    #   everything else -> RESTORE raw-G0 color
    restore_mask = (~visible) | uncertain | hidden | outside
    restore_mask = restore_mask.bool()
    restore_idx = torch.nonzero(restore_mask, as_tuple=False).squeeze(1)
    restore_count = int(restore_idx.numel())
    front_kept = int((~restore_mask).sum().item())

    if restore_count == 0:
        return {
            "enabled": True,
            "restored": 0,
            "front_kept": front_kept,
            "copied_dc": False,
            "copied_rest": False,
            "copy_opacity": False,
        }

    # Nearest donor search only for protected non-front rows. This avoids
    # touching the front rows at all and is faster than restoring the full cloud.
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError(
            "Strict Stage-1 non-front appearance protection requires scipy.cKDTree. "
            "Install it with: pip install scipy"
        ) from exc

    raw_xyz_np = raw_xyz.detach().float().cpu().numpy()
    protected_xyz_np = (
        gaussians._xyz.detach()
        .index_select(0, restore_idx)
        .float()
        .cpu()
        .numpy()
    )

    tree = cKDTree(raw_xyz_np)
    try:
        _, donor_idx_np = tree.query(protected_xyz_np, k=1, workers=-1)
    except TypeError:
        _, donor_idx_np = tree.query(protected_xyz_np, k=1)

    donor_idx = torch.as_tensor(donor_idx_np, dtype=torch.long)

    def copy_raw_masked(dst_name, bank_name):
        dst = getattr(gaussians, dst_name, None)
        src = donor_bank.get(bank_name, None)
        if not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
            return False
        if int(dst.shape[0]) != n:
            return False

        selected = src.index_select(0, donor_idx).to(
            device=dst.device,
            dtype=dst.dtype,
        )
        dst_idx = restore_idx.to(device=dst.device)
        expected_shape = (restore_count,) + tuple(dst.shape[1:])
        if tuple(selected.shape) != expected_shape:
            raise RuntimeError(
                f"Appearance restore shape mismatch for {dst_name}: "
                f"{tuple(selected.shape)} vs {expected_shape}"
            )
        dst.index_copy_(0, dst_idx, selected)
        return True

    copied_dc = copy_raw_masked("_features_dc", "features_dc")
    copied_rest = copy_raw_masked("_features_rest", "features_rest")

    # Preserve the old optional opacity behavior exactly: if explicitly enabled,
    # opacity is still restored for the whole cloud just as in the original file.
    copy_opacity = cfg_bool(cfg, "stage1_restore_original_opacity", False)
    copied_opacity = False
    if copy_opacity:
        current_xyz_np = gaussians._xyz.detach().float().cpu().numpy()
        try:
            _, all_donor_idx_np = tree.query(current_xyz_np, k=1, workers=-1)
        except TypeError:
            _, all_donor_idx_np = tree.query(current_xyz_np, k=1)
        all_donor_idx = torch.as_tensor(all_donor_idx_np, dtype=torch.long)
        dst = getattr(gaussians, "_opacity", None)
        src = donor_bank.get("opacity", None)
        if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor) and int(dst.shape[0]) == n:
            selected = src.index_select(0, all_donor_idx).to(device=dst.device, dtype=dst.dtype)
            if tuple(selected.shape) != tuple(dst.shape):
                raise RuntimeError(
                    f"Appearance restore shape mismatch for _opacity: "
                    f"{tuple(selected.shape)} vs {tuple(dst.shape)}"
                )
            dst.copy_(selected)
            copied_opacity = True

    return {
        "enabled": True,
        "restored": restore_count,
        "front_kept": front_kept,
        "copied_dc": bool(copied_dc),
        "copied_rest": bool(copied_rest),
        "copy_opacity": bool(copied_opacity),
    }



def prepare_stage1_lpips(cfg, device="cuda"):
    """Optional LPIPS-Alex evaluation model, loaded once outside the training loop."""
    if not cfg_bool(cfg, "stage1_eval_lpips", True):
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS evaluation is enabled but the lpips package is not installed. "
            "Run: pip install lpips (or set stage1_eval_lpips = false)."
        ) from exc
    net = str(getattr(cfg, "stage1_eval_lpips_net", "alex")).strip().lower()
    if net not in ("alex", "vgg", "squeeze"):
        raise ValueError(f"Invalid stage1_eval_lpips_net={net!r}")
    model = lpips.LPIPS(net=net).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate_stage1_lpips(rendered_neutral, target_neutral, lpips_model):
    """LPIPS on identically neutralized full-frame images, input normalized to [-1, 1]."""
    if lpips_model is None:
        return None
    pred = (rendered_neutral.clamp(0, 1)[None] * 2.0 - 1.0).float()
    truth = (target_neutral.clamp(0, 1)[None] * 2.0 - 1.0).float()
    return float(lpips_model(pred, truth).mean().item())


@torch.no_grad()
def evaluate_stage1_projection(render_rgb, render_alpha, target_rgb, target_mask, cfg, gaussian_count):
    """Identical pre/post fixed-camera metrics. RGB evaluated on a neutralized subject.

    LPIPS is calculated separately using one cached pretrained LPIPS model.
    """
    fg = (target_mask > 0.5)
    m = fg.float()
    alpha = render_alpha.clamp(0.0, 1.0)
    rgb = render_rgb.clamp(0.0, 1.0)
    tgt = target_rgb.clamp(0.0, 1.0)
    target_neutral = tgt * m[None]
    rendered_neutral = rgb * m[None]
    squared = (rgb - tgt).square().mean(dim=0)
    mse_fg = squared[fg].mean() if bool(fg.any()) else squared.new_tensor(float('nan'))
    mse_neutral = (rendered_neutral - target_neutral).square().mean()
    psnr_fg = -10.0 * torch.log10(mse_fg.clamp_min(1e-12))
    psnr_neutral = -10.0 * torch.log10(mse_neutral.clamp_min(1e-12))
    # Consistently compute SSIM on neutralized, full-frame 3-channel RGB.
    ssim_neutral = ssim(rendered_neutral[None], target_neutral[None]).clamp(-1.0, 1.0)
    missing, extra = structure_masks(target_mask, alpha, cfg)
    alpha_thr = cfg_float(cfg, 'stage1_evaluation_alpha_threshold', 0.5)
    prediction = alpha >= alpha_thr
    intersection = (prediction & fg).sum().float()
    union = (prediction | fg).sum().float()
    iou = intersection / union.clamp_min(1.0)
    miss_on_fg = (fg & ~prediction).sum().float() / fg.sum().float().clamp_min(1.0)
    extra_on_bg = ((~fg) & prediction).sum().float() / (~fg).sum().float().clamp_min(1.0)
    out = {
        'gaussians': int(gaussian_count),
        'psnr_foreground_db': float(psnr_fg.item()),
        'psnr_neutral_full_db': float(psnr_neutral.item()),
        'ssim_neutral_full': float(ssim_neutral.item()),
        'mask_iou': float(iou.item()),
        'missing_ratio_foreground': float(miss_on_fg.item()),
        'extra_ratio_background': float(extra_on_bg.item()),
        'missing_ratio_full_image_legacy': float(missing.mean().item()),
        'extra_ratio_full_image_legacy': float(extra.mean().item()),
        'alpha_threshold': float(alpha_thr),
        'lpips_neutral_full': None,
    }
    return out, rendered_neutral, target_neutral, missing, extra


def write_stage1_metric_file(path, metrics):
    with Path(path).open('w', encoding='utf-8') as f:
        f.write('MonoEv Stage-1 fixed-camera input-view evaluation\n')
        f.write('PSNR foreground uses RGB squared error over target foreground pixels.\n')
        f.write('Neutral full-frame PSNR/SSIM uses a common black background outside target mask.\n')
        f.write('IoU uses thresholded alpha; missing/extra normalized by foreground/background respectively.\n')
        f.write('Legacy missing/extra ratios normalized by entire image for compatibility.\n')
        for k, v in metrics.items():
            if v is None:
                f.write(f'{k} = unavailable (LPIPS evaluation disabled)\n')
            elif isinstance(v, float):
                f.write(f'{k} = {v:.9f}\n')
            else:
                f.write(f'{k} = {v}\n')

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/fan_stage1.ini")
    parser.add_argument("--override", default=None)
    parser.add_argument(
        "--g0", default=None,
        help="Override initial_gaussian_ply for this refinement round."
    )
    parser.add_argument(
        "--camera", default=None,
        help="Override camera_file with the fixed_camera.json from this round's alignment."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--diag_interval", type=int, default=None)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the E-D3DGS Gaussian renderer")
    cfg, _ = parse_utils.parse_ini(cli.config)
    if cli.override is not None:
        cfg = parse_utils.override_cfg(cli.override, cfg)
    iterations = (int(cli.iterations) if cli.iterations is not None else cfg_int(cfg, "stage1_iterations", 300))
    diag_interval = (int(cli.diag_interval) if cli.diag_interval is not None else cfg_int(cfg, "stage1_diag_interval", 25))
    if iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if diag_interval <= 0:
        raise ValueError("--diag_interval must be > 0")
    g0_value = cli.g0 if cli.g0 is not None else getattr(cfg, "initial_gaussian_ply", None)
    camera_value = cli.camera if cli.camera is not None else getattr(cfg, "camera_file", None)
    image_value = getattr(cfg, "t0_image_path", None)
    if g0_value is None:
        raise ValueError("Config is missing initial_gaussian_ply and --g0 was not supplied")
    if camera_value is None or str(camera_value).strip().lower() in ("", "none", "null"):
        raise ValueError("Stage-1 requires camera_file or --camera")
    if (image_value is None or str(image_value).strip().lower() in ("", "none", "null")):
        raise ValueError("Stage-1 requires t0_image_path")
    g0_path = Path(str(g0_value)).expanduser().resolve()
    camera_path = Path(str(camera_value)).expanduser().resolve()
    image_path = Path(str(image_value)).expanduser().resolve()
    if not g0_path.is_file():
        raise FileNotFoundError(f"Initial G0 does not exist: {g0_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"t0 image does not exist: {image_path}")
    if not camera_path.is_file():
        raise FileNotFoundError(f"Fixed camera does not exist: {camera_path}")
    output_dir = Path(cli.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_cfg = make_stage1_cfg(
        cfg, output_dir, g0_path, iterations, camera_file=camera_path
    )
    safe_state(stage1_cfg.quiet)
    # ---------------------------------------------------------------------
    # Load fixed camera + G0
    # ---------------------------------------------------------------------
    gaussians = GaussianModel(stage1_cfg.sh_degree)
    scene = Scene(stage1_cfg, gaussians, static_load_path=str(g0_path))
    cameras = list(scene.getTrainCameras())
    if len(cameras) != 1:
        raise RuntimeError("Stage-1 requires exactly ONE fixed camera, " f"found {len(cameras)}")
    camera = cameras[0]
    if stage1_cfg.load2gpu_on_the_fly:
        camera.load2device()
    if gaussians.get_xyz.shape[0] == 0:
        raise RuntimeError("Loaded G0 contains zero Gaussians")
    # ---------------------------------------------------------------------
    # Load TRUE first frame
    # ---------------------------------------------------------------------
    (target, png_alpha, width, height) = load_target_image(image_path, "cuda")
    if (width != int(camera.image_width) or height != int(camera.image_height)):
        raise ValueError(
            "t0 image and fixed_camera.json resolution do not match: "
            f"image={width}x{height}, "
            f"camera={camera.image_width}x"
            f"{camera.image_height}"
        )
    # Neutral internal renderer background.
    # The actual real-world image background is NOT assumed to be black.
    black_background = torch.zeros(3, dtype=torch.float32, device="cuda")
    # ---------------------------------------------------------------------
    # Initial alpha from G0, used to initialize automatic target segmentation.
    # ---------------------------------------------------------------------
    with torch.no_grad():
        initial_pkg = render(camera, gaussians, stage1_cfg, black_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
    # Need differentiable alpha function outside no_grad only for training.
    # Here a detached alpha is enough for mask initialization.
    direct_initial_alpha = _package_alpha(initial_pkg, height, width)
    if direct_initial_alpha is not None:
        initial_alpha = (direct_initial_alpha.detach().clamp(0.0, 1.0))
        initial_alpha_source = ("renderer")
    else:
        with torch.no_grad():
            white_background = torch.ones(3, dtype=torch.float32, device="cuda")
            initial_white_pkg = render(camera, gaussians, stage1_cfg, white_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
            initial_alpha = (1.0 - (initial_white_pkg["render"] - initial_pkg["render"]).mean(dim=0)).clamp(0.0, 1.0)
        initial_alpha_source = ("black-white-pair")
    # ---------------------------------------------------------------------
    # Target object mask
    # ---------------------------------------------------------------------
    target_mask, mask_source = (build_target_mask(target, png_alpha, initial_alpha, stage1_cfg))
    rgb_inside_distance = build_rgb_inside_distance_map(target_mask)
    detail_target = prepare_local_detail_target(
        target,
        target_mask,
        stage1_cfg,
    )
    initial_mask_dir = (output_dir / "target_mask_initialization")
    initial_mask_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(target, initial_mask_dir / "target.png")
    save_gray(initial_alpha, initial_mask_dir / "initial_g0_alpha.png")
    save_gray(target_mask, initial_mask_dir / "target_mask.png")
    # Normalized visualization of the hard in-mask radius budget.
    _dmax = rgb_inside_distance.max().clamp_min(1.0)
    save_gray((rgb_inside_distance / _dmax).clamp(0.0, 1.0), initial_mask_dir / "rgb_inside_distance.png")
    save_rgb(target * target_mask[None], initial_mask_dir / "target_object_neutralized.png")
    save_gray(detail_target["detail_map"], initial_mask_dir / "target_detail_map.png")

    # Load the LPIPS model once, only for before/after evaluation (not training loss).
    lpips_eval_model = prepare_stage1_lpips(stage1_cfg, device=target.device)

    # True pre-optimization evaluation: original loaded G0, same camera/target/mask.
    initial_metrics, initial_neutral, target_neutral, initial_missing_eval, initial_extra_eval = (
        evaluate_stage1_projection(
            initial_pkg['render'], initial_alpha, target, target_mask,
            stage1_cfg, gaussians.get_xyz.shape[0]
        )
    )
    initial_metrics['lpips_neutral_full'] = evaluate_stage1_lpips(
        initial_neutral, target_neutral, lpips_eval_model
    )
    eval_dir = output_dir / 'before_after_evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_stage1_metric_file(eval_dir / 'initial_metrics.txt', initial_metrics)
    save_rgb(initial_neutral, eval_dir / 'initial_render_neutral.png')
    save_rgb(target_neutral, eval_dir / 'target_neutral.png')
    save_gray(initial_alpha, eval_dir / 'initial_alpha.png')
    save_gray(initial_missing_eval, eval_dir / 'initial_missing.png')
    save_gray(initial_extra_eval, eval_dir / 'initial_extra.png')
    print('[Initial evaluation] ' + ', '.join(
        f'{k}={v:.6f}' for k, v in initial_metrics.items()
        if k in ('psnr_foreground_db', 'ssim_neutral_full', 'mask_iou')
    ))

    # ---------------------------------------------------------------------
    # Frozen DINOv2 target feature prior
    # ---------------------------------------------------------------------
    # The target feature is cached once. If disabled, no DINO model is loaded.
    feature_prior_state = prepare_feature_prior(
        target,
        target_mask,
        stage1_cfg,
    )
    print("[Feature prior] " + feature_prior_summary(feature_prior_state))

    # ---------------------------------------------------------------------
    # ORIGINAL G0 fixed-view depth reference
    # ---------------------------------------------------------------------
    # Freeze this once before any Stage-1 optimizer step. It is intentionally
    # NOT rebuilt when xyz/topology changes, because the purpose is to preserve
    # the depth structure of the original G0.
    g0_depth_reference = build_g0_depth_reference(
        initial_pkg,
        initial_alpha,
        target_mask,
        gaussians,
        camera,
        height,
        width,
        stage1_cfg,
    )

    if g0_depth_reference["mode"] == "rendered_depth":
        _g0_depth_vis = normalized_depth_for_save(
            g0_depth_reference["depth"],
            g0_depth_reference["mask"],
        )
        if _g0_depth_vis is not None:
            save_gray(
                _g0_depth_vis,
                initial_mask_dir / "initial_g0_depth.png",
            )

    print(
        f"G0 depth reference: mode={g0_depth_reference['mode']}, "
        f"source={g0_depth_reference['source']}"
    )

    # ---------------------------------------------------------------------
    # Gaussian optimizer
    # ---------------------------------------------------------------------
    gaussians.training_setup(stage1_cfg)

    # IMPORTANT: SH-DC is view-independent. Training it from ONE front image
    # inevitably bakes the front texture into other views. In strict mode we
    # therefore freeze color and use RGB only to drive geometry/opacity.
    preserve_original_appearance = cfg_bool(
        stage1_cfg, "stage1_preserve_original_appearance", True
    )
    train_color_dc = cfg_bool(
        stage1_cfg,
        "stage1_train_color_dc",
        not preserve_original_appearance,
    )
    set_appearance_trainability(
        gaussians,
        color_dc=train_color_dc,
        color_rest=False,
        opacity=True,
    )
    initial_count = int(gaussians.get_xyz.shape[0])
    strict_from_start = cfg_bool(stage1_cfg, "stage1_strict_contour_from_start", True)
    strict_surface_fill = cfg_bool(stage1_cfg, "stage1_strict_surface_fill", True)

    # In strict mode original G0 rows may be pruned from the first iteration.
    # Therefore no prefix may be treated as permanently protected after pruning.
    protected_count_runtime = 0 if strict_from_start else initial_count

    # Keep the original G0 footprint as an immutable reference. Original points
    # are protected from topology pruning, so this prefix correspondence remains stable.
    g0_scaling_raw = gaussians._scaling.detach().clone()
    g0_rotation_raw = gaussians._rotation.detach().clone()

    # Initial visibility partition is diagnostic only. Deletion gates are
    # recomputed from the CURRENT Gaussian cloud immediately before destructive
    # topology operations.
    initial_geometry_visibility = compute_stage1_visibility(
        gaussians,
        camera,
        stage1_cfg,
    )
    if initial_geometry_visibility is not None:
        print("[Visibility initial] " + visibility_summary(initial_geometry_visibility))

    # -----------------------------------------------------------------
    # ORIGINAL G0 donor bank for hidden-geometry recovery.
    # -----------------------------------------------------------------
    # This snapshot is taken BEFORE any Stage-1 move/add/prune operation.
    # Heavy tensors are immediately moved to CPU by g0_recovery.py.
    g0_donor_bank = build_g0_donor_bank(
        gaussians,
        initial_geometry_visibility,
        stage1_cfg,
    )
    print("[G0 donor] " + donor_bank_summary(g0_donor_bank))

    # -----------------------------------------------------------------
    # PERMANENT raw-G0 local surface teacher.
    # -----------------------------------------------------------------
    # Reuses the CPU donor-bank xyz instead of creating another full GPU copy.
    raw_g0_surface_prior = build_raw_g0_surface_prior(
        g0_donor_bank,
        initial_geometry_visibility,
        stage1_cfg,
    )
    print("[Surface prior] " + surface_prior_summary(raw_g0_surface_prior))

    # Fixed G0 reference + KNN graph for surface-preservation losses.
    # At this point the reference is exactly the loaded canonical G0.
    surface_reference = build_surface_regularizer_reference(
        gaussians.get_xyz.detach(),
        stage1_cfg,
        previous_reference=None,
    )
    lambda_anchor = max(0.0, cfg_float(stage1_cfg, "stage1_lambda_anchor", 0.10))
    lambda_smooth = max(0.0, cfg_float(stage1_cfg, "stage1_lambda_smooth", 0.20))
    lambda_depth = max(0.0, cfg_float(stage1_cfg, "stage1_lambda_depth", 0.10))
    lambda_scale_anchor = max(0.0, cfg_float(stage1_cfg, "stage1_lambda_scale_anchor", 0.20))
    lambda_overlap = max(0.0, cfg_float(stage1_cfg, "stage1_lambda_overlap", 0.20))
    lambda_surface_prior = max(
        0.0,
        cfg_float(stage1_cfg, "stage1_lambda_surface_prior", 0.08),
    )
    lambda_feature_prior = max(
        0.0,
        cfg_float(stage1_cfg, "stage1_lambda_feature_prior", 0.05),
    )

    max_ratio = cfg_float(stage1_cfg, "stage1_max_gaussian_ratio", 1.20)
    if strict_surface_fill:
        # Hole filling needs enough headroom after hard contour pruning.
        max_ratio = max(
            max_ratio,
            cfg_float(stage1_cfg, "stage1_strict_min_gaussian_ratio", 2.00),
        )
    max_total_points = max(initial_count, int(round(initial_count * max_ratio)))
    # CONTINUOUS add/delete policy: no iteration phases.
    topology_interval = max(
        1,
        cfg_int(stage1_cfg, "stage1_topology_interval", 5),
    )
    densify_stop_iter = iterations
    addition_start_iter = 1
    cleanup_start_iter = 1
    cleanup_interval = topology_interval
    grad_clip = cfg_float(stage1_cfg, "stage1_grad_clip", 1.0)
    lambda_coverage = max(0.0, cfg_float(stage1_cfg, "stage1_lambda_coverage", 0.10))
    coverage_refresh_interval = max(1, cfg_int(stage1_cfg, "stage1_coverage_refresh_interval", 10))
    coverage_target = None
    coverage_stats = {"spacing_median": 0.0, "target_median": 0.0}
    print("\n========== MonoEv Stage-1 Object-Aware Refinement ==========")
    print(f"G0: {g0_path}")
    print(f"target RGB: {image_path}")
    print(f"fixed camera: " f"{getattr(stage1_cfg, 'camera_file', 'from Scene')}")
    print(f"resolution: {width}x{height}")
    print(f"target mask source: {mask_source}")
    print(f"initial alpha source: {initial_alpha_source}")
    print("REAL TARGET BACKGROUND: IGNORED / NEUTRALIZED")
    print("Internal renderer background: BLACK only as a neutral compositing color")
    print(f"initial Gaussians: {initial_count:}")
    print(f"maximum Gaussians: {max_total_points:}")
    print(f"iterations: {iterations}")
    print(f"topology interval: {topology_interval}")
    print("topology policy: CONTINUOUS ADD + DELETE from iteration 1")
    print(f"topology add/refill interval: {topology_interval}")
    print("outside-contour centre deletion: EVERY optimizer iteration")
    print(f"strict contour from start: {strict_from_start}")
    print(f"strict surface fill throughout training: {strict_surface_fill}")
    print(f"cleanup interval: {cleanup_interval}")
    print(f"3D coverage: lambda={lambda_coverage:.3f}, beta={cfg_float(stage1_cfg, 'stage1_coverage_beta', 0.55):.3f}, refresh={coverage_refresh_interval}")
    print("RGB/SSIM supervision: FOREGROUND ONLY; target background never becomes black RGB supervision")
    print("HARD RGB MASK: projected centres outside the cleaned RGB subject are deleted")
    print("OFF-SCREEN Gaussian centres are deleted")
    print("FOOTPRINT: slight crossing allowed; moderate crossing shrunk; severe crossing pruned")
    print("G0 STRUCTURE: every new seed/split must remain near the permanent raw-G0 surface")
    print("DETAIL SPLIT: large high-error splats are replaced by smaller local children")
    print(f"New seed defaults: opacity={cfg_float(stage1_cfg, 'stage1_seed_surface_opacity', 0.18):.2f}, scale_factor={cfg_float(stage1_cfg, 'stage1_seed_surface_scale_factor', 1.00):.2f}")
    print(
        "Loss = "
        f"{cfg_float(stage1_cfg, 'stage1_lambda_rgb', 0.75):.2f}*RGB + "
        f"{cfg_float(stage1_cfg, 'stage1_lambda_color', 1.50):.2f}*Color + "
        f"{cfg_float(stage1_cfg, 'stage1_lambda_ssim', 0.25):.2f}*DSSIM + "
        f"{cfg_float(stage1_cfg, 'stage1_lambda_alpha', 2.0):.2f}*Alpha + "
        f"{cfg_float(stage1_cfg, 'stage1_lambda_opacity', 2.5):.2f}*Opacity + "
        f"{cfg_float(stage1_cfg, 'stage1_lambda_edge', 1.0):.2f}*Edge + "
        f"{cfg_float(stage1_cfg, 'stage1_lambda_missing', 4.0):.2f}*Missing + "
        f"{cfg_float(stage1_cfg, 'stage1_lambda_extra', 2.0):.2f}*Extra + "
        f"{lambda_coverage:.2f}*Coverage + "
        f"{lambda_anchor:.2f}*Anchor + "
        f"{lambda_smooth:.2f}*Smooth + "
        f"{lambda_depth:.2f}*G0Depth + "
        f"{lambda_scale_anchor:.2f}*ScaleAnchor + "
        f"{lambda_overlap:.2f}*Overlap + "
        f"{lambda_surface_prior:.2f}*RawSurfacePrior + "
        f"{lambda_feature_prior:.2f}*DINOFeature"
    )
    print("Trainable: xyz / rotation / scaling / opacity" + (" / SH-DC color" if train_color_dc else ""))
    print("Frozen: higher-order SH / camera" + (" / SH-DC color" if not train_color_dc else ""))
    print(f"Front-surface-only updates: {cfg_bool(stage1_cfg, 'stage1_front_surface_only_updates', True)}")
    print(f"Preserve original G0 appearance: {preserve_original_appearance}")
    print("Events: NOT USED")
    print("DeformModel: NOT USED")
    print("Anchor/Smooth KNN: fixed during ordinary iterations; rebuilt only after real add/prune topology changes")
    print(
        "Raw-G0 Surface Prior: permanent original-G0 teacher; "
        f"sample={cfg_int(stage1_cfg, 'stage1_surface_prior_sample_points', 8192)}, "
        f"every={cfg_int(stage1_cfg, 'stage1_surface_prior_every', 10)} iter"
    )
    print(
        "DINO Feature Prior: "
        f"enabled={cfg_bool(stage1_cfg, 'stage1_enable_feature_prior', False)}, "
        f"lambda={lambda_feature_prior:.4f}, "
        f"every={cfg_int(stage1_cfg, 'stage1_feature_every', 10)} iter, "
        f"input={cfg_int(stage1_cfg, 'stage1_feature_input_size', 224)}"
    )
    print("==========================================================\n")
    progress = tqdm(range(1, iterations + 1), desc="Stage1 object-aware G0 correction")
    ema_loss = 0.0
    topology_records = []
    for iteration in progress:
        # ================================================================
        # Forward
        # ================================================================
        gaussians.optimizer.zero_grad(set_to_none=True)
        package = render(camera, gaussians, stage1_cfg, black_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
        rendered = package["render"]
        if not torch.isfinite(rendered).all():
            raise FloatingPointError("Rendered image contains NaN/Inf at " f"iteration {iteration}")
        (render_alpha, alpha_source) = differentiable_render_alpha(camera, gaussians, stage1_cfg, package)
        if not torch.isfinite(render_alpha).all():
            raise FloatingPointError("Render alpha contains NaN/Inf at " f"iteration {iteration}")
        (total_loss, loss_diag) = object_aware_loss(rendered, render_alpha, target, target_mask, stage1_cfg)

        # ================================================================
        # FROZEN DINOV2 FEATURE PRIOR
        # ================================================================
        # DINO is frozen, but this call must remain outside torch.no_grad()
        # so feature gradients can flow through rendered RGB to the Gaussians.
        feature_loss, feature_diag = feature_prior_loss(
            rendered,
            target_mask,
            feature_prior_state,
            stage1_cfg,
            iteration,
            force=False,
        )
        total_loss = total_loss + lambda_feature_prior * feature_loss
        loss_diag["feature"] = feature_loss

        coverage_loss = total_loss * 0.0
        coverage_under_ratio = 0.0
        if lambda_coverage > 0.0:
            need_refresh = (
                coverage_target is None
                or coverage_target.shape[0] != gaussians.get_xyz.shape[0]
                or iteration == 1
                or (iteration - 1) % coverage_refresh_interval == 0
            )
            if need_refresh:
                coverage_target, coverage_stats = compute_coverage_target(gaussians, stage1_cfg)
            coverage_loss, coverage_under_ratio = local_coverage_loss(
                gaussians, coverage_target, package.get("visibility_filter", None), stage1_cfg
            )
            total_loss = total_loss + lambda_coverage * coverage_loss
        loss_diag["coverage"] = coverage_loss

        # ================================================================
        # G0 surface-preservation losses
        # ================================================================
        anchor_loss, smooth_loss = g0_anchor_and_smooth_loss(
            gaussians,
            surface_reference,
        )
        total_loss = (
            total_loss
            + lambda_anchor * anchor_loss
            + lambda_smooth * smooth_loss
        )
        loss_diag["anchor"] = anchor_loss
        loss_diag["smooth"] = smooth_loss

        # ================================================================
        # PERMANENT RAW-G0 LOCAL SURFACE PRIOR
        # ================================================================
        # Unlike Anchor/Smooth, this teacher never changes after Stage-1 starts.
        surface_prior_loss, surface_prior_diag = raw_g0_surface_prior_loss(
            gaussians,
            raw_g0_surface_prior,
            stage1_cfg,
            iteration,
            force=False,
        )
        total_loss = total_loss + lambda_surface_prior * surface_prior_loss
        loss_diag["surface_prior"] = surface_prior_loss

        # ================================================================
        # ORIGINAL G0 fixed-view depth preservation
        # ================================================================
        depth_loss, depth_source, depth_valid_ratio = g0_depth_preservation_loss(
            gaussians,
            camera,
            package,
            render_alpha,
            g0_depth_reference,
            target_mask,
            stage1_cfg,
        )
        total_loss = total_loss + lambda_depth * depth_loss
        loss_diag["depth"] = depth_loss

        # ================================================================
        # G0 footprint / surface continuity
        # ================================================================
        scale_anchor_loss, overlap_loss, overlap_gap_ratio = (
            g0_scale_anchor_and_overlap_loss(
                gaussians,
                surface_reference,
                g0_scaling_raw,
                initial_count,
                stage1_cfg,
            )
        )
        total_loss = (
            total_loss
            + lambda_scale_anchor * scale_anchor_loss
            + lambda_overlap * overlap_loss
        )
        loss_diag["scale_anchor"] = scale_anchor_loss
        loss_diag["overlap"] = overlap_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError("Stage-1 loss contains NaN/Inf at " f"iteration {iteration}")
        # ================================================================
        # Structure state used for local gradient/topology decisions
        # ================================================================
        with torch.no_grad():
            (missing_mask, extra_mask) = structure_masks(target_mask, render_alpha.detach(), stage1_cfg)
            target_object_detached = (target * target_mask[None])
            error_map = build_error_map(
                rendered.detach(),
                target_object_detached,
                render_alpha.detach(),
                target_mask,
                missing_mask,
                extra_mask,
                stage1_cfg,
            )
        # ================================================================
        # Backward
        # ================================================================
        total_loss.backward()
        (gaussian_weights, valid_gaussians) = gaussian_update_weights(
            gaussians, camera, error_map, extra_mask, stage1_cfg
        )

        # STRICT FRONT-SURFACE UPDATE GATE.
        # Hidden/back-side/uncertain Gaussians receive no single-view optimizer
        # update, so front RGB cannot move/recolor them through residual gradients.
        gaussian_weights, locked_gaussians, train_visibility = (
            _gate_single_view_updates(
                gaussians, camera, stage1_cfg, gaussian_weights
            )
        )
        apply_gaussian_gradient_gating(gaussians, gaussian_weights)
        _zero_optimizer_state_for_locked_gaussians(gaussians, locked_gaussians)

        local_densify_count = accumulate_local_densification_stats(
            gaussians,
            package,
            camera,
            error_map,
            extra_mask,
            stage1_cfg,
        )
        if grad_clip > 0.0:
            trainable = [
                p
                for p in (gaussians._xyz, gaussians._rotation, gaussians._scaling, gaussians._opacity, gaussians._features_dc)
                if p.requires_grad
                and p.grad is not None
            ]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        gaussians.optimizer.step()
        gaussians.update_learning_rate(iteration)

        # Preserve the original dense/smooth G0 splat footprint exactly.
        restore_original_g0_footprint(
            gaussians,
            g0_scaling_raw,
            g0_rotation_raw,
            initial_count,
            stage1_cfg,
        )

        # ABSOLUTE RGB contour projection after EVERY optimizer step.
        # Anything not projected inside the RGB subject mask is removed,
        # including off-screen and invalid projected centres.
        with torch.no_grad():
            hard_pkg = render(camera, gaussians, stage1_cfg, black_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
        hard_geometry_visibility = compute_stage1_visibility(
            gaussians,
            camera,
            stage1_cfg,
        )
        hard_guard_info = enforce_hard_rgb_support(
            gaussians,
            camera,
            target_mask,
            rgb_inside_distance,
            hard_pkg.get("visibility_filter", None),
            hard_pkg.get("radii", None),
            stage1_cfg,
            protected_count=protected_count_runtime,
            geometry_visibility=hard_geometry_visibility,
        )
        hard_changed = (
            hard_guard_info["hard_center_pruned"] > 0
            or hard_guard_info["hard_support_pruned"] > 0
            or hard_guard_info["hard_scale_shrunk"] > 0
        )
        if hard_changed:
            coverage_target = None

        # IMPORTANT FOR SPEED:
        # hard_scale_shrunk only changes Gaussian scale. It does NOT change
        # point count, xyz indexing, or the fixed G0 KNN graph, so NEVER
        # rebuild the Anchor/Smooth reference for scale-only hard-mask updates.
        hard_pruned = (
            hard_guard_info["hard_center_pruned"] > 0
            or hard_guard_info["hard_support_pruned"] > 0
        )
        continuous_refill_info = {
            "refill_passes": 0,
            "refill_added": 0,
            "refill_missing_px": 0,
        }

        if hard_pruned:
            continuous_refill_info = refill_missing_after_absolute_prune(
                gaussians,
                camera,
                target,
                target_mask,
                stage1_cfg,
                black_background,
                max_total_points,
                raw_g0_surface_prior=raw_g0_surface_prior,
            )

            surface_reference = build_surface_regularizer_reference(
                gaussians.get_xyz.detach(),
                stage1_cfg,
                previous_reference=None,
            )

        topology_changed = (
            hard_changed
            or int(
                continuous_refill_info.get(
                    "refill_added",
                    0,
                )
            ) > 0
        )
        # ================================================================
        # CONTINUOUS TOPOLOGY UPDATE: NO PHASES
        # ================================================================
        # Every topology tick from iteration 1 to the end can both:
        #   - add/refill missing foreground Gaussians;
        #   - delete any newly created/moved Gaussian whose projected centre
        #     is outside the RGB subject contour.
        topology_tick = (
            iteration == 1
            or iteration % topology_interval == 0
            or iteration == iterations
        )

        do_addition = topology_tick
        do_cleanup = topology_tick
        if do_addition or do_cleanup:
            before_topology = int(gaussians.get_xyz.shape[0])
            enable_native_densify = cfg_bool(stage1_cfg, "stage1_enable_native_densify", False)
            if do_addition and enable_native_densify:
                native_info = native_densify_with_budget(gaussians, scene, stage1_cfg, max_total_points)
            else:
                native_info = {
                    "native_before": before_topology,
                    "native_candidates": 0,
                    "native_budget": max(0, max_total_points - before_topology),
                    "native_after": before_topology,
                    "native_added": 0,
                }
            # Re-render after the optimizer step and optional native densification.
            with torch.no_grad():
                topo_pkg = render(camera, gaussians, stage1_cfg, black_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
            topo_alpha_direct = _package_alpha(topo_pkg, height, width)
            if topo_alpha_direct is not None:
                topo_alpha = topo_alpha_direct.detach().clamp(0.0, 1.0)
            else:
                with torch.no_grad():
                    white_background = torch.ones(3, dtype=torch.float32, device="cuda")
                    topo_white_pkg = render(camera, gaussians, stage1_cfg, white_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
                    topo_alpha = (1.0 - (topo_white_pkg["render"] - topo_pkg["render"]).mean(dim=0)).clamp(0.0, 1.0)
            topo_missing, topo_extra = structure_masks(target_mask, topo_alpha, stage1_cfg)
            topo_visibility = topo_pkg.get("visibility_filter", None)
            topo_radii = topo_pkg.get("radii", None)

            # ------------------------------------------------------------
            # Local residual/detail-driven split: large splats in locally wrong
            # regions become several smaller, G0-supported splats.
            # ------------------------------------------------------------
            detail_info = {
                "detail_candidates": 0,
                "detail_split_parents": 0,
                "detail_added": 0,
                "detail_g0_rejected": 0,
                "detail_mean_score": 0.0,
                "detail_mean_residual": 0.0,
                "detail_mean_radius": 0.0,
            }
            if do_addition:
                remaining_budget = max(
                    0,
                    max_total_points - int(gaussians.get_xyz.shape[0]),
                )
                detail_info = detail_aware_local_split(
                    gaussians,
                    camera,
                    target,
                    target_mask,
                    topo_pkg["render"].detach(),
                    topo_pkg,
                    detail_target,
                    raw_g0_surface_prior,
                    stage1_cfg,
                    remaining_budget,
                    iteration,
                )

                if int(detail_info.get("detail_added", 0)) > 0:
                    # Re-render because local splitting changed support before
                    # missing-region analysis.
                    with torch.no_grad():
                        topo_pkg = render(
                            camera, gaussians, stage1_cfg, black_background,
                            0.0, 0.0, 0.0, stage1_cfg.is_6dof
                        )
                    topo_alpha_direct = _package_alpha(topo_pkg, height, width)
                    if topo_alpha_direct is not None:
                        topo_alpha = topo_alpha_direct.detach().clamp(0.0, 1.0)
                    else:
                        with torch.no_grad():
                            white_background = torch.ones(3, dtype=torch.float32, device="cuda")
                            topo_white_pkg = render(
                                camera, gaussians, stage1_cfg, white_background,
                                0.0, 0.0, 0.0, stage1_cfg.is_6dof
                            )
                            topo_alpha = (
                                1.0
                                - (topo_white_pkg["render"] - topo_pkg["render"]).mean(dim=0)
                            ).clamp(0.0, 1.0)
                    topo_missing, topo_extra = structure_masks(
                        target_mask, topo_alpha, stage1_cfg
                    )
                    topo_visibility = topo_pkg.get("visibility_filter", None)
                    topo_radii = topo_pkg.get("radii", None)

            # Continuous missing-region refill.
            if do_addition:
                remaining_budget = max(
                    0,
                    max_total_points - int(gaussians.get_xyz.shape[0]),
                )
                seed_info = seed_missing_gaussians(
                    gaussians,
                    camera,
                    target,
                    target_mask,
                    topo_missing,
                    topo_visibility,
                    stage1_cfg,
                    remaining_budget,
                    raw_g0_surface_prior=raw_g0_surface_prior,
                )

                visibility_for_prune = None
                if topo_visibility is not None:
                    current_n = int(gaussians.get_xyz.shape[0])
                    old_n = int(topo_visibility.shape[0])
                    if current_n >= old_n:
                        visibility_for_prune = torch.cat(
                            [
                                topo_visibility.bool(),
                                torch.zeros(
                                    current_n - old_n,
                                    dtype=torch.bool,
                                    device=topo_visibility.device,
                                ),
                            ],
                            dim=0,
                        )

                geometry_visibility_for_prune = compute_stage1_visibility(
                    gaussians,
                    camera,
                    stage1_cfg,
                )
                extra_pruned = prune_extra_gaussians(
                    gaussians,
                    camera,
                    topo_extra,
                    visibility_for_prune,
                    stage1_cfg,
                    protected_count=protected_count_runtime,
                    geometry_visibility=geometry_visibility_for_prune,
                )
            else:
                seed_info = {
                    "seed_requested": 0,
                    "seed_added": 0,
                    "seed_mean_parent_distance_px": 0.0,
                    "seed_g0_rejected": 0,
                }
                extra_pruned = 0

            # Strong silhouette stray cleanup is independent from refill.
            stray_info = {
                "stray_candidates": 0,
                "stray_pruned": 0,
                "stray_hard": 0,
                "stray_soft": 0,
                "stray_isolated": 0,
                "stray_edge_radius": 0,
            }

            if do_cleanup:
                # Re-render because refill/extra-pruning may already have changed
                # topology before the cleanup pass.
                with torch.no_grad():
                    cleanup_pkg = render(
                        camera,
                        gaussians,
                        stage1_cfg,
                        black_background,
                        0.0, 0.0, 0.0,
                        stage1_cfg.is_6dof,
                    )

                cleanup_visibility = cleanup_pkg.get(
                    "visibility_filter",
                    None,
                )
                cleanup_radii = cleanup_pkg.get(
                    "radii",
                    None,
                )

                cleanup_geometry_visibility = compute_stage1_visibility(
                    gaussians,
                    camera,
                    stage1_cfg,
                )

                stray_info = prune_boundary_stray_gaussians(
                    gaussians,
                    camera,
                    target_mask,
                    cleanup_visibility,
                    cleanup_radii,
                    stage1_cfg,
                    protected_count=protected_count_runtime,
                    geometry_visibility=cleanup_geometry_visibility,
                )

            # Extra/stray pruning may have reindexed the cloud, therefore budget
            # gating is recomputed from the CURRENT rows.
            budget_geometry_visibility = compute_stage1_visibility(
                gaussians,
                camera,
                stage1_cfg,
            )
            budget_pruned = enforce_point_budget(
                gaussians,
                max_total_points,
                protected_count=protected_count_runtime,
                geometry_visibility=budget_geometry_visibility,
            )
            scale_visibility = None
            if topo_visibility is not None and topo_visibility.shape[0] == gaussians.get_xyz.shape[0]:
                scale_visibility = topo_visibility
            scale_floor_info = enforce_minimum_gaussian_scale(gaussians, scale_visibility, stage1_cfg, protected_count=protected_count_runtime)

            # Topology may have created new points or enlarged existing scales.
            # Re-render and enforce the RGB mask AGAIN before this update is accepted.
            with torch.no_grad():
                post_topo_pkg = render(camera, gaussians, stage1_cfg, black_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
            post_topo_geometry_visibility = compute_stage1_visibility(
                gaussians,
                camera,
                stage1_cfg,
            )
            post_topo_guard = enforce_hard_rgb_support(
                gaussians,
                camera,
                target_mask,
                rgb_inside_distance,
                post_topo_pkg.get("visibility_filter", None),
                post_topo_pkg.get("radii", None),
                stage1_cfg,
                protected_count=protected_count_runtime,
                geometry_visibility=post_topo_geometry_visibility,
            )

            # ------------------------------------------------------------
            # Strict local surface closure:
            # hard contour correction can expose a hole; refill it immediately,
            # then enforce the contour again. This happens at EVERY topology update.
            # ------------------------------------------------------------
            closure_added = 0
            closure_guard_pruned = 0
            closure_guard_shrunk = 0
            closure_passes = 0
            if strict_surface_fill:
                max_closure_passes = max(
                    1, cfg_int(stage1_cfg, "stage1_surface_closure_passes_per_update", 3)
                )
                for _closure_pass in range(max_closure_passes):
                    closure_passes += 1
                    with torch.no_grad():
                        closure_pkg = render(
                            camera, gaussians, stage1_cfg, black_background,
                            0.0, 0.0, 0.0, stage1_cfg.is_6dof
                        )
                    closure_alpha_direct = _package_alpha(closure_pkg, height, width)
                    if closure_alpha_direct is not None:
                        closure_alpha = closure_alpha_direct.detach().clamp(0.0, 1.0)
                    else:
                        with torch.no_grad():
                            closure_white = torch.ones(
                                3, dtype=torch.float32, device="cuda"
                            )
                            closure_white_pkg = render(
                                camera, gaussians, stage1_cfg, closure_white,
                                0.0, 0.0, 0.0, stage1_cfg.is_6dof
                            )
                            closure_alpha = (
                                1.0
                                - (
                                    closure_white_pkg["render"]
                                    - closure_pkg["render"]
                                ).mean(dim=0)
                            ).clamp(0.0, 1.0)

                    closure_missing, _closure_extra = structure_masks(
                        target_mask, closure_alpha, stage1_cfg
                    )
                    if not bool((closure_missing > 0.5).any()):
                        break

                    remaining_budget = max(
                        0, max_total_points - int(gaussians.get_xyz.shape[0])
                    )
                    if remaining_budget <= 0:
                        break

                    closure_seed = seed_missing_gaussians(
                        gaussians,
                        camera,
                        target,
                        target_mask,
                        closure_missing,
                        closure_pkg.get("visibility_filter", None),
                        stage1_cfg,
                        remaining_budget,
                        seed_erode_override=0,
                        raw_g0_surface_prior=raw_g0_surface_prior,
                    )
                    just_added = int(closure_seed.get("seed_added", 0))
                    closure_added += just_added
                    if just_added <= 0:
                        break

                    with torch.no_grad():
                        closure_guard_pkg = render(
                            camera, gaussians, stage1_cfg, black_background,
                            0.0, 0.0, 0.0, stage1_cfg.is_6dof
                        )
                    closure_geometry_visibility = compute_stage1_visibility(
                        gaussians,
                        camera,
                        stage1_cfg,
                    )
                    closure_guard = enforce_hard_rgb_support(
                        gaussians,
                        camera,
                        target_mask,
                        rgb_inside_distance,
                        closure_guard_pkg.get("visibility_filter", None),
                        closure_guard_pkg.get("radii", None),
                        stage1_cfg,
                        protected_count=protected_count_runtime,
                        geometry_visibility=closure_geometry_visibility,
                    )
                    closure_guard_pruned += int(
                        closure_guard.get("hard_center_pruned", 0)
                    ) + int(closure_guard.get("hard_support_pruned", 0))
                    closure_guard_shrunk += int(
                        closure_guard.get("hard_scale_shrunk", 0)
                    )

            after_topology = int(gaussians.get_xyz.shape[0])

            # ------------------------------------------------------------
            # Reference-safety bookkeeping
            # ------------------------------------------------------------
            # Any prune can reindex Gaussian rows. Anchor/Smooth/Overlap are
            # row-correspondence based, so a "pure append" reference update is
            # valid ONLY when absolutely no deletion/reindexing happened.
            native_added = int(native_info.get("native_added", 0))
            detail_added = int(detail_info.get("detail_added", 0))
            seed_added = int(seed_info.get("seed_added", 0))
            known_added = native_added + detail_added + seed_added + int(closure_added)

            detail_pruned = int(detail_info.get("detail_pruned_parents", 0))
            post_guard_pruned = (
                int(post_topo_guard.get("hard_center_pruned", 0))
                + int(post_topo_guard.get("hard_support_pruned", 0))
            )
            explicit_pruned = (
                detail_pruned
                + int(extra_pruned)
                + int(stray_info.get("stray_pruned", 0))
                + int(budget_pruned)
                + post_guard_pruned
                + int(closure_guard_pruned)
            )

            # Absolute safety net: even if a new pruning path forgets to report
            # its counter, point-count accounting still detects that this was
            # not a pure append operation.
            expected_after_without_prune = before_topology + known_added
            implicit_pruned = max(0, expected_after_without_prune - after_topology)

            topology_added = known_added > 0
            topology_pruned = explicit_pruned > 0 or implicit_pruned > 0
            topology_changed = (
                topology_changed
                or topology_added
                or topology_pruned
                or after_topology != before_topology
                or int(post_topo_guard.get("hard_scale_shrunk", 0)) > 0
                or int(closure_guard_shrunk) > 0
            )
            if topology_changed:
                coverage_target = None

            reference_rebuilt = False
            reference_extended = False
            if topology_pruned:
                # IMPORTANT: after ANY prune/reindex, old row correspondence is
                # invalid. Rebuild XYZ anchors and KNN edges from CURRENT rows.
                surface_reference = build_surface_regularizer_reference(
                    gaussians.get_xyz.detach(),
                    stage1_cfg,
                    previous_reference=None,
                )
                reference_rebuilt = True
            elif topology_added:
                # True pure append: old rows preserve identity, so existing
                # anchors can be retained and only appended rows get birth anchors.
                surface_reference = build_surface_regularizer_reference(
                    gaussians.get_xyz.detach(),
                    stage1_cfg,
                    previous_reference=surface_reference,
                )
                reference_extended = True

            # Defensive invariant: Anchor/Smooth must never see a stale-size
            # reference after a topology update.
            if (
                surface_reference is None
                or int(surface_reference.get("count", -1)) != after_topology
            ):
                surface_reference = build_surface_regularizer_reference(
                    gaussians.get_xyz.detach(),
                    stage1_cfg,
                    previous_reference=None,
                )
                reference_rebuilt = True
                reference_extended = False

            record = {
                "iteration": iteration,
                "mode": "continuous-add+delete",
                "before": before_topology,
                **native_info,
                **detail_info,
                **seed_info,
                "extra_pruned": extra_pruned,
                **stray_info,
                "budget_pruned": budget_pruned,
                **scale_floor_info,
                **post_topo_guard,
                "closure_passes": closure_passes,
                "closure_added": closure_added,
                "closure_guard_pruned": closure_guard_pruned,
                "closure_guard_shrunk": closure_guard_shrunk,
                "known_added": known_added,
                "detail_pruned": detail_pruned,
                "post_guard_pruned": post_guard_pruned,
                "explicit_pruned": explicit_pruned,
                "implicit_pruned": implicit_pruned,
                "reference_rebuilt": int(reference_rebuilt),
                "reference_extended": int(reference_extended),
                "after": after_topology,
                "net_change": after_topology - before_topology,
                "missing_ratio": float(topo_missing.mean().item()),
                "extra_ratio": float(topo_extra.mean().item()),
            }
            topology_records.append(record)
            print(
                f"\n[Topology {iteration}] mode={record['mode']} "
                f"before={before_topology:,}, native_added={native_added:,}, "
                f"detail_added={detail_added:,}, seed_added={seed_added:,}+closure{closure_added:,}, "
                f"g0_reject={seed_info.get('seed_g0_rejected', 0) + detail_info.get('detail_g0_rejected', 0):,}, "
                f"detail_pruned={detail_pruned:,}, extra_pruned={extra_pruned:,}, "
                f"stray_pruned={stray_info['stray_pruned']:,}/{stray_info['stray_candidates']:,}, "
                f"budget_pruned={budget_pruned:,}, post_pruned={post_guard_pruned:,}, "
                f"closure_pruned={closure_guard_pruned:,}, implicit_pruned={implicit_pruned:,}, "
                f"ref={'REBUILD' if reference_rebuilt else ('EXTEND' if reference_extended else 'KEEP')}, "
                f"after={after_topology:,}, net={after_topology - before_topology:+,}, "
                f"missing={topo_missing.mean().item():.6f}, extra={topo_extra.mean().item():.6f}"
            )
        # ================================================================
        # Console
        # ================================================================
        ema_loss = (0.4 * float(total_loss.detach()) + 0.6 * ema_loss)
        progress.set_postfix(
            {
                "loss": f"{ema_loss:.5f}",
                "RGB": f"{float(loss_diag['rgb'].detach()):.4f}",
                "Col": f"{float(loss_diag['color'].detach()):.4f}",
                "A": f"{float(loss_diag['alpha'].detach()):.4f}",
                "Op": f"{float(loss_diag['opacity'].detach()):.4f}",
                "Miss": f"{float(loss_diag['missing'].detach()):.4f}",
                "Cov": f"{float(loss_diag['coverage'].detach()):.4f}",
                "Anc": f"{float(loss_diag['anchor'].detach()):.4f}",
                "Sm": f"{float(loss_diag['smooth'].detach()):.4f}",
                "Dep": f"{float(loss_diag['depth'].detach()):.4f}",
                "ScA": f"{float(loss_diag['scale_anchor'].detach()):.4f}",
                "Ovl": f"{float(loss_diag['overlap'].detach()):.4f}",
                "Gap": f"{overlap_gap_ratio:.3f}",
                "N": f"{gaussians.get_xyz.shape[0]}",
                "dens": f"{local_densify_count}",
            }
        )
        # ================================================================
        # Diagnostics
        # ================================================================
        do_diag = (iteration == 1 or iteration % diag_interval == 0 or iteration == iterations )
        if do_diag:
            with torch.no_grad():
                diag_pkg = render(camera, gaussians, stage1_cfg, black_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
                diag_render = (diag_pkg["render"].clamp(0.0, 1.0))
            # Detached alpha for diagnostics.
            diag_direct_alpha = _package_alpha(diag_pkg, height, width)
            if diag_direct_alpha is not None:
                diag_alpha = (diag_direct_alpha.detach().clamp(0.0, 1.0))
                diag_alpha_source = ("renderer")
            else:
                with torch.no_grad():
                    white_background = torch.ones(3, dtype=torch.float32, device="cuda")
                    diag_white_pkg = render(camera, gaussians, stage1_cfg, white_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
                    diag_alpha = (1.0 - (diag_white_pkg["render"] - diag_pkg["render"]).mean(dim=0)).clamp(0.0, 1.0)
                diag_alpha_source = ("black-white-pair")
            with torch.no_grad():
                (diag_missing, diag_extra) = structure_masks(target_mask, diag_alpha, stage1_cfg)
            # Re-evaluate losses with differentiable graph not needed.
            with torch.no_grad():
                (diag_total, diag_loss_values) = object_aware_loss(diag_render, diag_alpha, target, target_mask, stage1_cfg)

                # Coverage is omitted here if disabled, matching the existing
                # diagnostic behavior. Anchor/Smooth are added explicitly.
                diag_anchor, diag_smooth = g0_anchor_and_smooth_loss(
                    gaussians,
                    surface_reference,
                )
                diag_total = (
                    diag_total
                    + lambda_anchor * diag_anchor
                    + lambda_smooth * diag_smooth
                )
                diag_loss_values["anchor"] = diag_anchor
                diag_loss_values["smooth"] = diag_smooth

                diag_depth, diag_depth_source, diag_depth_valid_ratio = (
                    g0_depth_preservation_loss(
                        gaussians,
                        camera,
                        diag_pkg,
                        diag_alpha,
                        g0_depth_reference,
                        target_mask,
                        stage1_cfg,
                    )
                )
                diag_total = diag_total + lambda_depth * diag_depth
                diag_loss_values["depth"] = diag_depth

                diag_scale_anchor, diag_overlap, diag_gap_ratio = (
                    g0_scale_anchor_and_overlap_loss(
                        gaussians,
                        surface_reference,
                        g0_scaling_raw,
                        initial_count,
                        stage1_cfg,
                    )
                )
                diag_total = (
                    diag_total
                    + lambda_scale_anchor * diag_scale_anchor
                    + lambda_overlap * diag_overlap
                )
                diag_loss_values["scale_anchor"] = diag_scale_anchor
                diag_loss_values["overlap"] = diag_overlap

                diag_error_map = (
                    build_error_map(
                        diag_render,
                        diag_loss_values["target_object"],
                        diag_alpha,
                        target_mask,
                        diag_missing,
                        diag_extra,
                        stage1_cfg,
                    )
                )
                (diag_weights, _) = gaussian_update_weights(gaussians, camera, diag_error_map, diag_extra, stage1_cfg)
                save_diagnostics(
                    output_dir,
                    iteration,
                    gaussians,
                    camera,
                    target,
                    diag_render,
                    target_mask,
                    diag_alpha,
                    diag_missing,
                    diag_extra,
                    diag_error_map,
                    diag_weights,
                    diag_total.detach(),
                    diag_loss_values,
                    diag_alpha_source,
                    mask_source,
                )
                print(
                    f"\n[Stage1 ITER {iteration}] "
                    f"Total={diag_total.item():.7f}, "
                    f"RGB={diag_loss_values['rgb'].item():.6f}, "
                    f"Color={diag_loss_values['color'].item():.6f}, "
                    f"DSSIM={diag_loss_values['dssim'].item():.6f}, "
                    f"Alpha={diag_loss_values['alpha'].item():.6f}, "
                    f"Opacity={diag_loss_values['opacity'].item():.6f}, "
                    f"Edge={diag_loss_values['edge'].item():.6f}, "
                    f"Missing={diag_loss_values['missing'].item():.6f}, "
                    f"Extra={diag_loss_values['extra'].item():.6f}, "
                    f"Anchor={diag_loss_values['anchor'].item():.6f}, "
                    f"Smooth={diag_loss_values['smooth'].item():.6f}, "
                    f"Depth={diag_loss_values['depth'].item():.6f}, "
                    f"ScaleAnchor={diag_loss_values['scale_anchor'].item():.6f}, "
                    f"Overlap={diag_loss_values['overlap'].item():.6f}, "
                    f"Gap={diag_gap_ratio:.4f}, "
                    f"DepthSrc={diag_depth_source}, "
                    f"DepthValid={diag_depth_valid_ratio:.4f}, "
                    f"missing_ratio={diag_missing.mean().item():.6f}, "
                    f"extra_ratio={diag_extra.mean().item():.6f}, "
                    f"N={gaussians.get_xyz.shape[0]:,}"
                )
    progress.close()
    # =====================================================================
    # Final strict surface closure before save.
    # Repeat: hard contour -> detect every missing foreground pixel -> seed ->
    # hard contour again. No restoration of the original G0 footprint follows.
    # =====================================================================
    final_hard_info = {
        "hard_center_pruned": 0,
        "hard_support_pruned": 0,
        "hard_scale_shrunk": 0,
    }
    final_fill_added = 0
    final_fill_passes = max(
        1, cfg_int(stage1_cfg, "stage1_final_surface_closure_passes", 12)
    )

    for _final_pass in range(final_fill_passes):
        with torch.no_grad():
            _guard_pkg = render(
                camera, gaussians, stage1_cfg, black_background,
                0.0, 0.0, 0.0, stage1_cfg.is_6dof
            )

        _final_geometry_visibility = compute_stage1_visibility(
            gaussians,
            camera,
            stage1_cfg,
        )
        _guard_info = enforce_hard_rgb_support(
            gaussians,
            camera,
            target_mask,
            rgb_inside_distance,
            _guard_pkg.get("visibility_filter", None),
            _guard_pkg.get("radii", None),
            stage1_cfg,
            protected_count=protected_count_runtime,
            geometry_visibility=_final_geometry_visibility,
        )
        for _k in final_hard_info:
            final_hard_info[_k] += int(_guard_info[_k])

        # Re-render after shrink/prune because those operations may expose holes.
        with torch.no_grad():
            _fill_pkg = render(
                camera, gaussians, stage1_cfg, black_background,
                0.0, 0.0, 0.0, stage1_cfg.is_6dof
            )
        _fill_alpha_direct = _package_alpha(_fill_pkg, height, width)
        if _fill_alpha_direct is not None:
            _fill_alpha = _fill_alpha_direct.detach().clamp(0.0, 1.0)
        else:
            with torch.no_grad():
                _fill_white = torch.ones(3, dtype=torch.float32, device="cuda")
                _fill_white_pkg = render(
                    camera, gaussians, stage1_cfg, _fill_white,
                    0.0, 0.0, 0.0, stage1_cfg.is_6dof
                )
                _fill_alpha = (
                    1.0
                    - (
                        _fill_white_pkg["render"]
                        - _fill_pkg["render"]
                    ).mean(dim=0)
                ).clamp(0.0, 1.0)

        _final_missing, _final_extra = structure_masks(
            target_mask, _fill_alpha, stage1_cfg
        )
        _missing_count = int((_final_missing > 0.5).sum().item())

        print(
            f"[Final closure {_final_pass + 1}/{final_fill_passes}] "
            f"N={int(gaussians.get_xyz.shape[0]):,}, "
            f"missing_px={_missing_count:,}, "
            f"hard_center_pruned={int(_guard_info.get('hard_center_pruned', 0)):,}, "
            f"hard_support_pruned={int(_guard_info.get('hard_support_pruned', 0)):,}, "
            f"hard_scale_shrunk={int(_guard_info.get('hard_scale_shrunk', 0)):,}"
        )

        if _missing_count == 0 and sum(int(v) for v in _guard_info.values()) == 0:
            break

        remaining_budget = max(
            0, max_total_points - int(gaussians.get_xyz.shape[0])
        )
        if _missing_count > 0 and remaining_budget > 0:
            _fill_seed = seed_missing_gaussians(
                gaussians,
                camera,
                target,
                target_mask,
                _final_missing,
                _fill_pkg.get("visibility_filter", None),
                stage1_cfg,
                remaining_budget,
                seed_erode_override=0,
                raw_g0_surface_prior=raw_g0_surface_prior,
            )
            final_fill_added += int(_fill_seed.get("seed_added", 0))

    # One last hard pass after the final fill. In strict mode we deliberately
    # DO NOT call restore_original_g0_footprint(), because that would undo
    # contour-safe scale shrinkage.
    with torch.no_grad():
        _last_pkg = render(
            camera, gaussians, stage1_cfg, black_background,
            0.0, 0.0, 0.0, stage1_cfg.is_6dof
        )
    _last_geometry_visibility = compute_stage1_visibility(
        gaussians,
        camera,
        stage1_cfg,
    )
    _last_guard = enforce_hard_rgb_support(
        gaussians,
        camera,
        target_mask,
        rgb_inside_distance,
        _last_pkg.get("visibility_filter", None),
        _last_pkg.get("radii", None),
        stage1_cfg,
        protected_count=protected_count_runtime,
        geometry_visibility=_last_geometry_visibility,
    )
    for _k in final_hard_info:
        final_hard_info[_k] += int(_last_guard[_k])

    # -----------------------------------------------------------------
    # ORIGINAL-G0 HIDDEN GEOMETRY RECOVERY
    # -----------------------------------------------------------------
    # At this point all ordinary Stage-1 refinement is complete.  We now
    # restore only raw-G0 Gaussians that:
    #   (a) were hidden in the original fixed view, and
    #   (b) no longer have local 3D coverage in the corrected G0.
    #
    # The recovery step is intentionally placed AFTER cleanup/pruning so the
    # recovered back-side structure cannot be deleted again by a later
    # single-view cleanup stage.
    g0_recovery_info = recover_missing_hidden_gaussians(
        gaussians,
        g0_donor_bank,
        stage1_cfg,
    )
    print("[G0 recovery] " + recovery_summary(g0_recovery_info))

    # After G0 recovery, apply the SAME absolute contour rule again.
    # Hidden/uncertain recovered Gaussians are not exempt if their projected
    # centres lie outside the RGB subject or outside the camera frame.
    if int(g0_recovery_info.get("recovered_count", 0)) > 0:
        with torch.no_grad():
            _recovery_pkg = render(
                camera,
                gaussians,
                stage1_cfg,
                black_background,
                0.0, 0.0, 0.0,
                stage1_cfg.is_6dof,
            )

        _recovery_geometry_visibility = compute_stage1_visibility(
            gaussians,
            camera,
            stage1_cfg,
        )

        _recovery_guard = enforce_hard_rgb_support(
            gaussians,
            camera,
            target_mask,
            rgb_inside_distance,
            _recovery_pkg.get("visibility_filter", None),
            _recovery_pkg.get("radii", None),
            stage1_cfg,
            protected_count=protected_count_runtime,
            geometry_visibility=_recovery_geometry_visibility,
        )

        for _k in final_hard_info:
            final_hard_info[_k] += int(_recovery_guard[_k])

        g0_recovery_info["post_recovery_visible_pruned"] = int(
            _recovery_guard.get("hard_center_pruned", 0)
            + _recovery_guard.get("hard_support_pruned", 0)
        )
    else:
        g0_recovery_info["post_recovery_visible_pruned"] = 0

    # -----------------------------------------------------------------
    # FINAL NON-FRONT APPEARANCE PROTECTION
    # -----------------------------------------------------------------
    # Keep the improved RGB color ONLY on the confidently visible front.
    # Hidden / uncertain / outside Gaussians recover original-G0 SH color.
    appearance_restore_info = _restore_original_g0_appearance(
        gaussians,
        g0_donor_bank,
        camera,
        stage1_cfg,
    )
    print(
        "[Appearance restore] "
        f"enabled={appearance_restore_info.get('enabled', False)}, "
        f"restored_nonfront={appearance_restore_info.get('restored', 0):,}, "
        f"front_kept={appearance_restore_info.get('front_kept', 0):,}, "
        f"dc={appearance_restore_info.get('copied_dc', False)}, "
        f"rest={appearance_restore_info.get('copied_rest', False)}, "
        f"opacity={appearance_restore_info.get('copy_opacity', False)}"
    )

    scene.save(iterations)
    corrected_path = (output_dir / "corrected_g0.ply")
    gaussians.save_ply(str(corrected_path))
    final_count = int(gaussians.get_xyz.shape[0])
    final_geometry_visibility = compute_stage1_visibility(
        gaussians,
        camera,
        stage1_cfg,
    )
    if final_geometry_visibility is not None:
        print("[Visibility final] " + visibility_summary(final_geometry_visibility))

    with torch.no_grad():
        final_pkg = render(camera, gaussians, stage1_cfg, black_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
        final_render = (final_pkg["render"].clamp(0.0, 1.0))
    final_direct_alpha = _package_alpha(final_pkg, height, width)
    if final_direct_alpha is not None:
        final_alpha = (final_direct_alpha.detach().clamp(0.0, 1.0))
    else:
        with torch.no_grad():
            white_background = torch.ones(3, dtype=torch.float32, device="cuda")
            final_white_pkg = render(camera, gaussians, stage1_cfg, white_background, 0.0, 0.0, 0.0, stage1_cfg.is_6dof)
            final_alpha = (1.0 - (final_white_pkg["render"] - final_pkg["render"]).mean(dim=0)).clamp(0.0, 1.0)
    with torch.no_grad():
        (final_loss, final_diag) = object_aware_loss(final_render, final_alpha, target, target_mask, stage1_cfg)
        final_coverage_target, final_coverage_stats = compute_coverage_target(gaussians, stage1_cfg)
        final_coverage_loss, final_coverage_under_ratio = local_coverage_loss(
            gaussians, final_coverage_target, final_pkg.get("visibility_filter", None), stage1_cfg
        )
        final_diag["coverage"] = final_coverage_loss
        final_loss = final_loss + lambda_coverage * final_coverage_loss

        final_surface_prior_loss, final_surface_prior_diag = raw_g0_surface_prior_loss(
            gaussians,
            raw_g0_surface_prior,
            stage1_cfg,
            iterations,
            force=True,
        )
        final_diag["surface_prior"] = final_surface_prior_loss
        final_loss = final_loss + lambda_surface_prior * final_surface_prior_loss

        final_feature_loss, final_feature_diag = feature_prior_loss(
            final_render,
            target_mask,
            feature_prior_state,
            stage1_cfg,
            iterations,
            force=True,
        )
        final_diag["feature"] = final_feature_loss
        final_loss = final_loss + lambda_feature_prior * final_feature_loss

        (final_missing, final_extra) = structure_masks(target_mask, final_alpha, stage1_cfg)
    # Evaluate the actual FINAL Gaussian cloud after recovery, appearance restore,
    # save, and the final render; no training-step or topology changes follow.
    final_metrics, final_neutral, _, final_missing_eval, final_extra_eval = (
        evaluate_stage1_projection(
            final_render, final_alpha, target, target_mask, stage1_cfg, final_count
        )
    )
    final_metrics['lpips_neutral_full'] = evaluate_stage1_lpips(
        final_neutral, target_neutral, lpips_eval_model
    )
    write_stage1_metric_file(eval_dir / 'final_metrics.txt', final_metrics)
    save_rgb(final_neutral, eval_dir / 'final_render_neutral.png')
    save_gray(final_alpha, eval_dir / 'final_alpha.png')
    save_gray(final_missing_eval, eval_dir / 'final_missing.png')
    save_gray(final_extra_eval, eval_dir / 'final_extra.png')
    with (eval_dir / 'metrics_comparison.txt').open('w', encoding='utf-8') as f:
        f.write('Metric | Initial G0 | Final G0* | Delta (final-initial)\n')
        for key, v0 in initial_metrics.items():
            v1 = final_metrics.get(key)
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                f.write(f'{key} | {v0:.9f} | {v1:.9f} | {v1 - v0:+.9f}\n')
            else:
                f.write(f'{key} | {v0} | {v1} | unavailable\n')
    print('[Final evaluation] ' + ', '.join(
        f'{k}={v:.6f}' for k, v in final_metrics.items()
        if k in ('psnr_foreground_db', 'ssim_neutral_full', 'mask_iou')
    ))

    topology_log_path = (output_dir / "topology_log.txt")
    with topology_log_path.open("w", encoding="utf-8") as f:
        for record in topology_records:
            f.write(" ".join(f"{key}={value}" for key, value in record.items()) + "\n")
    summary_path = (output_dir / "stage1_summary.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("MonoEv Stage-1 object-aware canonical G0 refinement\n")
        f.write(f"input_g0 = {g0_path}\n")
        f.write(f"target_rgb = {image_path}\n")
        f.write(f"camera_file = " f"{getattr(stage1_cfg, 'camera_file', 'Scene default')}\n")
        f.write(f"mask_source = {mask_source}\n")
        f.write("real_target_background = ignored / neutralized\n")
        f.write("internal_renderer_background = black neutral compositing only\n")
        f.write("target_mask_definition = all RGB pixels remaining after background removal\n")
        f.write("hard_rgb_mask = enabled; only confidently visible front-surface Gaussian centres/support are constrained by target RGB foreground\n")
        f.write(f"visibility_protection = {cfg_bool(stage1_cfg, 'stage1_enable_visibility_protection', True)}\n")
        f.write(f"front_surface_only_updates = {cfg_bool(stage1_cfg, 'stage1_front_surface_only_updates', True)}\n")
        f.write(f"uncertain_update_weight = {cfg_float(stage1_cfg, 'stage1_uncertain_update_weight', 0.0)}\n")
        f.write(f"preserve_original_appearance = {cfg_bool(stage1_cfg, 'stage1_preserve_original_appearance', True)}\n")
        f.write(f"train_color_dc = {train_color_dc}\n")
        f.write(f"appearance_restore_count = {appearance_restore_info.get('restored', 0)}\n")
        if final_geometry_visibility is not None:
            f.write(f"visibility_visible = {final_geometry_visibility['num_visible']}\n")
            f.write(f"visibility_hidden = {final_geometry_visibility['num_hidden']}\n")
            f.write(f"visibility_uncertain = {final_geometry_visibility['num_uncertain']}\n")
            f.write(f"visibility_outside = {final_geometry_visibility['num_outside']}\n")
        f.write(f"g0_recovery_enabled = {g0_recovery_info.get('enabled', False)}\n")
        f.write(f"g0_recovery_donor_total = {g0_recovery_info.get('donor_total', 0)}\n")
        f.write(f"g0_recovery_hidden_eligible = {g0_recovery_info.get('eligible_hidden', 0)}\n")
        f.write(f"g0_recovery_uncertain_eligible = {g0_recovery_info.get('eligible_uncertain', 0)}\n")
        f.write(f"g0_recovery_candidates = {g0_recovery_info.get('candidate_count', 0)}\n")
        f.write(f"g0_recovery_recovered = {g0_recovery_info.get('recovered_count', 0)}\n")
        f.write(f"g0_recovery_threshold = {g0_recovery_info.get('distance_threshold', 0.0):.9f}\n")
        f.write(f"g0_recovery_raw_spacing = {g0_recovery_info.get('spacing_median', 0.0):.9f}\n")
        f.write(f"g0_recovery_post_visible_pruned = {g0_recovery_info.get('post_recovery_visible_pruned', 0)}\n")
        f.write(f"final_hard_center_pruned = {final_hard_info['hard_center_pruned']}\n")
        f.write(f"final_hard_support_pruned = {final_hard_info['hard_support_pruned']}\n")
        f.write(f"final_hard_scale_shrunk = {final_hard_info['hard_scale_shrunk']}\n")
        f.write(f"iterations = {iterations}\n")
        f.write("topology_policy = continuous_add_delete_local_split_g0_coherent\n")
        f.write(f"clean_main_subject_mask = {cfg_bool(stage1_cfg, 'stage1_clean_main_subject_mask', True)}\n")
        f.write(f"g0_birth_structure_gate = {cfg_bool(stage1_cfg, 'stage1_enable_g0_birth_structure_gate', True)}\n")
        f.write(f"g0_birth_max_distance_mult = {cfg_float(stage1_cfg, 'stage1_g0_birth_max_distance_mult', 8.0)}\n")
        f.write(f"detail_densification = {cfg_bool(stage1_cfg, 'stage1_enable_detail_densification', True)}\n")
        f.write(f"detail_min_screen_radius_px = {cfg_float(stage1_cfg, 'stage1_detail_min_screen_radius_px', 4.0)}\n")
        f.write(f"footprint_tolerance_px = {cfg_float(stage1_cfg, 'stage1_footprint_tolerance_px', 1.5)}\n")
        f.write(f"footprint_prune_overflow_ratio = {cfg_float(stage1_cfg, 'stage1_footprint_prune_overflow_ratio', 0.68)}\n")
        f.write("addition_start_iter = 1\n")
        f.write(f"densify_stop_iter = {iterations}\n")
        f.write("cleanup_start_iter = 1\n")
        f.write(f"cleanup_interval = {topology_interval}\n")
        f.write(f"lambda_coverage = {lambda_coverage}\n")
        f.write(f"lambda_surface_prior = {lambda_surface_prior}\n")
        f.write(f"lambda_feature_prior = {lambda_feature_prior}\n")
        f.write(f"feature_prior_enabled = {feature_prior_state.enabled}\n")
        f.write(f"feature_prior_backend = {feature_prior_state.backend}\n")
        f.write(f"feature_prior_model = {feature_prior_state.model_name}\n")
        f.write(f"feature_prior_input_size = {feature_prior_state.input_size}\n")
        f.write(f"feature_prior_grid = {feature_prior_state.grid_hw[0]}x{feature_prior_state.grid_hw[1]}\n")
        f.write(f"feature_prior_final_patch = {final_feature_diag.get('patch', 0.0):.9f}\n")
        f.write(f"feature_prior_final_global = {final_feature_diag.get('global', 0.0):.9f}\n")
        f.write(f"feature_prior_final_mean_cosine = {final_feature_diag.get('mean_cosine', 0.0):.9f}\n")
        f.write(f"surface_prior_enabled = {cfg_bool(stage1_cfg, 'stage1_enable_surface_prior', True)}\n")
        f.write(f"surface_prior_spacing = {raw_g0_surface_prior.get('spacing', 0.0):.9f}\n")
        f.write(f"surface_prior_sample_points = {cfg_int(stage1_cfg, 'stage1_surface_prior_sample_points', 8192)}\n")
        f.write(f"surface_prior_every = {cfg_int(stage1_cfg, 'stage1_surface_prior_every', 10)}\n")
        f.write(f"surface_prior_final_valid_ratio = {final_surface_prior_diag.get('valid_ratio', 0.0):.9f}\n")
        f.write(f"surface_prior_final_plane = {final_surface_prior_diag.get('plane', 0.0):.9f}\n")
        f.write(f"surface_prior_final_tangent = {final_surface_prior_diag.get('tangent', 0.0):.9f}\n")
        f.write(f"coverage_beta = {cfg_float(stage1_cfg, 'stage1_coverage_beta', 0.55)}\n")
        f.write(f"coverage_spacing_median = {final_coverage_stats['spacing_median']:.9f}\n")
        f.write(f"coverage_target_median = {final_coverage_stats['target_median']:.9f}\n")
        f.write(f"coverage_under_ratio = {final_coverage_under_ratio:.9f}\n")
        f.write(f"before_after_metrics = {eval_dir / 'metrics_comparison.txt'}\n")
        f.write(f"initial_gaussians = {initial_count}\n")
        f.write(f"final_gaussians = {final_count}\n")
        f.write(f"net_gaussian_change = " f"{final_count - initial_count}\n")
        f.write(f"final_total_loss = " f"{final_loss.item():.9f}\n")
        for key in ("rgb", "color", "dssim", "alpha", "alpha_bce", "alpha_dice", "opacity", "opacity_inside", "opacity_outside", "edge", "missing", "extra", "coverage", "anchor", "smooth", "surface_prior", "feature", "depth", "scale_anchor", "overlap"):
                if key in final_diag:
                    f.write(f"final_{key} = {final_diag[key].item():.9f}\n")
                else:
                    f.write(f"final_{key} = not_computed\n")

        f.write(f"final_missing_ratio = " f"{final_missing.mean().item():.9f}\n")
        f.write(f"final_extra_ratio = " f"{final_extra.mean().item():.9f}\n")
        f.write(f"corrected_g0 = {corrected_path}\n")
        f.write(
            "trainable = xyz, rotation, scaling, opacity"
            + (", SH-DC color" if train_color_dc else "")
            + "\n"
        )
        f.write(
            "frozen = higher-order SH, fixed camera"
            + (", SH-DC color" if not train_color_dc else "")
            + "\n"
        )
        f.write("event_supervision = none\n")
        f.write("deform_model = none\n")
    if stage1_cfg.load2gpu_on_the_fly:
        camera.load2device("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    print("\nStage-1 complete.")
    print(f"Target mask source: {mask_source}")
    print(f"Initial Gaussians: {initial_count:}")
    print(f"Final Gaussians: {final_count:}")
    print(f"Net change: " f"{final_count - initial_count:+}")
    print(f"Final loss: " f"{final_loss.item():.7f}")
    print(f"Final missing ratio: " f"{final_missing.mean().item():.7f}")
    print(f"Final extra ratio: " f"{final_extra.mean().item():.7f}")
    print(f"Final coverage loss: " f"{final_coverage_loss.item():.7f}")
    print(f"Coverage under-scale ratio: " f"{final_coverage_under_ratio:.7f}")
    print(
        "Final raw-G0 surface prior: "
        f"{final_surface_prior_loss.item():.7f} "
        f"(valid={final_surface_prior_diag.get('valid_ratio', 0.0):.4f})"
    )
    print(
        "Final DINO feature prior: "
        f"{final_feature_loss.item():.7f} "
        f"(cos={final_feature_diag.get('mean_cosine', 0.0):.4f})"
    )
    print("[G0 recovery final] " + recovery_summary(g0_recovery_info))
    print(f"Corrected G0: {corrected_path}")
    print("Stage-2 checkpoint: " f"{output_dir}/point_cloud/" f"iteration_{iterations}/point_cloud.ply")
    print(f"Diagnostics: {output_dir}/diagnostics")
    print(f"Topology log: {topology_log_path}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
