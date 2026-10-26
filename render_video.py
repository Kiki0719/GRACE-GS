"""
Render MonoEv checkpoints with one fixed camera.

This renderer is kept consistent with MonoEv-v1 training:
- one canonical Gaussian point cloud
- one fixed camera
- normalized deformation time in [0, 1]
- d_xyz is enabled
- d_rotation / d_scaling are disabled unless the config explicitly enables them

It also writes diagnostic renders so canonical-G0 problems can be separated
from deformation problems.
"""

import inspect
import math
import subprocess
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.deform_model import DeformModel
from utils import parse_utils
from utils.general_utils import safe_state


def cfg_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def update_cfg(cfg, updates):
    values = cfg._asdict().copy()
    values.update(updates)
    Config = namedtuple("RenderConfig", values.keys())
    return Config(**values)


def load_deform(deform, model_path, iteration):
    checkpoint_path = (
        model_path
        / "deform"
        / f"iteration_{iteration}"
        / "deform.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))

    parameters = inspect.signature(deform.load_weights).parameters

    print(f"\nLoading deformation: {checkpoint_path}")
    print(f"load_weights interface: {inspect.signature(deform.load_weights)}")

    if "iteration" in parameters:
        deform.load_weights(str(model_path), iteration=iteration)
    else:
        deform.load_weights(str(checkpoint_path))

    deform.deform.eval()
    return checkpoint_path


def event_time_scale(unit):
    unit = str(unit).strip().lower()
    table = {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "µs": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
    }
    if unit not in table:
        raise ValueError(
            f"Unsupported event_time_unit={unit!r}; use s/ms/us/ns"
        )
    return table[unit]


def infer_event_duration(cfg):
    """
    Infer physical event-stream duration directly from cfg.event_path.
    """
    event_path = getattr(cfg, "event_path", None)
    if event_path in (None, "", "None", "none", "null"):
        return None

    path = Path(str(event_path)).expanduser()
    if not path.is_file():
        return None

    try:
        if path.suffix.lower() == ".npz":
            payload = np.load(str(path), allow_pickle=True)
            if "events" in payload:
                events = payload["events"]
            elif len(payload.files) == 1:
                events = payload[payload.files[0]]
            else:
                return None
        else:
            events = np.load(str(path), mmap_mode="r", allow_pickle=True)

        if events.ndim != 2 or events.shape[0] < 2 or events.shape[1] < 3:
            return None

        t0 = float(events[0, 2])
        t1 = float(events[-1, 2])
        scale = event_time_scale(getattr(cfg, "event_time_unit", "s"))
        duration = (t1 - t0) * scale

        if math.isfinite(duration) and duration > 0:
            return duration
    except Exception as error:
        print(f"[WARNING] Could not infer event duration: {error}")

    return None


def effective_head_flags(cfg):
    """
    MonoEv-v1 defaults:
        d_xyz      = ON
        d_rotation = OFF
        d_scaling  = OFF
    """
    use_rotation = cfg_bool(
        getattr(cfg, "enable_deform_rotation", False)
    )
    use_scaling = cfg_bool(
        getattr(cfg, "enable_deform_scaling", False)
    )
    return use_rotation, use_scaling


@torch.no_grad()
def deform_points(
    deform,
    points,
    fid,
    cfg,
    subtract_t0_deform=False,
):
    times = torch.full(
        (points.shape[0], 1),
        float(fid),
        device=points.device,
        dtype=points.dtype,
    )

    d_xyz, d_rotation, d_scaling = deform.step(
        points.detach(),
        times,
    )

    # Optional rescue/debug mode for checkpoints with a large t=0 gauge drift.
    # Normal MonoEv-v1 rendering should leave this OFF.
    if subtract_t0_deform:
        if bool(getattr(cfg, "is_6dof", False)):
            raise ValueError(
                "--subtract_t0_deform is only supported for is_6dof=False"
            )
        zero_times = torch.zeros_like(times)
        d_xyz_t0, _, _ = deform.step(
            points.detach(),
            zero_times,
        )
        d_xyz = d_xyz - d_xyz_t0

    use_rotation, use_scaling = effective_head_flags(cfg)

    # CRITICAL: training MonoEv-v1 zeroes these heads before rendering.
    if not use_rotation:
        d_rotation = torch.zeros_like(d_rotation)

    if not use_scaling:
        d_scaling = torch.zeros_like(d_scaling)

    return d_xyz, d_rotation, d_scaling


@torch.no_grad()
def deform_all(
    gaussians,
    deform,
    cfg,
    fid,
    subtract_t0_deform=False,
):
    xyz = gaussians.get_xyz.detach()

    chunk_size = max(
        1,
        int(getattr(cfg, "deform_chunk_size", 4096)),
    )

    offsets = []
    rotations = []
    scales = []

    for start in range(0, xyz.shape[0], chunk_size):
        end = min(start + chunk_size, xyz.shape[0])
        points = xyz[start:end]

        d_xyz, d_rotation, d_scaling = deform_points(
            deform,
            points,
            fid,
            cfg,
            subtract_t0_deform=subtract_t0_deform,
        )

        offsets.append(d_xyz)
        rotations.append(d_rotation)
        scales.append(d_scaling)

    return (
        torch.cat(offsets, dim=0),
        torch.cat(rotations, dim=0),
        torch.cat(scales, dim=0),
    )


@torch.no_grad()
def render_frame(
    camera,
    gaussians,
    deform,
    cfg,
    background,
    fid,
    subtract_t0_deform=False,
):
    d_xyz, d_rotation, d_scaling = deform_all(
        gaussians,
        deform,
        cfg,
        fid,
        subtract_t0_deform=subtract_t0_deform,
    )

    image = render(
        camera,
        gaussians,
        cfg,
        background,
        d_xyz,
        d_rotation,
        d_scaling,
        bool(cfg.is_6dof),
    )["render"]

    if not torch.isfinite(image).all().item():
        raise FloatingPointError(
            f"Rendered image contains NaN/Inf at time={fid}"
        )

    return image.clamp(0.0, 1.0)


@torch.no_grad()
def render_canonical_g0(
    camera,
    gaussians,
    cfg,
    background,
):
    """
    Render checkpoint PLY with exactly zero temporal deformation.
    """
    if bool(cfg.is_6dof):
        raise NotImplementedError(
            "Canonical zero-deformation diagnostic expects is_6dof=False"
        )

    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)

    image = render(
        camera,
        gaussians,
        cfg,
        background,
        d_xyz,
        d_rotation,
        d_scaling,
        False,
    )["render"]

    return image.clamp(0.0, 1.0)


@torch.no_grad()
def diagnose_deformation(
    gaussians,
    deform,
    cfg,
    subtract_t0_deform=False,
):
    xyz = gaussians.get_xyz.detach()
    if xyz.shape[0] == 0:
        raise RuntimeError("Loaded Gaussian model is empty")

    count = min(8192, xyz.shape[0])
    indices = torch.linspace(
        0,
        xyz.shape[0] - 1,
        steps=count,
        device=xyz.device,
    ).long()
    points = xyz[indices]

    use_rotation, use_scaling = effective_head_flags(cfg)

    print("\n========== 形变时间检查 ==========")
    print(f"高斯总数: {xyz.shape[0]}")
    print(f"检查点数: {count}")
    print(f"is_6dof: {cfg.is_6dof}")
    print("实际渲染头:")
    print("  d_xyz      : ON")
    print(f"  d_rotation : {'ON' if use_rotation else 'OFF'}")
    print(f"  d_scaling  : {'ON' if use_scaling else 'OFF'}")
    print(
        "  subtract_t0_deform : "
        f"{'ON' if subtract_t0_deform else 'OFF'}"
    )

    raw_reference = None
    effective_reference = None

    for fid in [0.0, 0.25, 0.5, 0.75, 1.0]:
        times = torch.full(
            (count, 1),
            float(fid),
            dtype=points.dtype,
            device=points.device,
        )

        raw_outputs = deform.step(points, times)

        effective_outputs = deform_points(
            deform,
            points,
            fid,
            cfg,
            subtract_t0_deform=subtract_t0_deform,
        )

        if raw_reference is None:
            raw_reference = tuple(v.clone() for v in raw_outputs)
            effective_reference = tuple(
                v.clone() for v in effective_outputs
            )

        print(f"\ntime={fid:.2f}")

        for name, raw_value, raw_initial in zip(
            ["d_xyz", "d_rotation", "d_scaling"],
            raw_outputs,
            raw_reference,
        ):
            if not torch.isfinite(raw_value).all().item():
                raise FloatingPointError(
                    f"{name} contains NaN/Inf at time={fid}"
                )

            raw_difference = (raw_value - raw_initial).abs()

            print(
                f"  RAW {name}: "
                f"幅值均值={raw_value.abs().mean().item():.8e}, "
                f"相对t0变化均值={raw_difference.mean().item():.8e}, "
                f"相对t0变化最大值={raw_difference.max().item():.8e}"
            )

        effective_xyz = effective_outputs[0]
        effective_xyz_initial = effective_reference[0]
        effective_delta = (
            effective_xyz - effective_xyz_initial
        ).abs()

        print(
            "  EFFECTIVE d_xyz: "
            f"幅值均值={effective_xyz.abs().mean().item():.8e}, "
            f"相对t0变化均值={effective_delta.mean().item():.8e}, "
            f"相对t0变化最大值={effective_delta.max().item():.8e}"
        )

    print("\n=================================\n")


@torch.no_grad()
def diagnose_render(
    camera,
    gaussians,
    deform,
    cfg,
    background,
    out_dir,
    subtract_t0_deform=False,
):
    debug_dir = out_dir / "motion_check"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== 实际渲染变化检查 ==========")
    print("先保存 canonical_g0.png（完全不施加 deformation）。")

    canonical = render_canonical_g0(
        camera,
        gaussians,
        cfg,
        background,
    )
    save_image(
        canonical,
        str(debug_dir / "canonical_g0.png"),
    )

    reference = None
    reference_uint8 = None

    for fid in [0.0, 0.25, 0.5, 0.75, 1.0]:
        image = render_frame(
            camera,
            gaussians,
            deform,
            cfg,
            background,
            fid,
            subtract_t0_deform=subtract_t0_deform,
        )

        tag = f"{fid:.2f}".replace(".", "_")

        save_image(
            image,
            str(debug_dir / f"render_t_{tag}.png"),
        )

        image_uint8 = (
            image * 255.0 + 0.5
        ).clamp(0, 255).to(torch.uint8)

        if reference is None:
            reference = image.clone()
            reference_uint8 = image_uint8.clone()

            canonical_diff = (image - canonical).abs()
            save_image(
                (canonical_diff * 10.0).clamp(0.0, 1.0),
                str(debug_dir / "diff_x10_t0_vs_canonical.png"),
            )

            canonical_uint8 = (
                canonical * 255.0 + 0.5
            ).clamp(0, 255).to(torch.uint8)

            canonical_changed = (
                image_uint8 != canonical_uint8
            ).any(dim=0).float().mean().item()

            print(
                "time=0.00 vs canonical: "
                f"像素差均值={canonical_diff.mean().item():.8e}, "
                f"最大值={canonical_diff.max().item():.8e}, "
                f"PNG变化像素比例={canonical_changed:.6%}"
            )

        difference = (image - reference).abs()

        changed_pixels = (
            image_uint8 != reference_uint8
        ).any(dim=0).float().mean().item()

        print(
            f"time={fid:.2f}: "
            f"相对t0像素差均值={difference.mean().item():.8e}, "
            f"最大值={difference.max().item():.8e}, "
            f"PNG变化像素比例={changed_pixels:.6%}"
        )

        if fid > 0.0:
            save_image(
                (difference * 10.0).clamp(0.0, 1.0),
                str(debug_dir / f"diff_x10_t_{tag}.png"),
            )

    print(f"检查图片保存到: {debug_dir}")
    print(
        "canonical_g0.png = checkpoint PLY 在 fixed camera 下、"
        "零 deformation 的基准。"
    )
    print(
        "diff_x10_t0_vs_canonical.png = t=0 deformation "
        "相对 G0 的变化。"
    )
    print("====================================\n")


def find_latest_iteration(model_path):
    candidates = []

    for path in (model_path / "point_cloud").glob("iteration_*"):
        suffix = path.name.removeprefix("iteration_")
        if not suffix.isdigit():
            continue

        iteration = int(suffix)
        ply_path = path / "point_cloud.ply"
        deform_path = (
            model_path
            / "deform"
            / f"iteration_{iteration}"
            / "deform.pth"
        )

        if ply_path.is_file() and deform_path.is_file():
            candidates.append(iteration)

    if not candidates:
        raise FileNotFoundError(
            f"No complete Gaussian/deform checkpoints in {model_path}"
        )

    return max(candidates)


def main():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument("--config", required=True)

    parser.add_argument(
        "--override",
        default=None,
        help="Same configuration override used for training, if any",
    )

    parser.add_argument(
        "--output",
        "--model_path",
        dest="model_path",
        default="/root/autodl-tmp/Mono_output",
    )

    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="-1 selects the latest complete checkpoint",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help=(
            "Explicit frame count spanning normalized time [0,1]. "
            "If omitted, infer frame count from event duration and FPS."
        ),
    )

    parser.add_argument(
        "--diagnose_only",
        action="store_true",
        help="Save diagnostic renders without making the video",
    )

    parser.add_argument(
        "--subtract_t0_deform",
        action="store_true",
        help=(
            "Subtract each Gaussian's d_xyz(t=0) from d_xyz(t). "
            "Use only to inspect/rescue checkpoints with canonical gauge drift; "
            "OFF is the faithful training-time render."
        ),
    )

    cli = parser.parse_args()

    if not math.isfinite(cli.fps) or cli.fps <= 0:
        parser.error("--fps must be positive and finite")

    if cli.num_frames is not None and cli.num_frames < 2:
        parser.error("--num_frames must be at least 2")

    model_path = Path(cli.model_path).expanduser().resolve()

    iteration = cli.iteration
    if iteration == -1:
        iteration = find_latest_iteration(model_path)

    ply_path = (
        model_path
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )

    deform_path = (
        model_path
        / "deform"
        / f"iteration_{iteration}"
        / "deform.pth"
    )

    for path in (ply_path, deform_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    cfg, _ = parse_utils.parse_ini(cli.config)

    if cli.override is not None:
        cfg = parse_utils.override_cfg(cli.override, cfg)

    cfg = update_cfg(
        cfg,
        {
            "model_path": str(model_path),
            "initial_gaussian_ply": str(ply_path),
            "train_select": [0],
            "test_select": [],
            "event_assist": False,
        },
    )

    safe_state(bool(getattr(cfg, "quiet", False)))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this renderer")

    gaussians = GaussianModel(cfg.sh_degree)

    scene = Scene(
        cfg,
        gaussians,
        static_load_path=str(ply_path),
    )

    cameras = list(scene.getTrainCameras())

    if len(cameras) != 1:
        raise RuntimeError(
            f"Expected one fixed camera, found {len(cameras)}"
        )

    if gaussians.get_xyz.shape[0] == 0:
        raise RuntimeError("Loaded Gaussian model is empty")

    camera = cameras[0]

    if cfg_bool(getattr(cfg, "load2gpu_on_the_fly", False)):
        camera.load2device()

    deform = DeformModel(cfg)
    loaded_deform_path = load_deform(
        deform,
        model_path,
        iteration,
    )

    background = torch.tensor(
        [1.0, 1.0, 1.0]
        if cfg_bool(getattr(cfg, "white_background", True))
        else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )

    out_dir = model_path / f"render_video_iter_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_rotation, use_scaling = effective_head_flags(cfg)

    print(f"\nGaussian checkpoint: {ply_path}")
    print(f"Deform checkpoint: {loaded_deform_path}")
    print(f"Selected iteration: {iteration}")
    print(f"Gaussian count: {gaussians.get_xyz.shape[0]}")
    print(
        "Renderer deformation heads: "
        f"d_xyz=ON, "
        f"d_rotation={'ON' if use_rotation else 'OFF'}, "
        f"d_scaling={'ON' if use_scaling else 'OFF'}"
    )
    print(
        "t0 deformation subtraction: "
        f"{'ON' if cli.subtract_t0_deform else 'OFF'}"
    )

    diagnose_deformation(
        gaussians,
        deform,
        cfg,
        subtract_t0_deform=cli.subtract_t0_deform,
    )

    diagnose_render(
        camera,
        gaussians,
        deform,
        cfg,
        background,
        out_dir,
        subtract_t0_deform=cli.subtract_t0_deform,
    )

    if cli.diagnose_only:
        if cfg_bool(getattr(cfg, "load2gpu_on_the_fly", False)):
            camera.load2device("cpu")
        print("Diagnosis complete. No video requested.")
        return

    duration = infer_event_duration(cfg)

    if cli.num_frames is None:
        if duration is None:
            print(
                "[WARNING] Could not infer event duration; "
                "using 100 frames over normalized time [0,1]."
            )
            num_frames = 100
        else:
            num_frames = max(
                2,
                int(math.ceil(duration * cli.fps)),
            )
    else:
        num_frames = cli.num_frames

    time_values = np.linspace(
        0.0,
        1.0,
        num_frames,
        dtype=np.float64,
    )

    full_dir = out_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)

    for old_png in full_dir.glob("*.png"):
        old_png.unlink()

    print(
        f"\nFrames: {num_frames}, "
        f"FPS: {cli.fps}, "
        f"video duration: {num_frames / cli.fps:.4f}s"
    )

    if duration is not None:
        print(f"Physical event duration: {duration:.6f}s")

    print(
        f"Normalized time: "
        f"{time_values[0]:.6f} -> {time_values[-1]:.6f}"
    )

    for idx, fid in enumerate(
        tqdm(time_values, desc="Rendering MonoEv")
    ):
        image = render_frame(
            camera,
            gaussians,
            deform,
            cfg,
            background,
            float(fid),
            subtract_t0_deform=cli.subtract_t0_deform,
        )

        save_image(
            image,
            str(full_dir / f"{idx:05d}.png"),
        )

    if cfg_bool(getattr(cfg, "load2gpu_on_the_fly", False)):
        camera.load2device("cpu")

    mp4_path = out_dir / f"video_iter_{iteration}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(cli.fps),
        "-start_number",
        "0",
        "-i",
        str(full_dir / "%05d.png"),
        "-frames:v",
        str(num_frames),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]

    print("\nMaking video...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Video encoding failed: {error}")
        print(f"PNG frames are available in: {full_dir}")
        return

    print(f"\nSaved video: {mp4_path}")
    print(f"Saved PNG frames: {full_dir}")


if __name__ == "__main__":
    main()
