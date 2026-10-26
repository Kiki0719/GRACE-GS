import os
from pathlib import Path
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils import parse_utils


CONFIG_PATH = "/E-D3DGS/config/fan_event_only_3000_rerun.ini"

MODEL_PATH = (
    "/root/autodl-tmp/E-D3DGS_output/"
    "fan_event_only_3000_rerun"
)

ITERATION = 3000

PLY_PATH = os.path.join(
    MODEL_PATH,
    "static_point_cloud",
    f"iteration_{ITERATION}",
    "point_cloud.ply",
)


def build_config():
    cfg, _ = parse_utils.parse_ini(CONFIG_PATH)
    cfg_dict = cfg._asdict().copy()

    cfg_dict["model_path"] = MODEL_PATH

    # 静态离线渲染不需要再次加载事件流
    cfg_dict["event_assist"] = False

    # 与 train.py 保持完全相同的训练/测试视角划分
    image_root = Path(cfg_dict["source_path"]) / "rgb"
    image_count = len(list(image_root.rglob("*.*")))

    train_select = np.linspace(
        0,
        image_count - 1,
        cfg_dict["train_rgb_num"],
        dtype=int,
    )

    test_select = np.zeros(
        len(train_select) - 1,
        dtype=np.int64,
    )

    for i in range(len(train_select) - 1):
        test_select[i] = round(
            (train_select[i] + train_select[i + 1]) / 2
        )

    cfg_dict["train_select"] = sorted(train_select.tolist())
    cfg_dict["test_select"] = sorted(test_select.tolist())

    Config = namedtuple("Config", cfg_dict.keys())
    return Config(**cfg_dict)


def main():
    if not os.path.isfile(PLY_PATH):
        raise FileNotFoundError(f"PLY not found: {PLY_PATH}")

    args = build_config()

    print("Loading:", PLY_PATH)

    gaussians = GaussianModel(args.sh_degree)

    scene = Scene(
        args,
        gaussians,
        static_load_path=PLY_PATH,
    )

    pipeline = SimpleNamespace(
        debug=False,
        convert_SHs_python=False,
        convert_shs_python=False,
        compute_cov3D_python=False,
        compute_cov3d_python=False,
    )

    background_color = (
        [1.0, 1.0, 1.0]
        if args.white_background
        else [0.0, 0.0, 0.0]
    )

    background = torch.tensor(
        background_color,
        dtype=torch.float32,
        device="cuda",
    )

    output_root = os.path.join(
        MODEL_PATH,
        "test",
        f"ours_{ITERATION}",
    )

    render_path = os.path.join(output_root, "renders")
    gt_path = os.path.join(output_root, "gt")
    depth_path = os.path.join(output_root, "depth")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    views = scene.getTestCameras()

    print("Test cameras:", len(views))

    with torch.no_grad():
        for idx, view in enumerate(
            tqdm(views, desc="Rendering test views")
        ):
            if getattr(args, "load2gpu_on_the_fly", False):
                view.load2device()

            result = render(
                view,
                gaussians,
                pipeline,
                background,
                0.0,   # d_xyz：静态 warm-up 不形变
                0.0,   # d_rotation
                0.0,   # d_scaling
                getattr(args, "is_6dof", False),
            )

            image = torch.clamp(result["render"], 0.0, 1.0)
            gt = torch.clamp(
                view.original_image.to("cuda"),
                0.0,
                1.0,
            )

            depth = result["depth"]
            depth = depth / (depth.max() + 1e-8)

            filename = f"{idx:05d}.png"

            torchvision.utils.save_image(
                image,
                os.path.join(render_path, filename),
            )

            torchvision.utils.save_image(
                gt,
                os.path.join(gt_path, filename),
            )

            torchvision.utils.save_image(
                depth,
                os.path.join(depth_path, filename),
            )

            if getattr(args, "load2gpu_on_the_fly", False):
                view.load2device("cpu")

    print("Rendering complete.")
    print("Results:", output_root)


if __name__ == "__main__":
    main()

