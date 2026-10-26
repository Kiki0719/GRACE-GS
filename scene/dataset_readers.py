#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# MonoEv-3DGS adaptation:
# - keeps COLMAP and ordinary Blender behavior compatible;
# - MonoEv stage-3 loads one fixed camera from fixed_camera.json;
# - MonoEv stage-3 does not require t0 RGB or temporal camera records;
# - keeps a non-zero normalization radius for a fixed camera;
# - creates only a one-point bootstrap cloud because stage-3 later loads improved G0.
#
import os
import sys
import json
from pathlib import Path
from typing import NamedTuple, Optional
import cv2 as cv
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image: Optional[object]
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    K = K / K[2, 2]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return K, pose

def getNerfppNorm(cam_info):
    if len(cam_info) == 0:
        raise RuntimeError("Cannot normalize an empty camera list")
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    cam_centers = np.hstack(cam_centers)
    center = np.mean(cam_centers, axis=1, keepdims=True)
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = float(np.max(dist))
    translate = -center.flatten()
    radius = diagonal * 1.1
    if radius < 1e-6:
        radius = 1.0
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            raise AssertionError(
                "Colmap camera model not handled: only PINHOLE or SIMPLE_PINHOLE are supported"
            )
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        fid = int(image_name) / max(num_frames - 1, 1)
        cam_infos.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                fid=fid,
            )
        )
    sys.stdout.write("\n")
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.zeros((positions.shape[0], 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3D data to .ply; this happens only once.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

def _frame_uid(frame, fallback_idx):
    raw = str(frame.get("file_path", fallback_idx)).replace("\\\\", "/").split("/")[-1]
    try:
        return int(Path(raw).stem)
    except ValueError:
        return int(fallback_idx)

def _frame_timestamp_ns(frame, idx, total_frames):
    if "time" in frame:
        return float(frame["time"]) * 1e9
    if total_frames <= 1:
        return 0.0
    return float(idx) / float(total_frames - 1) * 1e9

def _pose_from_blender_frame(frame):
    matrix = np.linalg.inv(np.asarray(frame["transform_matrix"], dtype=np.float64))
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    return R, T

def _load_blender_rgb(image_path, white_background):
    if not image_path.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {image_path}")
    image = Image.open(image_path)
    im_data = np.array(image.convert("RGBA"), dtype=np.float32) / 255.0
    bg = np.array([1.0, 1.0, 1.0], dtype=np.float32) if white_background else np.array(
        [0.0, 0.0, 0.0], dtype=np.float32
    )
    arr = im_data[:, :, :3] * im_data[:, :, 3:4] + bg[None, None, :] * (
        1.0 - im_data[:, :, 3:4]
    )
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "RGB")

def readCamerasFromTransforms(
    path,
    white_background,
    select,
    extension=".png",
    mono_fixed_camera=False,
    t0_index=0,
    t0_image_path=None,
):
    cam_infos = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as json_file:
        contents = json.load(json_file)
    fovx = float(contents["camera_angle_x"])
    frames = contents["frames"]
    if not frames:
        raise RuntimeError(f"No frames found in {path}")
    if not 0 <= int(t0_index) < len(frames):
        raise ValueError(f"t0_index={t0_index} outside [0, {len(frames) - 1}]")
    fixed_R = fixed_T = None
    t0_image = None
    t0_size = None
    if mono_fixed_camera:
        reference_frame = frames[int(t0_index)]
        fixed_R, fixed_T = _pose_from_blender_frame(reference_frame)
        if t0_image_path is None or str(t0_image_path).strip().lower() in ("", "none", "null"):
            raise ValueError("MonoEv stage-3 requires t0_image_path in the config")
        t0_path = Path(os.path.expanduser(str(t0_image_path)))
        if not t0_path.is_absolute():
            t0_path = path.parent / t0_path
        t0_path = t0_path.resolve()
        t0_image = _load_blender_rgb(t0_path, white_background)
        t0_size = t0_image.size
        print(
            f"[MonoEv] fixed camera from frame {t0_index}; "
            f"only t0 image is loaded: {t0_path}"
        )
    for idx in select:
        idx = int(idx)
        if not 0 <= idx < len(frames):
            raise IndexError(f"Selected frame {idx} outside [0, {len(frames) - 1}]")
        frame = frames[idx]
        frame_idx = _frame_uid(frame, idx)
        frame_time = _frame_timestamp_ns(frame, idx, len(frames))
        if mono_fixed_camera:
            R = fixed_R.copy()
            T = fixed_T.copy()
        else:
            R, T = _pose_from_blender_frame(frame)
        image_path = (path.parent / frame["file_path"]).with_suffix(extension)
        image_name = image_path.name
        if mono_fixed_camera:
            if idx == int(t0_index):
                image = t0_image.copy()
                image_path = t0_path
                image_name = t0_path.name
                width, height = image.size
            else:
                image = None
                image_path = ""
                image_name = f"temporal_{idx:06d}"
                width, height = t0_size
        else:
            image = _load_blender_rgb(image_path, white_background)
            width, height = image.size
        fovy = focal2fov(fov2focal(fovx, width), height)
        cam_infos.append(
            CameraInfo(
                uid=frame_idx,
                R=R,
                T=T,
                FovY=fovy,
                FovX=fovx,
                image=image,
                image_path=str(image_path),
                image_name=image_name,
                width=width,
                height=height,
                fid=frame_time,
            )
        )
    return cam_infos


def readMonoEvFixedCamera(camera_file):
    """
    Load exactly one fixed camera for MonoEv stage-3 from fixed_camera.json.

    Expected fields written by camera_alignment.py:
        width, height
        fx, fy, cx, cy
        fov_x_deg, fov_y_deg
        R, T

    Only R, T, image size and FoV are required by the current E-D3DGS Camera.
    The original cx/cy values are preserved only for validation because the
    standard 3DGS projection used by this repo assumes a centered principal point.
    """
    camera_file = Path(camera_file).expanduser().resolve()
    if not camera_file.is_file():
        raise FileNotFoundError(
            f"MonoEv fixed camera file does not exist: {camera_file}"
        )

    with camera_file.open("r", encoding="utf-8") as f:
        camera = json.load(f)

    width = camera.get("width", camera.get("image_width", None))
    height = camera.get("height", camera.get("image_height", None))
    if width is None or height is None:
        raise KeyError(
            "fixed_camera.json must contain width/height "
            "or image_width/image_height"
        )
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid camera resolution: width={width}, height={height}"
        )

    if "R" not in camera or "T" not in camera:
        raise KeyError("fixed_camera.json must contain R and T")

    R = np.asarray(camera["R"], dtype=np.float32)
    T = np.asarray(camera["T"], dtype=np.float32).reshape(-1)
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3, 3), got {R.shape}")
    if T.shape != (3,):
        raise ValueError(f"Expected T shape (3,), got {T.shape}")
    if not np.isfinite(R).all() or not np.isfinite(T).all():
        raise ValueError("R/T in fixed_camera.json contain NaN or Inf")

    if "fov_x_deg" in camera:
        FovX = float(np.deg2rad(float(camera["fov_x_deg"])))
    elif "fx" in camera:
        fx = float(camera["fx"])
        if fx <= 0:
            raise ValueError(f"fx must be positive, got {fx}")
        FovX = float(focal2fov(fx, width))
    else:
        raise KeyError("fixed_camera.json must contain fov_x_deg or fx")

    if "fov_y_deg" in camera:
        FovY = float(np.deg2rad(float(camera["fov_y_deg"])))
    elif "fy" in camera:
        fy = float(camera["fy"])
        if fy <= 0:
            raise ValueError(f"fy must be positive, got {fy}")
        FovY = float(focal2fov(fy, height))
    elif "fx" in camera:
        fx = float(camera["fx"])
        if fx <= 0:
            raise ValueError(f"fx must be positive, got {fx}")
        FovY = float(focal2fov(fx, height))
    else:
        raise KeyError("fixed_camera.json must contain fov_y_deg, fy, or fx")

    cx = float(camera.get("cx", width / 2.0))
    cy = float(camera.get("cy", height / 2.0))
    expected_cx = width / 2.0
    expected_cy = height / 2.0
    if abs(cx - expected_cx) > 1.0 or abs(cy - expected_cy) > 1.0:
        print(
            "[MonoEv WARNING] fixed_camera.json principal point is not centered: "
            f"cx={cx:.3f}, cy={cy:.3f}; current E-D3DGS projection assumes "
            f"cx={expected_cx:.3f}, cy={expected_cy:.3f}."
        )

    W2C = getWorld2View2(R, T)
    if not np.isfinite(W2C).all():
        raise ValueError("Camera world-to-view matrix contains NaN or Inf")
    C2W = np.linalg.inv(W2C)
    camera_center = C2W[:3, 3]

    print("[MonoEv] loaded ONE fixed camera from fixed_camera.json")
    print(f"[MonoEv] camera file: {camera_file}")
    print(f"[MonoEv] resolution: {width}x{height}")
    if "fx" in camera:
        print(f"[MonoEv] fx={float(camera['fx']):.6f}")
    if "fy" in camera:
        print(f"[MonoEv] fy={float(camera['fy']):.6f}")
    print(f"[MonoEv] cx={cx:.6f}, cy={cy:.6f}")
    print(
        f"[MonoEv] FovX={np.rad2deg(FovX):.6f} deg, "
        f"FovY={np.rad2deg(FovY):.6f} deg"
    )
    print(f"[MonoEv] camera center: {camera_center.tolist()}")
    print("[MonoEv] no t0 RGB image is loaded")

    return [
        CameraInfo(
            uid=0,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=None,
            image_path="",
            image_name="fixed_camera",
            width=width,
            height=height,
            fid=0.0,
            depth=None,
        )
    ]

def _is_mono_stage3(args):
    initial_gaussian_ply = getattr(args, "initial_gaussian_ply", None)
    if initial_gaussian_ply is None:
        return False
    return str(initial_gaussian_ply).strip().lower() not in ("", "none", "null")

def _make_monoev_bootstrap_cloud(model_path):
    os.makedirs(model_path, exist_ok=True)
    xyz = np.zeros((1, 3), dtype=np.float32)
    rgb = np.full((1, 3), 127, dtype=np.uint8)
    pcd = BasicPointCloud(
        points=xyz,
        colors=rgb.astype(np.float32) / 255.0,
        normals=np.zeros((1, 3), dtype=np.float32),
    )
    ply_path = os.path.join(model_path, "monoev_bootstrap_pcd.ply")
    storePly(ply_path, xyz, rgb)
    return pcd, ply_path

def readNerfSyntheticInfo(path, white_background, eval, args, extension=".png"):
    mono_stage3 = _is_mono_stage3(args)

    if mono_stage3:
        print("[MonoEv] Stage-3 fixed-camera mode")

        camera_file = getattr(args, "camera_file", None)
        if camera_file is None or str(camera_file).strip().lower() in ("", "none", "null"):
            camera_file = os.path.join(path, "fixed_camera.json")
        else:
            camera_file = os.path.expanduser(str(camera_file))
            if not os.path.isabs(camera_file):
                camera_file = os.path.join(path, camera_file)
        camera_file = os.path.abspath(camera_file)

        train_cam_infos = readMonoEvFixedCamera(camera_file)
        test_cam_infos = []
        nerf_normalization = getNerfppNorm(train_cam_infos)

        pcd, ply_path = _make_monoev_bootstrap_cloud(args.model_path)
        print(
            "[MonoEv] using one-point bootstrap cloud; "
            "train_monoev_stage3.py will replace it with initial_gaussian_ply"
        )
        print(f"[MonoEv] camera source: {camera_file}")
        print("[MonoEv] number of train cameras: 1")

        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=nerf_normalization,
            ply_path=ply_path,
        )

    camera_file = getattr(args, "camera_file", None)
    if camera_file is None or str(camera_file).strip().lower() in ("", "none", "null"):
        camera_file = os.path.join(path, "transforms.json")
    else:
        camera_file = os.path.expanduser(str(camera_file))
        if not os.path.isabs(camera_file):
            camera_file = os.path.join(path, camera_file)

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        camera_file,
        white_background,
        args.train_select,
        extension,
        mono_fixed_camera=False,
    )

    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        camera_file,
        white_background,
        args.test_select,
        extension,
        mono_fixed_camera=False,
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if args.random_points:
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
        )
        ply_path = os.path.join(args.model_path, "input_pcd.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd_path_value = getattr(args, "pcd_path", None)
        if pcd_path_value is None or str(pcd_path_value).strip().lower() in ("", "none", "null"):
            raise ValueError(
                "pcd_path must be provided when random_points=False outside MonoEv stage-3"
            )
        pcd_path = Path(pcd_path_value)
        if pcd_path.suffix == ".bin":
            xyz, rgb, _ = read_points3D_binary(pcd_path)
            ply_path = os.path.join(args.model_path, "input_pcd.ply")
            storePly(ply_path, xyz, rgb)
        else:
            ply_path = str(pcd_path)
        try:
            pcd = fetchPly(ply_path)
        except Exception:
            pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}
