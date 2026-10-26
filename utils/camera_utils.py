#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# MonoEv-3DGS camera utilities:
# - supports fixed CameraInfo entries without RGB images;
# - uses CameraInfo.width/height when image=None;
# - creates a neutral placeholder only for Camera-class compatibility;
# - placeholder images must NOT be used as RGB supervision;
# - preserves original E-D3DGS / 3DGS behavior for normal RGB cameras.
#

import json
import numpy as np
import torch

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal


WARNED = False


def _compute_resolution(args, orig_w, orig_h, resolution_scale):
    global WARNED

    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), "
                        "rescaling to 1.6K.\n"
                        "If this is not desired, explicitly specify resolution."
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1.0
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError(f"Invalid camera resolution after scaling: {resolution}")

    return resolution


def _placeholder_image(resolution, white_background):
    """
    Placeholder image used only because scene.cameras.Camera expects an image tensor.

    This is NOT RGB ground truth and must never be used as RGB supervision
    in MonoEv Stage-3.
    """
    width, height = resolution
    value = 1.0 if white_background else 0.0

    return torch.full(
        (3, height, width),
        fill_value=value,
        dtype=torch.float32,
    )


def loadCam(args, id, cam_info, resolution_scale):
    """
    Convert CameraInfo into the runtime Camera object.

    Normal RGB datasets:
        cam_info.image != None
        -> resize and load RGB normally.

    MonoEv fixed-camera Stage-3:
        cam_info.image == None
        -> use cam_info.width / cam_info.height
        -> create placeholder tensor only for Camera compatibility.
    """
    has_gt_image = cam_info.image is not None

    if has_gt_image:
        orig_w, orig_h = cam_info.image.size
    else:
        orig_w = int(cam_info.width)
        orig_h = int(cam_info.height)

        if orig_w <= 0 or orig_h <= 0:
            raise ValueError(
                f"CameraInfo(uid={cam_info.uid}) has image=None but invalid "
                f"resolution {orig_w}x{orig_h}"
            )

    resolution = _compute_resolution(
        args,
        orig_w,
        orig_h,
        resolution_scale,
    )

    loaded_mask = None

    if has_gt_image:
        resized_image = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image[:3, ...]

        if resized_image.shape[0] == 4:
            loaded_mask = resized_image[3:4, ...]
    else:
        gt_image = _placeholder_image(
            resolution,
            bool(getattr(args, "white_background", False)),
        )

    camera = Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device if not args.load2gpu_on_the_fly else "cpu",
        fid=cam_info.fid,
        depth=cam_info.depth,
    )

    # MonoEv metadata.
    # camera.original_image may be only a placeholder when has_gt_image=False.
    camera.has_gt_image = has_gt_image
    camera.is_monoev_fixed_camera = not has_gt_image

    return camera


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for camera_id, cam_info in enumerate(cam_infos):
        camera_list.append(
            loadCam(
                args,
                camera_id,
                cam_info,
                resolution_scale,
            )
        )

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)

    pos = W2C[:3, 3]
    rot = W2C[:3, :3]

    serializable_array_2d = [x.tolist() for x in rot]

    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }

    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a Nerfies-style JSON camera into memory."""
    with open(path, "r", encoding="utf-8") as fp:
        camera_json = json.load(fp)

    if "tangential" in camera_json:
        camera_json["tangential_distortion"] = camera_json["tangential"]

    return dict(
        orientation=np.array(camera_json["orientation"]),
        position=np.array(camera_json["position"]),
        focal_length=camera_json["focal_length"] * scale,
        principal_point=np.array(camera_json["principal_point"]) * scale,
        skew=camera_json["skew"],
        pixel_aspect_ratio=camera_json["pixel_aspect_ratio"],
        radial_distortion=np.array(camera_json["radial_distortion"]),
        tangential_distortion=np.array(camera_json["tangential_distortion"]),
        image_size=np.array(
            (
                int(round(camera_json["image_size"][0] * scale)),
                int(round(camera_json["image_size"][1] * scale)),
            )
        ),
    )