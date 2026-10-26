
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

# MonoEv-3DGS Scene adaptation:
# - Stage-3 directly loads the canonical Gaussian G0;
# - uses exactly ONE fixed camera from fixed_camera.json;
# - no t0 RGB image is required;
# - no transforms.json is required for MonoEv Stage-3;
# - no temporal camera sequence is created;
# - event timestamps control DeformModel(t), not camera poses;
# - no DSD/static-dynamic split is required;
# - legacy COLMAP / Blender / other dataset behavior is preserved.

import json
import os
import shutil

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.deform_model import DeformModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.event_utils import load_event_stream, get_interpolator


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        dynamic_gaussians=None,
        static_load_path=None,
        dynamic_load_path=None,
        resolution_scales=[1.0],
    ):
        self.args = args
        self.model_path = args.model_path
        self.static_load_path = static_load_path
        self.dynamic_load_path = dynamic_load_path

        self.gaussians = gaussians
        self.dynamic_gaussians = dynamic_gaussians

        self.train_cameras = {}
        self.test_cameras = {}
        self.event_list = []

        # Fixed-camera MonoEv does not need camera-pose interpolation.
        self.rot_interpolator = None
        self.trans_interpolator = None
        self.tss_bd_start = None
        self.tss_bd_end = None

        os.makedirs(self.model_path, exist_ok=True)

        # ============================================================
        # Detect MonoEv Stage-3
        # ============================================================
        initial_gaussian_ply = getattr(args, "initial_gaussian_ply", None)

        mono_stage3 = (
            initial_gaussian_ply is not None
            and str(initial_gaussian_ply).strip().lower() not in ("", "none", "null")
        )

        self.mono_stage3 = mono_stage3

        # ============================================================
        # Resolve G0 path
        # ============================================================
        resolved_initial_gaussian_ply = None

        if mono_stage3:
            resolved_initial_gaussian_ply = os.path.expanduser(
                str(initial_gaussian_ply)
            )

            if not os.path.isabs(resolved_initial_gaussian_ply):
                resolved_initial_gaussian_ply = os.path.join(
                    args.source_path,
                    resolved_initial_gaussian_ply,
                )

            resolved_initial_gaussian_ply = os.path.abspath(
                resolved_initial_gaussian_ply
            )

            if not os.path.isfile(resolved_initial_gaussian_ply):
                raise FileNotFoundError(
                    f"MonoEv canonical G0 does not exist: "
                    f"{resolved_initial_gaussian_ply}"
                )

        # ============================================================
        # Resolve camera_file
        #
        # MonoEv default:
        #   source_path/fixed_camera.json
        # ============================================================
        camera_file = getattr(args, "camera_file", None)

        if camera_file is not None and str(camera_file).strip().lower() not in (
            "",
            "none",
            "null",
        ):
            camera_file = os.path.expanduser(str(camera_file))

            if not os.path.isabs(camera_file):
                camera_file = os.path.join(
                    args.source_path,
                    camera_file,
                )

            camera_file = os.path.abspath(camera_file)

        elif mono_stage3:
            camera_file = os.path.abspath(
                os.path.join(
                    args.source_path,
                    "fixed_camera.json",
                )
            )

        else:
            camera_file = None

        # ============================================================
        # Load scene information
        # ============================================================

        # ------------------------------------------------------------
        # MonoEv Stage-3 has highest priority.
        #
        # Even if source_path contains transforms.json or sparse/,
        # Stage-3 must use fixed_camera.json.
        # ------------------------------------------------------------
        if mono_stage3:
            if camera_file is None or not os.path.isfile(camera_file):
                raise FileNotFoundError(
                    "MonoEv Stage-3 requires fixed_camera.json.\n"
                    f"Expected camera file: {camera_file}"
                )

            print("[MonoEv] Stage-3 fixed-camera mode")
            print(f"[MonoEv] G0: {resolved_initial_gaussian_ply}")
            print(f"[MonoEv] camera: {camera_file}")

            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval,
                args,
            )

        # ------------------------------------------------------------
        # Standard COLMAP
        # ------------------------------------------------------------
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.eval,
            )

        # ------------------------------------------------------------
        # Explicit Blender camera/transforms file
        # ------------------------------------------------------------
        elif camera_file is not None:
            if not os.path.isfile(camera_file):
                raise FileNotFoundError(
                    f"camera_file does not exist: {camera_file}"
                )

            print(f"Found camera file from config: {camera_file}")

            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval,
                args,
            )

        # ------------------------------------------------------------
        # Standard Blender
        # ------------------------------------------------------------
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms.json file, assuming Blender data set!")

            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval,
                args,
            )

        # ------------------------------------------------------------
        # DTU
        # ------------------------------------------------------------
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            if "DTU" not in sceneLoadTypeCallbacks:
                raise RuntimeError("DTU dataset callback is not registered")

            print("Found cameras_sphere.npz file, assuming DTU data set!")

            scene_info = sceneLoadTypeCallbacks["DTU"](
                args.source_path,
                "cameras_sphere.npz",
                "cameras_sphere.npz",
            )

        # ------------------------------------------------------------
        # Nerfies
        # ------------------------------------------------------------
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            if "nerfies" not in sceneLoadTypeCallbacks:
                raise RuntimeError("Nerfies dataset callback is not registered")

            print("Found dataset.json file, assuming Nerfies data set!")

            scene_info = sceneLoadTypeCallbacks["nerfies"](
                args.source_path,
                args.eval,
            )

        # ------------------------------------------------------------
        # PlenopticVideo / Neu3D
        # ------------------------------------------------------------
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            if "plenopticVideo" not in sceneLoadTypeCallbacks:
                raise RuntimeError(
                    "PlenopticVideo dataset callback is not registered"
                )

            print("Found poses_bounds.npy file, assuming Neu3D/PlenopticVideo data set!")

            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](
                args.source_path,
                args.eval,
                24,
            )

        else:
            raise AssertionError(
                "Could not recognize scene type!"
            )

        self.scene_info = scene_info

        # ============================================================
        # Validate MonoEv fixed-camera requirement
        # ============================================================
        if mono_stage3:
            if len(scene_info.train_cameras) != 1:
                raise RuntimeError(
                    "MonoEv Stage-3 must contain exactly ONE fixed camera, "
                    f"but dataset_readers returned {len(scene_info.train_cameras)}."
                )

            if len(scene_info.test_cameras) != 0:
                print(
                    "[MonoEv WARNING] test cameras were returned in Stage-3. "
                    "They will be ignored."
                )
                scene_info.test_cameras.clear()

        # ============================================================
        # Save input.ply and cameras.json
        # ============================================================
        if self.static_load_path is None:

            # --------------------------------------------------------
            # MonoEv:
            # copy the actual canonical Gaussian G0 as input.ply
            # instead of copying the one-point bootstrap cloud.
            # --------------------------------------------------------
            if mono_stage3:
                dst_input_ply = os.path.join(
                    self.model_path,
                    "input.ply",
                )

                shutil.copyfile(
                    resolved_initial_gaussian_ply,
                    dst_input_ply,
                )

                print(
                    f"[MonoEv] copied canonical G0 to: "
                    f"{dst_input_ply}"
                )

            # --------------------------------------------------------
            # Legacy behavior
            # --------------------------------------------------------
            elif scene_info.ply_path and os.path.isfile(scene_info.ply_path):
                with open(scene_info.ply_path, "rb") as src_file, open(
                    os.path.join(self.model_path, "input.ply"),
                    "wb",
                ) as dest_file:
                    dest_file.write(src_file.read())

            # --------------------------------------------------------
            # Save camera metadata
            # --------------------------------------------------------
            json_cams = []
            camlist = []

            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)

            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)

            for camera_id, cam in enumerate(camlist):
                json_cams.append(
                    camera_to_JSON(
                        camera_id,
                        cam,
                    )
                )

            with open(
                os.path.join(self.model_path, "cameras.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    json_cams,
                    file,
                    indent=2,
                )

        # ============================================================
        # Camera normalization
        # ============================================================
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_center = -scene_info.nerf_normalization["translate"]

        # ============================================================
        # Build runtime Camera objects
        # ============================================================
        for resolution_scale in resolution_scales:

            print("Loading Training Cameras")

            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras,
                resolution_scale,
                args,
            )

            print("Loading Test Cameras")

            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras,
                resolution_scale,
                args,
            )

            if mono_stage3:
                runtime_cameras = self.train_cameras[resolution_scale]

                if len(runtime_cameras) != 1:
                    raise RuntimeError(
                        "MonoEv runtime camera list must contain exactly "
                        f"one camera, got {len(runtime_cameras)}."
                    )

                fixed_camera = runtime_cameras[0]

                print(
                    "[MonoEv] Runtime fixed camera loaded: "
                    f"{fixed_camera.image_width}x{fixed_camera.image_height}"
                )

                print(
                    "[MonoEv] FoV: "
                    f"FovX={fixed_camera.FoVx:.6f}, "
                    f"FovY={fixed_camera.FoVy:.6f}"
                )

                print(
                    "[MonoEv] RGB GT available: "
                    f"{getattr(fixed_camera, 'has_gt_image', False)}"
                )

        # ============================================================
        # Load events
        # ============================================================
        if bool(getattr(args, "event_assist", False)):
            event_scale = resolution_scales[0]

            if event_scale not in self.train_cameras:
                raise KeyError(
                    f"Resolution scale {event_scale} not found in "
                    "training cameras."
                )

            event_cameras = self.train_cameras[event_scale]

            if mono_stage3 and len(event_cameras) != 1:
                raise RuntimeError(
                    "MonoEv event loading requires exactly one "
                    "fixed camera."
                )

            self.event_list = load_event_stream(
                event_cameras,
                args,
            )

            # --------------------------------------------------------
            # MonoEv:
            # camera pose never changes, therefore camera-pose
            # interpolation is meaningless and intentionally disabled.
            #
            # Event timestamps will later be passed directly into
            # DeformModel(t).
            # --------------------------------------------------------
            if mono_stage3:
                self.rot_interpolator = None
                self.trans_interpolator = None
                self.tss_bd_start = None
                self.tss_bd_end = None

                print(
                    "[MonoEv] Fixed camera: camera-pose interpolation disabled."
                )

            # --------------------------------------------------------
            # Legacy dynamic-camera behavior
            # --------------------------------------------------------
            else:
                (
                    self.rot_interpolator,
                    self.trans_interpolator,
                    self.tss_bd_start,
                    self.tss_bd_end,
                ) = get_interpolator(args)

        # ============================================================
        # Gaussian initialization
        # ============================================================

        # ------------------------------------------------------------
        # MonoEv Stage-3:
        # load canonical Gaussian G0 directly.
        # ------------------------------------------------------------
        if mono_stage3:
            print(
                f"[MonoEv] Loading canonical Gaussian G0: "
                f"{resolved_initial_gaussian_ply}"
            )

            self.gaussians.load_ply(
                resolved_initial_gaussian_ply
            )

            print(
                f"[MonoEv] Canonical G0 loaded: "
                f"{self.gaussians.get_xyz.shape[0]} Gaussians"
            )

        # ------------------------------------------------------------
        # Load saved static/dynamic Gaussians
        # ------------------------------------------------------------
        elif self.static_load_path is not None:
            og_number_points = None

            if scene_info.point_cloud is not None:
                og_number_points = len(
                    scene_info.point_cloud.points
                )

            if og_number_points is None:
                self.gaussians.load_ply(
                    self.static_load_path
                )
            else:
                self.gaussians.load_ply(
                    self.static_load_path,
                    og_number_points=og_number_points,
                )

            if (
                self.dynamic_load_path is not None
                and self.dynamic_gaussians is not None
            ):
                if og_number_points is None:
                    self.dynamic_gaussians.load_ply(
                        self.dynamic_load_path
                    )
                else:
                    self.dynamic_gaussians.load_ply(
                        self.dynamic_load_path,
                        og_number_points=og_number_points,
                    )

        # ------------------------------------------------------------
        # Standard point-cloud initialization
        # ------------------------------------------------------------
        else:
            if scene_info.point_cloud is None:
                raise RuntimeError(
                    "Scene point cloud is None and no "
                    "initial_gaussian_ply/static_load_path was provided."
                )

            self.gaussians.create_from_pcd(
                scene_info.point_cloud,
                self.cameras_extent,
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path,
            "point_cloud",
            f"iteration_{iteration}",
        )

        os.makedirs(
            point_cloud_path,
            exist_ok=True,
        )

        self.gaussians.save_ply(
            os.path.join(
                point_cloud_path,
                "point_cloud.ply",
            )
        )

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getEvents(self):
        return self.event_list