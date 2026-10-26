from pathlib import Path
import bpy
import json
import math


# ============================================================
# 当前 Python 文件所在目录
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_JSON = SCRIPT_DIR / "camera_intrinsics.json"


# ============================================================
# Sensor fit
# ============================================================

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == "AUTO":
        return "HORIZONTAL" if size_x >= size_y else "VERTICAL"

    return sensor_fit


# ============================================================
# 提取相机内参
# ============================================================

def get_camera_intrinsics(cam_obj, scene):

    cam = cam_obj.data

    # --------------------------------------------------------
    # 实际渲染分辨率
    # --------------------------------------------------------

    scale = scene.render.resolution_percentage / 100.0

    width = scene.render.resolution_x * scale
    height = scene.render.resolution_y * scale

    pixel_aspect_x = scene.render.pixel_aspect_x
    pixel_aspect_y = scene.render.pixel_aspect_y

    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x

    # --------------------------------------------------------
    # Blender Camera 参数
    # --------------------------------------------------------

    focal_mm = cam.lens

    sensor_fit = get_sensor_fit(
        cam.sensor_fit,
        width * pixel_aspect_x,
        height * pixel_aspect_y
    )

    if sensor_fit == "HORIZONTAL":

        sensor_size_mm = cam.sensor_width
        view_fac_px = width

    else:

        sensor_size_mm = cam.sensor_height
        view_fac_px = pixel_aspect_ratio * height

    # --------------------------------------------------------
    # focal length: mm -> pixel
    # --------------------------------------------------------

    pixel_size_mm_per_px = (
        sensor_size_mm /
        focal_mm /
        view_fac_px
    )

    fx = 1.0 / pixel_size_mm_per_px

    fy = fx / pixel_aspect_ratio

    # --------------------------------------------------------
    # 主点 cx cy
    # --------------------------------------------------------

    cx = width / 2.0 - cam.shift_x * view_fac_px

    cy = (
        height / 2.0
        + cam.shift_y
        * view_fac_px
        / pixel_aspect_ratio
    )

    # --------------------------------------------------------
    # FOV
    # --------------------------------------------------------

    fov_x = 2.0 * math.atan(
        width / (2.0 * fx)
    )

    fov_y = 2.0 * math.atan(
        height / (2.0 * fy)
    )

    return {

        "width": int(round(width)),
        "height": int(round(height)),

        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),

        "fov_x_deg": float(math.degrees(fov_x)),
        "fov_y_deg": float(math.degrees(fov_y)),

        "lens_mm": float(cam.lens),

        "sensor_width_mm": float(cam.sensor_width),
        "sensor_height_mm": float(cam.sensor_height),

        "sensor_fit": str(cam.sensor_fit),

        "shift_x": float(cam.shift_x),
        "shift_y": float(cam.shift_y),

        "pixel_aspect_x": float(pixel_aspect_x),
        "pixel_aspect_y": float(pixel_aspect_y)
    }


# ============================================================
# 提取 Camera-to-World
# ============================================================

def get_camera_pose(cam_obj):

    matrix = cam_obj.matrix_world

    location = matrix.to_translation()
    rotation = matrix.to_quaternion()

    # 4x4 Camera-to-World matrix
    c2w = [
        [float(matrix[i][j]) for j in range(4)]
        for i in range(4)
    ]

    return {

        "position": {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z)
        },

        "quaternion": {
            "w": float(rotation.w),
            "x": float(rotation.x),
            "y": float(rotation.y),
            "z": float(rotation.z)
        },

        "camera_to_world": c2w
    }


# ============================================================
# 找相机
# ============================================================

def find_camera(scene):

    # 优先读取当前 Scene 正在使用的相机
    if scene.camera is not None:
        return scene.camera

    cameras = [
        obj
        for obj in scene.objects
        if obj.type == "CAMERA"
    ]

    if len(cameras) == 0:
        return None

    return cameras[0]


# ============================================================
# 搜索当前文件夹全部 Blender 文件
# ============================================================

def find_blender_files():

    files = []

    files.extend(
        SCRIPT_DIR.glob("*.blend")
    )

    files.extend(
        SCRIPT_DIR.glob("*.blender")
    )

    return sorted(
        set(files),
        key=lambda p: p.name.lower()
    )


# ============================================================
# 处理一个 Blender
# ============================================================

def process_blender_file(blend_file):

    print()
    print("=" * 80)
    print(f"Processing: {blend_file.name}")
    print("=" * 80)

    # 打开 Blender 工程
    bpy.ops.wm.open_mainfile(
        filepath=str(blend_file)
    )

    scene = bpy.context.scene

    cam_obj = find_camera(scene)

    if cam_obj is None:

        print(
            f"[ERROR] {blend_file.name}: "
            f"没有找到 Camera"
        )

        return None

    intrinsics = get_camera_intrinsics(
        cam_obj,
        scene
    )

    pose = get_camera_pose(
        cam_obj
    )

    # --------------------------------------------------------
    # K 矩阵
    # --------------------------------------------------------

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]

    result = {

        "camera_name": cam_obj.name,

        "resolution": {
            "width": intrinsics["width"],
            "height": intrinsics["height"]
        },

        "intrinsics": {

            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,

            "K": K
        },

        "fov": {

            "fov_x_deg":
                intrinsics["fov_x_deg"],

            "fov_y_deg":
                intrinsics["fov_y_deg"]
        },

        "blender_camera": {

            "lens_mm":
                intrinsics["lens_mm"],

            "sensor_width_mm":
                intrinsics["sensor_width_mm"],

            "sensor_height_mm":
                intrinsics["sensor_height_mm"],

            "sensor_fit":
                intrinsics["sensor_fit"],

            "shift_x":
                intrinsics["shift_x"],

            "shift_y":
                intrinsics["shift_y"],

            "pixel_aspect_x":
                intrinsics["pixel_aspect_x"],

            "pixel_aspect_y":
                intrinsics["pixel_aspect_y"]
        },

        "extrinsics": pose
    }

    # --------------------------------------------------------
    # 输出到终端
    # --------------------------------------------------------

    print(
        f"Camera: {cam_obj.name}"
    )

    print(
        f"Resolution: "
        f"{intrinsics['width']} x "
        f"{intrinsics['height']}"
    )

    print()
    print("[Intrinsics]")

    print(
        f"fx = {fx:.10f}"
    )

    print(
        f"fy = {fy:.10f}"
    )

    print(
        f"cx = {cx:.10f}"
    )

    print(
        f"cy = {cy:.10f}"
    )

    print()
    print("[FOV]")

    print(
        f"FOVx = "
        f"{intrinsics['fov_x_deg']:.10f} deg"
    )

    print(
        f"FOVy = "
        f"{intrinsics['fov_y_deg']:.10f} deg"
    )

    print()
    print("[Camera position]")

    print(
        f"x = {pose['position']['x']:.10f}"
    )

    print(
        f"y = {pose['position']['y']:.10f}"
    )

    print(
        f"z = {pose['position']['z']:.10f}"
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("Blender Camera Parameter Extractor")
    print("=" * 80)

    print()
    print("Script directory:")
    print(SCRIPT_DIR)

    blender_files = find_blender_files()

    print()
    print(
        f"Found {len(blender_files)} Blender files."
    )

    if len(blender_files) == 0:

        print()
        print(
            "没有找到 .blend / .blender 文件。"
        )

        print(
            "请确保 .py 和 Blender 文件"
            "位于同一个文件夹。"
        )

        return

    print()

    for file in blender_files:
        print(f"  {file.name}")

    # ========================================================
    # JSON 最终结构
    # ========================================================

    all_results = {}

    # --------------------------------------------------------
    # 逐个处理
    # --------------------------------------------------------

    for blend_file in blender_files:

        try:

            result = process_blender_file(
                blend_file
            )

            if result is not None:

                all_results[
                    blend_file.name
                ] = result

        except Exception as e:

            print()
            print(
                f"[ERROR] "
                f"{blend_file.name}"
            )

            print(str(e))

    # --------------------------------------------------------
    # 保存 JSON
    # --------------------------------------------------------

    if len(all_results) == 0:

        print()
        print(
            "没有成功提取任何相机参数。"
        )

        return

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)

    print()
    print(
        f"成功处理："
        f"{len(all_results)} 个文件"
    )

    print()
    print("JSON 保存到：")

    print(
        OUTPUT_JSON
    )


if __name__ == "__main__":
    main()