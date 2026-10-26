"""Read-only rendering adapter for the E-D3DGS repository. No training writes."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'Stage1') not in sys.path:
    sys.path.insert(0, str(ROOT / 'Stage1'))


def load_model(ply, camera_json, config, workspace):
    from scene import Scene, GaussianModel
    from utils import parse_utils
    from gaussian_utils.refinement_utils import make_stage1_cfg
    for p in (ply, camera_json, config):
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU CUDA required for 3D Gaussian Splatting.')
    cfg, _ = parse_utils.parse_ini(str(config))
    args = make_stage1_cfg(cfg, Path(workspace), Path(ply), 1, camera_file=Path(camera_json))
    gs = GaussianModel(args.sh_degree)
    scene = Scene(args, gs, static_load_path=str(ply))
    cams = list(scene.getTrainCameras())
    if len(cams) != 1:
        raise RuntimeError(f'Expected one fixed camera, got {len(cams)}')
    cam = cams[0]
    if args.load2gpu_on_the_fly:
        cam.load2device()
    if cam.world_view_transform.device != gs.get_xyz.device:
        cam.load2device()
    xyz = gs.get_xyz.detach().cpu().numpy()
    return gs, cam, xyz




def make_orbit_camera(
    cam, xyz,
    yaw_deg=0.0,
    pitch_deg=0.0,
    distance_scale=1.0,
    orbit_center=None
):
    original_w2c = (
        cam.world_view_transform.detach()
        .cpu().numpy().T.astype(np.float64)
    )
    c2w = np.linalg.inv(original_w2c)

    eye = c2w[:3, 3].copy()
    R0 = c2w[:3, :3].copy()

    center = (
        np.asarray(orbit_center, dtype=np.float64)
        if orbit_center is not None
        else np.median(xyz, axis=0)
    )

    # Use the original camera vertical axis as the orbit axis.
    # This keeps Front / Left / Right / Back fully consistent.
    axis = R0[:, 1].copy()
    axis /= max(np.linalg.norm(axis), 1e-12)

    yaw = float(yaw_deg)
    pitch = float(pitch_deg)

    # 1) Yaw: rotate the original front camera around the object center.
    yaw_rotation = Rotation.from_rotvec(
        np.deg2rad(yaw) * axis
    ).as_matrix()

    new_eye = center + float(distance_scale) * (
        yaw_rotation @ (eye - center)
    )
    new_R = yaw_rotation @ R0

    # 2) Optional pitch: rotate around the CURRENT camera right axis.
    if abs(pitch) > 1e-8:
        pitch_axis = new_R[:, 0].copy()
        pitch_axis /= max(np.linalg.norm(pitch_axis), 1e-12)

        pitch_rotation = Rotation.from_rotvec(
            np.deg2rad(pitch) * pitch_axis
        ).as_matrix()

        new_eye = center + pitch_rotation @ (new_eye - center)
        new_R = pitch_rotation @ new_R

    new_c2w = np.eye(4, dtype=np.float64)
    new_c2w[:3, :3] = new_R
    new_c2w[:3, 3] = new_eye

    new_w2c = np.linalg.inv(new_c2w)

    matrix = torch.as_tensor(
        new_w2c.T.copy(),
        device=cam.world_view_transform.device,
        dtype=cam.world_view_transform.dtype
    )

    projection = cam.projection_matrix.to(
        device=matrix.device,
        dtype=matrix.dtype
    )

    return SimpleNamespace(
        FoVx=cam.FoVx,
        FoVy=cam.FoVy,
        image_width=cam.image_width,
        image_height=cam.image_height,
        world_view_transform=matrix,
        full_proj_transform=matrix @ projection,
        camera_center=torch.linalg.inv(matrix)[3, :3],
    )

def splat_image(gs, cam, xyz, yaw=0, pitch=0, zoom=1.0, background='Black', orbit_center=None):
    from gaussian_renderer import render
    view = make_orbit_camera(cam, xyz, yaw, pitch, zoom, orbit_center=orbit_center)
    bg = torch.ones(3, device=gs.get_xyz.device) if background == 'White' else torch.zeros(3, device=gs.get_xyz.device)
    pipe = SimpleNamespace(debug=False)
    with torch.inference_mode():
        # The customized renderer requires additive deformation tensors.
        package = render(view, gs, pipe, bg,
                         torch.zeros_like(gs.get_xyz),
                         torch.zeros_like(gs.get_rotation),
                         torch.zeros_like(gs.get_scaling), False)
    rgb = package['render'].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.uint8(np.round(rgb * 255))


def splat_diagnostic_view(gs, cam, record, center, background="Black"):
    """
    Render using the EXACT diagnostic camera pose convention
    defined by Stage1/camera_alignment.py.
    """
    from gaussian_renderer import render
    from camera_alignment import look_at_to_rt

    params = record["params"]
    az, el, roll, distance, offx, offy = map(float, params[:6])

    R, T, _ = look_at_to_rt(
        np.asarray(center, dtype=np.float64),
        distance,
        az,
        el,
        roll,
        offx,
        offy,
    )

    from utils.graphics_utils import getWorld2View2

    w2c = getWorld2View2(R, T)

    device = gs.get_xyz.device
    dtype = gs.get_xyz.dtype

    view_matrix = torch.as_tensor(
        w2c.T.copy(),
        device=device,
        dtype=dtype,
    )

    projection = cam.projection_matrix.to(
        device=device,
        dtype=dtype,
    )

    virtual_camera = SimpleNamespace(
        FoVx=cam.FoVx,
        FoVy=cam.FoVy,
        image_width=cam.image_width,
        image_height=cam.image_height,
        world_view_transform=view_matrix,
        full_proj_transform=view_matrix @ projection,
        camera_center=torch.linalg.inv(view_matrix)[3, :3],
    )

    bg = (
        torch.ones(3, device=device, dtype=dtype)
        if background == "White"
        else torch.zeros(3, device=device, dtype=dtype)
    )

    pipe = SimpleNamespace(debug=False)

    with torch.no_grad():
        result = render(
            virtual_camera,
            gs,
            pipe,
            bg,
            torch.zeros_like(gs.get_xyz),
            torch.zeros_like(gs.get_rotation),
            torch.zeros_like(gs.get_scaling),
            False,
        )

    rgb = (
        result["render"]
        .detach()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    return np.uint8(np.round(rgb * 255))
