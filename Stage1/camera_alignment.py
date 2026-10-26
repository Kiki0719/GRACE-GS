import sys
from pathlib import Path

# ============================================================
# Make E-D3DGS project root importable
# camera_alignment.py is located at:
# /E-D3DGS/Stage1/camera_alignment.py
# therefore parent.parent = /E-D3DGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



"""Region-balanced foreground-aware camera alignment with FIXED intrinsics.
Run from the E-D3DGS repository root:
    python Stage1/camera_alignment.py --config CONFIG.ini

Required INI entry in [Camera Alignment]:
    camera_intrinsics_path = /root/autodl-tmp/Mono_Data/Fan/camera_intrinsics.json

Only the six extrinsic pose parameters are optimized. JSON fx/fy/cx/cy
are read once and used unchanged at the target resolution. The legacy
10-value record remains for output compatibility; its last four values
are derived constants, never optimization variables. Gaussian geometry,
foreground separation, global masked SSIM + region-balanced IoU/Recall/SSIM scoring, and output layout are retained.
Input JSON width/height must match the target image; no silent rescaling.
"""

import argparse
import configparser
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

_FIXED_INTRINSICS = None

def cfg_get(cfg, section, key, cast=str, default=None, required=False):
    if not cfg.has_section(section):
        if required:
            raise KeyError(f"Missing section [{section}]")
        return default
    if not cfg.has_option(section, key):
        if required:
            raise KeyError(f"Missing config item [{section}] {key}")
        return default
    value = cfg.get(section, key)
    if cast is bool:
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return cast(value)

def resolve_input_path(cfg, config_path, key):
    value = cfg_get(cfg, "Path", key, required=True)
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path.resolve()
    source_value = cfg_get(cfg, "Path", "source_path", default=None)
    if source_value is not None and str(source_value).strip().lower() not in ("", "none", "null"):
        source = Path(os.path.expanduser(str(source_value)))
        if not source.is_absolute():
            source = Path(config_path).resolve().parent / source
        resolved = source / path
        if resolved.exists():
            return resolved.resolve()
    return (Path(config_path).resolve().parent / path).resolve()

def resolve_output_dir(cfg, config_path):
    value = cfg_get(cfg, "Camera Alignment", "camera_alignment_output", default=None)
    if value is not None and str(value).strip().lower() not in ("", "none", "null"):
        path = Path(os.path.expanduser(str(value)))
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        return path.resolve()
    model_path = Path(os.path.expanduser(cfg_get(cfg, "Path", "model_path", required=True)))
    if not model_path.is_absolute():
        model_path = Path(config_path).resolve().parent / model_path
    return (model_path / "camera_alignment").resolve()

def load_fixed_intrinsics(cfg, config_path, width, height):
    """Read calibrated pinhole intrinsics without fitting or changing their values."""
    value = cfg_get(cfg, "Camera Alignment", "camera_intrinsics_path", required=True)
    path = Path(os.path.expanduser(value.strip()))
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    path = path.resolve()
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Camera intrinsics JSON must be an object")
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        if key not in data:
            raise ValueError(f"Missing {key!r} in intrinsics JSON: {path}")
    if float(data['width']) != width or float(data['height']) != height:
        raise ValueError(
            f"Intrinsics resolution {data['width']}x{data['height']} does not "
            f"match t0 image {width}x{height}. Export intrinsics for this image size."
        )
    values = [float(data[k]) for k in ('fx', 'fy', 'cx', 'cy')]
    if not np.isfinite(values).all() or min(values[:2]) <= 0:
        raise ValueError("fx/fy must be positive; all intrinsics must be finite")
    if data.get('camera_type', 'PERSP') not in ('PERSP', 'PINHOLE', 'pinhole'):
        raise ValueError("Only perspective pinhole intrinsics are supported")
    fx, fy, cx, cy = values
    if 'K' in data:
        K = np.asarray(data['K'], dtype=np.float64)
        expected = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        if K.shape != (3, 3) or not np.allclose(K, expected, rtol=1e-7, atol=1e-7):
            raise ValueError("JSON K disagrees with fx/fy/cx/cy, or has unsupported skew")
    result = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
                  source_path=str(path))
    print(f"[fixed intrinsics] source={path}", flush=True)
    print(f"[fixed intrinsics] fx={fx}, fy={fy}, cx={cx}, cy={cy}; pose only", flush=True)
    return result


def fixed_intrinsics_at_size(width, height):
    if _FIXED_INTRINSICS is None:
        raise RuntimeError("Fixed intrinsics have not been loaded from INI/JSON")
    k = _FIXED_INTRINSICS
    # The current pipeline uses the target resolution throughout. Keep explicit
    # scaling for diagnostic callers that resize an image, not as an optimization.
    sx, sy = float(width) / k['width'], float(height) / k['height']
    fx, fy = k['fx'] * sx, k['fy'] * sy
    cx, cy = k['cx'] * sx, k['cy'] * sy
    return fx, fy, cx, cy, fov_from_focal(fx, width), fov_from_focal(fy, height)


def load_rgb(path, device):
    """Load RGB and preserve a useful PNG alpha plane when present."""
    pil = Image.open(path)
    png_alpha = None
    if "A" in pil.getbands():
        rgba = np.asarray(pil.convert("RGBA"), dtype=np.float32) / 255.0
        arr = rgba[..., :3]
        alpha_arr = rgba[..., 3]
        # Fully opaque alpha is not a foreground mask.
        if float(alpha_arr.min()) < 0.999:
            png_alpha = torch.from_numpy(alpha_arr).to(
                device=device, dtype=torch.float32
            ).clamp(0.0, 1.0)
    else:
        arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(
        device=device, dtype=torch.float32
    )
    return ten, png_alpha, pil.width, pil.height

def resize_image(image, width, height):
    return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]

def save_rgb(image, path):
    arr = torch.clamp(image.detach(), 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path)

def save_mask(mask, path):
    arr = torch.clamp(mask.detach(), 0.0, 1.0).cpu().numpy()
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path)


def save_rgba(rgb, alpha, path):
    """Save RGB with an explicit alpha channel so foreground/background stay separated."""
    rgb_arr = torch.clamp(rgb.detach(), 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    alpha_arr = torch.clamp(alpha.detach(), 0.0, 1.0).cpu().numpy()
    rgba = np.concatenate([rgb_arr, alpha_arr[..., None]], axis=-1)
    Image.fromarray(np.clip(rgba * 255.0, 0, 255).astype(np.uint8), mode="RGBA").save(path)


def decompose_render_foreground(rendered, render_alpha, background, alpha_min=0.15):
    """Recover foreground RGB from standard alpha compositing.

    rendered = foreground_premultiplied + background * (1 - alpha)
    foreground_color = foreground_premultiplied / alpha
    """
    alpha = render_alpha.clamp(0.0, 1.0)
    bg = background.to(device=rendered.device, dtype=rendered.dtype).view(3, 1, 1)
    premult = rendered - bg * (1.0 - alpha[None])
    color = premult / alpha.clamp_min(float(alpha_min))[None]
    return premult.clamp(0.0, 1.0), color.clamp(0.0, 1.0)

def estimate_border_background(target, fallback):
    """Use a constant background only when the image border supports it."""
    arr = target.detach().permute(1,2,0).cpu().numpy()
    h,w = arr.shape[:2]
    k = max(1, int(min(h,w)*.02))
    samples = np.concatenate([arr[:k].reshape(-1,3), arr[-k:].reshape(-1,3),
                              arr[:,:k].reshape(-1,3), arr[:,-k:].reshape(-1,3)])
    color = np.median(samples,axis=0)
    deviation = float(np.quantile(np.linalg.norm(samples-color,axis=1),.90))
    accepted = deviation < .08
    bg = torch.as_tensor(color,device=target.device,dtype=target.dtype) if accepted else fallback
    return bg, {'method': 'border_median' if accepted else 'configured_fallback',
                'border_deviation_q90': deviation, 'rgb': bg.cpu().tolist()}

def mask_shape_features(mask):
    binary=np.ascontiguousarray(mask.astype(np.uint8))
    h,w=binary.shape
    edge=binary-cv2.erode(binary,np.ones((3,3),np.uint8))
    distance=cv2.distanceTransform(1-edge,cv2.DIST_L2,3)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(binary,8)
    parts=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area>=max(32,.003*h*w):
            part=labels==i
            parts.append({'mask':part,'center':centroids[i]/np.array([w,h]),
                          'size':np.array([bw/w,bh/h]),'area':int(area)})
    parts=sorted(parts,key=lambda p:p['area'],reverse=True)[:12]
    ys,xs=np.where(binary)
    diameter=math.hypot(float(np.ptp(xs)+1),float(np.ptp(ys)+1)) if len(xs) else math.hypot(h,w)
    return {'edge':edge.astype(bool),'distance':distance,'parts':parts,
            'diagonal':max(diameter,1.)}

def detailed_shape_loss(render_binary,target_features):
    from scipy.optimize import linear_sum_assignment
    rf=mask_shape_features(render_binary);tf=target_features
    if not rf['edge'].any() or not tf['edge'].any():
        return 1.,1.
    contour=float(.5*(tf['distance'][rf['edge']].mean()+rf['distance'][tf['edge']].mean())/tf['diagonal'])
    rp,tp=rf['parts'],tf['parts']
    if not rp or not tp:
        return contour,1.
    cost=np.zeros((len(rp),len(tp)))
    for i,a in enumerate(rp):
        for j,b in enumerate(tp):
            union=np.count_nonzero(a['mask']|b['mask'])
            iou=np.count_nonzero(a['mask']&b['mask'])/max(union,1)
            center=np.linalg.norm(a['center']-b['center'])
            size=np.abs(np.log(np.maximum(a['size'],1e-6)/np.maximum(b['size'],1e-6))).mean()
            cost[i,j]=1-iou+.5*center+.1*min(float(size),3.)
    rows,cols=linear_sum_assignment(cost)
    unmatched=max(len(rp),len(tp))-len(rows)
    component=float((cost[rows,cols].sum()+unmatched)/max(len(rp),len(tp)))
    return contour,component

def internal_edge_features(rgb,mask):
    """Generic internal image edges, NOT semantic hubs or keypoint matches."""
    arr=rgb.detach().clamp(0,1).permute(1,2,0).cpu().numpy()
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    binary=np.ascontiguousarray(mask.astype(np.uint8))
    interior=cv2.erode(binary,np.ones((5,5),np.uint8))>0
    values=gray[binary>0]
    if len(values)<16:
        return []
    lo,hi=np.quantile(values,[.02,.98])
    # Contrast normalization reduces sensitivity to a global lighting difference.
    gray=np.clip((gray-lo)/max(float(hi-lo),.10),0,1)
    gray=(gray*255).astype(np.uint8)
    result=[]
    for sigma in (1.,2.):
        blurred=cv2.GaussianBlur(gray,(0,0),sigma)
        edge=(cv2.Canny(blurred,40,100,L2gradient=True)>0)&interior
        distance=cv2.distanceTransform((~edge).astype(np.uint8),cv2.DIST_L2,3)
        result.append({'edge':edge,'distance':distance})
    return result

def internal_edge_loss(render_features,target_features):
    # Ten-pixel truncation at 640px prevents a missing edge dominating everything.
    if not target_features:
        return 0.
    if not render_features:
        return 1.
    values=[]
    for rf,tf in zip(render_features,target_features):
        if not tf['edge'].any():
            continue
        if not rf['edge'].any():
            values.append(1.);continue
        forward=np.minimum(tf['distance'][rf['edge']],10.).mean()
        backward=np.minimum(rf['distance'][tf['edge']],10.).mean()
        values.append(float((forward+backward)/20.))
    return float(np.mean(values)) if values else 0.

def normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n

def robust_geometry(xyz):
    pts = xyz.detach().cpu().numpy().astype(np.float64)

    low = np.quantile(pts, 0.01, axis=0)
    high = np.quantile(pts, 0.99, axis=0)

    center = 0.5 * (low + high)

    dist = np.linalg.norm(
        pts - center[None],
        axis=1
    )

    radius50 = float(np.quantile(dist, 0.50))
    radius90 = float(np.quantile(dist, 0.90))
    radius98 = float(np.quantile(dist, 0.98))

    radius = max(radius98, 1e-4)

    print(
        "[geometry center] "
        f"low={low.tolist()} "
        f"high={high.tolist()} "
        f"center={center.tolist()}"
    )

    return (
        center,
        radius,
        radius50,
        radius90,
        radius98,
    )

def look_at_to_rt(target, distance, azimuth_deg, elevation_deg, roll_deg=0.0, offset_x=0.0, offset_y=0.0):
    target = np.asarray(target, dtype=np.float64)
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    roll = math.radians(float(roll_deg))
    camera_center = target + float(distance) * np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ], dtype=np.float64)
    forward = normalize(target - camera_center)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = normalize(np.cross(forward, world_up))
    down = normalize(np.cross(forward, right))
    shifted_target = target + right * float(offset_x) + down * float(offset_y)
    forward = normalize(shifted_target - camera_center)
    right = normalize(np.cross(forward, world_up))
    down = normalize(np.cross(forward, right))
    if abs(roll) > 1e-12:
        right_new = right * math.cos(roll) + down * math.sin(roll)
        down_new = -right * math.sin(roll) + down * math.cos(roll)
        right, down = right_new, down_new
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_center
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3]).astype(np.float32)
    T = w2c[:3, 3].astype(np.float32)
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0.999:
        raise RuntimeError("Camera basis must be a proper rotation")
    check = getWorld2View2(R, T)
    target_h = np.concatenate([shifted_target, [1.0]])
    target_cam = check @ target_h
    if not np.isfinite(target_cam).all() or target_cam[2] <= 0:
        raise RuntimeError(f"Constructed camera is invalid: target camera-z={target_cam[2]}")
    return R, T, camera_center

def fovy_from_fovx(fov_x_deg, width, height):
    fovx = math.radians(float(fov_x_deg))
    fx = width / (2.0 * math.tan(fovx / 2.0))
    return 2.0 * math.atan(height / (2.0 * fx))

def fov_from_focal(focal, pixels):
    focal = max(float(focal), 1e-9)
    return math.degrees(2.0 * math.atan(float(pixels) / (2.0 * focal)))

def upgrade_camera_params(params, width, height):
    """Keep legacy ten-value records, but overwrite all intrinsic slots from JSON."""
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    if p.size not in (6, 7, 10):
        raise ValueError(f"Expected 6, 7 or 10 camera parameters, got {p.size}")
    fx, fy, cx, cy, fovx, fovy = fixed_intrinsics_at_size(width, height)
    return np.concatenate([p[:6], np.array([
        fovx, fovy, cx - width * 0.5, cy - height * 0.5
    ], dtype=np.float64)])


def projection_from_intrinsics(znear, zfar, fx, fy, cx, cy, width, height, device, dtype):
    """3DGS perspective matrix with free fx/fy/cx/cy."""
    fx = float(fx)
    fy = float(fy)
    cx = float(cx)
    cy = float(cy)
    width = float(width)
    height = float(height)
    P = torch.zeros((4, 4), device=device, dtype=dtype)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 2.0 * cx / width - 1.0
    P[1, 2] = 2.0 * cy / height - 1.0
    P[3, 2] = 1.0
    P[2, 2] = float(zfar) / (float(zfar) - float(znear))
    P[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear))
    return P

def camera_intrinsics_from_params(width, height, params):
    # Deliberately ignore intrinsic values in view/prior parameter vectors.
    # Return JSON values directly instead of a focal -> FOV -> focal round trip.
    return fixed_intrinsics_at_size(width, height)


def make_camera(width, height, image, target_center, params, device):
    p = upgrade_camera_params(params, width, height)
    az, el, roll, distance, offx, offy = [float(x) for x in p[:6]]
    fx, fy, cx, cy, fov_x_deg, fov_y_deg = camera_intrinsics_from_params(
        width, height, p
    )
    R, T, camera_center = look_at_to_rt(
        target_center, distance, az, el, roll, offx, offy
    )
    cam = Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=math.radians(fov_x_deg),
        FoVy=math.radians(fov_y_deg),
        image=image,
        gt_alpha_mask=None,
        image_name="t0",
        uid=0,
        data_device=device,
        fid=0.0,
        depth=None,
    )
    dtype = cam.world_view_transform.dtype
    dev = cam.world_view_transform.device
    # Preserve the repository's exact z/depth convention and only replace the
    # four intrinsic entries. Camera.projection_matrix is stored transposed.
    P = cam.projection_matrix.transpose(0, 1).clone()
    P[0, 0] = 2.0 * float(fx) / float(width)
    P[1, 1] = 2.0 * float(fy) / float(height)
    P[0, 2] = 2.0 * float(cx) / float(width) - 1.0
    P[1, 2] = 2.0 * float(cy) / float(height) - 1.0
    cam.projection_matrix = P.transpose(0, 1).contiguous()
    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0)
        .bmm(cam.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    # Keep exact extrinsics/intrinsics even on E-D3DGS forks that do not store R.
    cam.R = R
    cam.T = T
    cam.k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=dev,
    )
    cam.alignment_intrinsics = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "fov_x_deg": fov_x_deg, "fov_y_deg": fov_y_deg,
    }
    return cam, camera_center

def render_static(camera, gaussians, pipe, background):
    # Static canonical G0: pass scalar zero offsets exactly as E-D3DGS does
    # when deformation has already been baked / is absent.
    package = render(
        camera,
        gaussians,
        pipe,
        background,
        0.0,
        0.0,
        0.0,
        False,
    )
    package["raw_render"] = package["render"]
    package["render"] = torch.clamp(package["render"], 0.0, 1.0)
    return package

def background_color_target_mask(target, cfg):
    """Foreground extraction by removing the estimated border background.

    This follows the Stage-1 object/background separation policy:
      - estimate background RGB from image borders using a robust median;
      - compare in Lab space when OpenCV is available;
      - choose the threshold from measured border variation + a margin;
      - clean the binary foreground mask morphologically.

    G0 / rendered alpha is deliberately NOT used to decide target foreground.
    """
    image = target.detach().permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    h, w, _ = image.shape

    def mask_option(key, cast, default):
        return cfg_get(cfg, "Camera Alignment", key, cast, default)

    border_px = max(2, mask_option("camera_mask_border_px", int, 12))
    border_px = min(border_px, max(2, min(h, w) // 8))

    top = image[:border_px].reshape(-1, 3)
    bottom = image[-border_px:].reshape(-1, 3)
    if h > 2 * border_px:
        left = image[border_px:h-border_px, :border_px].reshape(-1, 3)
        right = image[border_px:h-border_px, -border_px:].reshape(-1, 3)
        border_rgb = np.concatenate([top, bottom, left, right], axis=0)
    else:
        border_rgb = np.concatenate([top, bottom], axis=0)

    bg_rgb = np.median(border_rgb, axis=0)

    if cv2 is not None:
        image_u8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
        bg_u8 = np.clip(bg_rgb[None, None] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        image_lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(bg_u8, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]

        distance = np.linalg.norm(image_lab - bg_lab[None, None], axis=-1)

        border_lab_parts = [
            image_lab[:border_px].reshape(-1, 3),
            image_lab[-border_px:].reshape(-1, 3),
        ]
        if h > 2 * border_px:
            border_lab_parts.extend([
                image_lab[border_px:h-border_px, :border_px].reshape(-1, 3),
                image_lab[border_px:h-border_px, -border_px:].reshape(-1, 3),
            ])
        border_lab = np.concatenate(border_lab_parts, axis=0)
        border_distance = np.linalg.norm(border_lab - bg_lab[None], axis=-1)

        q = float(np.clip(
            mask_option("camera_mask_bg_distance_quantile", float, 0.995),
            0.50, 0.9999
        ))
        background_noise = float(np.quantile(border_distance, q))
        threshold = max(
            mask_option("camera_mask_bg_min_lab_distance", float, 12.0),
            background_noise + mask_option("camera_mask_bg_distance_margin", float, 5.0),
        )
        foreground = (distance > threshold).astype(np.uint8)
    else:
        distance = np.mean(np.abs(image - bg_rgb[None, None]), axis=-1)
        threshold = mask_option("camera_mask_bg_rgb_distance", float, 0.06)
        foreground = (distance > threshold).astype(np.uint8)

    if cv2 is not None:
        close_radius = max(0, mask_option("camera_mask_close_radius", int, 3))
        if close_radius > 0:
            k = 2 * close_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        open_radius = max(0, mask_option("camera_mask_open_radius", int, 1))
        if open_radius > 0:
            k = 2 * open_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground, connectivity=8
        )
        min_component = max(
            1, mask_option("camera_mask_min_component_pixels", int, 80)
        )
        cleaned = np.zeros_like(foreground)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_component:
                cleaned[labels == i] = 1
        if cleaned.sum() > 0:
            foreground = cleaned

        # Fill small enclosed holes, but never fill the outside background.
        inverse = (1 - foreground).astype(np.uint8)
        num_holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
            inverse, connectivity=8
        )
        max_hole = max(0, mask_option("camera_mask_max_hole_pixels", int, 800))
        for i in range(1, num_holes):
            ys, xs = np.where(hole_labels == i)
            if ys.size == 0:
                continue
            touches_border = (
                ys.min() == 0 or ys.max() == h - 1
                or xs.min() == 0 or xs.max() == w - 1
            )
            area = int(hole_stats[i, cv2.CC_STAT_AREA])
            if (not touches_border) and area <= max_hole:
                foreground[hole_labels == i] = 1

        smooth_kernel = max(0, mask_option("camera_mask_smooth_kernel", int, 7))
        if smooth_kernel >= 3:
            if smooth_kernel % 2 == 0:
                smooth_kernel += 1
            soft = cv2.GaussianBlur(
                foreground.astype(np.float32),
                (smooth_kernel, smooth_kernel), 0
            )
            foreground = (
                soft >= mask_option("camera_mask_smooth_threshold", float, 0.50)
            ).astype(np.uint8)

    mask = torch.from_numpy(foreground.astype(np.float32)).to(
        device=target.device, dtype=target.dtype
    ).clamp(0.0, 1.0)
    return mask, bg_rgb, float(threshold)


def build_target_mask(target, png_alpha, cfg, config_path, width, height):
    """Final target mask.

    Priority:
      1. explicit [Path] t0_mask_path
      2. useful PNG alpha
      3. Lab background-color removal

    The Gaussian model is never used to define target foreground.
    """
    mask_value = cfg_get(cfg, "Path", "t0_mask_path", default=None)

    if (
        mask_value is not None
        and str(mask_value).strip().lower() not in ("", "none", "null")
    ):
        mask_path = resolve_input_path(cfg, config_path, "t0_mask_path")
        mask_pil = Image.open(mask_path).convert("L")
        if mask_pil.size != (width, height):
            raise ValueError(
                f"t0 mask dimensions {mask_pil.size} do not match image {(width, height)}"
            )
        mask = torch.from_numpy(
            (np.asarray(mask_pil, dtype=np.float32) / 255.0)
        ).to(device=target.device, dtype=target.dtype)
        threshold = cfg_get(
            cfg, "Camera Alignment", "camera_mask_explicit_threshold", float, 0.5
        )
        mask = (mask >= threshold).float()
        source = str(mask_path)
        bg_info = None

    elif png_alpha is not None:
        threshold = cfg_get(
            cfg, "Camera Alignment", "camera_mask_png_alpha_threshold", float, 0.5
        )
        mask = (png_alpha >= threshold).float()
        source = "PNG alpha"
        bg_info = None

    else:
        mask, bg_rgb, bg_threshold = background_color_target_mask(target, cfg)
        source = (
            "background-color removal "
            f"bg_rgb={np.asarray(bg_rgb).tolist()} threshold={float(bg_threshold):.4f}"
        )
        bg_info = {
            "rgb": np.asarray(bg_rgb).tolist(),
            "threshold": float(bg_threshold),
        }

    mask = (mask >= 0.5).float().clamp(0.0, 1.0)
    min_pixels = max(
        1, cfg_get(
            cfg, "Camera Alignment",
            "camera_mask_min_target_foreground_pixels", int, 128
        )
    )
    foreground_pixels = int(mask.sum().item())
    if foreground_pixels < min_pixels:
        raise RuntimeError(
            f"Target mask contains too few foreground pixels: {foreground_pixels}; "
            f"source={source}"
        )
    return mask, source, bg_info


def mask_bbox(mask):
    ys, xs = torch.where(mask > 0.5)
    if xs.numel() == 0:
        return None
    x0 = float(xs.min().item())
    x1 = float(xs.max().item())
    y0 = float(ys.min().item())
    y1 = float(ys.max().item())
    return x0, y0, x1, y1

def mask_iou(a, b):
    aa = a > 0.5
    bb = b > 0.5
    union = (aa | bb).float().sum()
    if union.item() <= 0:
        return 0.0
    return float(((aa & bb).float().sum() / union).item())


def build_balanced_region_cells(mask, rows=9, cols=9, min_foreground_pixels=20):
    """Build equal-vote cells inside the GT foreground bounding box.

    Only cells containing enough GT foreground are retained. Every retained cell gets
    one vote later regardless of area, so a large part cannot numerically drown a thin
    handle, stem, support, leg, arm, or other small structure.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    min_foreground_pixels = max(1, int(min_foreground_pixels))
    bb = mask_bbox(mask)
    if bb is None:
        return []
    x0, y0, x1, y1 = bb
    x0 = int(math.floor(x0)); y0 = int(math.floor(y0))
    x1 = int(math.floor(x1)) + 1; y1 = int(math.floor(y1)) + 1
    h, w = mask.shape
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    bw, bh = x1 - x0, y1 - y0
    target_bin = mask > 0.5
    cells = []
    for r in range(rows):
        cy0 = y0 + int(round(r * bh / rows))
        cy1 = y0 + int(round((r + 1) * bh / rows))
        cy1 = max(cy0 + 1, cy1)
        for c in range(cols):
            cx0 = x0 + int(round(c * bw / cols))
            cx1 = x0 + int(round((c + 1) * bw / cols))
            cx1 = max(cx0 + 1, cx1)
            fg = int(target_bin[cy0:cy1, cx0:cx1].sum().item())
            if fg < min_foreground_pixels:
                continue
            cells.append({
                'row': int(r), 'col': int(c),
                'x0': int(cx0), 'x1': int(cx1),
                'y0': int(cy0), 'y1': int(cy1),
                'target_foreground_pixels': int(fg),
            })
    return cells


def region_balanced_silhouette_loss(alpha, target_mask, cells, alpha_threshold=0.05):
    """Equal-weight regional IoU and GT-recall/missing losses.

    For each retained cell k:
        L_iou,k  = 1 - IoU(render_k, target_k)
        L_miss,k = 1 - Recall(render_k w.r.t. target_k)
    The mean is over CELLS rather than pixels.
    """
    render_bin = alpha > float(alpha_threshold)
    target_bin = target_mask > 0.5
    if not cells:
        h, w = target_mask.shape
        cells = [{
            'row': 0, 'col': 0, 'x0': 0, 'x1': int(w),
            'y0': 0, 'y1': int(h),
            'target_foreground_pixels': int(target_bin.sum().item()),
        }]
    ious, recalls, precisions, records = [], [], [], []
    for cell in cells:
        y0, y1 = cell['y0'], cell['y1']
        x0, x1 = cell['x0'], cell['x1']
        t = target_bin[y0:y1, x0:x1]
        r = render_bin[y0:y1, x0:x1]
        target_count = int(t.sum().item())
        render_count = int(r.sum().item())
        inter = int((t & r).sum().item())
        union = int((t | r).sum().item())
        iou = float(inter / max(union, 1))
        recall = float(inter / max(target_count, 1))
        precision = float(inter / max(render_count, 1)) if render_count > 0 else 0.0
        ious.append(iou); recalls.append(recall); precisions.append(precision)
        records.append({
            **cell,
            'render_foreground_pixels': render_count,
            'intersection_pixels': inter,
            'union_pixels': union,
            'iou': iou, 'recall': recall, 'precision': precision,
        })
    mean_iou = float(np.mean(ious))
    mean_recall = float(np.mean(recalls))
    diag = {
        'active_regions': int(len(records)),
        'mean_region_iou': mean_iou,
        'worst_region_iou': float(np.min(ious)),
        'mean_region_recall': mean_recall,
        'worst_region_recall': float(np.min(recalls)),
        'mean_region_precision': float(np.mean(precisions)),
        'regions': records,
    }
    return 1.0 - mean_iou, 1.0 - mean_recall, diag


def region_balanced_ssim_loss(
    rendered,
    target,
    render_alpha,
    target_mask,
    cells,
    background,
    cfg=None,
):
    """
    Region-balanced foreground SSIM.

    IoU checks foreground occupancy. This term checks whether the local
    appearance/structure inside each active region actually matches the target.

    SSIM is measured only on GT foreground support inside each active cell.
    Missing rendered foreground is deliberately NOT removed from the support;
    otherwise missing parts could receive an artificially good SSIM.
    """
    def camera_option(key, cast, default):
        if cfg is None:
            return default
        return cfg_get(cfg, "Camera Alignment", key, cast, default)

    def erode_binary(mask, pixels):
        pixels = max(0, int(pixels))
        if pixels == 0:
            return mask
        k = 2 * pixels + 1
        return 1.0 - F.max_pool2d(
            (1.0 - mask)[None, None],
            k,
            stride=1,
            padding=pixels,
        )[0, 0]

    alpha_min = camera_option(
        "region_ssim_alpha_min",
        float,
        camera_option("camera_rgb_alpha_min", float, 0.15),
    )
    mask_erode = max(
        0,
        camera_option("region_ssim_mask_erode", int, 1),
    )
    min_pixels = max(
        1,
        camera_option("region_ssim_min_pixels", int, 8),
    )
    window = max(
        3,
        camera_option("region_ssim_window", int, 7),
    )
    if window % 2 == 0:
        window += 1
    padding = window // 2

    _, render_color = decompose_render_foreground(
        rendered,
        render_alpha,
        background,
        alpha_min=alpha_min,
    )

    x = render_color[None]
    y = target[None]
    pool = lambda v: F.avg_pool2d(
        v,
        window,
        stride=1,
        padding=padding,
    )
    ux = pool(x)
    uy = pool(y)
    vx = (pool(x * x) - ux * ux).clamp_min(0.0)
    vy = (pool(y * y) - uy * uy).clamp_min(0.0)
    cov = pool(x * y) - ux * uy

    ssim_map = (
        (2.0 * ux * uy + 0.01 ** 2)
        * (2.0 * cov + 0.03 ** 2)
        / (
            (ux * ux + uy * uy + 0.01 ** 2)
            * (vx + vy + 0.03 ** 2)
        )
    )
    ssim_map = ssim_map[0].mean(0).clamp(0.0, 1.0)

    target_fg = (target_mask > 0.5).float()
    eroded_fg = erode_binary(target_fg, mask_erode)
    records = []
    scores = []

    for cell in cells:
        y0, y1 = cell["y0"], cell["y1"]
        x0, x1 = cell["x0"], cell["x1"]

        support = eroded_fg[y0:y1, x0:x1]
        support_pixels = int((support > 0.5).sum().item())

        if support_pixels < min_pixels:
            support = target_fg[y0:y1, x0:x1]
            support_pixels = int((support > 0.5).sum().item())

        if support_pixels <= 0:
            continue

        local_map = ssim_map[y0:y1, x0:x1]
        denom = max(float(support.sum().item()), 1.0)
        local_score = float((local_map * support).sum().item() / denom)
        local_score = float(np.clip(local_score, 0.0, 1.0))
        scores.append(local_score)

        records.append({
            "row": int(cell["row"]),
            "col": int(cell["col"]),
            "x0": int(x0),
            "x1": int(x1),
            "y0": int(y0),
            "y1": int(y1),
            "support_pixels": int(support_pixels),
            "ssim": local_score,
        })

    if not scores:
        support = target_fg
        denom = float(support.sum().item())
        mean_ssim = (
            float((ssim_map * support).sum().item() / denom)
            if denom > 0
            else 0.0
        )
        mean_ssim = float(np.clip(mean_ssim, 0.0, 1.0))
        worst_ssim = mean_ssim
    else:
        mean_ssim = float(np.mean(scores))
        worst_ssim = float(np.min(scores))

    diag = {
        "active_ssim_regions": int(len(records)),
        "mean_region_ssim": mean_ssim,
        "worst_region_ssim": worst_ssim,
        "region_ssim_window": int(window),
        "region_ssim_min_pixels": int(min_pixels),
        "region_ssim_mask_erode": int(mask_erode),
        "regions": records,
    }

    return 1.0 - mean_ssim, 1.0 - worst_ssim, diag


def save_region_grid_preview(target, cells, path):
    arr = (target.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    for idx, cell in enumerate(cells):
        x0, x1, y0, y1 = cell['x0'], cell['x1'], cell['y0'], cell['y1']
        cv2.rectangle(arr, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (255, 0, 0), 1)
        cv2.putText(arr, str(idx), (x0 + 2, min(y1 - 2, y0 + 13)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(arr).save(path)


def _angle_difference_deg(a, b):
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def camera_prior_loss(params, anchor, radius):
    """Gentle normalized quadratic prior around an already-good rough camera."""
    if anchor is None:
        return 0.0
    p = np.asarray(params, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    scales = np.array([
        5.0, 5.0, 5.0,
        max(abs(float(a[3])) * 0.05, float(radius) * 0.08, 1e-6),
        max(float(radius) * 0.10, 1e-6),
        max(float(radius) * 0.10, 1e-6),
        2.0, 2.0, 8.0, 8.0,
    ], dtype=np.float64)
    delta = p - a
    delta[0] = _angle_difference_deg(p[0], a[0])
    delta[2] = _angle_difference_deg(p[2], a[2])
    z = delta[:6] / scales[:6]
    return float(np.mean(z * z))


def load_camera_prior_params(cfg, config_path, g0_path, width, height, explicit_prior_path=None):
    """Load rough-camera JSON.

    Priority:
      1) explicit_prior_path from CLI/orchestrator;
      2) [Camera Alignment] camera_prior_path in the INI;
      3) fixed_camera.json next to the input G0 (legacy fallback).
    """
    prior_paths = []
    if explicit_prior_path is not None:
        explicit_text = str(explicit_prior_path).strip()
        if explicit_text.lower() in ('none', 'null', 'off', 'disable', 'disabled'):
            print('[camera prior] explicitly disabled', flush=True)
            return None, None
        if explicit_text:
            q = Path(os.path.expanduser(explicit_text))
            if not q.is_absolute():
                q = Path(config_path).resolve().parent / q
            prior_paths.append(q.resolve())
    prior_value = cfg_get(cfg, 'Camera Alignment', 'camera_prior_path', default=None)
    if prior_value is not None and str(prior_value).strip().lower() not in ('', 'none', 'null'):
        q = Path(os.path.expanduser(str(prior_value)))
        if not q.is_absolute():
            q = Path(config_path).resolve().parent / q
        prior_paths.append(q.resolve())
    prior_paths.append((Path(g0_path).resolve().parent / 'fixed_camera.json').resolve())
    prior_path = next((q for q in prior_paths if q.is_file()), None)
    if prior_path is None:
        return None, None
    try:
        data = json.loads(prior_path.read_text(encoding='utf-8'))
        fovx = float(data.get('fov_x_deg', 50.0))
        fovy = float(data.get('fov_y_deg', math.degrees(fovy_from_fovx(fovx, width, height))))
        cx_delta = float(data.get('cx_delta_px', float(data.get('cx', width * 0.5)) - width * 0.5))
        cy_delta = float(data.get('cy_delta_px', float(data.get('cy', height * 0.5)) - height * 0.5))
        params = np.array([
            float(data['azimuth_deg']), float(data['elevation_deg']),
            float(data.get('roll_deg', 0.0)), float(data['distance']),
            float(data.get('target_offset_x', 0.0)), float(data.get('target_offset_y', 0.0)),
            fovx, fovy, cx_delta, cy_delta,
        ], dtype=np.float64)
        return params, str(prior_path)
    except Exception as exc:
        print(f'[camera prior] Failed to parse {prior_path}: {exc}; prior disabled.', flush=True)
        return None, None


def geometry_loss(render_alpha, target_mask):
    render_bin = (render_alpha > 0.05).float()
    target_bin = (target_mask > 0.5).float()
    iou = mask_iou(render_bin, target_bin)
    rb = mask_bbox(render_bin)
    tb = mask_bbox(target_bin)
    if rb is None or tb is None:
        return 10.0, {"iou": 0.0, "center": 10.0, "scale": 10.0, "mask_l1": 1.0}
    h, w = target_mask.shape
    rcx = (rb[0] + rb[2]) * 0.5 / w
    rcy = (rb[1] + rb[3]) * 0.5 / h
    tcx = (tb[0] + tb[2]) * 0.5 / w
    tcy = (tb[1] + tb[3]) * 0.5 / h
    center_err = math.sqrt((rcx - tcx) ** 2 + (rcy - tcy) ** 2)
    rw = max((rb[2] - rb[0] + 1.0) / w, 1e-6)
    rh = max((rb[3] - rb[1] + 1.0) / h, 1e-6)
    tw = max((tb[2] - tb[0] + 1.0) / w, 1e-6)
    th = max((tb[3] - tb[1] + 1.0) / h, 1e-6)
    scale_err = abs(math.log(rw / tw)) + abs(math.log(rh / th))
    mask_l1 = float(torch.mean(torch.abs(render_alpha - target_bin)).item())
    loss = (1.0 - iou) + 0.35 * center_err + 0.20 * scale_err + 0.20 * mask_l1
    return loss, {"iou": iou, "center": center_err, "scale": scale_err, "mask_l1": mask_l1}

def target_bbox_fraction(mask):
    bb = mask_bbox(mask)
    if bb is None:
        return 0.6
    h, w = mask.shape
    return max((bb[2] - bb[0] + 1.0) / w, (bb[3] - bb[1] + 1.0) / h)

def fit_distance(radius, fov_deg, target_fraction):
    frac = float(np.clip(target_fraction, 0.15, 0.90))
    half_angle = math.atan(max(frac, 1e-4) * math.tan(math.radians(fov_deg) * 0.5))
    s = max(math.sin(half_angle), 1e-4)
    return max(radius * 1.08 / s, radius * 1.15)


def gaussian_diagnostics(gaussians):
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach()
    scaling = gaussians.get_scaling.detach()
    def q(t, vals):
        flat = t.float().reshape(-1)
        return [float(torch.quantile(flat, v).item()) for v in vals]
    return {
        "num_gaussians": int(xyz.shape[0]),
        "xyz_min": [float(x) for x in xyz.min(dim=0).values.cpu()],
        "xyz_max": [float(x) for x in xyz.max(dim=0).values.cpu()],
        "opacity_quantiles_01_50_99": q(opacity, [0.01, 0.50, 0.99]),
        "scale_quantiles_01_50_99": q(scaling, [0.01, 0.50, 0.99]),
    }

def render_view(gaussians, pipe, background, image, target_center,
                     params, device, image_mirror=False):
    if image_mirror:
        raise ValueError("Post-render mirroring is disabled; use proper camera rotations")
    h, w = image.shape[1:]
    camera, camera_center = make_camera(w, h, image, target_center, params, device)
    with torch.no_grad():
        package = render_static(camera, gaussians, pipe, background)
        alpha = package.get("alpha")
        if alpha is None:
            # C_white - C_black = 1-alpha for standard 3DGS compositing.
            # render_static preserves raw RGB to avoid clipping this difference.
            # An arbitrary matched background needs an explicit black/white pair.
            black = torch.zeros_like(background)
            white = torch.ones_like(background)
            pb = package if torch.equal(background, black) else render_static(camera, gaussians, pipe, black)
            pw = package if torch.equal(background, white) else render_static(camera, gaussians, pipe, white)
            alpha = (1-(pw['raw_render']-pb['raw_render']).mean(0)).clamp(0,1)
        else:
            alpha = alpha.squeeze()
            if alpha.shape != (h,w):
                raise ValueError(f"Unexpected alpha shape: {alpha.shape}")
            alpha = alpha.float().clamp(0,1)
    visibility = package.get("visibility_filter")
    visible_count = int(visibility.sum().item()) if visibility is not None else -1
    return camera, camera_center, package, alpha, visible_count

def diagnostic_views(
    gaussians, pipe, background, image, target_center,
    radius, target_fraction, fov, output_dir, device
):
    """
    Render the six deterministic canonical views and return those exact poses.

    These six views are the ONLY global camera-start views used later:
        front, right, back, left, top, bottom

    There is no second global-view grid and no separate preview-view stage.
    """
    diag_dir = output_dir / "diagnostic_views"
    diag_dir.mkdir(parents=True, exist_ok=True)

    views = [
        ("front",  0.0,   0.0),   # +Z
        ("right", 90.0,   0.0),   # +X
        ("back", 180.0,   0.0),   # -Z
        ("left", 270.0,   0.0),   # -X
        ("top",    0.0,  90.0),   # +Y，
        ("bottom", 0.0, -90.0),   # -Y，
    ]

    distance = fit_distance(radius, fov, target_fraction)
    records = []
    any_visible = False

    for name, az, el in views:
        params = np.array(
            [az, el, 0.0, distance, 0.0, 0.0, fov],
            dtype=np.float64,
        )

        _, _, package, alpha, visible_count = render_view(
            gaussians, pipe, background, image, target_center, params, device
        )

        alpha_area = float((alpha > 0.05).float().mean().item())
        valid = bool(visible_count > 0 or alpha_area > 1e-5)
        any_visible = any_visible or valid

        save_rgb(package["render"], diag_dir / f"{name}.png")
        save_mask(alpha, diag_dir / f"{name}_alpha.png")

        records.append({
            "name": name,
            "azimuth": float(az),
            "elevation": float(el),
            "roll": 0.0,
            "distance": float(distance),
            "target_offset_x": 0.0,
            "target_offset_y": 0.0,
            "fov_x_deg": float(fov),
            "visible_gaussians": int(visible_count),
            "alpha_area_ratio": float(alpha_area),
            "valid": valid,
            "params": params.tolist(),
        })

        print(
            f"[diagnostic view] {name:6s} "
            f"visible={visible_count} alpha_area={alpha_area:.6f}",
            flush=True,
        )

    with (diag_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "role": "only global camera-start views",
                "distance": float(distance),
                "views": records,
            },
            f,
            indent=2,
        )

    if not any_visible:
        raise RuntimeError(
            "G0 produced no visible Gaussian in the six diagnostic views. "
            "Check whether the PLY is a renderable 3DGS model for this repository."
        )

    return records

def evaluate_metrics(rendered, target, alpha, target_mask, background=None, cfg=None):
    foreground = appearance_score(
        rendered, target, target_mask,
        render_alpha=alpha, background=background, cfg=cfg
    )

    # Foreground-only PSNR using the same support logic conceptually: compare
    # de-composited foreground color against target foreground and normalize by
    # foreground pixels, never by background area.
    if background is not None:
        alpha_min = (
            cfg_get(cfg, "Camera Alignment", "camera_rgb_alpha_min", float, 0.15)
            if cfg is not None else 0.15
        )
        _, render_color = decompose_render_foreground(
            rendered, alpha, background, alpha_min=alpha_min
        )
    else:
        render_color = rendered

    support = (target_mask > 0.5).float()
    if alpha is not None:
        alpha_min = (
            cfg_get(cfg, "Camera Alignment", "camera_rgb_alpha_min", float, 0.15)
            if cfg is not None else 0.15
        )
        support = support * (alpha >= float(alpha_min)).float()
        if float(support.sum().item()) < 16.0:
            support = (target_mask > 0.5).float()

    sq = ((render_color - target) ** 2).mean(0)
    fg_mse = (sq * support).sum() / support.sum().clamp_min(1.0)
    fg_psnr = float((-10.0 * torch.log10(fg_mse.clamp_min(1e-12))).item())

    # Keep full-frame metrics explicitly named GLOBAL for diagnostics only.
    rb = rendered.unsqueeze(0)
    tb = target.unsqueeze(0)
    return {
        "Foreground_PSNR_dB": fg_psnr,
        "Foreground_SSIM_7x7": 1.0 - foreground['ssim'],
        "Foreground_MAE": foreground['rgb'],
        "Foreground_RGB_pixels": int(foreground['foreground_pixels']),
        "Global_PSNR_dB": float(psnr(rb, tb).mean().item()),
        "Global_SSIM": float(ssim(rb, tb).mean().item()),
        "Global_LPIPS": float(lpips(rb, tb).mean().item()),
        "Global_MAE": float(torch.mean(torch.abs(rendered - target)).item()),
        "Global_RMSE": float(torch.sqrt(torch.mean((rendered - target) ** 2)).item()),
        "MaskIoU": mask_iou(alpha > 0.05, target_mask > 0.5),
    }


@torch.no_grad()
def global_masked_ssim_loss(
    rendered,
    target,
    target_mask,
    render_alpha=None,
    background=None,
    cfg=None,
):
    """
    SSIM over the WHOLE target object mask.

    This is intentionally different from Region SSIM:
      - Global Masked SSIM asks whether the entire masked object appearance /
        structure is globally consistent.
      - Region SSIM gives every local cell an equal vote and catches local errors.

    Support policy:
      - use the GT target mask as the support;
      - do NOT restrict support to GT ∩ rendered-alpha, because that would let
        missing rendered structures disappear from the SSIM penalty;
      - optionally erode the GT mask by a small amount to reduce boundary noise.
    """
    mask = target_mask.float().clamp(0.0, 1.0)

    def option(key, cast, default):
        if cfg is None:
            return default
        return cfg_get(cfg, "Camera Alignment", key, cast, default)

    def erode_binary(m, pixels):
        pixels = max(0, int(pixels))
        if pixels == 0:
            return m
        k = 2 * pixels + 1
        return 1.0 - F.max_pool2d(
            (1.0 - m)[None, None],
            k,
            stride=1,
            padding=pixels,
        )[0, 0]

    mask_erode = max(
        0,
        option("global_mask_ssim_erode", int, 1),
    )
    window = max(
        3,
        option("global_mask_ssim_window", int, 7),
    )
    if window % 2 == 0:
        window += 1

    support = erode_binary(mask, mask_erode)
    if float(support.sum().item()) < 16.0:
        support = mask

    render_color = rendered
    if render_alpha is not None and background is not None:
        alpha_min = option(
            "global_mask_ssim_alpha_min",
            float,
            option("camera_rgb_alpha_min", float, 0.15),
        )
        _, render_color = decompose_render_foreground(
            rendered,
            render_alpha,
            background,
            alpha_min=alpha_min,
        )

    # Apply the GT mask to both images so the comparison is explicitly
    # "the whole object after masking", not the raw full frame.
    render_masked = render_color * mask[None]
    target_masked = target * mask[None]

    x = render_masked[None]
    y = target_masked[None]
    padding = window // 2
    pool = lambda v: F.avg_pool2d(
        v,
        window,
        stride=1,
        padding=padding,
    )

    ux = pool(x)
    uy = pool(y)
    vx = (pool(x * x) - ux * ux).clamp_min(0.0)
    vy = (pool(y * y) - uy * uy).clamp_min(0.0)
    cov = pool(x * y) - ux * uy

    ssim_map = (
        (2.0 * ux * uy + 0.01 ** 2)
        * (2.0 * cov + 0.03 ** 2)
        / (
            (ux * ux + uy * uy + 0.01 ** 2)
            * (vx + vy + 0.03 ** 2)
        )
    )

    ssim_map = ssim_map[0].mean(0).clamp(0.0, 1.0)

    denom = support.sum().clamp_min(1.0)
    global_mask_ssim = float(
        (ssim_map * support).sum().item()
        / float(denom.item())
    )
    global_mask_ssim = float(np.clip(global_mask_ssim, 0.0, 1.0))

    return 1.0 - global_mask_ssim, {
        "global_mask_ssim": global_mask_ssim,
        "global_mask_ssim_loss": 1.0 - global_mask_ssim,
        "support_pixels": int((support > 0.5).sum().item()),
        "window": int(window),
        "mask_erode": int(mask_erode),
    }


@torch.no_grad()
def appearance_score(rendered, target, mask, render_alpha=None, background=None, cfg=None):
    """Foreground-only RGB/SSIM/texture score.

    Target background never contributes to RGB/SSIM. When render alpha/background
    are available, the rendered foreground color is first de-composited from the
    renderer background, so a white/black/nonuniform target background cannot bias
    camera selection through appearance.
    """
    mask = mask.float().clamp(0.0, 1.0)

    def camera_option(key, cast, default):
        if cfg is None:
            return default
        return cfg_get(cfg, "Camera Alignment", key, cast, default)

    def erode_binary(m, pixels):
        pixels = max(0, int(pixels))
        if pixels == 0:
            return m
        k = 2 * pixels + 1
        return 1.0 - F.max_pool2d(
            (1.0 - m)[None, None], k, stride=1, padding=pixels
        )[0, 0]

    def weighted_mean(value, weight):
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(0)
        channels = value.shape[0] if value.ndim == 3 else 1
        return (value * weight).sum() / (weight.sum() * channels).clamp_min(1e-8)

    photo_mask = erode_binary(
        mask, camera_option("camera_rgb_mask_erode", int, 1)
    )
    if float(photo_mask.sum().item()) < 16.0:
        photo_mask = mask

    render_color = rendered
    color_support = photo_mask
    if render_alpha is not None and background is not None:
        alpha_min = camera_option("camera_rgb_alpha_min", float, 0.15)
        _, render_color = decompose_render_foreground(
            rendered, render_alpha, background, alpha_min=alpha_min
        )
        color_support = (
            (photo_mask > 0.5) & (render_alpha >= float(alpha_min))
        ).float()
        support_erode = camera_option("camera_rgb_support_erode", int, 1)
        color_support = erode_binary(color_support, support_erode)
        if float(color_support.sum().item()) < 16.0:
            color_support = (
                (photo_mask > 0.5) & (render_alpha >= float(alpha_min))
            ).float()
        if float(color_support.sum().item()) < 16.0:
            color_support = photo_mask

    rgb = weighted_mean(
        F.smooth_l1_loss(
            render_color, target, reduction="none",
            beta=camera_option("camera_rgb_smooth_l1_beta", float, 0.03),
        ),
        color_support,
    )

    # Local SSIM map, normalized ONLY over foreground support.
    x, y = render_color[None], target[None]
    pool = lambda v: F.avg_pool2d(v, 7, stride=1, padding=3)
    ux, uy = pool(x), pool(y)
    vx = (pool(x * x) - ux * ux).clamp_min(0)
    vy = (pool(y * y) - uy * uy).clamp_min(0)
    cov = pool(x * y) - ux * uy
    smap = (
        (2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)
        / ((ux * ux + uy * uy + .01 ** 2) * (vx + vy + .03 ** 2))
    )
    structural = weighted_mean(
        1 - smap[0].mean(0).clamp(-1, 1), color_support
    )

    # Texture/gradient score is also foreground-only.
    interior = erode_binary(
        color_support,
        camera_option("camera_texture_mask_erode", int, 2),
    )
    if float(interior.sum().item()) < 16.0:
        interior = color_support

    texture_terms = []
    for dim in (1, 2):
        dr = torch.diff(render_color, dim=dim)
        dt = torch.diff(target, dim=dim)
        if dim == 2:
            support = interior[:, 1:] * interior[:, :-1]
        else:
            support = interior[1:, :] * interior[:-1, :]
        magnitude = dt.abs().mean(0)
        strength = weighted_mean(magnitude, support).clamp_min(.01)
        weight = support * (1 + (magnitude / strength).clamp(max=4))
        error = (dr - dt).abs().mean(0)
        texture_terms.append(weighted_mean(error, weight))
    texture = torch.stack(texture_terms).mean()

    return {
        'rgb': float(rgb),
        'ssim': float(structural),
        'texture': float(texture),
        'foreground_pixels': int((color_support > 0.5).sum().item()),
    }


def powell_camera_parameters(initial, evaluate, radius, fov_min, fov_max,
                             maxfev=600, optimize_fov=False, fine=False,
                             label='Powell', principal_limit_px=32.0):
    """Bounded Powell over six extrinsic parameters; intrinsics always fixed.
    Parameter order:
      az, el, roll, distance, target_off_x, target_off_y,
      fov_x_deg, fov_y_deg, cx_delta_px, cy_delta_px
    """
    from scipy.optimize import minimize
    anchor = np.asarray(initial['params'], dtype=np.float64).copy()
    if anchor.size != 10:
        raise ValueError("powell_camera_parameters expects upgraded 10-D params")
    spread = np.array([
        45.0, 30.0, 45.0,
        max(anchor[3] * 0.80, radius * 0.80),
        radius, radius,
        10.0, 10.0,
        principal_limit_px, principal_limit_px,
    ], dtype=np.float64)
    if fine:
        spread = np.array([
            8.0, 8.0, 8.0,
            max(anchor[3] * 0.10, radius * 0.12),
            radius * 0.18, radius * 0.18,
            2.5, 2.5,
            min(12.0, principal_limit_px),
            min(12.0, principal_limit_px),
        ], dtype=np.float64)
    lower, upper = anchor - spread, anchor + spread
    lower[1], upper[1] = max(lower[1], -85.0), min(upper[1], 85.0)
    lower[3], upper[3] = max(lower[3], radius * 1.05), min(upper[3], radius * 30.0)
    lower[4:6] = np.maximum(lower[4:6], -radius * 2.0)
    upper[4:6] = np.minimum(upper[4:6], radius * 2.0)
    # Never unlock intrinsics, even if a legacy caller passes optimize_fov=True.
    optimize_fov = False
    dims = list(range(6))
    dims = [d for d in dims if upper[d] - lower[d] > 1e-12]
    best = initial
    evaluations = 0
    trace = [{
        'evaluation': 0,
        'loss': float(best['loss']),
        'iou': float(best['diag']['iou']),
        'params': anchor.tolist(),
    }]
    def objective(z):
        nonlocal best, evaluations
        p = anchor.copy()
        p[dims] += np.asarray(z, dtype=np.float64) * spread[dims]
        rec = evaluate(p)
        evaluations += 1
        if rec is not None and np.isfinite(rec['loss']) and rec['loss'] < best['loss']:
            best = rec
            trace.append({
                'evaluation': evaluations,
                'loss': float(best['loss']),
                'iou': float(best['diag']['iou']),
                'params': best['params'].tolist(),
            })
        if evaluations % 100 == 0:
            print(
                f'[{label}] evaluations={evaluations}, '
                f'best_loss={best["loss"]:.7f}, IoU={best["diag"]["iou"]:.6f}',
                flush=True,
            )
        return float(rec['loss']) if rec is not None and np.isfinite(rec['loss']) else 1e6
    bounds = [
        ((lower[d] - anchor[d]) / spread[d],
         (upper[d] - anchor[d]) / spread[d])
        for d in dims
    ]
    result = minimize(
        objective,
        np.zeros(len(dims), dtype=np.float64),
        method='Powell',
        bounds=bounds,
        options={
            'maxfev': max(30, int(maxfev)),
            'maxiter': 160 if fine else 100,
            'xtol': 2e-5 if fine else 5e-4,
            'ftol': 1e-8 if fine else 1e-6,
        },
    )
    return best, {
        'method': 'bounded_normalized_Powell_extrinsics_only',
        'label': label,
        'evaluations': evaluations,
        'optimizer_success': bool(result.success),
        'stop_reason': str(result.message),
        'intrinsics_optimized': bool(optimize_fov),
        'trace': trace,
    }

def differential_evolution_camera_parameters(initial, evaluate, radius, fov_min, fov_max,
                                             principal_limit_px=32.0, maxiter=10,
                                             popsize=5, label='Region-balanced DE'):
    """Small deterministic global refinement around the best Powell result.
    This is deliberately local: the diagnostic-view/Powell stage already resolves
    the gross orientation. DE is used to escape the small Powell basin that is
    visible as several-pixel red/green contour disagreement.
    """
    from scipy.optimize import differential_evolution
    anchor = np.asarray(initial['params'], dtype=np.float64).copy()
    if anchor.size != 10:
        raise ValueError("differential_evolution_camera_parameters expects 10-D params")
    span = np.array([
        5.0, 5.0, 4.0,
        max(anchor[3] * 0.07, radius * 0.08),
        radius * 0.10, radius * 0.10,
        2.0, 2.0,
        min(10.0, principal_limit_px),
        min(10.0, principal_limit_px),
    ], dtype=np.float64)
    lo = anchor - span
    hi = anchor + span
    lo[1], hi[1] = max(lo[1], -85.0), min(hi[1], 85.0)
    lo[3], hi[3] = max(lo[3], radius * 1.05), min(hi[3], radius * 30.0)
    lo[4:6] = np.maximum(lo[4:6], -radius * 2.0)
    hi[4:6] = np.minimum(hi[4:6], radius * 2.0)
    best = initial
    evaluations = 0
    trace = []
    def objective(p):
        nonlocal best, evaluations
        trial = anchor.copy()
        trial[:6] = np.asarray(p, dtype=np.float64)
        rec = evaluate(trial)
        evaluations += 1
        if rec is not None and np.isfinite(rec['loss']) and rec['loss'] < best['loss']:
            best = rec
            trace.append({
                'evaluation': evaluations,
                'loss': float(best['loss']),
                'iou': float(best['diag']['iou']),
                'params': best['params'].tolist(),
            })
        if evaluations % 100 == 0:
            print(
                f'[{label}] evaluations={evaluations}, '
                f'best_loss={best["loss"]:.7f}, IoU={best["diag"]["iou"]:.6f}',
                flush=True,
            )
        return float(rec['loss']) if rec is not None and np.isfinite(rec['loss']) else 1e6
    result = differential_evolution(
        objective,
        bounds=list(zip(lo[:6].tolist(), hi[:6].tolist())),
        maxiter=max(1, int(maxiter)),
        popsize=max(3, int(popsize)),
        tol=1e-5,
        atol=1e-6,
        seed=0,
        polish=False,
        workers=1,
        updating='immediate',
    )
    return best, {
        'method': 'local_differential_evolution_extrinsics_only',
        'intrinsics_optimized': False,
        'label': label,
        'evaluations': evaluations,
        'optimizer_success': bool(result.success),
        'stop_reason': str(result.message),
        'trace': trace,
    }

def soft_geometry_loss(alpha, mask):
    """Continuous silhouette overlap and first/second spatial moments."""
    a=alpha.clamp(0,1); m=mask.float().clamp(0,1)
    eps=1e-6
    dice=1-(2*(a*m).sum()+eps)/(a.sum()+m.sum()+eps)
    h,w=m.shape
    ys=torch.linspace(0,1,h,device=a.device,dtype=a.dtype)[:,None]
    xs=torch.linspace(0,1,w,device=a.device,dtype=a.dtype)[None,:]
    def moments(v):
        mass=v.sum().clamp_min(eps)
        cx=(v*xs).sum()/mass;cy=(v*ys).sum()/mass
        sx=((v*(xs-cx)**2).sum()/mass+eps).sqrt()
        sy=((v*(ys-cy)**2).sum()/mass+eps).sqrt()
        return cx,cy,sx,sy
    acx,acy,asx,asy=moments(a);mcx,mcy,msx,msy=moments(m)
    center=(acx-mcx).abs()+(acy-mcy).abs()
    scale=(torch.log(asx/msx)).abs()+(torch.log(asy/msy)).abs()
    l1=(a-m).abs().mean()
    total=dice+.35*center+.2*scale+.2*l1
    return float(total), {'iou':mask_iou(a>.05,m>.5),'soft_dice_loss':float(dice),
                         'center':float(center),'scale':float(scale),'mask_l1':float(l1)}

def strict_silhouette_loss(alpha, mask, target_features, alpha_threshold=0.05,
                           iou_weight=24.0, chamfer_weight=0.18,
                           p95_weight=0.035, component_weight=0.35,
                           area_weight=0.25):
    """Pixel-level structure loss.
    The previous contour term normalized pixel error by the whole object
    diagonal, so several-pixel misses were cheap. This version keeps contour
    distance in pixels and puts hard-mask IoU first.
    """
    rb = (alpha > float(alpha_threshold)).detach().cpu().numpy().astype(np.uint8)
    tb = (mask > 0.5).detach().cpu().numpy().astype(np.uint8)
    intersection = np.count_nonzero((rb > 0) & (tb > 0))
    union = np.count_nonzero((rb > 0) | (tb > 0))
    iou = float(intersection / max(union, 1))
    rf = mask_shape_features(rb)
    tf = target_features
    if not rf['edge'].any() or not tf['edge'].any():
        return 1e3, {
            'iou': 0.0,
            'boundary_mean_px': 1e3,
            'boundary_p95_px': 1e3,
            'component': 1e3,
            'area_log_error': 1e3,
        }
    fwd = tf['distance'][rf['edge']]
    bwd = rf['distance'][tf['edge']]
    boundary_mean = 0.5 * (float(fwd.mean()) + float(bwd.mean()))
    boundary_p95 = 0.5 * (
        float(np.quantile(fwd, 0.95)) + float(np.quantile(bwd, 0.95))
    )
    _, component = detailed_shape_loss(rb.astype(bool), tf)
    ar = max(float(rb.mean()), 1e-8)
    at = max(float(tb.mean()), 1e-8)
    area_log_error = abs(math.log(ar / at))
    total = (
        float(iou_weight) * (1.0 - iou)
        + float(chamfer_weight) * boundary_mean
        + float(p95_weight) * boundary_p95
        + float(component_weight) * component
        + float(area_weight) * area_log_error
    )
    return float(total), {
        'iou': iou,
        'boundary_mean_px': boundary_mean,
        'boundary_p95_px': boundary_p95,
        'component': float(component),
        'area_log_error': float(area_log_error),
    }
@torch.no_grad()
def main(
    config_path,
    g0_override=None,
    output_override=None,
    prior_override=None,
    camera_output_override=None,
    prior_only=False,
):
    global _FIXED_INTRINSICS
    _FIXED_INTRINSICS = None
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError('This version requires scipy: python -m pip install scipy') from exc
    cfg = configparser.ConfigParser()
    if not cfg.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"Cannot read config: {config_path}")
    if g0_override is None:
        g0_path = resolve_input_path(cfg, config_path, "initial_gaussian_ply")
    else:
        g0_path = Path(os.path.expanduser(str(g0_override)))
        if not g0_path.is_absolute():
            g0_path = Path(config_path).resolve().parent / g0_path
        g0_path = g0_path.resolve()

    image_path = resolve_input_path(cfg, config_path, "t0_image_path")

    if output_override is None:
        output_dir = resolve_output_dir(cfg, config_path)
    else:
        output_dir = Path(os.path.expanduser(str(output_override)))
        if not output_dir.is_absolute():
            output_dir = Path(config_path).resolve().parent / output_dir
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not g0_path.is_file():
        raise FileNotFoundError(f"G0 does not exist: {g0_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"t0 image does not exist: {image_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by E-D3DGS Gaussian renderer")
    device = "cuda"
    target, png_alpha, width, height = load_rgb(image_path, device)
    if width != 640:
        raise ValueError(f'This version uses a fixed 640px image width; got {width}x{height}. '
                         'Use the matching 640px t0 image and mask.')
    print(f'[resolution] Fixed 640px width ({width}x{height}); no 160/320px stages.')
    _FIXED_INTRINSICS = load_fixed_intrinsics(cfg, config_path, width, height)
    sh_degree = cfg_get(cfg, "Train", "sh_degree", int, 3)
    white_background = cfg_get(cfg, "RGB Data Preprocessing", "white_background", bool, True)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    background_info = {'method': 'configured', 'rgb': background.cpu().tolist()}
    if cfg_get(cfg, 'Camera Alignment', 'camera_align_match_background', bool, True):
        background, background_info = estimate_border_background(target, background)
    print('[background]', background_info, flush=True)
    pipe = SimpleNamespace(
        convert_SHs_python=cfg_get(cfg, "Other", "convert_shs_python", bool, False),
        compute_cov3D_python=cfg_get(cfg, "Other", "compute_cov3d_python", bool, False),
        debug=cfg_get(cfg, "Other", "debug", bool, False),
        is_6dof=False,
    )
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(g0_path))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        value = getattr(gaussians, name, None)
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)
    diagnostics = gaussian_diagnostics(gaussians)
    center, radius, r50, r90, r98 = robust_geometry(gaussians.get_xyz)
    diagnostics["robust_center"] = center.tolist()
    diagnostics["radius_q50_q90_q98"] = [r50, r90, r98]
    with (output_dir / "g0_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"G0: {g0_path}")
    print(f"t0 image: {image_path}")
    print(f"Image size: {width}x{height}")
    print(f"G0 points: {diagnostics['num_gaussians']}")
    print(f"Robust center: {center.tolist()}")
    print(f"Robust radius q98: {radius:.6f}")
    print(f"Opacity q01/q50/q99: {diagnostics['opacity_quantiles_01_50_99']}")
    print(f"Scale q01/q50/q99: {diagnostics['scale_quantiles_01_50_99']}")
    target_mask, target_mask_source, target_bg_info = build_target_mask(
        target, png_alpha, cfg, config_path, width, height
    )
    save_mask(target_mask, output_dir / "target_mask.png")
    save_rgba(target, target_mask, output_dir / "target_foreground_rgba.png")
    save_rgb(target * target_mask[None], output_dir / "target_foreground_rgb.png")
    save_rgb(target * (1.0 - target_mask[None]), output_dir / "target_background_rgb.png")
    print(f"[target mask] source={target_mask_source}")
    target_fraction = target_bbox_fraction(target_mask)
    print(f"Target foreground ratio: {float(target_mask.mean().item()):.6f}")
    print(f"Target bbox fraction: {target_fraction:.6f}")
    # The original 640px image is used throughout.
    init_fov = fixed_intrinsics_at_size(width, height)[4]
    # Kept only for helper bounds; the calibrated intrinsics themselves are fixed.
    fov_min, fov_max = 0.0, 180.0

    if prior_only:
        diagnostic_records = []
        print(
            "[1/5] PRIOR-ONLY mode: skip six diagnostic views. "
            "Round 2 will start directly from the Round-1 best camera."
        )
        print(
            "[2/5] The Round-1 fixed_camera.json is the ONLY Round-2 "
            "alignment start pose."
        )
    else:
        print("[1/5] Rendering the six diagnostic views...")
        diagnostic_records = diagnostic_views(
            gaussians,
            pipe,
            background,
            target,
            center,
            radius,
            target_fraction,
            init_fov,
            output_dir,
            device,
        )

        print(
            "[2/5] The six diagnostic views are the ONLY global start views "
            "for all subsequent alignment."
        )

    if cfg_get(cfg, 'Camera Alignment', 'camera_align_allow_mirror', bool, False):
        print('[config] Legacy image mirroring is disabled in this version.')

    def option(key, cast, default):
        return cfg_get(cfg, 'Camera Alignment', key, cast, default)

    # v10: region-balanced camera objective.
    region_weights = {
        'global_iou': option('region_global_iou_weight', float, 0.50),
        'region_iou': option('region_equal_iou_weight', float, 1.50),
        'region_missing': option('region_missing_weight', float, 2.00),
        'global_mask_ssim': option('global_mask_ssim_weight', float, 2.00),
        'region_ssim': option('region_ssim_weight', float, 2.00),
        'worst_region_ssim': option('region_worst_ssim_weight', float, 1.00),
        'pose_prior': option('region_pose_prior_weight', float, 0.05),
    }
    region_rows = max(1, option('region_grid_rows', int, 8))
    region_cols = max(1, option('region_grid_cols', int, 8))
    region_min_fg = max(1, option('region_min_foreground_pixels', int, 20))
    region_ssim_min_pixels = max(
        1, option('region_ssim_min_pixels', int, 8)
    )
    region_ssim_mask_erode = max(
        0, option('region_ssim_mask_erode', int, 1)
    )
    region_ssim_window = max(
        3, option('region_ssim_window', int, 7)
    )
    if region_ssim_window % 2 == 0:
        region_ssim_window += 1

    # Region-balanced silhouette remains the primary camera objective.
    # Appearance terms are now foreground-only and may be enabled as weak tie-breakers.
    weights = {
        'soft_geometry': option('structure_soft_geometry_weight', float, 0.0),
        'edge': option('structure_internal_edge_weight', float, 0.0),
        'rgb': option('structure_rgb_weight', float, 0.0),
        'ssim': option('structure_ssim_weight', float, 0.0),
        'texture': option('structure_texture_weight', float, 0.0),
    }
    strict_weights = {
        'iou_weight': option('strict_iou_weight', float, 24.0),
        'chamfer_weight': option('strict_chamfer_weight', float, 0.18),
        'p95_weight': option('strict_p95_weight', float, 0.035),
        'component_weight': option('strict_component_weight', float, 0.35),
        'area_weight': option('strict_area_weight', float, 0.25),
    }
    alpha_threshold = option('structure_alpha_threshold', float, 0.05)
    principal_limit_px = 0.0  # Legacy principal-point search settings are ignored.
    prior_params, prior_source = load_camera_prior_params(
        cfg, config_path, g0_path, width, height, explicit_prior_path=prior_override
    )
    if prior_params is not None:
        prior_params = upgrade_camera_params(prior_params, width, height)
        print(f'[camera prior] using {prior_source}')
        print(f'[camera prior] params={prior_params.tolist()}')
    else:
        print('[camera prior] none; pose-prior term is disabled')

    if prior_only and prior_params is None:
        raise RuntimeError(
            "--prior-only requires a valid --camera-prior fixed_camera.json. "
            "Round 2 must start from the Round-1 best external parameters."
        )
    if not (0.0 < alpha_threshold < 1.0):
        raise ValueError('structure_alpha_threshold must be in (0,1)')
    if min(weights.values()) < 0 or min(strict_weights.values()) < 0 or min(region_weights.values()) < 0:
        raise ValueError('Alignment weights must be nonnegative')
    print('[region-balanced weights]', region_weights)
    print(f'[region grid] {region_rows}x{region_cols}, min GT foreground={region_min_fg}px/cell')
    print(
        f'[region SSIM] window={region_ssim_window}, '
        f'min pixels={region_ssim_min_pixels}, '
        f'mask erode={region_ssim_mask_erode}px'
    )
    print('[diagnostic-only weights]', weights)
    print('[strict silhouette diagnostic weights]', strict_weights)
    print('[intrinsics] JSON values locked; all optimization stages use 6-DoF extrinsics only')

    target_regions = build_balanced_region_cells(
        target_mask, rows=region_rows, cols=region_cols,
        min_foreground_pixels=region_min_fg
    )
    if not target_regions:
        raise RuntimeError('No valid region-balanced cells could be built from target mask')
    save_region_grid_preview(target, target_regions, output_dir / 'region_grid_target.png')
    with (output_dir / 'region_grid.json').open('w', encoding='utf-8') as f:
        json.dump({
            'rows': region_rows, 'cols': region_cols,
            'min_foreground_pixels': region_min_fg,
            'active_regions': target_regions,
        }, f, indent=2)
    print(f'[region grid] active cells={len(target_regions)}')

    cache = {}
    target_shape_cache = {}
    edge_cache = {}
    region_cache = {tuple(target_mask.shape): target_regions}
    def bound(p):
        p = upgrade_camera_params(p, width, height)
        p[0] %= 360.0
        p[1] = np.clip(p[1], -85.0, 85.0)
        p[2] = (p[2] + 180.0) % 360.0 - 180.0
        p[3] = np.clip(p[3], radius * 1.05, radius * 30.0)
        p[4:6] = np.clip(p[4:6], -radius * 2.0, radius * 2.0)
        return p
    def score(p, img, mask):
        p = bound(p)
        key = (tuple(img.shape), tuple(np.round(p, 7)))
        if key in cache:
            return cache[key]
        _, _, pkg, alpha, visible = render_view(
            gaussians, pipe, background, img, center, p, device
        )
        if visible == 0 or not torch.isfinite(pkg['render']).all():
            return None

        shape_key = tuple(mask.shape)
        if shape_key not in target_shape_cache:
            target_shape_cache[shape_key] = mask_shape_features(
                (mask > 0.5).cpu().numpy()
            )
        if shape_key not in region_cache:
            scale = (mask.shape[0] * mask.shape[1]) / float(height * width)
            min_fg_here = max(1, int(round(region_min_fg * scale)))
            region_cache[shape_key] = build_balanced_region_cells(
                mask, rows=region_rows, cols=region_cols,
                min_foreground_pixels=min_fg_here
            )

        # Strict silhouette is computed for diagnostics, but no longer drives selection.
        strict_loss, strict_diag = strict_silhouette_loss(
            alpha, mask, target_shape_cache[shape_key],
            alpha_threshold=alpha_threshold, **strict_weights
        )
        global_iou_loss = 1.0 - float(strict_diag['iou'])
        region_iou_loss, region_missing_loss, region_diag = region_balanced_silhouette_loss(
            alpha, mask, region_cache[shape_key], alpha_threshold=alpha_threshold
        )

        global_mask_ssim_loss, global_mask_ssim_diag = global_masked_ssim_loss(
            pkg['render'],
            img,
            mask,
            render_alpha=alpha,
            background=background,
            cfg=cfg,
        )

        (
            region_ssim_loss,
            worst_region_ssim_loss,
            region_ssim_diag,
        ) = region_balanced_ssim_loss(
            pkg['render'],
            img,
            alpha,
            mask,
            region_cache[shape_key],
            background,
            cfg=cfg,
        )
        prior_loss = camera_prior_loss(p, prior_params, radius) if prior_params is not None else 0.0

        # Optional diagnostics/tie terms. Defaults are zero.
        soft_geo, soft_diag = soft_geometry_loss(alpha, mask)
        app = appearance_score(
            pkg['render'], img, mask,
            render_alpha=alpha, background=background, cfg=cfg
        )
        app['edge'] = 0.0
        if weights['edge'] > 0:
            if shape_key not in edge_cache:
                edge_cache[shape_key] = internal_edge_features(
                    img, (mask > 0.5).cpu().numpy()
                )
            app['edge'] = internal_edge_loss(
                internal_edge_features(
                    pkg['render'], (alpha > alpha_threshold).cpu().numpy()
                ),
                edge_cache[shape_key],
            )

        total = (
            region_weights['global_iou'] * global_iou_loss
            + region_weights['region_iou'] * region_iou_loss
            + region_weights['region_missing'] * region_missing_loss
            + region_weights['global_mask_ssim'] * global_mask_ssim_loss
            + region_weights['region_ssim'] * region_ssim_loss
            + region_weights['worst_region_ssim'] * worst_region_ssim_loss
            + region_weights['pose_prior'] * prior_loss
            + weights['soft_geometry'] * soft_geo
            + weights['edge'] * app['edge']
            + weights['rgb'] * app['rgb']
            + weights['ssim'] * app['ssim']
            + weights['texture'] * app['texture']
        )
        if not np.isfinite(total):
            return None
        diag = dict(soft_diag)
        diag.update({
            'iou': float(strict_diag['iou']),
            'global_iou_loss': float(global_iou_loss),
            'region_iou_loss': float(region_iou_loss),
            'region_missing_loss': float(region_missing_loss),
            'global_mask_ssim_loss': float(global_mask_ssim_loss),
            'global_mask_ssim': float(
                global_mask_ssim_diag['global_mask_ssim']
            ),
            'region_ssim_loss': float(region_ssim_loss),
            'worst_region_ssim_loss': float(worst_region_ssim_loss),
            'pose_prior_loss': float(prior_loss),
            'mean_region_iou': float(region_diag['mean_region_iou']),
            'worst_region_iou': float(region_diag['worst_region_iou']),
            'mean_region_recall': float(region_diag['mean_region_recall']),
            'worst_region_recall': float(region_diag['worst_region_recall']),
            'mean_region_ssim': float(region_ssim_diag['mean_region_ssim']),
            'worst_region_ssim': float(region_ssim_diag['worst_region_ssim']),
            'active_regions': int(region_diag['active_regions']),
            'active_ssim_regions': int(region_ssim_diag['active_ssim_regions']),
        })
        rec = dict(
            loss=float(total), strict_loss=float(strict_loss), geo_loss=float(soft_geo),
            global_iou_loss=float(global_iou_loss),
            region_iou_loss=float(region_iou_loss),
            region_missing_loss=float(region_missing_loss),
            global_mask_ssim_loss=float(global_mask_ssim_loss),
            global_mask_ssim_diag=global_mask_ssim_diag,
            region_ssim_loss=float(region_ssim_loss),
            worst_region_ssim_loss=float(worst_region_ssim_loss),
            pose_prior_loss=float(prior_loss),
            region_diag=region_diag,
            region_ssim_diag=region_ssim_diag,
            params=p, diag=diag, strict_diag=strict_diag,
            visible=visible, image_mirror=False, **app,
        )
        cache[key] = rec
        return rec
    # ------------------------------------------------------------------
    # Global / initial alignment stage
    #
    # Round 1:
    #   six diagnostic views -> six Powell runs -> best diagnostic view
    #
    # Round 2 (--prior-only):
    #   Round-1 fixed_camera.json -> ONE Powell run -> continue polishing
    #
    # There is no new diagnostic/global direction search in Round 2.
    # ------------------------------------------------------------------
    budget = max(60, option('camera_align_powell_maxfev', int, 500))
    powell_runs = []
    initial_alignment_record = None  # pre-Powell initial candidate, immutable baseline


    if prior_only:
        # --------------------------------------------------------------
        # ROUND 2: the Round-1 best external parameters are the ONLY seed.
        # --------------------------------------------------------------
        prior_seed = score(prior_params, target, target_mask)
        if prior_seed is None:
            raise RuntimeError(
                "Round-1 best camera could not render/score against the "
                "Round-1 refined G0."
            )

        prior_seed = dict(prior_seed)
        prior_seed["diagnostic_view"] = "round1_best_external"
        initial_alignment_record = dict(prior_seed)


        with (output_dir / "round2_prior_initial_score.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "mode": "round1_best_external_only",
                    "source_camera": prior_source,
                    "initial_loss": float(prior_seed["loss"]),
                    "global_iou": float(prior_seed["diag"]["iou"]),
                    "mean_region_iou": float(
                        prior_seed["diag"]["mean_region_iou"]
                    ),
                    "mean_region_recall": float(
                        prior_seed["diag"]["mean_region_recall"]
                    ),
                    "worst_region_recall": float(
                        prior_seed["diag"]["worst_region_recall"]
                    ),
                    "mean_region_ssim": float(
                        prior_seed["diag"]["mean_region_ssim"]
                    ),
                    "worst_region_ssim": float(
                        prior_seed["diag"]["worst_region_ssim"]
                    ),
                    "params": prior_seed["params"].tolist(),
                },
                f,
                indent=2,
                allow_nan=False,
            )

        print(
            "[3/5] Round-2 direct Powell from Round-1 best external "
            "parameters only..."
        )

        selected, report = powell_camera_parameters(
            prior_seed,
            lambda p: score(p, target, target_mask),
            radius,
            fov_min,
            fov_max,
            maxfev=budget,
            optimize_fov=False,
            fine=False,
            label="Round-2 Powell from Round-1 best camera",
            principal_limit_px=principal_limit_px,
        )
        selected = dict(selected)
        selected["diagnostic_view"] = "round1_best_external"

        report["width"] = width
        report["height"] = height
        report["start_mode"] = "round1_best_external_only"
        report["source_camera"] = prior_source
        powell_runs.append(report)

        selected_diagnostic_view = "round1_best_external"

        print(
            f"[4/5] Round-2 prior result: "
            f"loss={selected['loss']:.7f}, "
            f"globalIoU={selected['diag']['iou']:.5f}, "
            f"meanRegionIoU={selected['diag']['mean_region_iou']:.5f}, "
            f"globalMaskSSIM={selected['diag']['global_mask_ssim']:.5f}, "
            f"meanRegionSSIM={selected['diag']['mean_region_ssim']:.5f}, "
            f"worstRegionSSIM={selected['diag']['worst_region_ssim']:.5f}, "
            f"worstRecall={selected['diag']['worst_region_recall']:.5f}",
            flush=True,
        )

        with (output_dir / "round2_prior_optimized.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "mode": "round1_best_external_only",
                    "source_camera": prior_source,
                    "optimized_loss": float(selected["loss"]),
                    "global_iou": float(selected["diag"]["iou"]),
                    "mean_region_iou": float(
                        selected["diag"]["mean_region_iou"]
                    ),
                    "mean_region_recall": float(
                        selected["diag"]["mean_region_recall"]
                    ),
                    "worst_region_recall": float(
                        selected["diag"]["worst_region_recall"]
                    ),
                    "mean_region_ssim": float(
                        selected["diag"]["mean_region_ssim"]
                    ),
                    "worst_region_ssim": float(
                        selected["diag"]["worst_region_ssim"]
                    ),
                    "params": selected["params"].tolist(),
                },
                f,
                indent=2,
                allow_nan=False,
            )

        with (output_dir / "camera_powell_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "method": (
                        "round2_round1_best_external_direct_"
                        "fixed_640px_region_balanced_Powell"
                    ),
                    "global_view_policy": "no new view search",
                    "source_camera": prior_source,
                    "runs": powell_runs,
                },
                f,
                indent=2,
                allow_nan=False,
            )

    else:
        # --------------------------------------------------------------
        # ROUND 1: score and optimize ONLY the six diagnostic views.
        # --------------------------------------------------------------
        initial_diagnostic_scores = []
        valid_diagnostic_runs = []

        for view_index, view in enumerate(diagnostic_records, start=1):
            if not bool(view.get("valid", True)):
                print(
                    f"[diagnostic view] {view['name']} skipped: "
                    "no visible support.",
                    flush=True,
                )
                continue

            rec = score(view["params"], target, target_mask)
            if rec is None:
                print(
                    f"[diagnostic view] {view['name']} skipped: "
                    "invalid alignment score.",
                    flush=True,
                )
                continue

            rec = dict(rec)
            rec["diagnostic_view"] = view["name"]

            initial_diagnostic_scores.append({
                "view": view["name"],
                "loss": float(rec["loss"]),
                "global_iou": float(rec["diag"]["iou"]),
                "mean_region_iou": float(rec["diag"]["mean_region_iou"]),
                "mean_region_recall": float(
                    rec["diag"]["mean_region_recall"]
                ),
                "worst_region_recall": float(
                    rec["diag"]["worst_region_recall"]
                ),
                "mean_region_ssim": float(
                    rec["diag"]["mean_region_ssim"]
                ),
                "worst_region_ssim": float(
                    rec["diag"]["worst_region_ssim"]
                ),
                "params": rec["params"].tolist(),
            })

            valid_diagnostic_runs.append(rec)

            print(
                f"[diagnostic score] {view['name']:6s} "
                f"loss={rec['loss']:.7f}, "
                f"globalIoU={rec['diag']['iou']:.5f}, "
                f"meanRegionIoU={rec['diag']['mean_region_iou']:.5f}, "
                f"globalMaskSSIM={rec['diag']['global_mask_ssim']:.5f}, "
                f"meanRegionSSIM={rec['diag']['mean_region_ssim']:.5f}, "
                f"worstRegionSSIM={rec['diag']['worst_region_ssim']:.5f}, "
                f"worstRecall={rec['diag']['worst_region_recall']:.5f}",
                flush=True,
            )

        if not valid_diagnostic_runs:
            raise RuntimeError(
                "None of the six diagnostic views produced a valid "
                "alignment score."
            )

        # Before any Powell pass, freeze the best of the six scored raw views.
        initial_alignment_record = dict(min(valid_diagnostic_runs, key=lambda r: r["loss"]))


        with (output_dir / "diagnostic_view_initial_scores.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "policy": "diagnostic_views_only",
                    "views": initial_diagnostic_scores,
                },
                f,
                indent=2,
                allow_nan=False,
            )

        print(
            f"[3/5] Powell optimization from all "
            f"{len(valid_diagnostic_runs)} valid diagnostic views..."
        )

        optimized_diagnostic_runs = []

        for run_index, seed in enumerate(
            valid_diagnostic_runs, start=1
        ):
            view_name = seed["diagnostic_view"]
            label = (
                f"Powell diagnostic {view_name} "
                f"({run_index}/{len(valid_diagnostic_runs)})"
            )

            optimized, report = powell_camera_parameters(
                seed,
                lambda p: score(p, target, target_mask),
                radius,
                fov_min,
                fov_max,
                maxfev=budget,
                optimize_fov=False,
                fine=False,
                label=label,
                principal_limit_px=principal_limit_px,
            )

            optimized = dict(optimized)
            optimized["diagnostic_view"] = view_name
            optimized_diagnostic_runs.append(optimized)

            report["width"] = width
            report["height"] = height
            report["diagnostic_view"] = view_name
            powell_runs.append(report)

            print(
                f"[{label}] loss={optimized['loss']:.7f}, "
                f"globalIoU={optimized['diag']['iou']:.5f}, "
                f"meanRegionIoU="
                f"{optimized['diag']['mean_region_iou']:.5f}, "
                f"globalMaskSSIM="
                f"{optimized['diag']['global_mask_ssim']:.5f}, "
                f"meanRegionSSIM="
                f"{optimized['diag']['mean_region_ssim']:.5f}, "
                f"worstRegionSSIM="
                f"{optimized['diag']['worst_region_ssim']:.5f}, "
                f"worstRecall="
                f"{optimized['diag']['worst_region_recall']:.5f}",
                flush=True,
            )

        if not optimized_diagnostic_runs:
            raise RuntimeError(
                "All diagnostic-view Powell optimizations failed."
            )

        print("[4/5] Ranking optimized diagnostic views...")

        ranked_diagnostic_runs = sorted(
            optimized_diagnostic_runs,
            key=lambda r: (
                float(r["loss"]),
                -float(r["diag"]["worst_region_ssim"]),
                -float(r["diag"]["worst_region_recall"]),
                -float(r["diag"]["mean_region_ssim"]),
                -float(r["diag"]["mean_region_iou"]),
                -float(r["diag"]["iou"]),
                float(r["strict_diag"]["boundary_mean_px"]),
            ),
        )

        diagnostic_ranking = []
        for rank, rec in enumerate(
            ranked_diagnostic_runs, start=1
        ):
            diagnostic_ranking.append({
                "rank": int(rank),
                "diagnostic_view": rec["diagnostic_view"],
                "full_resolution_loss": float(rec["loss"]),
                "global_iou": float(rec["diag"]["iou"]),
                "mean_region_iou": float(
                    rec["diag"]["mean_region_iou"]
                ),
                "mean_region_recall": float(
                    rec["diag"]["mean_region_recall"]
                ),
                "worst_region_recall": float(
                    rec["diag"]["worst_region_recall"]
                ),
                "mean_region_ssim": float(
                    rec["diag"]["mean_region_ssim"]
                ),
                "worst_region_ssim": float(
                    rec["diag"]["worst_region_ssim"]
                ),
                "params": rec["params"].tolist(),
            })

            print(
                f"[diagnostic ranking] rank={rank}, "
                f"view={rec['diagnostic_view']}, "
                f"loss={rec['loss']:.7f}, "
                f"globalIoU={rec['diag']['iou']:.5f}, "
                f"meanRegionIoU="
                f"{rec['diag']['mean_region_iou']:.5f}, "
                f"globalMaskSSIM="
                f"{rec['diag']['global_mask_ssim']:.5f}, "
                f"meanRegionSSIM="
                f"{rec['diag']['mean_region_ssim']:.5f}, "
                f"worstRegionSSIM="
                f"{rec['diag']['worst_region_ssim']:.5f}, "
                f"worstRecall="
                f"{rec['diag']['worst_region_recall']:.5f}",
                flush=True,
            )

        with (output_dir / "diagnostic_view_ranking.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "evaluation_width": width,
                    "evaluation_height": height,
                    "selection_metric": (
                        "region_balanced_full_resolution_loss"
                    ),
                    "ranked_diagnostic_views": diagnostic_ranking,
                },
                f,
                indent=2,
                allow_nan=False,
            )

        with (output_dir / "camera_powell_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "method": (
                        "diagnostic_views_only_fixed_640px_"
                        "region_balanced_Powell"
                    ),
                    "global_view_policy": (
                        "front/right/back/left/top/bottom only"
                    ),
                    "runs": powell_runs,
                },
                f,
                indent=2,
                allow_nan=False,
            )

        selected = ranked_diagnostic_runs[0]
        selected_diagnostic_view = selected["diagnostic_view"]

        print(
            f"[diagnostic winner] {selected_diagnostic_view}, "
            f"loss={selected['loss']:.7f}",
            flush=True,
        )

    if prior_only:
        print(
            "[5/5] Region-balanced local polishing from the "
            "Round-1 best external parameters..."
        )
    else:
        print(
            "[5/5] Region-balanced polishing from diagnostic winner..."
        )
    before_selected = selected
    _, _, before_package, before_alpha, _ = render_view(
        gaussians, pipe, background, target, center, selected['params'], device
    )
    before_render = before_package['render']
    before_metrics = evaluate_metrics(
        before_render, target, before_alpha, target_mask,
        background=background, cfg=cfg
    )
    save_rgb(before_render, output_dir / 'before_camera_polish.png')
    restart_records = [selected]
    anchor = selected['params'].copy()
    angle = option('structure_restart_angle_deg', float, 5.0)
    for da, de in (
        (-angle, 0), (angle, 0), (0, -angle), (0, angle),
        (-angle, -angle), (-angle, angle), (angle, -angle), (angle, angle)
    ):
        p = anchor.copy()
        p[0] += da
        p[1] += de
        rec = score(p, target, target_mask)
        if rec is not None:
            restart_records.append(rec)
    others = sorted(restart_records[1:], key=lambda r: r['loss'])
    seeds = [selected] + others[:max(0, option('structure_local_restarts', int, 5) - 1)]
    local_runs = []
    for index, seed in enumerate(seeds):
        rec, report = powell_camera_parameters(
            seed,
            lambda p: score(p, target, target_mask),
            radius,
            fov_min,
            fov_max,
            maxfev=max(80, option('structure_restart_maxfev', int, 700)),
            optimize_fov=False,
            fine=True,
            label=f'Structure restart {index+1}/{len(seeds)}',
            principal_limit_px=principal_limit_px,
        )
        local_runs.append(report)
        if rec['loss'] < selected['loss']:
            selected = rec
    de_report = None
    if option('camera_align_use_de_refine', bool, True):
        selected, de_report = differential_evolution_camera_parameters(
            selected,
            lambda p: score(p, target, target_mask),
            radius,
            fov_min,
            fov_max,
            principal_limit_px=principal_limit_px,
            maxiter=option('camera_align_de_maxiter', int, 10),
            popsize=option('camera_align_de_popsize', int, 5),
            label='Region-balanced DE',
        )
    selected, polish_report = powell_camera_parameters(
        selected,
        lambda p: score(p, target, target_mask),
        radius,
        fov_min,
        fov_max,
        maxfev=max(100, option('camera_align_powell_final_maxfev', int, 1800)),
        optimize_fov=False,
        fine=True,
        label='Final 6-DoF fixed-intrinsics Powell polish',
        principal_limit_px=principal_limit_px,
    )
    # Keep the refinement on the trajectory that started from the selected
    # diagnostic view. Do not jump to an unrelated cached pose from another
    # diagnostic-view optimization, because that would lose view provenance.
    polish_report['nearby_pose_runs'] = local_runs
    polish_report['differential_evolution'] = de_report
    polish_report['selected_start_view'] = selected_diagnostic_view
    polish_report['selected_diagnostic_view'] = (
        None if prior_only else selected_diagnostic_view
    )
    polish_report['alignment_start_mode'] = (
        'round1_best_external_only'
        if prior_only else 'diagnostic_views'
    )
    polish_report['round1_seed_camera'] = (
        prior_source if prior_only else None
    )
    polish_report['comparison_resolution'] = [width, height]
    polish_report['before_loss'] = float(before_selected['loss'])
    polish_report['after_loss'] = float(selected['loss'])
    polish_report['loss_reduction'] = float(before_selected['loss'] - selected['loss'])
    polish_report['before_params'] = before_selected['params'].tolist()
    polish_report['after_params'] = selected['params'].tolist()
    polish_report['parameter_order'] = [
        'azimuth_deg', 'elevation_deg', 'roll_deg',
        'distance', 'target_offset_x', 'target_offset_y',
        'fov_x_deg', 'fov_y_deg', 'cx_delta_px', 'cy_delta_px'
    ]
    polish_report['intrinsics_optimized'] = False
    polish_report['fixed_intrinsics'] = dict(_FIXED_INTRINSICS)
    polish_report['optimized_parameter_indices'] = list(range(6))
    polish_report['before_metrics'] = before_metrics
    polish_report['strict_silhouette_diagnostics'] = selected['strict_diag']
    polish_report['region_balanced_diagnostics'] = selected['region_diag']
    polish_report['global_mask_ssim_diagnostics'] = (
        selected['global_mask_ssim_diag']
    )
    polish_report['region_ssim_diagnostics'] = selected['region_ssim_diag']
    print(
        f"[polish result] loss {before_selected['loss']:.7f} -> {selected['loss']:.7f}, "
        f"globalIoU {before_selected['diag']['iou']:.6f} -> {selected['diag']['iou']:.6f}, "
        f"meanRegionIoU {before_selected['diag']['mean_region_iou']:.6f} -> "
        f"{selected['diag']['mean_region_iou']:.6f}, "
        f"meanRegionSSIM {before_selected['diag']['mean_region_ssim']:.6f} -> "
        f"{selected['diag']['mean_region_ssim']:.6f}, "
        f"worstRegionSSIM {before_selected['diag']['worst_region_ssim']:.6f} -> "
        f"{selected['diag']['worst_region_ssim']:.6f}, "
        f"worstRecall {before_selected['diag']['worst_region_recall']:.6f} -> "
        f"{selected['diag']['worst_region_recall']:.6f}"
    )
    params = selected['params'].copy()
    image_mirror = False
    normal_loss = float(selected['loss'])
    mirror_loss = None
    mirror_improvement = float('-inf')
    mirror_min_improvement = 0.0
    final_camera, camera_center, final_package, final_alpha, visible_count = render_view(
        gaussians, pipe, background, target, center, params, device,
        image_mirror=image_mirror
    )
    final_render = final_package["render"]
    target_shape=mask_shape_features((target_mask>.5).cpu().numpy())
    final_shape=mask_shape_features((final_alpha>alpha_threshold).cpu().numpy())
    contour_view=np.zeros((height,width,3),dtype=np.uint8)
    contour_view[:,:,0]=target_shape['edge'].astype(np.uint8)*255
    contour_view[:,:,1]=final_shape['edge'].astype(np.uint8)*255
    Image.fromarray(contour_view).save(output_dir/'structure_contour_overlay.png')
    target_edges=internal_edge_features(target,(target_mask>.5).cpu().numpy())
    render_edges=internal_edge_features(final_render,(final_alpha>alpha_threshold).cpu().numpy())
    edge_view=np.zeros((height,width,3),dtype=np.uint8)
    if target_edges: edge_view[:,:,0]=target_edges[0]['edge'].astype(np.uint8)*255
    if render_edges: edge_view[:,:,1]=render_edges[0]['edge'].astype(np.uint8)*255
    Image.fromarray(edge_view).save(output_dir/'structure_edge_overlay.png')
    final_geo_loss, final_geo_diag = geometry_loss(final_alpha, target_mask)
    metrics = evaluate_metrics(
        final_render, target, final_alpha, target_mask,
        background=background, cfg=cfg
    )
    metrics.update({
        'MeanRegionIoU': float(selected['diag']['mean_region_iou']),
        'WorstRegionIoU': float(selected['diag']['worst_region_iou']),
        'MeanRegionRecall': float(selected['diag']['mean_region_recall']),
        'WorstRegionRecall': float(selected['diag']['worst_region_recall']),
        'GlobalMaskedSSIM': float(selected['diag']['global_mask_ssim']),
        'GlobalMaskedSSIMLoss': float(selected['global_mask_ssim_loss']),
        'MeanRegionSSIM': float(selected['diag']['mean_region_ssim']),
        'WorstRegionSSIM': float(selected['diag']['worst_region_ssim']),
        'RegionSSIMLoss': float(selected['region_ssim_loss']),
        'WorstRegionSSIMLoss': float(selected['worst_region_ssim_loss']),
        'RegionBalancedLoss': float(selected['loss']),
    })
    # --------------------------------------------------------------
    # Unified PRE-search versus FINAL pose metrics, measured on the
    # SAME G0, fixed camera intrinsics, image, mask, and background.
    # Round 1 initial = best *unoptimized* of six diagnostic views.
    # Round 2 initial = supplied prior pose BEFORE Powell.
    # (before_metrics above refers only to before LOCAL polishing.)
    # --------------------------------------------------------------
    if initial_alignment_record is None:
        raise RuntimeError("Initial alignment candidate was not captured")
    initial_pose = np.asarray(initial_alignment_record["params"], dtype=np.float64).copy()
    _, _, initial_package, initial_alpha, _ = render_view(
        gaussians, pipe, background, target, center, initial_pose, device
    )
    initial_render = initial_package["render"]
    initial_metrics = evaluate_metrics(
        initial_render, target, initial_alpha, target_mask,
        background=background, cfg=cfg
    )
    initial_diag = initial_alignment_record["diag"]
    initial_metrics.update({
        "MeanRegionIoU": float(initial_diag["mean_region_iou"]),
        "WorstRegionIoU": float(initial_diag["worst_region_iou"]),
        "MeanRegionRecall": float(initial_diag["mean_region_recall"]),
        "WorstRegionRecall": float(initial_diag["worst_region_recall"]),
        "GlobalMaskedSSIM": float(initial_diag["global_mask_ssim"]),
        "MeanRegionSSIM": float(initial_diag["mean_region_ssim"]),
        "WorstRegionSSIM": float(initial_diag["worst_region_ssim"]),
        "RegionBalancedLoss": float(initial_alignment_record["loss"]),
    })
    eval_dir = output_dir / "before_after_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(initial_render, eval_dir / "initial_render.png")
    save_mask(initial_alpha, eval_dir / "initial_alpha.png")
    save_rgb(final_render, eval_dir / "final_render.png")
    save_mask(final_alpha, eval_dir / "final_alpha.png")
    save_rgb(target, eval_dir / "target.png")
    save_mask(target_mask, eval_dir / "target_mask.png")
    metric_delta = {
        key: float(metrics[key]) - float(initial_metrics[key])
        for key in initial_metrics.keys()
        if key in metrics
        and isinstance(initial_metrics[key], (int, float))
        and isinstance(metrics[key], (int, float))
    }
    mode_desc = (
        "prior_pose_before_powell" if prior_only
        else "best_of_six_unoptimized_diagnostic_views"
    )
    for name, payload in (
        ("initial_metrics.json", initial_metrics),
        ("final_metrics.json", metrics),
        ("metrics_comparison.json", {
            "initial_definition": mode_desc,
            "initial_view": initial_alignment_record.get("diagnostic_view", "unknown"),
            "note": "Camera alignment only. Same original G0 and input image; pose changes.",
            "initial_pose": initial_pose.tolist(),
            "final_pose": params.tolist(),
            "initial": initial_metrics,
            "final": metrics,
            "delta_final_minus_initial": metric_delta,
        }),
    ):
        with (eval_dir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
    lines = [
        "MonoEv Camera Alignment: Initial vs Final",
        "Initial definition: " + mode_desc,
        "Initial view: " + str(initial_alignment_record.get("diagnostic_view", "unknown")),
        "Delta = final - initial; PSNR/SSIM/IoU higher better, LPIPS/MAE lower better.",
        "",
        f"{'Metric':<30} {'Initial':>16} {'Final':>16} {'Delta':>16}",
    ]
    for key in initial_metrics:
        if key in metric_delta:
            lines.append(
                f"{key:<30} {initial_metrics[key]:>16.6f} "
                f"{metrics[key]:>16.6f} {metric_delta[key]:>+16.6f}"
            )
    (eval_dir / "metrics_comparison.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[alignment evaluation] Saved initial and final metrics: {eval_dir}", flush=True)

    polish_report['after_metrics'] = metrics
    with (output_dir / 'camera_polish_report.json').open('w', encoding='utf-8') as f:
        json.dump(polish_report, f, indent=2, allow_nan=False)
    save_rgb(torch.cat([target, before_render, final_render], dim=2),
             output_dir / 'camera_polish_comparison.png')
    print(
        f"Final geometry: loss={final_geo_loss:.6f}, "
        f"IoU={final_geo_diag['iou']:.6f}, "
        f"center_err={final_geo_diag['center']:.6f}, "
        f"scale_err={final_geo_diag['scale']:.6f}"
    )
    fx, fy, cx, cy, fov_x, fov_y_deg = camera_intrinsics_from_params(
        width, height, params
    )
    fov_y = math.radians(fov_y_deg)
    fixed_camera = {
        "type": "MonoEv fixed virtual camera",
        "alignment_version": (
            "round2_prior_only_global_mask_region_ssim_640px_fixed_intrinsics_v16"
            if prior_only
            else "diagnostic_views_only_global_mask_region_ssim_640px_fixed_intrinsics_v16"
        ),
        "alignment_start_mode": (
            "round1_best_external_only"
            if prior_only
            else "diagnostic_views"
        ),
        "camera_convention": "3DGS R stored transposed; camera x-right/y-down/z-forward",
        "post_render_transform": "none",
        "background_rgb": background.cpu().tolist(),
        "background_selection": background_info,
        "target_foreground_extraction": {
            "source": target_mask_source,
            "background_separation": target_bg_info,
            "foreground_pixels": int(target_mask.sum().item()),
        },
        "alignment_score": float(selected["loss"]),
        "alignment_weights": {
            "region_balanced": region_weights,
            "diagnostic_optional": weights,
        },
        "strict_silhouette_diagnostic_weights": strict_weights,
        "alpha_threshold": float(alpha_threshold),
        "region_grid": {
            "rows": int(region_rows),
            "cols": int(region_cols),
            "min_foreground_pixels": int(region_min_fg),
            "active_regions": int(selected["region_diag"]["active_regions"]),
            "ssim_min_pixels": int(region_ssim_min_pixels),
            "ssim_mask_erode": int(region_ssim_mask_erode),
            "ssim_window": int(region_ssim_window),
            "active_ssim_regions": int(
                selected["region_ssim_diag"]["active_ssim_regions"]
            ),
        },
        "rough_camera_prior": {
            "enabled": bool(prior_params is not None),
            "source": prior_source,
            "weight": float(region_weights["pose_prior"]),
        },
        "alignment_terms": {
            "global_iou": float(selected["diag"]["iou"]),
            "global_iou_loss": float(selected["global_iou_loss"]),
            "region_iou_loss": float(selected["region_iou_loss"]),
            "region_missing_loss": float(selected["region_missing_loss"]),
            "global_mask_ssim": float(
                selected["diag"]["global_mask_ssim"]
            ),
            "global_mask_ssim_loss": float(
                selected["global_mask_ssim_loss"]
            ),
            "region_ssim_loss": float(selected["region_ssim_loss"]),
            "worst_region_ssim_loss": float(
                selected["worst_region_ssim_loss"]
            ),
            "pose_prior_loss": float(selected["pose_prior_loss"]),
            "mean_region_iou": float(selected["diag"]["mean_region_iou"]),
            "worst_region_iou": float(selected["diag"]["worst_region_iou"]),
            "mean_region_recall": float(selected["diag"]["mean_region_recall"]),
            "worst_region_recall": float(selected["diag"]["worst_region_recall"]),
            "mean_region_ssim": float(selected["diag"]["mean_region_ssim"]),
            "worst_region_ssim": float(selected["diag"]["worst_region_ssim"]),
            "global_mask_ssim_diagnostics": selected[
                "global_mask_ssim_diag"
            ],
            "region_ssim_diagnostics": selected["region_ssim_diag"],
            "strict_diagnostics": selected["strict_diag"],
            "soft_geometry_diagnostic": float(selected["geo_loss"]),
            "edge_diagnostic": float(selected["edge"]),
            "rgb_diagnostic": float(selected["rgb"]),
            "ssim_diagnostic": float(selected["ssim"]),
            "texture_diagnostic": float(selected["texture"]),
        },
        "soft_geometry_loss": float(selected['geo_loss']),
        "selected_start_view": selected_diagnostic_view,
        "selected_diagnostic_view": (
            None if prior_only else selected_diagnostic_view
        ),
        "round1_seed_camera": (
            prior_source if prior_only else None
        ),
        "pre_polish_alignment_score": float(before_selected["loss"]),
        "camera_polish_loss_reduction": polish_report['loss_reduction'],
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "fov_x_deg": fov_x,
        "fov_y_deg": float(fov_y_deg),
        "cx_delta_px": float(params[8]),
        "cy_delta_px": float(params[9]),
        "projection_matrix": final_camera.projection_matrix.detach().cpu().numpy().tolist(),
        "full_proj_transform": final_camera.full_proj_transform.detach().cpu().numpy().tolist(),
        "R": np.asarray(final_camera.R).tolist(),
        "T": np.asarray(final_camera.T).tolist(),
        "camera_center": np.asarray(camera_center).tolist(),
        "target_center": center.tolist(),
        "azimuth_deg": float(params[0]),
        "elevation_deg": float(params[1]),
        "roll_deg": float(params[2]),
        "image_mirror": bool(image_mirror),
        "mirror_axis": "horizontal" if image_mirror else "none",
        "mirror_selection": {
            "normal_refined_loss": normal_loss,
            "mirror_refined_loss": mirror_loss,
            "mirror_improvement": (
                None if not np.isfinite(mirror_improvement)
                else float(mirror_improvement)
            ),
            "minimum_required_improvement": float(mirror_min_improvement),
        },
        "distance": float(params[3]),
        "target_offset_x": float(params[4]),
        "target_offset_y": float(params[5]),
        "visible_gaussians": int(visible_count),
        "metrics": metrics,
        "source_g0": str(g0_path),
        "source_t0_image": str(image_path),
    }
    # Stage-1 pipeline contract: the camera belongs to THIS alignment round.
    # Never write it back into the input G0 directory, otherwise round1/round2
    # overwrite each other and refinement cannot be chained deterministically.
    if camera_output_override is None:
        camera_save_path = output_dir / "fixed_camera.json"
    else:
        camera_save_path = Path(os.path.expanduser(str(camera_output_override)))
        if not camera_save_path.is_absolute():
            camera_save_path = Path(config_path).resolve().parent / camera_save_path
        camera_save_path = camera_save_path.resolve()
    camera_save_path.parent.mkdir(parents=True, exist_ok=True)
    with camera_save_path.open("w", encoding="utf-8") as f:
        json.dump(fixed_camera, f, indent=2)
    # Diagnostics stay in this round's alignment directory.
    with (output_dir / "camera_alignment_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_rgb(final_render, output_dir / "aligned_render.png")
    save_mask(final_alpha, output_dir / "render_alpha.png")
    final_premult, final_foreground_color = decompose_render_foreground(
        final_render, final_alpha, background,
        alpha_min=cfg_get(cfg, "Camera Alignment", "camera_rgb_alpha_min", float, 0.15)
    )
    save_rgba(final_foreground_color, final_alpha, output_dir / "aligned_render_foreground_rgba.png")
    save_rgb(final_premult, output_dir / "aligned_render_foreground_premult.png")
    overlay = 0.5 * final_render + 0.5 * target
    save_rgb(overlay, output_dir / "alignment_overlay.png")
    print(f"Saved camera: {camera_save_path}")
    print(f"Saved metrics: {output_dir / 'camera_alignment_metrics.json'}")
    print(f"Saved render: {output_dir / 'aligned_render.png'}")
    print(f"Saved alpha: {output_dir / 'render_alpha.png'}")
    print(f"Saved overlay: {output_dir / 'alignment_overlay.png'}")
    print(
        f"Final: visible={visible_count}, FOVx/FOVy={fov_x:.4f}/{fov_y_deg:.4f}, "
        f"fx/fy={fx:.3f}/{fy:.3f}, cx/cy={cx:.3f}/{cy:.3f}, "
        f"az/el/roll={params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}, "
        f"mirror={image_mirror}, distance={params[3]:.6f}"
    )
    print(
        f"Region-balanced: globalIoU={selected['diag']['iou']:.6f}, "
        f"meanRegionIoU={selected['diag']['mean_region_iou']:.6f}, "
        f"globalMaskSSIM={selected['diag']['global_mask_ssim']:.6f}, "
        f"meanRegionSSIM={selected['diag']['mean_region_ssim']:.6f}, "
        f"worstRegionSSIM={selected['diag']['worst_region_ssim']:.6f}, "
        f"meanRecall={selected['diag']['mean_region_recall']:.6f}, "
        f"worstRecall={selected['diag']['worst_region_recall']:.6f}, "
        f"activeRegions={selected['diag']['active_regions']}"
    )
    print(
        f"Boundary diagnostics: mean={selected['strict_diag']['boundary_mean_px']:.3f}px, "
        f"p95={selected['strict_diag']['boundary_p95_px']:.3f}px"
    )
    print(
        f"Metrics: MaskIoU={metrics['MaskIoU']:.6f}, "
        f"MeanRegionIoU={metrics['MeanRegionIoU']:.6f}, "
        f"WorstRegionRecall={metrics['WorstRegionRecall']:.6f}, "
        f"ForegroundPSNR={metrics['Foreground_PSNR_dB']:.4f} dB, "
        f"ForegroundSSIM={metrics['Foreground_SSIM_7x7']:.6f}, "
        f"ForegroundMAE={metrics['Foreground_MAE']:.6f}, "
        f"GlobalSSIM={metrics['Global_SSIM']:.6f}, "
        f"GlobalLPIPS={metrics['Global_LPIPS']:.6f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed-intrinsics camera alignment for one Stage-1 round."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--g0", default=None,
        help="Override [Path] initial_gaussian_ply for this round."
    )
    parser.add_argument(
        "--output", default=None,
        help="Override [Camera Alignment] camera_alignment_output for this round."
    )
    parser.add_argument(
        "--camera-prior", default=None,
        help="Optional fixed_camera.json used as a pose prior (round2 uses round1 camera)."
    )
    parser.add_argument(
        "--camera-output", default=None,
        help="Exact fixed_camera.json output path. Default: <output>/fixed_camera.json."
    )
    parser.add_argument(
        "--prior-only",
        action="store_true",
        help=(
            "Use --camera-prior as the ONLY initial external pose. "
            "Skips diagnostic-view search. Intended for Round 2."
        ),
    )
    args = parser.parse_args()
    main(
        args.config,
        g0_override=args.g0,
        output_override=args.output,
        prior_override=args.camera_prior,
        camera_output_override=args.camera_output,
        prior_only=args.prior_only,
    )
