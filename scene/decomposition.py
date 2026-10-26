"""Initial-event moving-part extraction for fixed-camera E-D3DGS.

This build uses only a short event segment at the very beginning to identify which canonical Gaussian part starts moving.
Positive events are visualized as red, negative events as blue. Spatially
paired red/blue edges are cleaned, connected, and contour-filled to produce a
2D moving-part mask. The mask is unprojected through the canonical Gaussian
depth map and converted to dynamic/static Gaussian masks by 3D proximity.

The canonical render is used for depth only. Render-vs-event photometric
residuals do not decide whether a pixel is dynamic. This is intentional for
monocular fixed-camera event training where an imperfect G0 should not create
false dynamics simply because its appearance or geometry differs from reality.

Build: 2026-10-06-red-blue-contour-v4
"""



import glob

import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

from pathlib import Path



import cv2

import numpy as np

import torch

from plyfile import PlyData, PlyElement

from scipy.spatial import cKDTree

from sklearn.neighbors import NearestNeighbors

from torchvision import transforms


# Allow direct execution with:
#   python /E-D3DGS/scene/decomposition.py --config ...
# without relying on train.py to prepare Python paths.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from gaussian_renderer import render

from scene import Scene

from scene.gaussian_model import GaussianModel

from utils.event_utils import interpolating_virtual_camera

from utils.project_utils import unproject





EVENT_DSD_BUILD = "2026-10-11-main-contour-depth-bounded-v7"





def remove_images(path):

    """Remove PNG visualization files from one directory."""

    for image_file in glob.glob(os.path.join(path, "*.png")):

        os.remove(image_file)





def _positive_odd(value, minimum=1):

    """Convert a kernel size to a positive odd integer."""

    value = max(int(value), int(minimum))

    if value % 2 == 0:

        value += 1

    return value





def _timestamp_to_float(timestamp):

    """Convert a scalar tensor/NumPy scalar/Python number to float."""

    if torch.is_tensor(timestamp):

        return float(timestamp.detach().cpu().item())

    return float(timestamp)





def _move_camera(camera, device):

    """Move a camera only when the on-the-fly loading mode is enabled."""

    camera.load2device(device)





def render_to_log_photovoltage(image, args):

    """Use exactly the same differentiable sensor response as training."""

    if image.ndim != 3 or image.shape[0] != 3:

        raise ValueError(f"Expected renderer output (3,H,W), got {tuple(image.shape)}")

    response = torch.tensor(

        [

            float(getattr(args, "event_response_r", 0.299)),

            float(getattr(args, "event_response_g", 0.587)),

            float(getattr(args, "event_response_b", 0.114)),

        ],

        device=image.device,

        dtype=image.dtype,

    )

    response = response / response.sum().clamp_min(1e-8)

    photovoltage = (

        torch.clamp(image, min=0.0) * response[:, None, None]

    ).sum(dim=0)

    return torch.log(

        photovoltage.clamp_min(float(getattr(args, "event_log_eps", 1e-4)))

    )





def _threshold_pair(fixed_threshold):

    """Return validated scalar positive/negative contrast thresholds."""

    if torch.is_tensor(fixed_threshold):

        values = fixed_threshold.detach().reshape(-1)

        if values.numel() == 1:

            positive_threshold = negative_threshold = float(values[0].item())

        elif values.numel() == 2:

            positive_threshold = float(values[0].item())

            negative_threshold = float(values[1].item())

        else:

            raise ValueError("A threshold tensor must contain one or two values")

    elif isinstance(fixed_threshold, np.ndarray):

        values = fixed_threshold.reshape(-1)

        if values.size == 1:

            positive_threshold = negative_threshold = float(values[0])

        elif values.size == 2:

            positive_threshold = float(values[0])

            negative_threshold = float(values[1])

        else:

            raise ValueError("A threshold array must contain one or two values")

    elif isinstance(fixed_threshold, (tuple, list)):

        if len(fixed_threshold) != 2:

            raise ValueError("A threshold sequence must contain exactly (C+, C-)")

        positive_threshold = float(fixed_threshold[0])

        negative_threshold = float(fixed_threshold[1])

    else:

        positive_threshold = negative_threshold = float(fixed_threshold)



    if (

        not np.isfinite(positive_threshold)

        or not np.isfinite(negative_threshold)

        or positive_threshold <= 0.0

        or negative_threshold <= 0.0

    ):

        raise ValueError("Event contrast thresholds must be finite and positive")

    return positive_threshold, negative_threshold





def fixed_threshold_event_frame(ECM, fixed_threshold, device, dtype):

    """Merge an ECM into a signed event log-intensity-change image.



    Args:

        ECM: Event count map with shape [2,B,H,W].  Channel 0 contains

            positive counts and channel 1 contains negative counts.

        fixed_threshold: Either one scalar threshold C or a (C+, C-) pair.

        device: Output device.

        dtype: Output floating-point dtype.



    Returns:

        Tensor with shape [H,W]: C+ * N_pos - C- * N_neg.

    """

    if ECM.ndim != 4 or ECM.shape[0] != 2:

        raise ValueError(

            "Expected ECM shape [2,B,H,W], but received "

            f"{tuple(ECM.shape)}."

        )



    positive_threshold, negative_threshold = _threshold_pair(fixed_threshold)



    ECM = ECM.to(device=device, dtype=dtype)

    positive_count = ECM[0].sum(dim=0)

    negative_count = ECM[1].sum(dim=0)

    return (

        positive_threshold * positive_count

        - negative_threshold * negative_count

    )





def polarity_event_residual(

    ECM,

    rendered_change,

    fixed_threshold,

    charbonnier_eps=1e-3,

):

    """Compare positive and negative event counts without cancellation.



    A single signed event frame loses information whenever both polarities are

    present at one pixel.  Here the rendered log-intensity change is converted

    to predicted positive/negative counts and each polarity is compared with

    its own observed count map.  This is the same representation used by the

    event reconstruction objective during training.



    Returns:

        residual: [H,W] robust polarity-separated count error.

        positive_count: [H,W] observed positive event count.

        negative_count: [H,W] observed negative event count.

        predicted_positive: [H,W] count predicted by the static render.

        predicted_negative: [H,W] count predicted by the static render.

    """

    if ECM.ndim != 4 or ECM.shape[0] != 2:

        raise ValueError(

            "Expected ECM shape [2,B,H,W], but received "

            f"{tuple(ECM.shape)}."

        )

    if tuple(ECM.shape[-2:]) != tuple(rendered_change.shape[-2:]):

        raise ValueError(

            "ECM/rendered-change resolution mismatch: "

            f"ECM={tuple(ECM.shape[-2:])}, "

            f"rendered_change={tuple(rendered_change.shape[-2:])}."

        )



    positive_threshold, negative_threshold = _threshold_pair(fixed_threshold)

    ECM = ECM.to(

        device=rendered_change.device,

        dtype=rendered_change.dtype,

    )

    positive_count = ECM[0].sum(dim=0)

    negative_count = ECM[1].sum(dim=0)



    predicted_positive = torch.relu(rendered_change) / positive_threshold

    predicted_negative = torch.relu(-rendered_change) / negative_threshold



    positive_error = predicted_positive - positive_count

    negative_error = predicted_negative - negative_count

    eps = max(float(charbonnier_eps), 0.0)

    if eps > 0.0:

        positive_penalty = torch.sqrt(positive_error.square() + eps * eps) - eps

        negative_penalty = torch.sqrt(negative_error.square() + eps * eps) - eps

    else:

        positive_penalty = positive_error.abs()

        negative_penalty = negative_error.abs()



    residual = 0.5 * (positive_penalty + negative_penalty)

    return (

        residual,

        positive_count,

        negative_count,

        predicted_positive,

        predicted_negative,

    )





def _save_tensor_image(tensor, path):

    """Save a CHW tensor as a clamped PNG."""

    image = transforms.ToPILImage()(torch.clamp(tensor.detach().cpu(), 0.0, 1.0))

    image.save(path)





def _prepare_visualization_directories():

    directories = [

        "vis_results/Event_decomp/Static_render",

        "vis_results/Event_decomp/Event_polarity",

        "vis_results/Event_decomp/Paired_support",

        "vis_results/Event_decomp/Dynamic_contour",

        "vis_results/Event_decomp/Event_activity",

        "vis_results/Event_decomp/Dynamic_mask",

        "vis_results/Event_decomp/Masked_dynamic",

        "vis_results/Event_decomp/depth",

    ]



    for directory in directories:

        Path(directory).mkdir(exist_ok=True, parents=True)

        remove_images(directory)





def residual_to_dynamic_mask(residual_map, activity_map, args, device):

    """Convert an event residual map into a cleaned binary dynamic mask.



    Args:

        residual_map: [H,W] absolute static/event residual.

        activity_map: [H,W] Boolean event-activity map.

        args: Configuration object.

        device: Device on which the returned mask is placed.



    Returns:

        dynamic_mask: [1,H,W] float tensor.

        residual_vis: [H,W] uint8 visualization.

        activity_support: [H,W] uint8 dilated activity support.

        used_threshold: Integer threshold in the 0--255 visualization scale.

    """

    residual_np = residual_map.detach().float().cpu().numpy()

    activity_np = activity_map.detach().cpu().numpy().astype(np.uint8)



    blur_kernel = _positive_odd(

        getattr(args, "event_dsd_blur_kernel", 3),

        minimum=1,

    )

    if blur_kernel > 1:

        residual_np = cv2.GaussianBlur(residual_np, (blur_kernel, blur_kernel), 0)



    activity_dilation = _positive_odd(

        getattr(args, "event_dsd_activity_dilation", 5),

        minimum=1,

    )

    activity_kernel = np.ones(

        (activity_dilation, activity_dilation),

        dtype=np.uint8,

    )

    activity_support = cv2.dilate(activity_np, activity_kernel, iterations=1)



    valid_values = residual_np[activity_support > 0]

    valid_values = valid_values[np.isfinite(valid_values)]



    if valid_values.size == 0:

        empty = torch.zeros(

            (1, residual_np.shape[0], residual_np.shape[1]),

            dtype=torch.float32,

            device=device,

        )

        return empty, np.zeros_like(activity_np), activity_support, 0



    percentile = float(getattr(args, "event_dsd_normalize_percentile", 99.0))

    percentile = min(max(percentile, 50.0), 100.0)

    upper_bound = max(float(np.percentile(valid_values, percentile)), 1e-6)



    residual_normalized = np.nan_to_num(

        residual_np / upper_bound,

        nan=0.0,

        posinf=1.0,

        neginf=0.0,

    )

    residual_vis = (np.clip(residual_normalized, 0.0, 1.0) * 255.0).astype(

        np.uint8

    )



    support_values = residual_vis[activity_support > 0]

    if support_values.size == 0 or int(support_values.max()) == 0:

        empty = torch.zeros(

            (1, residual_np.shape[0], residual_np.shape[1]),

            dtype=torch.float32,

            device=device,

        )

        return empty, residual_vis, activity_support, 0



    # Otsu is computed on supported pixels, so a large zero-valued background

    # does not force the threshold towards zero.

    otsu_threshold, _ = cv2.threshold(

        support_values.reshape(-1, 1),

        0,

        255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU,

    )

    # A very small Otsu threshold makes nearly the whole event-support region

    # dynamic.  Keep a meaningful lower bound while still allowing Otsu to

    # choose a stricter threshold when the residual distribution is bimodal.

    threshold_floor = int(getattr(args, "event_dsd_otsu_floor", 24))

    used_threshold = max(int(round(float(otsu_threshold))), threshold_floor)



    binary_mask = (residual_vis >= used_threshold).astype(np.uint8)

    binary_mask *= activity_support



    close_kernel_size = _positive_odd(

        getattr(args, "event_dsd_close_kernel", 5),

        minimum=1,

    )

    if close_kernel_size > 1:

        close_kernel = np.ones(

            (close_kernel_size, close_kernel_size),

            dtype=np.uint8,

        )

        binary_mask = cv2.morphologyEx(

            binary_mask,

            cv2.MORPH_CLOSE,

            close_kernel,

        )



    mask_dilation = int(getattr(args, "event_dsd_mask_dilation", 0))

    if mask_dilation > 0:

        binary_mask = cv2.dilate(

            binary_mask,

            np.ones((3, 3), dtype=np.uint8),

            iterations=mask_dilation,

        )

        binary_mask *= activity_support



    minimum_area = max(int(getattr(args, "event_dsd_min_component", 64)), 1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(

        binary_mask,

        connectivity=8,

    )



    filtered_mask = np.zeros_like(binary_mask)

    for component_index in range(1, component_count):

        component_area = stats[component_index, cv2.CC_STAT_AREA]

        if component_area >= minimum_area:

            filtered_mask[labels == component_index] = 1



    # Deliberately do not use event_dsd_max_pixel_ratio or top-k selection.

    # The amount of motion is an output of the event evidence, not a target.



    dynamic_mask = torch.from_numpy(filtered_mask).to(

        device=device,

        dtype=torch.float32,

    )

    return dynamic_mask.unsqueeze(0), residual_vis, activity_support, used_threshold





def polarity_contour_to_dynamic_mask(positive_count, negative_count, args, device):
    """Build a motion mask directly from red/blue event contours.

    This function deliberately does *not* compare events with a rendered
    brightness change. Positive events are treated as the red contour and
    negative events as the blue contour. A reliable motion seed requires the
    two polarities to be spatially close, because a moving brightness edge
    normally produces a leading/trailing polarity pair. The paired support is
    then morphologically connected and its external contours are filled so
    that the interior of a moving part, not only its event edge, is selected.

    Args:
        positive_count: [H,W] accumulated positive event counts.
        negative_count: [H,W] accumulated negative event counts.
        args: Configuration object.
        device: Device for the returned torch mask.

    Returns:
        dynamic_mask: [1,H,W] float tensor, 1 = dynamic.
        polarity_vis: [H,W,3] uint8 BGR visualization (red=+, blue=-).
        activity_support: [H,W] uint8 union of accepted polarity activity.
        paired_support: [H,W] uint8 red/blue spatially paired support.
    """
    if positive_count.ndim != 2 or negative_count.ndim != 2:
        raise ValueError(
            "positive_count and negative_count must both have shape [H,W]."
        )
    if positive_count.shape != negative_count.shape:
        raise ValueError(
            "Positive/negative event-count maps must have the same resolution."
        )

    positive_np = positive_count.detach().float().cpu().numpy()
    negative_np = negative_count.detach().float().cpu().numpy()
    positive_np = np.nan_to_num(positive_np, nan=0.0, posinf=0.0, neginf=0.0)
    negative_np = np.nan_to_num(negative_np, nan=0.0, posinf=0.0, neginf=0.0)

    min_count = max(int(getattr(args, "event_dsd_min_event_count", 1)), 1)
    positive_active = (positive_np >= min_count).astype(np.uint8)
    negative_active = (negative_np >= min_count).astype(np.uint8)
    activity_support = np.logical_or(positive_active, negative_active).astype(np.uint8)

    # Red/blue visualization: OpenCV uses BGR ordering.
    polarity_vis = np.zeros(
        (positive_np.shape[0], positive_np.shape[1], 3), dtype=np.uint8
    )
    polarity_vis[..., 2] = positive_active * 255
    polarity_vis[..., 0] = negative_active * 255

    if int(activity_support.sum()) == 0:
        empty = torch.zeros(
            (1, positive_np.shape[0], positive_np.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        return empty, polarity_vis, activity_support, np.zeros_like(activity_support)

    # A moving boundary should have opposite-polarity evidence nearby.  The
    # kernel controls how far red and blue edges may be separated by motion.
    pair_kernel_size = _positive_odd(
        getattr(args, "event_dsd_polarity_pair_kernel", 31), minimum=1
    )
    pair_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pair_kernel_size, pair_kernel_size)
    )
    positive_near = cv2.dilate(positive_active, pair_kernel, iterations=1)
    negative_near = cv2.dilate(negative_active, pair_kernel, iterations=1)
    paired_support = np.logical_or(
        np.logical_and(positive_active > 0, negative_near > 0),
        np.logical_and(negative_active > 0, positive_near > 0),
    ).astype(np.uint8)

    allow_single = bool(getattr(args, "event_dsd_allow_single_polarity", False))
    if int(paired_support.sum()) == 0 and allow_single:
        paired_support = activity_support.copy()

    if int(paired_support.sum()) == 0:
        empty = torch.zeros(
            (1, positive_np.shape[0], positive_np.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        return empty, polarity_vis, activity_support, paired_support

    # Expand the paired seeds only through pixels supported by actual events.
    # This keeps isolated sensor noise out while preserving the full red/blue
    # edge structure around a moving object part.
    seed_dilation = max(int(getattr(args, "event_dsd_pair_dilation", 1)), 0)

    # The intersection of dilated red and blue supports creates a corridor
    # between the opposite-polarity edges. Without this bridge, contour filling
    # would often keep only two thin event arcs instead of the moving surface
    # lying between them.
    polarity_bridge = np.logical_and(positive_near > 0, negative_near > 0)
    motion_edges = np.logical_or(paired_support > 0, polarity_bridge).astype(np.uint8)

    if seed_dilation > 0:
        motion_edges = cv2.dilate(
            motion_edges,
            np.ones((3, 3), dtype=np.uint8),
            iterations=seed_dilation,
        )

    activity_dilation = max(int(getattr(args, "event_dsd_contour_activity_dilation", 2)), 0)
    expanded_activity = activity_support.copy()
    if activity_dilation > 0:
        expanded_activity = cv2.dilate(
            expanded_activity,
            np.ones((3, 3), dtype=np.uint8),
            iterations=activity_dilation,
        )

    # Keep the bridge close to actual event evidence; this prevents a large
    # pairing kernel from painting remote silent regions as dynamic.
    motion_edges = np.logical_and(
        motion_edges > 0, expanded_activity > 0
    ).astype(np.uint8)

    # Join broken event arcs. This is important for sparse event streams where
    # the red/blue boundary of one moving part is not perfectly continuous.
    close_kernel_size = _positive_odd(
        getattr(args, "event_dsd_contour_close_kernel", 11), minimum=1
    )
    if close_kernel_size > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        motion_edges = cv2.morphologyEx(
            motion_edges, cv2.MORPH_CLOSE, close_kernel, iterations=1
        )

    # Remove tiny edge islands before contour filling.
    minimum_area = max(int(getattr(args, "event_dsd_min_component", 64)), 1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        motion_edges, connectivity=8
    )
    cleaned_edges = np.zeros_like(motion_edges)
    for component_index in range(1, component_count):
        if stats[component_index, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned_edges[labels == component_index] = 1

    if int(cleaned_edges.sum()) == 0:
        empty = torch.zeros(
            (1, positive_np.shape[0], positive_np.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        return empty, polarity_vis, activity_support, paired_support

    fill_contours = bool(getattr(args, "event_dsd_fill_contours", True))
    if fill_contours:
        contours, _ = cv2.findContours(
            (cleaned_edges * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        filled = np.zeros_like(cleaned_edges)
        if contours:
            # --------------------------------------------------------------
            # DOMINANT / MAIN-CONTOUR GATE
            # --------------------------------------------------------------
            # Sparse red/blue satellites outside the real moving-part contour
            # must NOT create extra dynamic islands.  This was the main reason
            # that a static handle/background part could be pulled into the
            # dynamic Gaussian set: every sufficiently large external contour
            # used to be filled independently.
            #
            # We now rasterize every contour, measure its actual filled area,
            # and keep only the dominant contour.  Any event point/component
            # outside this large contour is discarded before 3D unprojection,
            # projected-candidate construction, and Gaussian manifold growth.
            contour_fills = []
            contour_areas = []
            for contour in contours:
                contour_mask = np.zeros_like(cleaned_edges, dtype=np.uint8)
                cv2.drawContours(
                    contour_mask,
                    [contour],
                    -1,
                    color=1,
                    thickness=cv2.FILLED,
                )
                contour_fills.append(contour_mask)
                contour_areas.append(int(contour_mask.sum()))

            main_index = int(np.argmax(np.asarray(contour_areas, dtype=np.int64)))
            main_filled = contour_fills[main_index]
            main_area = contour_areas[main_index]

            # A tiny optional margin is allowed around the *main* contour only.
            # It never resurrects disconnected satellite event islands.
            main_margin = max(
                int(getattr(args, "event_dsd_main_contour_margin", 0)), 0
            )
            if main_margin > 0:
                main_gate = cv2.dilate(
                    main_filled,
                    np.ones((3, 3), dtype=np.uint8),
                    iterations=main_margin,
                )
            else:
                main_gate = main_filled

            # Reject all peripheral event evidence outside the dominant contour.
            # Keeping this as an explicit intersection is important because the
            # red/blue pair kernel can otherwise make isolated external points
            # look like legitimate paired motion evidence.
            cleaned_edges = np.logical_and(
                cleaned_edges > 0, main_gate > 0
            ).astype(np.uint8)
            filled = main_filled.copy()

            print(
                "[Event DSD] dominant contour gate:",
                f"contours={len(contours)},",
                f"main_area={main_area},",
                f"discarded_contours={max(len(contours) - 1, 0)}",
            )
        else:
            filled = cleaned_edges.copy()
    else:
        # Even without contour filling, retain only the largest connected event
        # component so isolated peripheral event points cannot seed static parts.
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned_edges.astype(np.uint8), connectivity=8
        )
        if component_count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            main_component = 1 + int(np.argmax(areas))
            filled = (labels == main_component).astype(np.uint8)
        else:
            filled = cleaned_edges.copy()

    # Optional final dilation gives a small safety margin around the accepted
    # dominant contour only. It cannot bring discarded satellite components back.
    mask_dilation = max(int(getattr(args, "event_dsd_mask_dilation", 1)), 0)
    if mask_dilation > 0:
        filled = cv2.dilate(
            filled,
            np.ones((3, 3), dtype=np.uint8),
            iterations=mask_dilation,
        )

    # The dominant contour itself is the final motion region. We still verify
    # that it has enough area and contains both polarities (unless explicitly
    # allowing single-polarity evidence), but we never re-add another component.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled.astype(np.uint8), connectivity=8
    )
    filtered_mask = np.zeros_like(filled, dtype=np.uint8)
    if component_count > 1:
        valid_components = [
            component_index
            for component_index in range(1, component_count)
            if stats[component_index, cv2.CC_STAT_AREA] >= minimum_area
        ]
        if valid_components:
            main_component = max(
                valid_components,
                key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]),
            )
            component = labels == main_component
            accept_component = True
            if not allow_single:
                component_u8 = component.astype(np.uint8)
                check_kernel_size = max(pair_kernel_size, 3)
                check_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (check_kernel_size, check_kernel_size)
                )
                expanded_component = cv2.dilate(
                    component_u8, check_kernel, iterations=1
                )
                has_positive = bool(
                    np.any((expanded_component > 0) & (positive_active > 0))
                )
                has_negative = bool(
                    np.any((expanded_component > 0) & (negative_active > 0))
                )
                accept_component = has_positive and has_negative
            if accept_component:
                filtered_mask[component] = 1

    dynamic_mask = torch.from_numpy(filtered_mask).to(
        device=device, dtype=torch.float32
    )
    return dynamic_mask.unsqueeze(0), polarity_vis, activity_support, paired_support



def _build_interval_specs(frame_index, total_frame, event_list):

    """Return all full event intervals adjacent to one training frame."""

    intervals = []



    # event_list[i] represents the interval from camera i to camera i+1.

    if frame_index > 0 and frame_index - 1 < len(event_list):

        past = event_list[frame_index - 1]

        intervals.append(

            {

                "ECM": past["ECM"],

                "timestamp": past["tss"][0],

                "part_flag": "past",

            }

        )



    if frame_index < total_frame - 1 and frame_index < len(event_list):

        future = event_list[frame_index]

        intervals.append(

            {

                "ECM": future["ECM"],

                "timestamp": future["tss"][-1],

                "part_flag": "future",

            }

        )



    return intervals





def _filter_unprojected_points(all_points, scene, args):

    """Apply the original camera-extent and nearest-neighbour filters."""

    if all_points.shape[0] < 2:

        raise RuntimeError("Event DSD produced fewer than two unprojected points.")



    device = all_points.device

    camera_center = torch.as_tensor(

        scene.cameras_center,

        device=device,

        dtype=all_points.dtype,

    )

    distances_to_center = torch.norm(all_points - camera_center, dim=1)

    all_points = all_points[

        distances_to_center <= 0.6 * float(scene.cameras_extent)

    ]



    if all_points.shape[0] < 2:

        raise RuntimeError(

            "Too few event DSD points remain after camera-extent filtering."

        )



    points_np = all_points.detach().cpu().numpy()

    neighbors = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(points_np)

    neighbor_distances, _ = neighbors.kneighbors(points_np)

    nearest_distances = neighbor_distances[:, 1]



    strength = float(getattr(args, "outlier_filtering_strength", 0.0))

    distance_threshold = (

        float(nearest_distances.mean())

        - strength * float(nearest_distances.std())

    )



    # Preserve the original E-D3DGS rule when it is valid.  If a large

    # filtering strength makes the threshold non-positive, use the median so

    # that the whole decomposition does not collapse to an empty point cloud.

    if not np.isfinite(distance_threshold) or distance_threshold <= 0.0:

        distance_threshold = float(np.median(nearest_distances))



    filtered_points = points_np[nearest_distances < distance_threshold]

    if filtered_points.shape[0] < 2:

        raise RuntimeError(

            "All event DSD points were removed by outlier filtering. "

            "Reduce outlier_filtering_strength or inspect the dynamic masks."

        )



    return filtered_points





def _write_point_cloud(path, points):

    """Write an XYZ NumPy point array to PLY."""

    vertex = np.array(

        [tuple(point) for point in points],

        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],

    )

    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)





def _gaussian_masks_from_points(unprojected_points, gaussians, args):

    """Convert dynamic 3D samples to evidence-based dynamic/buffer masks.



    Membership is determined by 3D distance to red/blue event-contour samples. No

    desired dynamic/static ratio or top-k point count is imposed.

    """

    gaussian_points = gaussians.get_xyz.detach().cpu().numpy()

    if gaussian_points.shape[0] < 2:

        raise RuntimeError("At least two Gaussians are required for DSD.")



    gaussian_neighbors = NearestNeighbors(

        n_neighbors=2,

        algorithm="ball_tree",

    ).fit(gaussian_points)

    gaussian_distances, _ = gaussian_neighbors.kneighbors(gaussian_points)

    radius_percentile = float(

        getattr(args, "event_dsd_manifold_percentile", 50.0)

    )

    radius_percentile = min(max(radius_percentile, 5.0), 95.0)

    manifold_radius = float(

        np.percentile(gaussian_distances[:, 1], radius_percentile)

    )



    if not np.isfinite(manifold_radius) or manifold_radius <= 0.0:

        raise RuntimeError("Could not estimate a valid Gaussian manifold radius.")



    dynamic_tree = cKDTree(unprojected_points)

    distances_to_dynamic, _ = dynamic_tree.query(gaussian_points, k=1)



    dynamic_radius_scale = float(getattr(args, "dynamic_radius_scale", 1.0))

    buffer_radius_scale = float(getattr(args, "buffer_radius_scale", 1.5))

    if dynamic_radius_scale <= 0.0 or buffer_radius_scale <= 0.0:

        raise ValueError("dynamic_radius_scale and buffer_radius_scale must be positive.")



    dynamic_candidates = distances_to_dynamic < (

        dynamic_radius_scale * manifold_radius

    )

    buffer_candidates = distances_to_dynamic < (

        buffer_radius_scale * manifold_radius

    )

    buffer_candidates = np.logical_or(buffer_candidates, dynamic_candidates)



    gaussian_count = gaussian_points.shape[0]

    dynamic_mask = dynamic_candidates.astype(bool, copy=True)



    dynamic_count = int(dynamic_mask.sum())

    use_buffer = bool(getattr(args, "buffer_strategy", False))

    if use_buffer:

        prune_static_mask = buffer_candidates.astype(bool, copy=True)

    else:

        # Hard decomposition: remove from the static model only points that

        # have actually been classified as dynamic.  Nearby but uncertain

        # points remain static rather than disappearing from both models.

        prune_static_mask = dynamic_mask.copy()



    post_static_count = gaussian_count - int(prune_static_mask.sum())

    post_dynamic_ratio = dynamic_count / max(post_static_count + dynamic_count, 1)



    print(

        "[Event DSD] raw dynamic candidates:",

        int(dynamic_candidates.sum()),

        "selected dynamic:",

        dynamic_count,

        "static prune mask:",

        int(prune_static_mask.sum()),

        "buffer strategy:",

        use_buffer,

        "post-decomposition ratio:",

        f"{post_dynamic_ratio:.4f}",

    )

    return dynamic_mask, prune_static_mask, manifold_radius





def event_decomposition(
    scene: Scene,
    gaussians: GaussianModel,
    background: torch.Tensor,
    event_list,
    fixed_threshold,
    args,
):
    """Perform red/blue event-contour dynamic/static decomposition.

    Dynamic/static membership is decided directly from event polarity geometry:
    positive events form the red contour, negative events form the blue contour,
    and spatially paired red/blue edges define motion evidence.  The event
    contour is cleaned and filled in 2D, then unprojected with the canonical
    Gaussian depth map and converted to a 3D Gaussian mask.

    The canonical render is used only for depth/unprojection and diagnostics;
    its brightness change does not participate in dynamic/static classification.
    This prevents an imperfect G0 appearance or geometry from creating a false
    dynamic residual merely because the static render cannot explain an event.

    Returns:
        dynamic_gaussian_mask: NumPy bool array used to initialize dynamic GS.
        prune_static_gaussian_mask: NumPy bool array removed from the static
            GS. It equals the dynamic mask unless buffer_strategy is enabled.
    """
    if len(event_list) == 0:
        raise RuntimeError("Event DSD received an empty event list.")

    # Kept in the signature for compatibility with existing training code.
    # Red/blue contour DSD does not need the event contrast threshold.
    _ = fixed_threshold

    print(f"[Event DSD] decomposition build: {EVENT_DSD_BUILD}")
    print("[Event DSD] mode: red/blue polarity contour -> 2D mask -> 3D Gaussians")

    train_cameras = scene.getTrainCameras().copy()
    total_frame = len(train_cameras)
    if total_frame < 2:
        raise RuntimeError("Event DSD requires at least two training cameras.")
    if len(event_list) < total_frame - 1:
        raise RuntimeError(
            "Event DSD expected at least total_frame - 1 event intervals, but got "
            f"{len(event_list)} intervals for {total_frame} cameras."
        )

    visualization_enabled = bool(getattr(args, "vis_decomposition", False))
    on_the_fly = bool(getattr(args, "load2gpu_on_the_fly", False))
    if visualization_enabled:
        _prepare_visualization_directories()

    device = gaussians.get_xyz.device
    unprojected_point_sets = []
    processed_frames = 0
    selected_pixels = 0
    observed_positive_events = 0.0
    observed_negative_events = 0.0

    with torch.no_grad():
        for frame_index, train_camera in enumerate(train_cameras):
            if on_the_fly:
                _move_camera(train_camera, device)

            # The render is intentionally used only to obtain visible-surface
            # depth and to create diagnostics. It does NOT decide motion.
            current_package = render(
                train_camera,
                gaussians,
                args,
                background,
                0.0,
                0.0,
                0.0,
                args.is_6dof,
            )
            current_image = current_package["render"]
            depth = current_package["depth"]

            interval_specs = _build_interval_specs(
                frame_index,
                total_frame,
                event_list,
            )
            if len(interval_specs) == 0:
                if on_the_fly:
                    _move_camera(train_camera, "cpu")
                continue

            positive_accum = None
            negative_accum = None

            for interval in interval_specs:
                ECM = interval["ECM"]
                if ECM.ndim != 4 or ECM.shape[0] != 2:
                    raise ValueError(
                        "Expected ECM shape [2,B,H,W], but received "
                        f"{tuple(ECM.shape)}."
                    )
                if tuple(ECM.shape[-2:]) != tuple(current_image.shape[-2:]):
                    raise RuntimeError(
                        "Event/render resolution mismatch during DSD: "
                        f"ECM={tuple(ECM.shape[-2:])}, "
                        f"render={tuple(current_image.shape[-2:])}."
                    )

                ECM = ECM.to(device=device, dtype=torch.float32)
                positive_count = ECM[0].sum(dim=0)
                negative_count = ECM[1].sum(dim=0)

                if positive_accum is None:
                    positive_accum = positive_count.clone()
                    negative_accum = negative_count.clone()
                else:
                    positive_accum += positive_count
                    negative_accum += negative_count

                observed_positive_events += float(positive_count.sum().item())
                observed_negative_events += float(negative_count.sum().item())

            if positive_accum is None or negative_accum is None:
                if on_the_fly:
                    _move_camera(train_camera, "cpu")
                continue

            (
                dynamic_pixel_mask,
                polarity_vis,
                activity_support,
                paired_support,
            ) = polarity_contour_to_dynamic_mask(
                positive_accum,
                negative_accum,
                args,
                device,
            )

            processed_frames += 1
            frame_selected_pixels = int(dynamic_pixel_mask.sum().item())
            selected_pixels += frame_selected_pixels

            if visualization_enabled:
                _save_tensor_image(
                    current_image,
                    "vis_results/Event_decomp/Static_render/"
                    f"image_{frame_index:03d}.png",
                )
                cv2.imwrite(
                    "vis_results/Event_decomp/Event_polarity/"
                    f"image_{frame_index:03d}.png",
                    polarity_vis,
                )
                cv2.imwrite(
                    "vis_results/Event_decomp/Paired_support/"
                    f"image_{frame_index:03d}.png",
                    paired_support * 255,
                )
                cv2.imwrite(
                    "vis_results/Event_decomp/Event_activity/"
                    f"image_{frame_index:03d}.png",
                    activity_support * 255,
                )
                dynamic_np = (
                    dynamic_pixel_mask.squeeze(0).detach().cpu().numpy() > 0.5
                ).astype(np.uint8)
                contour_vis = polarity_vis.copy()
                contours, _ = cv2.findContours(
                    dynamic_np * 255,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 1)
                cv2.imwrite(
                    "vis_results/Event_decomp/Dynamic_contour/"
                    f"image_{frame_index:03d}.png",
                    contour_vis,
                )
                _save_tensor_image(
                    dynamic_pixel_mask,
                    "vis_results/Event_decomp/Dynamic_mask/"
                    f"image_{frame_index:03d}.png",
                )

                output_background = (
                    torch.ones_like(current_image)
                    if args.white_background
                    else torch.zeros_like(current_image)
                )
                masked_dynamic = (
                    dynamic_pixel_mask * current_image
                    + (1.0 - dynamic_pixel_mask) * output_background
                )
                _save_tensor_image(
                    masked_dynamic,
                    "vis_results/Event_decomp/Masked_dynamic/"
                    f"image_{frame_index:03d}.png",
                )

            # Convert the 2D red/blue motion region into 3D samples using the
            # current canonical depth. Only pixels with a valid rendered depth
            # can seed a dynamic Gaussian.
            points = unproject(depth, train_camera.w2c, train_camera.k)
            points = points.reshape(points.shape[0], -1)
            depth_flat = depth.reshape(-1)
            mask_flat = dynamic_pixel_mask.reshape(-1).bool()
            valid_depth = torch.isfinite(depth_flat) & (depth_flat > 0)
            valid_mask = mask_flat & valid_depth

            if valid_mask.any():
                unprojected_point_sets.append(points[:, valid_mask].T)

            if visualization_enabled:
                finite_positive_depth = depth[torch.isfinite(depth) & (depth > 0)]
                if finite_positive_depth.numel() > 0:
                    depth_scale = finite_positive_depth.max()
                else:
                    depth_scale = torch.tensor(1.0, device=depth.device)
                masked_depth = dynamic_pixel_mask * (depth / (depth_scale + 1e-8))
                _save_tensor_image(
                    masked_depth,
                    "vis_results/Event_decomp/depth/"
                    f"image_{frame_index:03d}.png",
                )

                positive_pixels = int((positive_accum >= max(int(getattr(args, "event_dsd_min_event_count", 1)), 1)).sum().item())
                negative_pixels = int((negative_accum >= max(int(getattr(args, "event_dsd_min_event_count", 1)), 1)).sum().item())
                paired_pixels = int(paired_support.sum())
                print(
                    f"[Event DSD] frame={frame_index:03d}, "
                    f"red_pixels={positive_pixels}, "
                    f"blue_pixels={negative_pixels}, "
                    f"paired_pixels={paired_pixels}, "
                    f"dynamic_pixels={frame_selected_pixels}"
                )

            if on_the_fly:
                _move_camera(train_camera, "cpu")

    if len(unprojected_point_sets) == 0:
        raise RuntimeError(
            "Red/blue event DSD did not find any valid dynamic 3D samples. "
            "Inspect Event_polarity, Paired_support and Dynamic_mask. If the "
            "event stream is sparse, reduce event_dsd_min_event_count or "
            "event_dsd_min_component, increase event_dsd_polarity_pair_kernel, "
            "or set event_dsd_allow_single_polarity=True only if necessary."
        )

    all_unprojected_points = torch.cat(unprojected_point_sets, dim=0)
    filtered_points = _filter_unprojected_points(
        all_unprojected_points,
        scene,
        args,
    )

    if visualization_enabled:
        _write_point_cloud(
            "vis_results/event_unproject_pcd.ply",
            filtered_points,
        )

    dynamic_gaussian_mask, prune_static_gaussian_mask, manifold_radius = (
        _gaussian_masks_from_points(filtered_points, gaussians, args)
    )

    if dynamic_gaussian_mask.sum() == 0:
        raise RuntimeError(
            "Red/blue event DSD found dynamic 3D samples but selected zero "
            "Gaussians. Increase dynamic_radius_scale."
        )

    if visualization_enabled:
        gaussian_points = gaussians.get_xyz.detach().cpu().numpy()
        _write_point_cloud(
            "vis_results/event_dynamic_pcd.ply",
            gaussian_points[dynamic_gaussian_mask],
        )

    print(
        "[Event DSD] processed frames:",
        processed_frames,
        "selected pixels:",
        selected_pixels,
    )
    print("[Event DSD] manifold radius:", manifold_radius)
    print(
        "[Event DSD] dynamic Gaussians:",
        int(dynamic_gaussian_mask.sum()),
        "/",
        len(dynamic_gaussian_mask),
    )
    print(
        "[Event DSD] static-prune Gaussians:",
        int(prune_static_gaussian_mask.sum()),
        "/",
        len(prune_static_gaussian_mask),
    )
    print(
        "[Event DSD] observed events (+/-):",
        f"{observed_positive_events:.0f}",
        "/",
        f"{observed_negative_events:.0f}",
    )

    return dynamic_gaussian_mask, prune_static_gaussian_mask



# =============================================================================
# Standalone fixed-camera continuous-event diagnostic entry point
# =============================================================================


def _standalone_build_args(config_path):
    """Parse the normal E-D3DGS INI without importing/running train.py."""
    from utils import parse_utils

    cfg, print_format = parse_utils.parse_ini(config_path)
    values = dict(cfg._asdict())

    # Only defaults needed by Scene/render/decomposition are supplied here.
    # Values present in the INI always win.
    defaults = {
        "sh_degree": 3,
        "random_points": False,
        "event_assist": True,
        "bin_num": 64,
        "fixed_threshold": 0.2,
        "white_background": False,
        "load2gpu_on_the_fly": False,
        "data_device": "cuda",
        "is_6dof": False,
        "debug": False,
        "quiet": False,
        # Red/blue contour DSD defaults.
        "event_dsd_min_event_count": 1,
        "event_dsd_polarity_pair_kernel": 31,
        "event_dsd_pair_dilation": 1,
        "event_dsd_contour_activity_dilation": 2,
        "event_dsd_contour_close_kernel": 11,
        "event_dsd_fill_contours": True,
        "event_dsd_main_contour_margin": 0,
        "event_dsd_min_component": 64,
        "event_dsd_mask_dilation": 1,
        "event_dsd_allow_single_polarity": False,
        "dynamic_radius_scale": 1.0,
        "buffer_radius_scale": 1.5,
        "buffer_strategy": False,
        "event_response_r": 0.299,
        "event_response_g": 0.587,
        "event_response_b": 0.114,
        "event_log_eps": 1e-4,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)

    args = SimpleNamespace(**values)
    return args, print_format


def _event_counts_from_continuous_events(events, height, width):
    """Convert Nx4 [x,y,t,p] events into positive/negative [H,W] count maps.

    This deliberately bypasses temporal bin construction because standalone DSD
    only needs polarity geometry. Invalid coordinates are discarded. Polarity
    is interpreted using p>0 as positive and p<0 as negative.
    """
    if not torch.is_tensor(events):
        events = torch.as_tensor(events)
    if events.ndim != 2 or events.shape[1] < 4:
        raise ValueError(
            "Continuous event tensor must have shape [N,4+] containing x,y,t,p; "
            f"got {tuple(events.shape)}"
        )

    # Keep counting on CPU. This avoids copying a multi-million-event stream to
    # GPU merely to produce a diagnostic mask.
    ev = events.detach().cpu()
    x = ev[:, 0].long()
    y = ev[:, 1].long()
    p = ev[:, 3]

    valid = (
        (x >= 0)
        & (x < int(width))
        & (y >= 0)
        & (y < int(height))
        & torch.isfinite(p.float())
    )
    x = x[valid]
    y = y[valid]
    p = p[valid]

    flat_index = y * int(width) + x
    num_pixels = int(height) * int(width)

    positive_index = flat_index[p > 0]
    negative_index = flat_index[p < 0]

    positive = torch.bincount(
        positive_index,
        minlength=num_pixels,
    ).reshape(int(height), int(width))
    negative = torch.bincount(
        negative_index,
        minlength=num_pixels,
    ).reshape(int(height), int(width))

    return positive, negative


def _save_standalone_scalar(tensor, path, signed=False):
    """Save HxW/1xHxW tensor as an 8-bit diagnostic image."""
    arr = tensor.detach().float().cpu().squeeze().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if signed:
        scale = max(float(np.max(np.abs(arr))), 1e-8)
        arr = 0.5 + 0.5 * arr / scale
    else:
        max_value = max(float(np.max(arr)), 1e-8)
        if max_value > 1.0:
            arr = arr / max_value
    cv2.imwrite(str(path), (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))


def _save_dynamic_gaussian_diagnostic_ply(path, xyz, dynamic_mask):
    """Write an RGB point cloud: dynamic=red, static=gray."""
    xyz = np.asarray(xyz, dtype=np.float32)
    dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz [N,3], got {xyz.shape}")
    if dynamic_mask.shape[0] != xyz.shape[0]:
        raise ValueError("dynamic mask length does not match Gaussian count")

    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["red"] = np.where(dynamic_mask, 255, 110).astype(np.uint8)
    vertex["green"] = np.where(dynamic_mask, 35, 110).astype(np.uint8)
    vertex["blue"] = np.where(dynamic_mask, 35, 110).astype(np.uint8)
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


def _select_initial_event_segment(events, event_record, args):
    """Return only the short event segment at the beginning of the sequence.

    Selection is temporal, not a fraction of event count.  By default the first
    150 ms are used.  ``event_dsd_initial_fraction`` is only a fallback when
    ``event_dsd_initial_duration_ms <= 0``.  A minimum number of events can
    enlarge the segment slightly so that sparse startup motion is still visible;
    ``event_dsd_initial_max_events`` caps memory/diagnostic cost.
    """
    if not torch.is_tensor(events):
        events = torch.as_tensor(events)
    ev = events.detach().cpu()
    if ev.ndim != 2 or ev.shape[1] < 4:
        raise ValueError(f"Expected continuous events [N,4+], got {tuple(ev.shape)}")
    if ev.shape[0] < 4:
        raise RuntimeError("Continuous event stream contains fewer than four events.")

    ts = ev[:, 2].to(torch.int64)
    if not bool(torch.all(ts[1:] >= ts[:-1])):
        order = torch.argsort(ts)
        ev = ev[order]
        ts = ts[order]

    n = int(ev.shape[0])
    t0 = int(ts[0].item())
    t1 = int(ts[-1].item())
    if t1 <= t0:
        raise RuntimeError("Continuous event stream has zero temporal duration.")

    duration_ms = float(getattr(args, "event_dsd_initial_duration_ms", 150.0))
    fraction = float(getattr(args, "event_dsd_initial_fraction", 0.03))
    fraction = min(max(fraction, 1e-6), 1.0)

    if duration_ms > 0.0:
        # MonoEv's continuous loader normalizes event timestamps to int64 ns.
        cutoff = t0 + int(round(duration_ms * 1_000_000.0))
        cutoff = min(cutoff, t1)
        selection_mode = f"first_{duration_ms:.3f}_ms"
    else:
        cutoff = t0 + int(round((t1 - t0) * fraction))
        cutoff = min(max(cutoff, t0 + 1), t1)
        selection_mode = f"first_{fraction:.6f}_duration"

    cutoff_tensor = torch.tensor(cutoff, dtype=ts.dtype)
    end_idx = int(torch.searchsorted(ts, cutoff_tensor, right=True).item())

    min_events = max(int(getattr(args, "event_dsd_initial_min_events", 5000)), 4)
    max_events = int(getattr(args, "event_dsd_initial_max_events", 250000))
    end_idx = max(end_idx, min(min_events, n))
    if max_events > 0:
        end_idx = min(end_idx, max_events)
    end_idx = min(max(end_idx, 4), n)

    initial = ev[:end_idx]
    actual_t0 = int(initial[0, 2].item())
    actual_t1 = int(initial[-1, 2].item())
    return initial, {
        "selection_mode": selection_mode,
        "index_start": 0,
        "index_end": end_idx,
        "events": end_idx,
        "t_start": actual_t0,
        "t_end": actual_t1,
        "duration_ms": (actual_t1 - actual_t0) / 1e6,
        "whole_stream_events": n,
        "whole_stream_duration_ms": (t1 - t0) / 1e6,
    }


def _project_gaussian_centers_to_image_mask(gaussians, camera, image_mask, dilation_px=2):
    """Return Gaussians whose centers project inside a 2D moving-part mask.

    Unlike depth unprojection, this considers *all* Gaussian centers along the
    camera ray.  It therefore recovers the back/side Gaussians of the same part
    instead of extracting only the visible front surface.
    """
    mask = np.asarray(image_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"image_mask must be HxW, got {mask.shape}")
    if int(dilation_px) > 0:
        mask = cv2.dilate(
            mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=int(dilation_px),
        )

    xyz = gaussians.get_xyz.detach().cpu().float()
    w2c = torch.as_tensor(camera.w2c, dtype=torch.float32).detach().cpu()
    k = torch.as_tensor(camera.k, dtype=torch.float32).detach().cpu()
    if tuple(w2c.shape) != (4, 4):
        raise ValueError(f"camera.w2c must be 4x4, got {tuple(w2c.shape)}")
    if tuple(k.shape) != (3, 3):
        raise ValueError(f"camera.k must be 3x3, got {tuple(k.shape)}")

    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    xyz1 = torch.cat([xyz, ones], dim=1)
    cam = (w2c @ xyz1.T).T[:, :3]
    z = cam[:, 2]
    pix_h = (k @ cam.T).T
    denom = pix_h[:, 2]
    u = pix_h[:, 0] / denom.clamp_min(1e-8)
    v = pix_h[:, 1] / denom.clamp_min(1e-8)

    h, w = mask.shape
    valid = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(z)
        & (z > 0)
        & (u >= 0)
        & (u <= w - 1)
        & (v >= 0)
        & (v <= h - 1)
    )
    candidate = np.zeros(xyz.shape[0], dtype=bool)
    if bool(valid.any()):
        idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        ui = torch.round(u[idx]).long().clamp(0, w - 1).numpy()
        vi = torch.round(v[idx]).long().clamp(0, h - 1).numpy()
        candidate[idx.numpy()] = mask[vi, ui] > 0
    return candidate


def _motion_depth_gate(gaussians, camera, depth, motion_mask, manifold_radius, args):
    """Reject rear Gaussians even if their centers project inside motion mask.

    Two constraints are applied in the *same camera-depth units*:
    (1) per-pixel visible depth + rear tolerance;
    (2) robust global 10--90% depth envelope of the visible motion surface.
    All guards are upper bounds, so they cannot introduce extra dynamic points.
    """
    depth_np = depth.detach().float().cpu().numpy().squeeze()
    if depth_np.ndim != 2:
        raise RuntimeError(f"Expected 2D render depth, got {depth_np.shape}")
    mask = np.asarray(motion_mask, dtype=bool)
    surface = depth_np[mask & np.isfinite(depth_np) & (depth_np > 0)]
    if surface.size < 8:
        raise RuntimeError("Insufficient valid event-region depth for rear depth guard")

    # The depth renderer and camera XYZ must use comparable z conventions.
    # For this project, render depth is camera-space positive z.
    median_depth = float(np.median(surface))
    spacing_scale = max(float(getattr(args, "event_dsd_rear_depth_spacing_scale", 12.0)), 0.0)
    relative = max(float(getattr(args, "event_dsd_rear_depth_relative", 0.015)), 0.0)
    margin = max(spacing_scale * float(manifold_radius), relative * median_depth, 1e-7)
    q10, q90 = np.percentile(surface, [10.0, 90.0])
    global_near = float(q10) - margin
    global_far = float(q90) + margin

    xyz = gaussians.get_xyz.detach().cpu().float()
    w2c = torch.as_tensor(camera.w2c, dtype=torch.float32).detach().cpu()
    K = torch.as_tensor(camera.k, dtype=torch.float32).detach().cpu()
    cam = (w2c @ torch.cat((xyz, torch.ones((len(xyz), 1))), dim=1).T).T[:, :3]
    z = cam[:, 2].numpy()
    ph = (K @ cam.T).T
    den = ph[:, 2].numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        u = ph[:, 0].numpy() / den
        v = ph[:, 1].numpy() / den
    h, w = depth_np.shape
    good = (np.isfinite(z) & (z > 0) & np.isfinite(u) & np.isfinite(v)
            & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1))
    accepted = np.zeros(len(z), dtype=bool)
    ids = np.flatnonzero(good)
    if ids.size:
        ui = np.rint(u[ids]).astype(np.int32).clip(0, w-1)
        vi = np.rint(v[ids]).astype(np.int32).clip(0, h-1)
        local = depth_np[vi, ui]
        local_ok = np.isfinite(local) & (local > 0) & (z[ids] <= local + margin)
        # Keep both front and back of blades in the robust moving-surface envelope;
        # remove much deeper motor body/stand even where those are locally visible.
        global_ok = (z[ids] >= global_near) & (z[ids] <= global_far)
        accepted[ids] = local_ok & global_ok

    info = {
        "rear_depth_margin": float(margin),
        "rear_depth_q10": float(q10),
        "rear_depth_q90": float(q90),
        "rear_depth_global_near": global_near,
        "rear_depth_global_far": global_far,
        "depth_guard_accepted_gaussians": int(accepted.sum()),
    }
    return accepted, info


def _grow_gaussian_part_from_initial_seeds(
    gaussians,
    seed_mask,
    projected_candidate_mask,
    manifold_radius,
    args,
):
    """Recover the whole connected Gaussian part from visible event seeds.

    The event mask supplies visible-surface seeds.  We then grow only through
    Gaussians whose centers project inside the same initial moving-part mask.
    This is the key step that turns a front-surface event observation into an
    extracted 3D Gaussian *part* while avoiding arbitrary whole-object growth.
    """
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    seed_mask = np.asarray(seed_mask, dtype=bool).copy()
    candidates = np.asarray(projected_candidate_mask, dtype=bool).copy()
    if seed_mask.shape[0] != xyz.shape[0] or candidates.shape[0] != xyz.shape[0]:
        raise ValueError("Gaussian seed/candidate mask length mismatch")

    candidates |= seed_mask
    selected = seed_mask.copy()
    frontier = np.flatnonzero(seed_mask)
    if frontier.size == 0:
        raise RuntimeError("Initial event mask produced zero Gaussian seeds.")

    connectivity_scale = float(getattr(args, "event_dsd_part_connectivity_scale", 3.0))
    max_steps = max(int(getattr(args, "event_dsd_part_max_growth_steps", 64)), 1)
    radius = max(float(manifold_radius) * connectivity_scale, 1e-12)
    tree = cKDTree(xyz)

    steps = 0
    while frontier.size > 0 and steps < max_steps:
        neighbour_lists = tree.query_ball_point(xyz[frontier], r=radius, workers=-1)
        nonempty = [np.asarray(v, dtype=np.int64) for v in neighbour_lists if len(v) > 0]
        if not nonempty:
            break
        neighbours = np.unique(np.concatenate(nonempty))
        new_idx = neighbours[candidates[neighbours] & (~selected[neighbours])]
        if new_idx.size == 0:
            break
        selected[new_idx] = True
        frontier = new_idx
        steps += 1

    return selected, {
        "connectivity_radius": radius,
        "connectivity_scale": connectivity_scale,
        "growth_steps": steps,
        "seed_gaussians": int(seed_mask.sum()),
        "projected_candidates": int(candidates.sum()),
        "grown_gaussians": int(selected.sum()),
    }


def _save_full_gaussian_subset_ply(path, gaussians, boolean_mask):
    """Save a true 3DGS PLY subset with SH/opacity/scale/rotation preserved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(boolean_mask, dtype=bool)
    indices_np = np.flatnonzero(mask)
    if indices_np.size == 0:
        raise RuntimeError(f"Cannot save empty Gaussian subset: {path}")
    indices = torch.from_numpy(indices_np).long().to(gaussians.get_xyz.device)

    if hasattr(gaussians, "save_ply_with_indices"):
        try:
            gaussians.save_ply_with_indices(str(path), indices)
            return
        except Exception as exc:
            print(f"[Initial DSD] save_ply_with_indices fallback: {exc}")

    required = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]
    if not all(hasattr(gaussians, name) for name in required):
        raise RuntimeError("GaussianModel lacks tensors needed to save a full 3DGS subset.")

    xyz = gaussians._xyz[indices].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc[indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest[indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacity = gaussians._opacity[indices].detach().cpu().numpy()
    scaling = gaussians._scaling[indices].detach().cpu().numpy()
    rotation = gaussians._rotation[indices].detach().cpu().numpy()

    if hasattr(gaussians, "construct_list_of_attributes"):
        names = gaussians.construct_list_of_attributes()
    else:
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
        names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        names += ["opacity"]
        names += [f"scale_{i}" for i in range(scaling.shape[1])]
        names += [f"rot_{i}" for i in range(rotation.shape[1])]

    attributes = np.concatenate(
        [xyz, normals, f_dc, f_rest, opacity, scaling, rotation], axis=1
    ).astype(np.float32, copy=False)
    if attributes.shape[1] != len(names):
        raise RuntimeError(
            f"Gaussian PLY attribute mismatch: {attributes.shape[1]} values vs {len(names)} names"
        )
    dtype = [(name, "f4") for name in names]
    elements = np.empty(attributes.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def standalone_continuous_event_decomposition(
    scene,
    gaussians,
    event_record,
    args,
    output_dir,
    max_unprojected_points=200000,
):
    """Extract one fixed moving Gaussian part from only the initial events.

    Logic:
      1) Use a short event segment at the *beginning* only.
      2) Red/blue polarity geometry identifies the 2D part that starts moving.
      3) Canonical G0 depth turns that mask into visible 3D Gaussian seeds.
      4) The seeds are grown through the connected Gaussian manifold while
         remaining inside the same 2D moving-part projection.
      5) The resulting Gaussian set is frozen as the dynamic membership for the
         entire sequence.  Later events do not re-classify static/dynamic parts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = scene.getTrainCameras().copy()
    if len(cameras) != 1:
        raise RuntimeError(
            "Initial-motion extraction requires exactly ONE fixed camera; "
            f"Scene returned {len(cameras)}."
        )
    camera = cameras[0]

    if "events" not in event_record:
        raise RuntimeError(
            "Expected one continuous event record containing event_record['events']."
        )
    events = event_record["events"]
    if not torch.is_tensor(events):
        events = torch.as_tensor(events)

    sensor_size = event_record.get("sensor_size", None)
    if sensor_size is None:
        height = int(getattr(args, "event_height", 0))
        width = int(getattr(args, "event_width", 0))
        if height <= 0 or width <= 0:
            raise RuntimeError(
                "Event record has no sensor_size and event_height/event_width are missing."
            )
    else:
        height, width = map(int, sensor_size)

    initial_events, initial_info = _select_initial_event_segment(events, event_record, args)

    device = gaussians.get_xyz.device
    if bool(getattr(args, "load2gpu_on_the_fly", False)):
        _move_camera(camera, device)

    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(getattr(args, "white_background", False))
        else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )

    print("=" * 78)
    print("INITIAL-EVENT MOVING GAUSSIAN PART EXTRACTION")
    print("=" * 78)
    print(f"build                    : {EVENT_DSD_BUILD}")
    print(f"gaussians                : {gaussians.get_xyz.shape[0]:,}")
    print(f"whole event stream       : {int(events.shape[0]):,} events")
    print(f"initial selection        : {initial_info['selection_mode']}")
    print(f"initial events           : {initial_info['events']:,}")
    print(f"initial actual duration  : {initial_info['duration_ms']:.3f} ms")
    print(f"output                   : {output_dir}")

    with torch.no_grad():
        package = render(
            camera,
            gaussians,
            args,
            background,
            0.0,
            0.0,
            0.0,
            bool(getattr(args, "is_6dof", False)),
        )
        render_image = package["render"]
        depth = package["depth"]

    render_h, render_w = map(int, render_image.shape[-2:])
    if (render_h, render_w) != (height, width):
        raise RuntimeError(
            "Fixed-camera render and event resolution differ: "
            f"render={(render_h, render_w)}, event={(height, width)}."
        )

    _save_tensor_image(render_image, output_dir / "g0_render.png")
    finite_depth = depth[torch.isfinite(depth) & (depth > 0)]
    if finite_depth.numel() > 0:
        depth_vis = depth / finite_depth.quantile(0.99).clamp_min(1e-8)
        _save_standalone_scalar(depth_vis, output_dir / "g0_depth.png")

    # ------------------------------------------------------------------
    # ONLY the initial short segment determines dynamic membership.
    # ------------------------------------------------------------------
    positive_count, negative_count = _event_counts_from_continuous_events(
        initial_events,
        height,
        width,
    )
    (
        initial_mask,
        polarity_vis,
        activity_support,
        paired_support,
    ) = polarity_contour_to_dynamic_mask(
        positive_count,
        negative_count,
        args,
        device,
    )

    initial_np = (initial_mask.squeeze(0).detach().cpu().numpy() > 0.5).astype(np.uint8)
    if int(initial_np.sum()) == 0:
        raise RuntimeError(
            "The initial event segment produced an empty moving-part mask. "
            "Increase event_dsd_initial_duration_ms or relax the red/blue contour thresholds."
        )

    cv2.imwrite(str(output_dir / "initial_event_polarity.png"), polarity_vis)
    cv2.imwrite(str(output_dir / "initial_event_activity.png"), activity_support * 255)
    cv2.imwrite(str(output_dir / "initial_paired_support.png"), paired_support * 255)
    _save_tensor_image(initial_mask, output_dir / "initial_dynamic_mask.png")

    contour_overlay = polarity_vis.copy()
    contours, _ = cv2.findContours(
        initial_np * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(str(output_dir / "initial_dynamic_contour.png"), contour_overlay)

    output_background = (
        torch.ones_like(render_image)
        if bool(getattr(args, "white_background", False))
        else torch.zeros_like(render_image)
    )
    masked_render = initial_mask * render_image + (1.0 - initial_mask) * output_background
    _save_tensor_image(masked_render, output_dir / "initial_g0_dynamic_region.png")

    # ------------------------------------------------------------------
    # Visible event region -> 3D front-surface seeds.
    # ------------------------------------------------------------------
    points = unproject(depth, camera.w2c, camera.k)
    points = points.reshape(points.shape[0], -1)
    depth_flat = depth.reshape(-1)
    mask_flat = initial_mask.reshape(-1).bool()
    valid_depth = torch.isfinite(depth_flat) & (depth_flat > 0)
    valid_mask = mask_flat & valid_depth
    if not bool(valid_mask.any()):
        raise RuntimeError(
            "Initial moving-part mask does not overlap valid G0 depth. "
            "Inspect initial_dynamic_mask.png and g0_depth.png."
        )

    dynamic_points = points[:, valid_mask].T.detach().cpu().numpy()
    if max_unprojected_points > 0 and dynamic_points.shape[0] > int(max_unprojected_points):
        selection = np.linspace(
            0,
            dynamic_points.shape[0] - 1,
            int(max_unprojected_points),
            dtype=np.int64,
        )
        dynamic_points_for_match = dynamic_points[selection]
    else:
        dynamic_points_for_match = dynamic_points
    _write_point_cloud(output_dir / "initial_unprojected_motion_surface.ply", dynamic_points_for_match)

    seed_mask, _, manifold_radius = _gaussian_masks_from_points(
        dynamic_points_for_match, gaussians, args
    )
    if int(seed_mask.sum()) == 0:
        raise RuntimeError(
            "Initial event surface was found but zero Gaussian seeds were matched. "
            "Increase dynamic_radius_scale."
        )

    # ------------------------------------------------------------------
    # Recover the entire Gaussian part, including non-visible/back-side GS.
    # ------------------------------------------------------------------
    projection_dilation = max(
        int(getattr(args, "event_dsd_projection_mask_dilation", 2)), 0
    )
    projected_candidates = _project_gaussian_centers_to_image_mask(
        gaussians,
        camera,
        initial_np,
        dilation_px=projection_dilation,
    )
    depth_allowed, depth_info = _motion_depth_gate(
        gaussians, camera, depth, initial_np, manifold_radius, args
    )
    rejected_by_depth = projected_candidates & ~depth_allowed
    projected_candidates &= depth_allowed
    seed_mask &= depth_allowed
    if not seed_mask.any():
        raise RuntimeError(
            "Depth gate removed all seeds. Increase --rear-depth-relative or "
            "--rear-depth-spacing-scale."
        )
    print("[Depth guard]", depth_info, "rejected projected candidates:",
          int(rejected_by_depth.sum()))
    np.save(output_dir / "depth_allowed_gaussian_mask.npy", depth_allowed)
    np.save(output_dir / "depth_rejected_gaussian_mask.npy", rejected_by_depth)
    dynamic_gaussian_mask, growth_info = _grow_gaussian_part_from_initial_seeds(
        gaussians,
        seed_mask,
        projected_candidates,
        manifold_radius,
        args,
    )
    growth_info.update(depth_info)
    growth_info["depth_guard_rejected_projected"] = int(rejected_by_depth.sum())
    prune_static_gaussian_mask = dynamic_gaussian_mask.copy()

    gaussian_xyz = gaussians.get_xyz.detach().cpu().numpy()
    _write_point_cloud(
        output_dir / "dynamic_gaussian_centers.ply",
        gaussian_xyz[dynamic_gaussian_mask],
    )
    _save_dynamic_gaussian_diagnostic_ply(
        output_dir / "gaussians_dynamic_red_static_gray.ply",
        gaussian_xyz,
        dynamic_gaussian_mask,
    )
    _save_full_gaussian_subset_ply(
        output_dir / "dynamic_gaussians.ply",
        gaussians,
        dynamic_gaussian_mask,
    )
    np.save(output_dir / "dynamic_gaussian_mask.npy", dynamic_gaussian_mask)
    np.save(output_dir / "prune_static_gaussian_mask.npy", prune_static_gaussian_mask)
    np.savetxt(
        output_dir / "dynamic_gaussian_indices.txt",
        np.flatnonzero(dynamic_gaussian_mask),
        fmt="%d",
    )

    dynamic_pixels = int(initial_mask.sum().item())
    valid_depth_pixels = int(valid_mask.sum().item())
    total_gaussians = int(dynamic_gaussian_mask.shape[0])
    dynamic_gaussians = int(dynamic_gaussian_mask.sum())
    dynamic_ratio = dynamic_gaussians / max(total_gaussians, 1)

    with (output_dir / "stats.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"build={EVENT_DSD_BUILD}\n")
        handle.write("mode=initial_segment_fixed_dynamic_membership\n")
        handle.write("later_events_used_for_membership=False\n")
        handle.write(f"event_source={event_record.get('source', 'unknown')}\n")
        for key, value in initial_info.items():
            handle.write(f"initial_{key}={value}\n")
        handle.write(f"initial_positive_events={int(positive_count.sum().item())}\n")
        handle.write(f"initial_negative_events={int(negative_count.sum().item())}\n")
        handle.write(f"initial_dynamic_pixels={dynamic_pixels}\n")
        handle.write(f"initial_valid_depth_pixels={valid_depth_pixels}\n")
        handle.write(f"manifold_radius={manifold_radius:.10f}\n")
        for key, value in growth_info.items():
            handle.write(f"{key}={value}\n")
        handle.write(f"dynamic_gaussians={dynamic_gaussians}\n")
        handle.write(f"total_gaussians={total_gaussians}\n")
        handle.write(f"dynamic_gaussian_ratio={dynamic_ratio:.8f}\n")
        handle.write("dynamic_membership_scope=t0_to_end\n")
        handle.write("dynamic_gaussian_ply=dynamic_gaussians.ply\n")

    print("-" * 78)
    print(f"initial dynamic pixels : {dynamic_pixels:,}")
    print(f"visible seed Gaussians : {growth_info['seed_gaussians']:,}")
    print(f"projected candidates   : {growth_info['projected_candidates']:,}")
    print(
        f"FINAL moving Gaussians : {dynamic_gaussians:,} / {total_gaussians:,} "
        f"({100.0 * dynamic_ratio:.2f}%)"
    )
    print(f"3D growth steps        : {growth_info['growth_steps']}")
    print(f"dynamic 3DGS PLY       : {output_dir / 'dynamic_gaussians.ply'}")
    print(f"fixed membership mask  : {output_dir / 'dynamic_gaussian_mask.npy'}")
    print("membership usage       : fixed for t0 -> end; later events do NOT reclassify")
    print("-" * 78)

    if bool(getattr(args, "load2gpu_on_the_fly", False)):
        _move_camera(camera, "cpu")

    return dynamic_gaussian_mask, prune_static_gaussian_mask


def _standalone_main():
    parser = ArgumentParser(
        description=(
            "Extract the Gaussian part that starts moving from only the short "
            "initial red/blue event segment. Later events never change membership."
        )
    )
    parser.add_argument("--config", required=True, help="Path to E-D3DGS INI config")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Default: <model_path>/initial_motion_decomposition",
    )
    parser.add_argument(
        "--initial-duration-ms",
        type=float,
        default=None,
        help="Use only this many milliseconds from the beginning (default: 150 ms)",
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=None,
        help="Fallback fraction of total duration when initial-duration-ms <= 0",
    )
    parser.add_argument(
        "--initial-min-events",
        type=int,
        default=None,
        help="Minimum startup events; sparse initial windows may expand to this count",
    )
    parser.add_argument(
        "--initial-max-events",
        type=int,
        default=None,
        help="Maximum startup events used for extraction",
    )
    parser.add_argument(
        "--projection-mask-dilation",
        type=int,
        default=None,
        help="Pixel margin when recovering non-visible/back-side Gaussians",
    )
    parser.add_argument(
        "--part-connectivity-scale",
        type=float,
        default=None,
        help="3D connectivity radius in units of Gaussian manifold spacing",
    )
    parser.add_argument(
        "--rear-depth-spacing-scale", type=float, default=None,
        help="Maximum allowed rear depth margin in Gaussian spacings (default 12)",
    )
    parser.add_argument(
        "--rear-depth-relative", type=float, default=None,
        help="Minimum rear depth margin as fraction of visible median depth (default 0.015)",
    )
    parser.add_argument(
        "--max-unprojected-points",
        type=int,
        default=200000,
        help="Maximum visible event-surface points used for seed matching",
    )
    cli = parser.parse_args()

    args, print_format = _standalone_build_args(cli.config)
    # Standalone initial-motion defaults. INI values win unless CLI overrides.
    defaults = {
        "event_dsd_initial_duration_ms": 150.0,
        "event_dsd_initial_fraction": 0.03,
        "event_dsd_initial_min_events": 5000,
        "event_dsd_initial_max_events": 250000,
        "event_dsd_projection_mask_dilation": 2,
        "event_dsd_main_contour_margin": 0,
        "event_dsd_part_connectivity_scale": 3.0,
        "event_dsd_part_max_growth_steps": 64,
        "event_dsd_rear_depth_spacing_scale": 12.0,
        "event_dsd_rear_depth_relative": 0.015,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if cli.initial_duration_ms is not None:
        args.event_dsd_initial_duration_ms = cli.initial_duration_ms
    if cli.initial_fraction is not None:
        args.event_dsd_initial_fraction = cli.initial_fraction
    if cli.initial_min_events is not None:
        args.event_dsd_initial_min_events = cli.initial_min_events
    if cli.initial_max_events is not None:
        args.event_dsd_initial_max_events = cli.initial_max_events
    if cli.projection_mask_dilation is not None:
        args.event_dsd_projection_mask_dilation = cli.projection_mask_dilation
    if cli.part_connectivity_scale is not None:
        args.event_dsd_part_connectivity_scale = cli.part_connectivity_scale

    if cli.rear_depth_spacing_scale is not None:
        args.event_dsd_rear_depth_spacing_scale = cli.rear_depth_spacing_scale
    if cli.rear_depth_relative is not None:
        args.event_dsd_rear_depth_relative = cli.rear_depth_relative

    from utils import parse_utils
    cfg, _ = parse_utils.parse_ini(cli.config)
    parse_utils.print_cfg(cfg, print_format)

    if not torch.cuda.is_available():
        raise RuntimeError("Standalone Gaussian rendering requires CUDA.")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    gaussians = GaussianModel(int(getattr(args, "sh_degree", 3)))
    scene = Scene(args, gaussians)

    event_list = scene.getEvents()
    if len(event_list) != 1:
        raise RuntimeError(
            "Initial-motion extraction expects exactly one continuous event record, "
            f"but Scene returned {len(event_list)}."
        )

    if cli.output is None:
        model_path = Path(str(getattr(args, "model_path", "/E-D3DGS/output")))
        output_dir = model_path / "initial_motion_decomposition"
    else:
        output_dir = Path(cli.output)

    standalone_continuous_event_decomposition(
        scene=scene,
        gaussians=gaussians,
        event_record=event_list[0],
        args=args,
        output_dir=output_dir,
        max_unprojected_points=cli.max_unprojected_points,
    )


if __name__ == "__main__":
    _standalone_main()
