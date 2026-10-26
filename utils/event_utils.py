from pathlib import Path
import os
import json
import random

import networkx as nx
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
from tqdm import tqdm

from scene.cameras import Camera


def binary_search_torch_tensor(t, l, r, x, side="left"):
    if r is None:
        r = len(t) - 1

    while l <= r:
        if t[l] == x:
            return l
        if t[r] == x:
            return r

        mid = l + (r - l) // 2
        midval = t[mid]

        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1

    if side == "left":
        return l
    return r


def events_to_image(xs, ys, ps, sensor_size=(180, 240)):
    xs_mask = (xs >= sensor_size[1]) | (xs < 0)
    ys_mask = (ys >= sensor_size[0]) | (ys < 0)
    mask = xs_mask | ys_mask

    xs = xs.clone()
    ys = ys.clone()
    ps = ps.clone()

    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0

    if xs.dtype is not torch.long:
        xs = xs.long()
    if ys.dtype is not torch.long:
        ys = ys.long()

    img = torch.zeros(
        list(sensor_size),
        device=xs.device,
        dtype=torch.int64,
    )
    img.index_put_((ys, xs), ps.long(), accumulate=True)
    return img


def events_to_stack(xs, ys, ts, ps, B, sensor_size=(400, 400)):
    """
    Convert events in one time interval into a [2, B, H, W] count tensor.

    Channel 0: positive-event counts
    Channel 1: negative-event counts stored as positive magnitudes

    tss_list contains B+1 temporal boundaries in the SAME timestamp unit as ts.
    MonoEv normalizes all event timestamps to int64 nanoseconds before calling
    this function.
    """
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")

    device = xs.device

    if len(ts) <= 3:
        return (
            torch.zeros(
                [2, B, sensor_size[0], sensor_size[1]],
                dtype=torch.int64,
                device=device,
            ),
            torch.zeros(B + 1, dtype=torch.int64, device=device),
        )

    assert len(xs) == len(ys) == len(ts) == len(ps)

    if not torch.all(ts[1:] >= ts[:-1]):
        order = torch.argsort(ts)
        xs = xs[order]
        ys = ys[order]
        ts = ts[order]
        ps = ps[order]

    ts = ts.to(torch.int64)
    dt = ts[-1] - ts[0]

    if dt <= 0:
        return (
            torch.zeros(
                [2, B, sensor_size[0], sensor_size[1]],
                dtype=torch.int64,
                device=device,
            ),
            torch.full(
                (B + 1,),
                int(ts[0].item()),
                dtype=torch.int64,
                device=device,
            ),
        )

    positives = []
    negatives = []
    tss_list = []

    # Use floating-point boundaries only for bin construction, while the
    # original event timestamps remain int64 nanoseconds.
    t0 = float(ts[0].item())
    t1 = float(ts[-1].item())
    delta_t = (t1 - t0) / float(B)

    for bi in range(B):
        tstart = t0 + delta_t * bi
        tend = t0 + delta_t * (bi + 1)

        tss_list.append(int(round(tstart)))

        beg = int(torch.searchsorted(
            ts,
            torch.tensor(int(np.floor(tstart)), dtype=ts.dtype, device=ts.device),
            right=False,
        ).item())

        end = int(torch.searchsorted(
            ts,
            torch.tensor(int(np.ceil(tend)), dtype=ts.dtype, device=ts.device),
            right=True,
        ).item())

        beg = max(0, min(beg, len(ts)))
        end = max(beg, min(end, len(ts)))

        xs_bin = xs[beg:end]
        ys_bin = ys[beg:end]
        ps_bin = ps[beg:end]

        if len(ps_bin) == 0:
            vp = torch.zeros(sensor_size, dtype=torch.int64, device=device)
            vn = torch.zeros(sensor_size, dtype=torch.int64, device=device)
        else:
            pos_p = torch.where(
                ps_bin > 0,
                torch.ones_like(ps_bin),
                torch.zeros_like(ps_bin),
            )
            neg_p = torch.where(
                ps_bin < 0,
                -torch.ones_like(ps_bin),
                torch.zeros_like(ps_bin),
            )

            vp = events_to_image(xs_bin, ys_bin, pos_p, sensor_size=sensor_size)
            vn = events_to_image(xs_bin, ys_bin, neg_p, sensor_size=sensor_size)

        positives.append(vp)
        negatives.append(vn)

    tss_list.append(int(ts[-1].item()))

    positives_b = torch.stack(positives)
    negatives_b = torch.stack(negatives)
    stack = torch.stack([positives_b, negatives_b])
    tss_tensor = torch.tensor(tss_list, dtype=torch.int64, device=device)

    return stack, tss_tensor


def add_noise(ECM, noise_std=1.0, noise_fraction=0.05):
    noise = (noise_std * torch.randn_like(ECM.float())).abs().int()

    if noise_fraction < 1.0:
        mask = torch.rand_like(ECM.float()) >= noise_fraction
        noise.masked_fill_(mask, 0)

    return ECM + noise


def _structured_events_to_array(stream, event_path):
    names = set(stream.dtype.names or [])

    if "x" not in names or "y" not in names:
        raise ValueError(f"{event_path}: structured events must contain x and y")

    if "timestamp" in names:
        t_name = "timestamp"
    elif "t" in names:
        t_name = "t"
    elif "ts" in names:
        t_name = "ts"
    else:
        raise ValueError(
            f"{event_path}: structured events must contain timestamp/t/ts"
        )

    if "polarity" in names:
        p_name = "polarity"
    elif "p" in names:
        p_name = "p"
    elif "pol" in names:
        p_name = "pol"
    else:
        raise ValueError(
            f"{event_path}: structured events must contain polarity/p/pol"
        )

    return np.column_stack([
        stream["x"],
        stream["y"],
        stream[t_name],
        stream[p_name],
    ])


def _load_event_npz(event_path):
    """
    Supported NPZ layouts:
      1) x, y, t, p
      2) x, y, timestamp, polarity
      3) event_stream structured array
      4) event_stream Nx4 array

    Returns an ndarray with columns [x, y, t, p].
    """
    data = np.load(event_path, allow_pickle=False)

    if all(key in data.files for key in ("x", "y", "t", "p")):
        return np.column_stack([
            data["x"],
            data["y"],
            data["t"],
            data["p"],
        ])

    if all(key in data.files for key in ("x", "y", "timestamp", "polarity")):
        return np.column_stack([
            data["x"],
            data["y"],
            data["timestamp"],
            data["polarity"],
        ])

    if "event_stream" not in data.files:
        raise ValueError(
            f"{event_path}: expected x/y/t/p arrays or event_stream"
        )

    stream = data["event_stream"]

    if stream.dtype.names is not None:
        return _structured_events_to_array(stream, event_path)

    stream = np.asarray(stream)

    if stream.ndim != 2 or stream.shape[1] != 4:
        raise ValueError(
            f"{event_path}: event_stream must have shape (N,4), got {stream.shape}"
        )

    return stream


def _load_event_npy(event_path):
    """
    Supported NPY layouts:
      1) plain Nx4 array, interpreted as [x, y, t, p]
      2) structured array with x/y/timestamp(or t)/polarity(or p)
    """
    stream = np.load(event_path, allow_pickle=False)

    if stream.dtype.names is not None:
        return _structured_events_to_array(stream, event_path)

    stream = np.asarray(stream)

    if stream.ndim != 2 or stream.shape[1] != 4:
        raise ValueError(
            f"{event_path}: .npy events must have shape (N,4), got {stream.shape}"
        )

    return stream


def _load_event_file(event_path):
    event_path = Path(event_path)
    suffix = event_path.suffix.lower()

    if suffix == ".npy":
        return _load_event_npy(event_path)

    if suffix == ".npz":
        return _load_event_npz(event_path)

    raise ValueError(
        f"Unsupported event file extension: {event_path.suffix}. "
        "MonoEv supports .npy and .npz."
    )


def _time_scale_to_ns(unit):
    unit = str(unit).strip().lower()

    table = {
        "s": 1e9,
        "sec": 1e9,
        "second": 1e9,
        "seconds": 1e9,
        "ms": 1e6,
        "millisecond": 1e6,
        "milliseconds": 1e6,
        "us": 1e3,
        "µs": 1e3,
        "microsecond": 1e3,
        "microseconds": 1e3,
        "ns": 1.0,
        "nanosecond": 1.0,
        "nanoseconds": 1.0,
    }

    if unit not in table:
        raise ValueError(
            f"Unsupported event_time_unit={unit!r}. "
            "Use s, ms, us, or ns."
        )

    return table[unit]


def _normalize_polarity(p):
    p = np.asarray(p)
    unique = np.unique(p)

    # Common exported event convention: 0 = negative, 1 = positive. ESIM often already uses -1/+1.
    if np.all(np.isin(unique, [0, 1, False, True])):
        return np.where(p > 0, 1, -1).astype(np.int8)

    # Already signed.
    if np.all(np.isin(unique, [-1, 1])):
        return p.astype(np.int8)

    # Generic fallback: positive -> +1, zero/negative -> -1.
    return np.where(p > 0, 1, -1).astype(np.int8)


def _prepare_continuous_events(raw_events, args, sensor_size):
    """
    Normalize a raw Nx4 array to:
        x: int64
        y: int64
        t: int64 nanoseconds, sorted and optionally zero-based
        p: int8 {-1, +1}
    """
    raw_events = np.asarray(raw_events)

    if raw_events.ndim != 2 or raw_events.shape[1] != 4:
        raise ValueError(
            f"Expected raw events with shape (N,4), got {raw_events.shape}"
        )

    if len(raw_events) == 0:
        raise RuntimeError("Event stream is empty")

    x = np.asarray(raw_events[:, 0], dtype=np.int64)
    y = np.asarray(raw_events[:, 1], dtype=np.int64)
    t = np.asarray(raw_events[:, 2], dtype=np.float64)
    p = _normalize_polarity(raw_events[:, 3])

    if not np.isfinite(t).all():
        raise ValueError("Event timestamps contain NaN/Inf")

    event_time_unit = getattr(args, "event_time_unit", "s")
    time_scale = _time_scale_to_ns(event_time_unit)

    t_ns = np.rint(t * time_scale).astype(np.int64)

    order = np.argsort(t_ns, kind="stable")
    x = x[order]
    y = y[order]
    t_ns = t_ns[order]
    p = p[order]

    zero_time = bool(getattr(args, "event_zero_time", True))
    original_t0_ns = int(t_ns[0])

    if zero_time:
        t_ns = t_ns - original_t0_ns

    H, W = sensor_size
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    invalid_count = int((~valid).sum())

    if invalid_count > 0:
        print(
            f"[MonoEv] Dropping {invalid_count} out-of-bounds events "
            f"for sensor {W}x{H}"
        )
        x = x[valid]
        y = y[valid]
        t_ns = t_ns[valid]
        p = p[valid]

    if len(t_ns) < 4:
        raise RuntimeError(
            f"Only {len(t_ns)} valid events remain after preprocessing"
        )

    events = np.column_stack([
        x.astype(np.int64),
        y.astype(np.int64),
        t_ns.astype(np.int64),
        p.astype(np.int64),
    ])

    meta = {
        "event_count": int(events.shape[0]),
        "time_unit_input": str(event_time_unit),
        "time_unit_internal": "ns",
        "zero_time": zero_time,
        "original_t0_ns": original_t0_ns,
        "t_start_ns": int(events[0, 2]),
        "t_end_ns": int(events[-1, 2]),
        "duration_ns": int(events[-1, 2] - events[0, 2]),
        "x_min": int(events[:, 0].min()),
        "x_max": int(events[:, 0].max()),
        "y_min": int(events[:, 1].min()),
        "y_max": int(events[:, 1].max()),
        "positive_events": int((events[:, 3] > 0).sum()),
        "negative_events": int((events[:, 3] < 0).sum()),
    }

    return events, meta


def _is_mono_stage3(args):
    initial_gaussian_ply = getattr(args, "initial_gaussian_ply", None)

    if initial_gaussian_ply is None:
        return False

    return str(initial_gaussian_ply).strip().lower() not in (
        "",
        "none",
        "null",
    )


def _resolve_event_path(args):
    event_path = getattr(args, "event_path", None)

    if event_path is None or str(event_path).strip().lower() in (
        "",
        "none",
        "null",
    ):
        raise ValueError(
            "MonoEv Stage-3 requires event_path pointing to one continuous "
            ".npy or .npz event stream."
        )

    event_path = Path(os.path.expanduser(str(event_path)))

    if not event_path.is_absolute():
        event_path = Path(args.source_path) / event_path

    event_path = event_path.resolve()

    if not event_path.exists():
        raise FileNotFoundError(f"Event path does not exist: {event_path}")

    return event_path


def _load_monoev_continuous_event_stream(sorted_cameras, args):
    """
    MonoEv Stage-3 event loader for one continuous ESIM .npy/.npz stream.

    Input:
        exactly ONE fixed camera
        exactly ONE continuous event file (.npy/.npz)

    Output:
        event_list containing one dictionary with the complete normalized
        event stream on CPU. Random temporal windows are created later by
        sampling_event_window().
    """
    if len(sorted_cameras) != 1:
        raise RuntimeError(
            "MonoEv Stage-3 requires exactly one fixed camera, "
            f"got {len(sorted_cameras)}"
        )

    camera = sorted_cameras[0]

    event_height = int(getattr(args, "event_height", camera.image_height))
    event_width = int(getattr(args, "event_width", camera.image_width))

    if event_height <= 0 or event_width <= 0:
        raise ValueError(
            f"Invalid event resolution: {event_width}x{event_height}"
        )

    if int(camera.image_width) != event_width or int(camera.image_height) != event_height:
        raise ValueError(
            "Fixed camera resolution and event resolution must match in "
            "MonoEv Stage-3. "
            f"camera={camera.image_width}x{camera.image_height}, "
            f"events={event_width}x{event_height}. "
            "Fix event_width/event_height in the config before training."
        )

    event_path = _resolve_event_path(args)

    if event_path.is_dir():
        raise ValueError(
            "MonoEv Stage-3 now expects one continuous event file, not an "
            f"event directory: {event_path}"
        )

    print(f"[MonoEv] Loading continuous event stream: {event_path}")

    raw_events = _load_event_file(event_path)

    events, meta = _prepare_continuous_events(
        raw_events,
        args,
        sensor_size=(event_height, event_width),
    )

    print(f"[MonoEv] events: {meta['event_count']:,}")
    print(
        f"[MonoEv] time: {meta['t_start_ns']} -> {meta['t_end_ns']} ns "
        f"(duration={meta['duration_ns'] / 1e9:.6f} s)"
    )
    print(
        f"[MonoEv] coordinates: x={meta['x_min']}..{meta['x_max']}, "
        f"y={meta['y_min']}..{meta['y_max']}"
    )
    print(
        f"[MonoEv] polarity: +={meta['positive_events']:,}, "
        f"-={meta['negative_events']:,}"
    )

    # Keep the full stream on CPU. A training iteration copies only its sampled
    # window to GPU, avoiding several million events living on the GPU.
    event_tensor = torch.from_numpy(events).to(torch.int64).cpu()

    return [{
        "events": event_tensor,
        "t_start_ns": int(meta["t_start_ns"]),
        "t_end_ns": int(meta["t_end_ns"]),
        "duration_ns": int(meta["duration_ns"]),
        "sensor_size": (event_height, event_width),
        "source": str(event_path),
        "meta": meta,
    }]


def _load_legacy_event_stream(sorted_cameras, args):
    """
    Original E-D3DGS-style loader:
    build one ECM interval between consecutive temporal camera records using
    a directory of NPZ chunks.
    """
    if len(sorted_cameras) < 2:
        raise RuntimeError("At least two temporal camera records are required")

    frame_id_list = [int(cam.colmap_id) for cam in sorted_cameras]

    event_path = getattr(args, "event_path", None)

    if event_path is None or str(event_path).strip().lower() in (
        "",
        "none",
        "null",
    ):
        event_root = Path(args.source_path) / "events"
    else:
        event_root = Path(os.path.expanduser(str(event_path)))
        if not event_root.is_absolute():
            event_root = Path(args.source_path) / event_root

    if not event_root.is_dir():
        raise FileNotFoundError(
            f"Legacy event directory does not exist: {event_root}"
        )

    event_paths = sorted(
        p for p in event_root.iterdir()
        if p.is_file() and p.suffix.lower() == ".npz"
    )

    if not event_paths:
        raise RuntimeError(f"No .npz event files found in {event_root}")

    j = 1
    event_acc = None
    event_list = []

    print("Loading Training Event stream")

    for i, event_path in tqdm(
        enumerate(event_paths),
        total=len(event_paths),
    ):
        event_i = _load_event_npz(event_path)

        if event_acc is None:
            event_acc = event_i
        else:
            event_acc = np.concatenate((event_acc, event_i), axis=0)

        if j < len(frame_id_list) and i == frame_id_list[j] - 1:
            event_acc = torch.from_numpy(np.asarray(event_acc))

            ECM, tss_ns = events_to_stack(
                xs=event_acc[:, 0],
                ys=event_acc[:, 1],
                ts=event_acc[:, 2],
                ps=event_acc[:, 3],
                B=args.bin_num,
                sensor_size=(args.event_height, args.event_width),
            )

            if bool(getattr(args, "spatially_varying_thresholds", False)):
                ECM = add_noise(
                    ECM,
                    noise_fraction=float(
                        getattr(args, "noise_fraction", 0.05)
                    ),
                )

            event_list.append({
                "ECM": ECM,
                "tss": tss_ns,
            })

            event_acc = None
            j += 1

    expected = len(sorted_cameras) - 1

    if len(event_list) != expected:
        raise RuntimeError(
            f"Expected {expected} event intervals for "
            f"{len(sorted_cameras)} temporal camera records, but constructed "
            f"{len(event_list)}. Check camera IDs and NPZ ordering."
        )

    return event_list


def load_event_stream(sorted_cameras, args):
    """
    Unified event loader.

    MonoEv Stage-3:
        one fixed camera + one continuous .npy/.npz event stream.

    Legacy E-D3DGS:
        multiple temporal camera records + directory of .npz chunks.
    """
    if _is_mono_stage3(args):
        return _load_monoev_continuous_event_stream(sorted_cameras, args)

    return _load_legacy_event_stream(sorted_cameras, args)


def sampling_event_window(event_list, args):
    """
    Sample an arbitrary continuous time window for MonoEv Stage-3.

    Returns:
        ECM:       [2, B, H, W], positive/negative event counts
        t_start:   int64 ns
        t_end:     int64 ns

    The training code should render G(t_start) and G(t_end) with the SAME
    fixed camera and compare their log-intensity change against this ECM.

    Window size is controlled by:
        event_slice_min_events
        event_slice_max_events

    This event-count-based sampling is robust to nonuniform event rates.
    """
    if not event_list:
        raise RuntimeError("event_list is empty")

    record = event_list[0]

    if "events" not in record:
        raise RuntimeError(
            "sampling_event_window() is for MonoEv continuous streams only"
        )

    events = record["events"]
    num_events = int(events.shape[0])

    min_events = int(getattr(args, "event_slice_min_events", 5000))
    max_events = int(getattr(args, "event_slice_max_events", 10000))

    if min_events < 4:
        min_events = 4

    if max_events < min_events:
        max_events = min_events

    max_events = min(max_events, num_events)
    min_events = min(min_events, max_events)

    if num_events < 4:
        raise RuntimeError(
            f"Not enough events to sample a training window: {num_events}"
        )

    window_events = random.randint(min_events, max_events)

    if num_events == window_events:
        start_idx = 0
    else:
        start_idx = random.randint(0, num_events - window_events)

    end_idx = start_idx + window_events
    sampled = events[start_idx:end_idx]

    t_start = int(sampled[0, 2].item())
    t_end = int(sampled[-1, 2].item())

    # Very rarely many events can share the same timestamp. Expand the window
    # to the right until a positive temporal span is obtained.
    if t_end <= t_start:
        probe_end = end_idx
        while probe_end < num_events and t_end <= t_start:
            probe_end = min(probe_end + max(32, window_events // 10), num_events)
            sampled = events[start_idx:probe_end]
            t_end = int(sampled[-1, 2].item())

        if t_end <= t_start:
            raise RuntimeError(
                "Sampled event window has zero temporal duration"
            )

    B = int(getattr(args, "bin_num", 64))
    sensor_size = tuple(record["sensor_size"])

    # Only the sampled window is transferred to GPU.
    sampled = sampled.cuda(non_blocking=True)

    ECM, tss = events_to_stack(
        xs=sampled[:, 0],
        ys=sampled[:, 1],
        ts=sampled[:, 2],
        ps=sampled[:, 3],
        B=B,
        sensor_size=sensor_size,
    )

    if bool(getattr(args, "spatially_varying_thresholds", False)):
        ECM = add_noise(
            ECM,
            noise_fraction=float(getattr(args, "noise_fraction", 0.05)),
        )

    return ECM, torch.tensor(t_start, dtype=torch.int64), torch.tensor(
        t_end,
        dtype=torch.int64,
    )


def sampling_event(event_list, train_idx, train_camera_num, args):
    """
    Legacy sampler kept for original E-D3DGS compatibility.

    MonoEv Stage-3 should NOT use this function anymore. Use
    sampling_event_window() so both t_start and t_end are explicitly known.
    """
    if _is_mono_stage3(args):
        raise RuntimeError(
            "MonoEv Stage-3 must use sampling_event_window(), not "
            "sampling_event(). The fixed-camera formulation requires explicit "
            "t_start and t_end for every sampled event window."
        )

    if train_camera_num < 2:
        raise RuntimeError("Need at least two temporal records")

    if len(event_list) != train_camera_num - 1:
        raise RuntimeError(
            f"event_list has {len(event_list)} intervals, "
            f"but train_camera_num={train_camera_num}"
        )

    use_past = (
        train_idx == train_camera_num - 1
        or (train_idx > 0 and np.random.random() < 0.5)
    )

    if use_past:
        interval_idx = train_idx - 1
        part_flag = "past"
    else:
        interval_idx = train_idx
        part_flag = "future"

    ECM = event_list[interval_idx]["ECM"].cuda(non_blocking=True)
    tss_list = event_list[interval_idx]["tss"]

    available_bins = ECM.shape[1]

    if available_bins < 2:
        raise RuntimeError(
            f"Event interval {interval_idx} has only {available_bins} bins"
        )

    sampled_idx = random.choice(range(1, available_bins))

    if part_flag == "past":
        ECM_sampled = ECM[:, -sampled_idx:, :, :]
        sampled_tss = tss_list[-(sampled_idx + 1)]
    else:
        ECM_sampled = ECM[:, :sampled_idx, :, :]
        sampled_tss = tss_list[sampled_idx]

    return ECM_sampled, sampled_tss, part_flag


def rgb_to_luma(rgb, esim=True):
    device = rgb.device

    if esim:
        r = 0.299
        g = 0.587
        b = 0.114
    else:
        r = 0.2126
        g = 0.7152
        b = 0.0722

    factors = torch.tensor(
        [r, g, b],
        device=device,
        dtype=rgb.dtype,
    )

    luma = torch.sum(rgb * factors[None, :], axis=-1)
    return luma[..., None]


def lin_log(color, linlog_thres=20):
    lin_slope = np.log(linlog_thres) / linlog_thres

    return torch.where(
        color < linlog_thres,
        lin_slope * color,
        torch.log(color),
    )


def make_event_frame(event_tensor, resolution):
    H, W = resolution

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    pos_count = torch.bincount(
        pos[:, 0].long() + pos[:, 1].long() * W,
        minlength=H * W,
    ).reshape(H, W)

    neg_count = torch.bincount(
        neg[:, 0].long() + neg[:, 1].long() * W,
        minlength=H * W,
    ).reshape(H, W)

    event_count = pos_count - neg_count
    return event_count.unsqueeze(-1).float()


def _blender_frame_to_pose(frame):
    matrix = np.linalg.inv(
        np.asarray(frame["transform_matrix"], dtype=np.float64)
    )

    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]

    Rt = np.eye(4, dtype=np.float64)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = T

    C2W = np.linalg.inv(Rt)
    return C2W[:3, :3], C2W[:3, 3]


def get_interpolator(args):
    """
    Legacy dynamic-camera interpolation.

    MonoEv Stage-3 has one physically fixed camera, so no camera interpolation
    exists. Scene.__init__ should skip this call, but returning Nones here makes
    accidental calls harmless.
    """
    if _is_mono_stage3(args):
        return None, None, None, None

    camera_file = getattr(args, "camera_file", None)

    if camera_file is None or str(camera_file).strip().lower() in (
        "",
        "none",
        "null",
    ):
        transforms_path = os.path.join(
            args.source_path,
            "transforms.json",
        )
    else:
        transforms_path = os.path.expanduser(str(camera_file))

        if not os.path.isabs(transforms_path):
            transforms_path = os.path.join(
                args.source_path,
                transforms_path,
            )

    with open(transforms_path, "r", encoding="utf-8") as json_file:
        contents = json.load(json_file)

    frames = contents["frames"]

    if not frames:
        raise RuntimeError(f"No frames in {transforms_path}")

    tss_ns = np.array(
        [
            float(frame.get("time", idx / max(len(frames) - 1, 1))) * 1e9
            for idx, frame in enumerate(frames)
        ],
        dtype=np.float64,
    )

    rots = []
    trans = []

    for frame in frames:
        rot, tra = _blender_frame_to_pose(frame)
        rots.append(rot)
        trans.append(tra)

    rots = np.stack(rots)
    trans = np.stack(trans)

    rot_interpolator = Slerp(
        tss_ns,
        Rotation.from_matrix(rots),
    )

    kind = "cubic" if len(tss_ns) >= 4 else "linear"

    trans_interpolator = interp1d(
        x=tss_ns,
        y=trans,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=(trans[0], trans[-1]),
    )

    tss_bd_start = float(np.min(tss_ns))
    tss_bd_end = float(np.max(tss_ns))

    return (
        rot_interpolator,
        trans_interpolator,
        tss_bd_start,
        tss_bd_end,
    )


def RT_from_rots_trans(rots, trans):
    assert rots.shape == (3, 3)
    assert trans.shape == (3,)

    rt_C2W = np.eye(4, dtype=np.float64)
    rt_C2W[:3, :3] = rots.copy()
    rt_C2W[:3, 3] = trans.copy()

    RT_W2C = np.linalg.inv(rt_C2W)

    R = RT_W2C[:3, :3].transpose()
    T = RT_W2C[:3, 3]

    return R, T


def interpolating_virtual_camera(event_tss, scene, viewpoint_cam, args):
    """
    Create a camera at an event timestamp.

    MonoEv:
        R/T/FoV are copied exactly from the single fixed camera.
        Only fid changes.

    Legacy:
        interpolate camera pose from transforms.json.
    """
    event_tss = float(
        event_tss.item()
        if torch.is_tensor(event_tss)
        else event_tss
    )

    if _is_mono_stage3(args):
        R_at_event = np.asarray(
            viewpoint_cam.R,
            dtype=np.float64,
        ).copy()

        T_at_event = np.asarray(
            viewpoint_cam.T,
            dtype=np.float64,
        ).copy()
    else:
        if scene.tss_bd_start is not None and event_tss < scene.tss_bd_start:
            event_tss = float(scene.tss_bd_start)
        elif scene.tss_bd_end is not None and event_tss > scene.tss_bd_end:
            event_tss = float(scene.tss_bd_end)

        rots_at_event = scene.rot_interpolator(event_tss).as_matrix()
        trans_at_event = scene.trans_interpolator(event_tss)

        R_at_event, T_at_event = RT_from_rots_trans(
            rots_at_event,
            trans_at_event,
        )

    image_height = int(getattr(viewpoint_cam, "height", args.event_height))
    image_width = int(getattr(viewpoint_cam, "width", args.event_width))

    event_camera = Camera(
        colmap_id=getattr(viewpoint_cam, "colmap_id", None),
        R=R_at_event,
        T=T_at_event,
        FoVx=viewpoint_cam.FoVx,
        FoVy=viewpoint_cam.FoVy,
        image=torch.zeros(
            3,
            image_height,
            image_width,
            dtype=torch.float32,
        ),
        gt_alpha_mask=None,
        image_name=f"event_{int(event_tss)}",
        uid=getattr(viewpoint_cam, "uid", None),
        data_device=(
            args.data_device
            if not args.load2gpu_on_the_fly
            else "cpu"
        ),
        fid=event_tss,
        depth=None,
    )

    event_camera.has_gt_image = False
    event_camera.is_monoev_fixed_camera = _is_mono_stage3(args)

    return event_camera


def merge_ECM_with_thres(ECM, threshold):
    ECM_with_thres = ECM * threshold

    pos = ECM_with_thres[0, :, :, :]
    neg = ECM_with_thres[1, :, :, :]

    pos = torch.sum(pos, dim=0)
    neg = torch.sum(neg, dim=0)

    merge_event_frame = pos - neg

    return merge_event_frame.unsqueeze(-1)


def find_max_connected_components(array):
    G = nx.Graph()
    G.add_edges_from(array)

    connected_components = sorted(
        nx.connected_components(G),
        key=len,
        reverse=True,
    )

    return [
        list(component)
        for component in connected_components
    ]
