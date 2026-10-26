#!/usr/bin/env python3
"""Restore missing 3D Gaussians using the original G0 as a geometric prior.

Five orthographic inspection directions are used, excluding the specified front.
The original PLY is NOT changed; the refined PLY is NOT overwritten.
Note: this is a conservative geometric restoration, not multi-view RGB training.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
from plyfile import PlyData, PlyElement

AXES = {
    '+x': np.array([1., 0., 0.]), '-x': np.array([-1., 0., 0.]),
    '+y': np.array([0., 1., 0.]), '-y': np.array([0., -1., 0.]),
    '+z': np.array([0., 0., 1.]), '-z': np.array([0., 0., -1.]),
}

def xyz(data):
    for key in ('x', 'y', 'z'):
        if key not in data.dtype.names:
            raise ValueError(f'Missing PLY Gaussian field {key!r}')
    return np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)

def closest_dist(query_xyz, reference_xyz, workers=4, chunk=100_000):
    tree = cKDTree(reference_xyz)
    out = np.empty(len(query_xyz), np.float32)
    for start in range(0, len(query_xyz), chunk):
        end = min(start + chunk, len(query_xyz))
        out[start:end] = tree.query(query_xyz[start:end], k=1, workers=workers)[0]
    return out

def projection_coords(xyz_arr, view, resolution, lo, span):
    dim = int(np.argmax(np.abs(view)))
    ax = [i for i in range(3) if i != dim]
    uv = np.floor((xyz_arr[:, ax] - lo[ax]) / span[ax] * (resolution - 1)).astype(np.int32)
    uv = np.clip(uv, 0, resolution - 1)
    depth = xyz_arr[:, dim] * view[dim]
    return uv[:, 0], uv[:, 1], depth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--g0', required=True, help='Original unrefined G0 PLY')
    ap.add_argument('--refined', required=True, help='Refined corrected_g0.ply')
    ap.add_argument('--out', required=True, help='Output PLY, distinct from inputs')
    ap.add_argument('--front-axis', choices=AXES, default='+z',
                    help='Which world coordinate direction roughly faces original camera. Exclude it from the 5 view tests.')
    ap.add_argument('--resolution', type=int, default=512)
    ap.add_argument('--min-distance-mult', type=float, default=2.0,
                    help='G0-restored candidates must be >= this x median G0 NN spacing from the refined PLY')
    ap.add_argument('--depth-margin-mult', type=float, default=1.5)
    ap.add_argument('--max-add-ratio', type=float, default=0.40,
                    help='Cap restored points at this fraction of original G0 (choose higher for larger missing areas)')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    original_path = Path(args.g0).resolve()
    refined_path = Path(args.refined).resolve()
    output_path = Path(args.out).resolve()
    if output_path in (original_path, refined_path):
        ap.error('--out must be a NEW path (will not overwrite either input).')
    print('[1/6] Read PLY...', flush=True)
    original_ply = PlyData.read(str(original_path))
    refined_ply = PlyData.read(str(refined_path))
    orig = original_ply['vertex'].data
    refined = refined_ply['vertex'].data
    raw_xyz, ref_xyz = xyz(orig), xyz(refined)
    if not len(raw_xyz) or not len(ref_xyz):
        raise ValueError('Empty Gaussian PLY')
    print(f'  G0={len(orig):,}, corrected={len(refined):,}', flush=True)

    print('[2/6] Estimate original surface spacing...', flush=True)
    rng = np.random.default_rng(2026)
    sampled = rng.choice(len(raw_xyz), min(len(raw_xyz), 20_000), replace=False)
    raw_tree = cKDTree(raw_xyz)
    spacing = raw_tree.query(raw_xyz[sampled], k=2, workers=args.workers)[0][:, 1]
    spacing = float(np.median(spacing[spacing > 1e-9]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Cannot estimate G0 point spacing')
    print(f'  median G0 NN spacing: {spacing:.8g}', flush=True)

    print('[3/6] Find original surface Gaussians missing in corrected G0...', flush=True)
    dist = closest_dist(raw_xyz, ref_xyz, workers=args.workers)
    candidates = np.flatnonzero(dist > args.min_distance_mult * spacing)
    print(f'  spatially missing candidates: {len(candidates):,}', flush=True)
    if len(candidates) == 0:
        print('No sufficiently missing G0 Gaussians; nothing to restore.')
        return

    print('[4/6] Five-direction projected depth/coverage check...', flush=True)
    lo = np.minimum(raw_xyz.min(axis=0), ref_xyz.min(axis=0))
    hi = np.maximum(raw_xyz.max(axis=0), ref_xyz.max(axis=0))
    pad = np.maximum((hi - lo) * 0.025, spacing * 2)
    lo -= pad
    hi += pad
    span = np.maximum(hi - lo, 1e-7)
    selected = np.zeros(len(candidates), dtype=bool)
    for key, direction in AXES.items():
        if key == args.front_axis:
            continue
        r_u, r_v, r_depth = projection_coords(ref_xyz, direction, args.resolution, lo, span)
        depth_map = np.full((args.resolution, args.resolution), -np.inf, dtype=np.float32)
        np.maximum.at(depth_map, (r_v, r_u), r_depth)
        depth_map = maximum_filter(depth_map, size=3, mode='constant', cval=-np.inf)
        c_u, c_v, c_depth = projection_coords(raw_xyz[candidates], direction, args.resolution, lo, span)
        local = depth_map[c_v, c_u]
        uncovered = ~np.isfinite(local)
        newly_visible = c_depth > local + args.depth_margin_mult * spacing
        match = uncovered | newly_visible
        selected |= match
        print(f'  {key}: missing in projection {np.count_nonzero(match):,}', flush=True)

    chosen = candidates[selected]
    limit = int(len(orig) * args.max_add_ratio)
    if len(chosen) > limit:
        # Most missing in 3D first: avoid adding near-duplicates when capped.
        chosen = chosen[np.argpartition(dist[chosen], -limit)[-limit:]] if limit else chosen[:0]
    print(f'[5/6] Restore {len(chosen):,} original Gaussians...', flush=True)

    # Use corrected PLY schema; retain the corrected cloud unchanged.
    # For any missing properties in original, use safe neutral defaults.
    additions = np.zeros(len(chosen), dtype=refined.dtype)
    original_fields = set(orig.dtype.names)
    missing_fields = []
    for field in refined.dtype.names:
        if field in original_fields:
            additions[field] = orig[field][chosen]
        else:
            missing_fields.append(field)
            if field == 'rot_0':
                additions[field] = 1.0
            elif field == 'opacity':
                additions[field] = 0.0
    if missing_fields:
        print('WARNING: G0 lacks corrected PLY fields:', missing_fields[:15])
        print('Those fields used neutral defaults. Check resulting rendering.')
    merged = np.empty(len(refined) + len(additions), dtype=refined.dtype)
    merged[:len(refined)] = refined
    merged[len(refined):] = additions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comments = list(getattr(refined_ply, 'comments', [])) + [
        'Restored original G0 Gaussians based on 5-view projected coverage and 3D novelty',
        f'front_axis_excluded={args.front_axis}, added={len(additions)}',
    ]
    print('[6/6] Write binary PLY...', flush=True)
    PlyData([PlyElement.describe(merged, 'vertex')], text=False, comments=comments).write(str(output_path))
    print(f'DONE: {output_path}')
    print(f'  corrected={len(refined):,} + restored={len(additions):,} => total={len(merged):,}')
    print('IMPORTANT: This restores a subset of ORIGINAL geometry; it does not recover unseen GT or guarantee front silhouette.')

if __name__ == '__main__':
    main()
