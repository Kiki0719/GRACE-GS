#!/usr/bin/env python3
"""MonoEv Stage-1: geometry-preservation diagnostics (no multiview render).

Compare original and corrected Gaussian centers, scales, and opacity without
pretending model-to-model agreement is ground-truth reconstruction quality.
Optional GT point-cloud/mesh requires an explicitly supplied GT-to-G0 transform
(or --gt-already-aligned). All distance values are in the G0 coordinate system.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--g0', required=True, help='Original Gaussian PLY')
    p.add_argument('--refined', required=True, help='Stage-1 corrected Gaussian PLY')
    p.add_argument('--out', required=True, help='Output directory')
    p.add_argument('--sample-points', type=int, default=100000, help='Deterministic sample per Gaussian model')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--voxel-ratios', type=float, nargs='+', default=[0.005, 0.01, 0.02], help='Voxel side / original bounding box diagonal')
    p.add_argument('--opacity-cutoff', type=float, default=0.05, help='Exclude near-transparent Gaussians from geometric coverage')
    p.add_argument('--gt', default=None, help='Optional ground-truth PLY (points or mesh)')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--gt-transform', default=None, help='JSON 4x4 matrix mapping GT coordinates to G0 coordinates')
    g.add_argument('--gt-already-aligned', action='store_true', help='Explicitly affirm GT and G0 are already registered in coordinates and scale')
    p.add_argument('--gt-samples', type=int, default=100000)
    p.add_argument('--threshold-ratios', type=float, nargs='+', default=[0.005, 0.01, 0.02], help='F-score thresholds as fractions of original bbox diagonal')
    return p.parse_args()


def read_ply(path):
    from plyfile import PlyData
    vertex = PlyData.read(str(path))['vertex'].data
    names = set(vertex.dtype.names)
    if not {'x', 'y', 'z'} <= names:
        raise ValueError(f'{path}: missing x/y/z in vertex PLY')
    pts = np.column_stack([vertex[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    if not np.all(np.isfinite(pts)):
        raise ValueError(f'{path}: NaN/Inf coordinates')
    opacity = None
    if 'opacity' in names:
        z = np.asarray(vertex['opacity'], dtype=np.float64)
        opacity = 1.0 / (1.0 + np.exp(-np.clip(z, -80, 80)))  # 3DGS logit
    scales = None
    keys = [f'scale_{i}' for i in range(3)]
    if set(keys) <= names:
        log_scale = np.column_stack([vertex[k] for k in keys]).astype(np.float64)
        scales = np.exp(np.clip(log_scale, -80, 80))  # 3DGS log-scales
    return pts, opacity, scales


def sample_indices(n, count, rng):
    return np.sort(rng.choice(n, min(n, count), replace=False))


def summary(v):
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return None
    return {'mean': float(np.mean(v)), 'median': float(np.median(v)),
            'p90': float(np.percentile(v, 90)), 'p95': float(np.percentile(v, 95)),
            'p99': float(np.percentile(v, 99)), 'max': float(np.max(v))}


def nearest(a, b):
    dist, _ = cKDTree(b).query(a, k=1, workers=-1)
    return np.asarray(dist)


def coverage(a, b, thresholds):
    # Exact distance to nearest sampled Gaussian *center*, not to Gaussian surface.
    da = nearest(a, b)
    db = nearest(b, a)
    return {
        'a_to_b_center_distance': summary(da),
        'b_to_a_center_distance': summary(db),
        'chamfer_l1_sampled_centers': float(np.mean(da) + np.mean(db)),
        'chamfer_l2_sampled_centers': float(np.mean(da**2) + np.mean(db**2)),
        'thresholds': {str(float(t)): {
            'a_covered_by_b': float(np.mean(da <= t)),
            'b_covered_by_a': float(np.mean(db <= t)),
            'fscore': float(2 * np.mean(da <= t) * np.mean(db <= t) / max(1e-12, np.mean(da <= t) + np.mean(db <= t)))
        } for t in thresholds}
    }


def voxel_set(points, origin, side):
    cells = np.floor((points - origin) / side).astype(np.int64)
    return set(map(tuple, np.unique(cells, axis=0)))


def knn_spacing(points):
    if len(points) < 2:
        return None
    d, _ = cKDTree(points).query(points, k=2, workers=-1)
    return summary(d[:, 1])


def load_gt(path, n, rng):
    from plyfile import PlyData
    ply = PlyData.read(path)
    vertices = ply['vertex'].data
    xyz = np.column_stack([vertices[k] for k in ('x','y','z')]).astype(np.float64)
    if 'face' in ply:
        # Surface area sampling, avoids using non-uniformly distributed mesh vertices as GT.
        import trimesh
        mesh = trimesh.load(path, process=False, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError('GT contains face data, but no usable triangular mesh')
        # trimesh sampling uses NumPy global RNG; preserve determinism.
        state = np.random.get_state()
        try:
            np.random.seed(int(rng.integers(0, 2**31-1)))
            xyz = trimesh.sample.sample_surface(mesh, n)[0]
        finally:
            np.random.set_state(state)
    elif len(xyz) > n:
        xyz = xyz[sample_indices(len(xyz), n, rng)]
    if not np.isfinite(xyz).all():
        raise ValueError('Invalid GT points')
    return xyz


def load_transform(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(obj, dict):
        if 'gt_to_g0' not in obj:
            raise ValueError('Transform JSON object must contain gt_to_g0 (4x4)')
        obj = obj['gt_to_g0']
    matrix = np.asarray(obj, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Transform must be a finite 4x4 matrix')
    return matrix


def main():
    args = get_args()
    if args.sample_points < 2 or args.gt_samples < 2:
        raise ValueError('Sampling sizes must be >=2')
    if (args.gt_transform or args.gt_already_aligned) and not args.gt:
        raise ValueError('--gt-transform/--gt-already-aligned requires --gt')
    if args.gt and not (args.gt_transform or args.gt_already_aligned):
        raise ValueError('GT frame cannot be assumed: pass --gt-transform or --gt-already-aligned')
    if any(t <= 0 for t in args.voxel_ratios + args.threshold_ratios):
        raise ValueError('All voxel/threshold ratios must be positive')
    rng = np.random.default_rng(args.seed)
    original_xyz, original_op, original_scale = read_ply(args.g0)
    refined_xyz, refined_op, refined_scale = read_ply(args.refined)
    if len(original_xyz) == 0 or len(refined_xyz) == 0:
        raise ValueError('Input Gaussian PLY is empty')
    # Use original *full* model bounding box as common scale reference.
    origin = original_xyz.min(axis=0)
    bbox_diag = float(np.linalg.norm(original_xyz.max(axis=0) - origin))
    if bbox_diag <= 0:
        raise ValueError('Original G0 bounding box diagonal is zero')
    # Excluding tiny opacity splats gives a more meaningful geometry-coverage diagnostic.
    def active(xyz, op, scales):
        if op is None:
            return xyz, op, scales
        mask = op >= args.opacity_cutoff
        if not mask.any():
            raise ValueError('Opacity threshold excluded all Gaussians')
        return xyz[mask], op[mask], (scales[mask] if scales is not None else None)
    a, ao, asc = active(original_xyz, original_op, original_scale)
    b, bo, bsc = active(refined_xyz, refined_op, refined_scale)
    ai = sample_indices(len(a), args.sample_points, rng)
    bi = sample_indices(len(b), args.sample_points, rng)
    a_s, b_s = a[ai], b[bi]
    thresholds = [float(q * bbox_diag) for q in args.threshold_ratios]
    dist = coverage(a_s, b_s, thresholds)
    voxels = {}
    for ratio in args.voxel_ratios:
        side = bbox_diag * ratio
        # Full active centers for voxel occupancy; may be mildly memory intensive.
        va = voxel_set(a, origin, side)
        vb = voxel_set(b, origin, side)
        both = len(va & vb)
        voxels[str(ratio)] = {
            'voxel_size': float(side), 'g0_occupied': len(va), 'g0_star_occupied': len(vb),
            'intersection': both, 'voxel_iou': both/max(1,len(va|vb)),
            'g0_cells_retained': both/max(1,len(va)),
            'new_g0_star_voxel_fraction': (len(vb)-both)/max(1,len(vb))
        }
    report = {
        'type': 'MonoEv_Stage1_geometry_preservation_NOT_absolute_accuracy',
        'inputs': {'g0': str(Path(args.g0).resolve()), 'g0_star': str(Path(args.refined).resolve())},
        'normalization': {'g0_bbox_diagonal': bbox_diag, 'units': 'G0 coordinate units',
                          'sample_points_max_per_model': args.sample_points, 'seed': args.seed,
                          'opacity_cutoff': args.opacity_cutoff},
        'counts': {'g0': len(original_xyz), 'g0_star': len(refined_xyz),
                   'g0_active': len(a), 'g0_star_active': len(b),
                   'count_growth_percent': float(100*(len(refined_xyz)/len(original_xyz)-1))},
        'geometry_agreement': dist,
        'normalized_center_distances': {
            'g0_to_g0_star_mean_over_diag': dist['a_to_b_center_distance']['mean']/bbox_diag,
            'g0_star_to_g0_mean_over_diag': dist['b_to_a_center_distance']['mean']/bbox_diag,
            'symmetric_chamfer_l1_over_diag': dist['chamfer_l1_sampled_centers']/bbox_diag
        },
        'voxel_occupancy': voxels,
        'local_center_spacing_on_equal_size_samples': {'g0': knn_spacing(a_s), 'g0_star': knn_spacing(b_s)},
        'gaussian_parameters': {
            'g0_opacity': summary(original_op) if original_op is not None else None,
            'g0_star_opacity': summary(refined_op) if refined_op is not None else None,
            'g0_max_axis_scale': summary(np.max(original_scale,axis=1)) if original_scale is not None else None,
            'g0_star_max_axis_scale': summary(np.max(refined_scale,axis=1)) if refined_scale is not None else None,
            'g0_axis_anisotropy': summary(np.max(original_scale,axis=1)/np.maximum(1e-12,np.min(original_scale,axis=1))) if original_scale is not None else None,
            'g0_star_axis_anisotropy': summary(np.max(refined_scale,axis=1)/np.maximum(1e-12,np.min(refined_scale,axis=1))) if refined_scale is not None else None,
        },
        'limitations': [
            'Nearest-center distances/voxel IoU measure change between two splat distributions, not closeness to true surfaces.',
            'Original and refined models have different counts; point sampling and density affect distance/knn values.',
            'Gaussian covariance, orientation and alpha compositing are NOT modeled by center-only geometry measures.',
            'A high preservation score does not imply accurate geometry; a low score does not imply degraded geometry.',
            'Input-view RGB fidelity should be reported separately from Stage-1 initial/final metrics.'
        ]
    }
    if args.gt:
        gt = load_gt(args.gt, args.gt_samples, rng)
        if args.gt_transform:
            matrix = load_transform(args.gt_transform)
            gt_h = np.column_stack([gt, np.ones(len(gt))]) @ matrix.T
            if np.any(np.abs(gt_h[:, 3]) < 1e-10):
                raise ValueError('Invalid homogeneous transformed GT points')
            gt = gt_h[:, :3] / gt_h[:, 3:]
        # GT mesh surface samples vs Gaussian centers: proxy, not mesh-to-mesh Chamfer.
        report['gt_geometry_proxy'] = {
            'gt': str(Path(args.gt).resolve()),
            'registration': str(Path(args.gt_transform).resolve()) if args.gt_transform else 'user_asserted_aligned',
            'evaluation': 'GT sampled mesh surface or GT points versus opacity-filtered Gaussian CENTERS',
            'g0': coverage(a_s, gt, thresholds),
            'g0_star': coverage(b_s, gt, thresholds),
            'warning': 'Center-to-mesh-point distances do not equal true Gaussian surface reconstruction accuracy.'
        }
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'geometry_metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = [
        'MonoEv Stage-1 geometry evaluation (NO multiview render)',
        'G0 vs G0*: preservation / geometry change, NOT GT quality',
        f"Gaussians: {len(original_xyz):,} -> {len(refined_xyz):,} ({report['counts']['count_growth_percent']:+.2f}%)",
        f'G0 bbox diagonal: {bbox_diag:.8g}',
        f"Symmetric center Chamfer L1 / diagonal: {report['normalized_center_distances']['symmetric_chamfer_l1_over_diag']:.6f}",
        f"G0->G0* mean NN distance / diagonal: {report['normalized_center_distances']['g0_to_g0_star_mean_over_diag']:.6f}",
        f"G0*->G0 mean NN distance / diagonal: {report['normalized_center_distances']['g0_star_to_g0_mean_over_diag']:.6f}",
    ]
    for q, m in voxels.items():
        lines.append(f"Voxel ratio {q}: IoU={m['voxel_iou']:.5f}, retained={m['g0_cells_retained']:.5f}, new_fraction={m['new_g0_star_voxel_fraction']:.5f}")
    if args.gt:
        lines.append('GT proxy computed (only valid if GT coordinates and scale are registered correctly).')
    lines += ['', 'For true quality use input-view metrics and verified-alignment GT evaluation.']
    (out / 'geometry_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    print('Saved:', out/'geometry_metrics.json')


if __name__ == '__main__':
    main()
