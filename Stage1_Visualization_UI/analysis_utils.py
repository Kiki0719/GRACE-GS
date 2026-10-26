"""Post-training diagnostics. PLY and camera objects are never mutated."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

C0 = 0.28209479177387814

def dc_rgb(gs, indices):
    feat = gs.get_features.detach().cpu().numpy()
    if feat.ndim != 3:
        raise ValueError(f'Unexpected SH feature shape {feat.shape}')
    # Native E-D3DGS: N x SH x RGB; some forks use N x RGB x SH.
    if feat.shape[-1] == 3:
        dc = feat[indices, 0, :]
    elif feat.shape[1] == 3:
        dc = feat[indices, :, 0]
    else:
        raise ValueError(f'Unrecognized SH feature layout {feat.shape}')
    return np.clip(.5 + C0 * dc, 0, 1)

def rgba_alpha(black_rgb, white_rgb):
    """Assuming standard over compositing: white - black = 255*(1-alpha)."""
    b = np.asarray(black_rgb, dtype=np.float32)/255
    w = np.asarray(white_rgb, dtype=np.float32)/255
    alpha = 1 - np.mean(w-b, axis=2)
    return np.clip(alpha, 0, 1)

def mask_metrics(alpha, target, threshold=.5):
    pred = alpha >= threshold
    gt = np.asarray(target) > 0
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    iou = tp/max(1,tp+fp+fn)
    precision = tp/max(1,tp+fp)
    recall = tp/max(1,tp+fn)
    # 0=background, 1=correct, 2=extra, 3=missing
    labels = np.zeros(pred.shape, np.uint8)
    labels[pred & gt] = 1
    labels[pred & ~gt] = 2
    labels[~pred & gt] = 3
    overlay = np.zeros((*pred.shape, 3), np.uint8)
    overlay[labels==1] = [60,175,115]
    overlay[labels==2] = [235,80,70]
    overlay[labels==3] = [55,125,235]
    return {'iou':iou,'precision':precision,'recall':recall,'extra_pixels':fp,'missing_pixels':fn}, overlay

def foreground_from_rgb(path, black_threshold=12):
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert('RGB'))
    return np.max(rgb, axis=2) > black_threshold

def resize_mask(mask, size):
    from PIL import Image
    if mask.shape != (size[1],size[0]):
        # NN for label mask; dimensions are (width,height).
        mask = np.asarray(Image.fromarray((mask.astype(np.uint8)*255)).resize(size, Image.Resampling.NEAREST)) > 0
    return mask

def nn_distances(query_xyz, reference_xyz, k=1):
    tree = cKDTree(reference_xyz)
    dist, idx = tree.query(query_xyz, k=k, workers=-1)
    return dist, idx

def symmetric_surface_dist(a,b,max_points=60000,seed=42):
    """Approximate bidirectional center-to-center distances; NOT tracked displacement."""
    rng = np.random.default_rng(seed)
    aa = a[rng.choice(len(a),min(len(a),max_points),replace=False)]
    bb = b[rng.choice(len(b),min(len(b),max_points),replace=False)]
    d_ab, _ = nn_distances(aa,bb)
    d_ba, _ = nn_distances(bb,aa)
    return aa,bb,d_ab,d_ba

def local_geometry(points, neighbors=12):
    tree=cKDTree(points)
    distance, index=tree.query(points,k=min(neighbors+1,len(points)),workers=-1)
    d=distance[:,1:]
    median=np.median(d,axis=1)
    p95=np.percentile(d,95,axis=1)
    return median,p95

def scale_anisotropy(gs, sample_idx):
    s=gs.get_scaling.detach().cpu().numpy()[sample_idx]
    return s.max(axis=1)/np.maximum(s.min(axis=1),1e-9), s
