# MonoEv Stage1 Visualization UI — Three Pages

This is an independent, read-only post-training UI. It does not mutate Stage1, any PLY, or the fixed camera JSON.

## Pages

1. **Auto Detail × 3:** fixed-front original G0 / corrected G0 / source RGB; automatically identifies three or four non-overlapping salient regions A–D. Shows labeled full images and enlarged crops, exports one annotated PNG.
2. **Four-view Scientific Figure:** Front, Left, Right, Back as **rows**; Original RGB, Original G0, Corrected G0 as **columns**. All Gaussian renders use the **same virtual camera poses and shared pivot**, chosen from corrected model coordinates. Only Front has ground truth; other RGB panels are deliberately blank (black or white, as selected). Generates a single 4×3 scientific figure PNG.
3. **KNN & Ellipsoid Diagnostics:** 3D centers with SH DC color, 1-NN spacing, KNN median, scale anisotropy and opacity. Metrics are geometry diagnostics, not a proof of topology or hole-free reconstruction.

## Run

```bash
conda activate ed3dgs
cd /E-D3DGS
pip install -r Stage1_Visualization_UI/requirements.txt
streamlit run Stage1_Visualization_UI/app.py --server.address 0.0.0.0 --server.port 8501
```

Use the sidebar to select Fan, Lamp or custom input paths. For Fan the round directory default is `/root/autodl-tmp/MonoEv/Stage1/Fan/round1`. Change it if the output is elsewhere. The input models are never overwritten.

## Accuracy notes

- Actual views use the project's `gaussian_renderer.render` rather than a scatter plot.
- The front camera is the supplied fixed camera; alternate views are virtual orbit views, with no camera-observed RGB ground truth.
- The two models must share the same world-space coordinate frame to make strict per-view comparison meaningful. If the original G0 has not yet been aligned to that frame, comparing them under the same fixed camera may reveal misalignment.
- Dynamic time series are intentionally not included.
- This code has passed syntax checks, but full CUDA execution must be validated in the AutoDL environment.
