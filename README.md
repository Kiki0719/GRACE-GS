# GRACE-GS

Gaussian Reconstruction, Alignment, and Continuity Enhancement.

Source export for the MonoEv / MERCY static initialization
and post-training visualization modules.

## Main modules

- `Stage1/` — Fixed-camera alignment and Gaussian refinement
- `Stage1_Visualization_UI/` — Streamlit visual inspection
- `scene/`, `gaussian_renderer/`, `utils/` — 3DGS runtime
- `submodules/` — CUDA extension source (if present)

## Important

This is a source snapshot, not yet a validated standalone build.

Training datasets, Gaussian PLY files, experiment outputs and
model weights are not included.

Check `EXPORT_REPORT.json` and `EXTERNAL_IMPORTS.txt` for
dependency review.

Sanitize experiment-specific absolute paths in INI files
before public release. Preserve applicable third-party licenses.
