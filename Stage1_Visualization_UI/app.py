"""MonoEv Stage1 read-only post-training inspection dashboard."""
from pathlib import Path
import io
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from scipy.spatial import cKDTree

from render_engine import load_model, splat_image
from analysis_utils import (dc_rgb, rgba_alpha, mask_metrics, foreground_from_rgb,
                            resize_mask, symmetric_surface_dist, local_geometry, scale_anisotropy)

st.set_page_config(page_title='MonoEv | Stage1 Inspector', page_icon='◉', layout='wide')
st.markdown('''<style>.block-container{padding-top:1rem;max-width:1660px}
[data-testid="stMetricValue"]{font-size:1.35rem}</style>''', unsafe_allow_html=True)
st.title('MonoEv · Stage1 Gaussian Repair Inspector')
st.caption('Post-training only · real Gaussian Splatting · original model files are read-only')

with st.sidebar:
    st.header('Data')
    # Automatically discover datasets from MonoEv/Stage1.
    preset_root = Path("/root/autodl-tmp/MonoEv/Stage1")
    data_root = Path("/root/autodl-tmp/Mono_Data")
    config_root = Path("/E-D3DGS/Stage1")

    preset_names = sorted(
        [d.name for d in preset_root.iterdir() if d.is_dir()],
        key=str.casefold
    ) if preset_root.is_dir() else []

    obj = st.selectbox("Preset", preset_names + ["Custom"])

    def find_existing(paths):
        for path in paths:
            if path.is_file():
                return str(path)
        return str(paths[0]) if paths else ""

    if obj == "Custom":
        pathbase = ""
        default_g0 = ""
        default_cfg = ""
        default_rgb = ""
    else:
        model_root = preset_root / obj
        data_dir = data_root / obj

        # Prefer round1, but allow an object with another round layout.
        rounds = sorted(
            [d for d in model_root.glob("round*") if d.is_dir()],
            key=lambda d: d.name
        )
        preferred_round = model_root / "round1"
        round_dir = (
            preferred_round if preferred_round.is_dir()
            else rounds[-1] if rounds
            else preferred_round
        )
        pathbase = str(round_dir)

        # Case-insensitive matching for the dataset directory.
        if not data_dir.is_dir() and data_root.is_dir():
            for candidate in data_root.iterdir():
                if candidate.is_dir() and candidate.name.casefold() == obj.casefold():
                    data_dir = candidate
                    break

        default_g0 = find_existing(
            [data_dir / f"{obj}.ply", data_dir / f"{obj.lower()}.ply"]
            + sorted(data_dir.glob("*.ply")) if data_dir.is_dir()
            else [data_dir / f"{obj}.ply"]
        )

        default_rgb = find_existing(
            [data_dir / f"{obj}.png", data_dir / f"{obj.lower()}.png"]
            + sorted(data_dir.glob("*.png"))
            + sorted(data_dir.glob("*.jpg"))
            if data_dir.is_dir()
            else [data_dir / f"{obj}.png"]
        )

        default_cfg = find_existing([
            config_root / f"{obj.lower()}_stage1.ini",
            config_root / f"{obj}_stage1.ini",
        ])

    directory = st.text_input(
        "Stage1 round directory", pathbase, key=f"dir_{obj}"
    )
    ply = st.text_input(
        "Corrected PLY",
        f"{directory}/refinement/corrected_g0.ply" if directory else "",
        key=f"ply_{obj}"
    )
    g0 = st.text_input(
        "Original G0 (optional)", default_g0, key=f"g0_{obj}"
    )
    camera = st.text_input(
        "Fixed camera JSON",
        f"{directory}/alignment/fixed_camera.json" if directory else "",
        key=f"camera_{obj}"
    )
    cfg = st.text_input(
        "Stage1 INI", default_cfg, key=f"cfg_{obj}"
    )
    rgb_path = st.text_input(
        "Original RGB", default_rgb, key=f"rgb_{obj}"
    )
    mask_path = st.text_input(
        "Saved target mask (optional)",
        f"{directory}/refinement/target_mask_initialization/target_mask.png"
        if directory else "",
        key=f"mask_{obj}"
    )
    st.divider()
    bg=st.selectbox('Render background',['Black','White'])
    distance=st.slider('Camera distance multiplier',.6,2.0,1.,.05)
    st.caption('Color mask is only used for diagnostics; nothing writes to the input models.')

@st.cache_resource(show_spinner='Loading model on CUDA GPU...',max_entries=3)
def get_model(p,c,i,version):
    workspace=Path('/tmp/monoev_stage1_inspector')
    workspace.mkdir(parents=True,exist_ok=True)
    return load_model(p,c,i,workspace)

@st.cache_data(show_spinner=False,max_entries=20)
def cached_render(p,c,i,version,yaw,pitch,zoom,background,pivot=None):
    gs,cam,xyz=get_model(p,c,i,version)
    return splat_image(gs,cam,xyz,yaw,pitch,zoom,background,orbit_center=pivot)

@st.cache_data(show_spinner=False,max_entries=4)
def diag_sample(xyz_key,sample_size,seed=42):
    # xyz_key is a bound numpy array; cached with its contents.
    rng=np.random.default_rng(seed)
    return rng.choice(len(xyz_key),min(len(xyz_key),sample_size),replace=False)

def png_bytes(image):
    buf=io.BytesIO();Image.fromarray(np.asarray(image,dtype=np.uint8)).save(buf,format='PNG');return buf.getvalue()

def download_image(image, filename, key):
    st.download_button('Download PNG',png_bytes(image),file_name=filename,mime='image/png',key=key)

def display_cloud(pts,values, title,colorscale='Turbo', size=2, categoric=False):
    if categoric:
        marker=dict(size=size,color=values,opacity=.9)
    else:
        marker=dict(size=size,color=values,colorscale=colorscale,colorbar=dict(title=title),opacity=.85)
    fig=go.Figure(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',marker=marker,hoverinfo='skip'))
    fig.update_layout(template='plotly_dark',height=590,margin=dict(l=0,r=0,b=0,t=0),scene=dict(aspectmode='data'))
    st.plotly_chart(fig,use_container_width=True)

missing=[p for p in (ply,camera,cfg) if not Path(p).is_file()]
if missing:
    st.warning('Missing required file(s): '+', '.join(missing));st.stop()
try:
    version=tuple(Path(p).stat().st_mtime_ns for p in (ply,camera,cfg))
    gs,cam,xyz=get_model(ply,camera,cfg,version)
except Exception as e:
    st.exception(e);st.stop()

cols=st.columns(4)
cols[0].metric('Final Gaussians',f'{len(xyz):,}')
for j,axis in enumerate('XYZ',start=1):cols[j].metric(f'{axis} extent',f'{np.ptp(xyz[:,j-1]):.3f}')

tabs = st.tabs(['Auto Detail 5 × 3', 'Four-view Scientific Figure'])

with tabs[0]:
    st.subheader('Automatic Front Detail Comparison · G0 / Corrected / RGB')
    st.caption('Same fixed front camera and image-space regions. Four diverse, high-detail areas are selected automatically; no manual ROI clicking.')
    if not Path(g0).is_file() or not Path(rgb_path).is_file():
        st.error('Provide Original G0 and Original RGB in the sidebar.')
    else:
        try:
            from PIL import ImageDraw
            from scipy.ndimage import gaussian_filter, uniform_filter, sobel
            roi_count = 4  # exactly four automatic regions; no 3x3 selector
            # The reference RGB is only valid for the calibrated front camera.
            st.info('RGB ground truth exists only for the calibrated FRONT view. The three-way comparison therefore uses Front (yaw=0, pitch=0).')
            v0 = (Path(g0).stat().st_mtime_ns, version[1], version[2])
            stamp = Path(rgb_path).stat().st_mtime_ns
            scene_key = (g0, ply, camera, cfg, rgb_path, v0, version, stamp, bg)
            if st.session_state.get('auto_detail_scene_key') != scene_key:
                with st.spinner('Automatically rendering front view and selecting details...'):
                    old_arr = cached_render(g0, camera, cfg, v0, 0, 0, 1.0, bg)
                    new_arr = cached_render(ply, camera, cfg, version, 0, 0, 1.0, bg)
                    h, w = new_arr.shape[:2]
                    if old_arr.shape[:2] != (h, w):
                        raise ValueError('Original and refined render resolutions differ')
                    ref = Image.open(rgb_path).convert('RGB')
                    if ref.size != (w, h):
                        ref = ref.resize((w, h), Image.Resampling.BICUBIC)
                        st.warning('RGB was resized to match render dimensions.')
                    st.session_state['auto_detail_images'] = (
                        Image.fromarray(np.asarray(old_arr, dtype=np.uint8)).convert('RGB'),
                        Image.fromarray(np.asarray(new_arr, dtype=np.uint8)).convert('RGB'), ref)
                    st.session_state['auto_detail_scene_key'] = scene_key
            if 'auto_detail_images' in st.session_state:
                if st.session_state.get('auto_detail_scene_key') != scene_key:
                    st.warning('Inputs changed; the front renders will refresh automatically.')
                else:
                    ims = st.session_state['auto_detail_images']
                    w, h = ims[0].size
                    roi = max(24, int(min(w, h) * 0.20))
                    roi = min(roi, w, h)
                    # Score combines textured RGB content with reconstruction difference.
                    original = np.asarray(ims[0], dtype=np.float32) / 255.0
                    refined = np.asarray(ims[1], dtype=np.float32) / 255.0
                    reference = np.asarray(ims[2], dtype=np.float32) / 255.0
                    gray = reference.mean(axis=2)
                    detail = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
                    detail /= max(float(np.quantile(detail, .99)), 1e-6)
                    disagreement = np.abs(original - refined).mean(axis=2)
                    improvement = np.abs(original-reference).mean(axis=2) - np.abs(refined-reference).mean(axis=2)
                    importance = (0.45*np.clip(detail,0,1) +
                                  0.35*np.clip(disagreement*4,0,1) +
                                  0.20*np.clip(np.abs(improvement)*4,0,1))
                    # Suppress blank background. Original RGB may be black on black.
                    bg_ref = np.median(np.concatenate((reference[:8].reshape(-1,3),
                                     reference[-8:].reshape(-1,3),
                                     reference[:,:8].reshape(-1,3),
                                     reference[:,-8:].reshape(-1,3))), axis=0)
                    foreground = (np.linalg.norm(reference-bg_ref,axis=2) > 0.065)
                    if foreground.sum() < 0.01*w*h:
                        foreground = (detail > .035)
                    foreground = gaussian_filter(foreground.astype(np.float32), max(1,roi/12))
                    score = uniform_filter(importance * foreground, size=roi, mode='nearest')
                    fg_coverage = uniform_filter((foreground > .3).astype(np.float32), size=roi, mode='nearest')
                    score[fg_coverage < .18] = -1
                    pad = roi//2
                    score[:pad,:] = -1; score[h-(roi-pad):,:] = -1
                    score[:,:pad] = -1; score[:,w-(roi-pad):] = -1
                    boxes=[]
                    used = score.copy()
                    for k in range(roi_count):
                        cy,cx = np.unravel_index(np.argmax(used),used.shape)
                        if used[cy,cx] <= 0:
                            break
                        x = int(np.clip(cx-pad,0,w-roi)); y=int(np.clip(cy-pad,0,h-roi))
                        boxes.append((x,y,roi,roi))
                        # Enforce spatial diversity and avoid adjacent duplicate crops.
                        sep = int(roi*1.15)
                        used[max(0,cy-sep):min(h,cy+sep+1),max(0,cx-sep):min(w,cx+sep+1)] = -1
                    if len(boxes) < roi_count:
                        st.warning(f'Only {len(boxes)} well-separated foreground detail areas were detected.')
                    colors = ['#0096A7','#E35245','#7951C4','#D99C2C']
                    labels = 'ABCD'
                    full = []
                    for image in ims:
                        marked=image.copy(); draw=ImageDraw.Draw(marked)
                        for k,(x,y,bw,bh) in enumerate(boxes):
                            color=colors[k]
                            for d in range(3):
                                draw.rectangle((x-d,y-d,x+bw+d,y+bh+d),outline=color)
                            draw.rectangle((x,y-25,x+27,y),fill=color)
                            draw.text((x+8,y-22),labels[k],fill='white')
                        full.append(marked)
                    names=['Original G0','Corrected G0','Original RGB']
                    st.markdown('#### Full front-view reference (A–D marked)')
                    for col,name,img in zip(st.columns(3), names, full):
                        col.image(img,caption=name,use_container_width=True)


                    # Five columns:
                    # Full Image | Detail A | Detail B | Detail C | Detail D
                    # Three rows:
                    # Original G0 | Corrected G0 | Original RGB

                    st.markdown(
                        "### 5 Columns × 3 Rows · Scientific Detail Comparison"
                    )

                    from PIL import ImageFont

                    tile = 360
                    gap = 14
                    header = 44
                    margin = 20
                    label_width = 165

                    ncols = 5
                    nrows = 3

                    sheet_w = (
                        margin * 2
                        + label_width
                        + ncols * tile
                        + (ncols - 1) * gap
                    )

                    sheet_h = (
                        margin * 2
                        + header
                        + nrows * (tile + header)
                        + (nrows - 1) * gap
                    )

                    sheet = Image.new(
                        "RGB",
                        (sheet_w, sheet_h),
                        "white"
                    )

                    painter = ImageDraw.Draw(sheet)

                    col_names = [
                        "Full Image",
                        "Detail A",
                        "Detail B",
                        "Detail C",
                        "Detail D"
                    ]

                    for j, title in enumerate(col_names):
                        xx = margin + label_width + j * (tile + gap)
                        color = "#202932" if j == 0 else colors[j - 1]
                        painter.text(
                            (xx + 12, margin + 12),
                            title,
                            fill=color
                        )

                    for i, name in enumerate(names):

                        top = (
                            margin + header
                            + i * (tile + header + gap)
                        )

                        painter.text(
                            (margin + 5, top + header + tile // 2),
                            name,
                            fill="#202932"
                        )

                        for j in range(ncols):

                            xx = (
                                margin + label_width
                                + j * (tile + gap)
                            )

                            painter.rectangle(
                                (xx, top, xx + tile, top + header),
                                fill="#f2f5f8"
                            )

                            painter.text(
                                (xx + 12, top + 12),
                                f"{name} | {col_names[j]}",
                                fill="#202932"
                            )

                            if j == 0:
                                # First column: complete image with ROI boxes.
                                source = full[i]

                                fitted = source.copy()
                                fitted.thumbnail(
                                    (tile - 12, tile - 12),
                                    Image.Resampling.LANCZOS
                                )

                                panel = Image.new(
                                    "RGB",
                                    (tile, tile),
                                    "white"
                                )

                                px = (tile - fitted.width) // 2
                                py = (tile - fitted.height) // 2

                                panel.paste(fitted, (px, py))

                            else:
                                # Remaining four columns: synchronized crops.
                                k = j - 1

                                if k < len(boxes):
                                    x, y, bw, bh = boxes[k]

                                    crop = ims[i].crop(
                                        (x, y, x + bw, y + bh)
                                    )

                                    panel = crop.resize(
                                        (tile, tile),
                                        Image.Resampling.LANCZOS
                                    )
                                else:
                                    panel = Image.new(
                                        "RGB",
                                        (tile, tile),
                                        "white"
                                    )

                            sheet.paste(
                                panel,
                                (xx, top + header)
                            )

                            # A/B/C/D detail panels use their ROI colors.
                            # The Full Image column remains unchanged.
                            if j > 0:
                                border_color = colors[j - 1]
                                border_width = 6

                                for d in range(border_width):
                                    painter.rectangle(
                                        (
                                            xx + d,
                                            top + header + d,
                                            xx + tile - 1 - d,
                                            top + header + tile - 1 - d
                                        ),
                                        outline=border_color
                                    )

                    st.image(
                        sheet,
                        caption=(
                            "Full Image + A/B/C/D | "
                            "Original G0 / Corrected G0 / Original RGB"
                        ),
                        use_container_width=True
                    )

                    st.download_button(
                        "Download 5×3 Scientific Figure",
                        data=png_bytes(np.asarray(sheet)),
                        file_name="monoev_detail_comparison_5x3.png",
                        mime="image/png",
                        key="auto_detail_download"
                    )
        except Exception as exc:
            st.exception(exc)




    # MONOEV_STAGE1_BEFORE_AFTER_METRICS_V1
    st.divider()
    st.subheader("Surface Reconstruction Metrics | Before vs After")
    st.caption(
        "Fixed front camera. RGB metrics against the original image; "
        "silhouette metrics use the estimated target mask and rendered alpha."
    )

    try:
        import pandas as pd
        import cv2
        from PIL import Image
        from skimage.metrics import structural_similarity

        required = [g0, ply, camera, cfg, rgb_path]
        if not all(Path(str(v)).is_file() for v in required):
            st.warning("Please check G0, corrected G0, camera, config and RGB paths.")
        else:
            original_version = (
                Path(g0).stat().st_mtime_ns,
                version[1],
                version[2],
            )

            def _render_metrics_model(path, ver):
                black = cached_render(
                    path, camera, cfg, ver,
                    0, 0, 1.0, "Black"
                )
                white = cached_render(
                    path, camera, cfg, ver,
                    0, 0, 1.0, "White"
                )

                black = np.asarray(black, dtype=np.float32) / 255.0
                white = np.asarray(white, dtype=np.float32) / 255.0

                # Alpha compositing:
                # white_render - black_render = 1 - alpha
                alpha = np.clip(
                    1.0 - np.mean(white - black, axis=2),
                    0.0, 1.0
                )
                return black, alpha

            with st.spinner("Calculating reconstruction metrics..."):
                before_rgb, before_alpha = _render_metrics_model(
                    g0, original_version
                )
                after_rgb, after_alpha = _render_metrics_model(
                    ply, version
                )

                h, w = after_rgb.shape[:2]

                target_pil = Image.open(rgb_path)
                original_size = target_pil.size
                rgba = np.asarray(
                    target_pil.convert("RGBA").resize(
                        (w, h), Image.Resampling.BICUBIC
                    ),
                    dtype=np.float32
                ) / 255.0

                target_rgb = rgba[:, :, :3]

                # Prefer meaningful PNG alpha.
                if np.min(rgba[:, :, 3]) < 0.999:
                    target_mask = rgba[:, :, 3] > 0.5
                    mask_source = "PNG alpha"
                else:
                    # Fallback: approximate foreground from border color.
                    border = np.concatenate([
                        target_rgb[0, :, :],
                        target_rgb[-1, :, :],
                        target_rgb[:, 0, :],
                        target_rgb[:, -1, :]
                    ], axis=0)

                    bg_rgb = np.median(border, axis=0)
                    target_mask = (
                        np.max(
                            np.abs(target_rgb - bg_rgb),
                            axis=2
                        ) > (12.0 / 255.0)
                    )
                    mask_source = "Estimated RGB background"

                if not target_mask.any():
                    raise ValueError("Target foreground mask is empty")

                # Common neutral black background for fair comparison.
                target_eval = target_rgb.copy()
                target_eval[~target_mask] = 0

                def evaluate(rendered, alpha):
                    rendered = rendered.copy()
                    rendered[~target_mask] = 0

                    diff = rendered - target_eval
                    foreground_diff = diff[target_mask]

                    mse = float(np.mean(foreground_diff ** 2))
                    psnr = -10.0 * np.log10(max(mse, 1e-12))
                    mae = float(np.mean(np.abs(foreground_diff)))

                    ssim = structural_similarity(
                        target_eval,
                        rendered,
                        channel_axis=2,
                        data_range=1.0,
                        win_size=7
                    )

                    prediction = alpha > 0.5
                    tp = np.count_nonzero(prediction & target_mask)
                    fp = np.count_nonzero(prediction & ~target_mask)
                    fn = np.count_nonzero(~prediction & target_mask)

                    iou = tp / max(tp + fp + fn, 1)
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)

                    return {
                        "PSNR (dB) ↑": psnr,
                        "SSIM ↑": float(ssim),
                        "MAE ↓": mae,
                        "Mask IoU ↑": iou,
                        "Mask Precision ↑": precision,
                        "Mask Recall ↑": recall,
                    }

                before = evaluate(before_rgb, before_alpha)
                after = evaluate(after_rgb, after_alpha)

                rows = []
                for metric in before:
                    a = before[metric]
                    b = after[metric]
                    delta = b - a
                    pct_change = (delta / abs(a) * 100.0) if abs(a) > 1e-12 else float("nan")
                    improved = delta < 0 if "MAE" in metric else delta > 0

                    rows.append({
                        "Metric": metric,
                        "Original G0": round(a, 5),
                        "Corrected G0": round(b, 5),
                        "Change (%)": f"{pct_change:+.2f}%" if np.isfinite(pct_change) else "N/A",
                        "Result": (
                            "Improved" if improved
                            else "Unchanged" if abs(delta) < 1e-8
                            else "Decreased"
                        )
                    })

                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )


                # Browser-native table capture (preserves Streamlit styling).
                import streamlit.components.v1 as components

                components.html("""
                <div style="margin:8px 0">
                  <button id="capture-metrics"
                    style="padding:9px 18px;border-radius:8px;
                    border:1px solid #bbb;background:#ffffff;
                    color:#222;font-weight:600;cursor:pointer;">
                    Capture Table PNG
                  </button>
                </div>

                <script>
                const btn = document.getElementById("capture-metrics");

                btn.addEventListener("click", async () => {
                    const parentDoc = window.parent.document;
                    const tables = parentDoc.querySelectorAll(
                        '[data-testid="stDataFrame"]'
                    );

                    if (!tables.length) {
                        alert("Metrics table not found.");
                        return;
                    }

                    const target = tables[tables.length - 1];

                    if (!window.html2canvas) {
                        const script = document.createElement("script");
                        script.src =
                          "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js";
                        document.head.appendChild(script);

                        await new Promise((resolve, reject) => {
                            script.onload = resolve;
                            script.onerror = reject;
                        });
                    }

                    const canvas = await html2canvas(target, {
                        backgroundColor: "#ffffff",
                        scale: 3,
                        useCORS: true
                    });

                    const a = document.createElement("a");
                    a.download = "stage1_reconstruction_metrics.png";
                    a.href = canvas.toDataURL("image/png");
                    a.click();
                });
                </script>
                """, height=58)

                st.caption(
                    f"Target mask: {mask_source} | "
                    f"Alpha threshold: 0.5 | "
                    f"Image resolution: {w} x {h} | "
                    "Positive change is better except MAE."
                )

                if original_size != (w, h):
                    st.warning(
                        "Original RGB was resized to match render resolution."
                    )

                if mask_source != "PNG alpha":
                    st.warning(
                        "The target mask is estimated from background color. "
                        "Mask IoU/Precision/Recall may differ from Stage1 "
                        "if its foreground extraction differs."
                    )

                st.download_button(
                    "Download Reconstruction Metrics CSV",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="stage1_reconstruction_metrics.csv",
                    mime="text/csv",
                    key="stage1_metrics_csv"
                )

    except Exception as exc:
        st.error("Metric calculation failed")
        st.exception(exc)


with tabs[1]:
    st.subheader("Diagnostic Four-view Scientific Comparison")
    st.caption(
        "Exact canonical camera poses from Stage1 alignment diagnostic_views. "
        "Both Gaussian models use identical poses."
    )

    import json
    from render_engine import splat_diagnostic_view
    from PIL import ImageDraw

    alignment_dir = Path(camera).parent
    diagnostic_json = (
        alignment_dir / "diagnostic_views" / "diagnostics.json"
    )

    if not diagnostic_json.is_file():
        st.error(f"Missing diagnostic camera data: {diagnostic_json}")
    elif not Path(g0).is_file():
        st.error(f"Original G0 not found: {g0}")
    elif not Path(rgb_path).is_file():
        st.error(f"Original RGB not found: {rgb_path}")
    else:
        try:
            diagnostic_data = json.loads(
                diagnostic_json.read_text(encoding="utf-8")
            )

            fixed_data = json.loads(
                Path(camera).read_text(encoding="utf-8")
            )

            if "target_center" not in fixed_data:
                raise RuntimeError(
                    "fixed_camera.json has no target_center"
                )

            center = fixed_data["target_center"]

            records = {
                v["name"].lower(): v
                for v in diagnostic_data["views"]
            }

            view_names = ["front", "left", "right", "back"]

            if not all(v in records for v in view_names):
                raise RuntimeError(
                    "Diagnostic JSON lacks one of front/left/right/back"
                )

            v0 = (
                Path(g0).stat().st_mtime_ns,
                version[1],
                version[2],
            )

            diagnostic_key = (
                g0, ply, camera, cfg, rgb_path, bg,
                v0, version,
                diagnostic_json.stat().st_mtime_ns,
            )

            if (
                st.button(
                    "Generate Diagnostic Four-view Figure",
                    type="primary",
                    key="diag_four_run",
                )
                or st.session_state.get("diag_four_key")
                    != diagnostic_key
            ):

                original_gs, original_cam, _ = get_model(
                    g0, camera, cfg, v0
                )

                corrected_gs, corrected_cam, _ = get_model(
                    ply, camera, cfg, version
                )

                comparison = []

                with st.spinner("Rendering canonical diagnostic views..."):
                    for name in view_names:
                        record = records[name]

                        before = splat_diagnostic_view(
                            original_gs,
                            original_cam,
                            record,
                            center,
                            bg,
                        )

                        after = splat_diagnostic_view(
                            corrected_gs,
                            corrected_cam,
                            record,
                            center,
                            bg,
                        )

                        comparison.append((name, before, after))

                H, W = comparison[0][1].shape[:2]

                ref = Image.open(rgb_path).convert("RGB")
                if ref.size != (W, H):
                    ref = ref.resize(
                        (W, H),
                        Image.Resampling.BICUBIC,
                    )

                # 4 columns: Left / Right / Back
                # 3 rows: Original RGB / Original G0 / Corrected G0
                panel = 384
                gap = 12
                top = 90
                header = 38
                outer = 22

                sheet_w = 4 * panel + 3 * gap + 2 * outer
                sheet_h = (
                    top + 2 * (panel + header)
                    + 2 * gap + outer
                )

                paper = Image.new(
                    "RGB", (sheet_w, sheet_h), "white"
                )
                draw = ImageDraw.Draw(paper)

                draw.text(
                    (outer, 15),
                    "MonoEv | Diagnostic Canonical Four-view Comparison",
                    fill="#17212d",
                )
                draw.text(
                    (outer, 43),
                    "Front / Left / Right / Back | Original G0 / Corrected G0",
                    fill="#64748b",
                )

                row_names = [
                    "Original G0",
                    "Corrected G0",
                ]

                for r, row_name in enumerate(row_names):
                    for c, (name, before, after) in enumerate(comparison):

                        x = outer + c * (panel + gap)
                        y = top + r * (panel + header + gap)

                        draw.rectangle(
                            (x, y, x + panel - 1, y + header - 1),
                            fill="#f0f3f7",
                        )
                        draw.text(
                            (x + 9, y + 10),
                            f"{row_name} | {name.title()}",
                            fill="#1c2938",
                        )

                        if r == 0:
                            image = Image.fromarray(before)
                        else:
                            image = Image.fromarray(after)

                        image = image.resize(
                            (panel, panel),
                            Image.Resampling.LANCZOS,
                        )
                        paper.paste(image, (x, y + header))

                data = png_bytes(np.asarray(paper))

                st.session_state["diag_four_figure"] = (
                    data, np.asarray(paper)
                )
                st.session_state["diag_four_key"] = diagnostic_key

            if (
                st.session_state.get("diag_four_key")
                == diagnostic_key
                and "diag_four_figure" in st.session_state
            ):
                data, image = st.session_state["diag_four_figure"]

                st.image(
                    image,
                    caption="Alignment diagnostic camera views | 4 columns x 2 rows",
                    use_container_width=True,
                )

                st.download_button(
                    "Download Diagnostic Four-view Scientific PNG",
                    data,
                    file_name="monoev_diagnostic_four_view.png",
                    mime="image/png",
                    key="diag_four_download",
                )

        except Exception as exc:
            st.exception(exc)





# MONOEV_RGB_CONTOUR_PAGE_V1


