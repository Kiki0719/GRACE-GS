# GRACE-GS

**Gaussian Reconstruction, Alignment, and Continuity Enhancement**

GRACE-GS 是 MERCY 项目的单目静态三维高斯初始化模块，主要包括固定内参相机标定、RGB 引导的 Gaussian Refinement，以及重建结果的可视化分析。

## 1. 项目结构

```text
GRACE-GS/
├── Stage1/
│   ├── train_stage1.py             # Stage1 完整训练入口
│   ├── camera_alignment.py         # 虚拟相机标定
│   ├── gaussian_refinement.py      # Gaussian 表面修复
│   ├── gaussian_utils/             # 高斯增密、几何优化及辅助函数
│   └── *_stage1.ini                # 不同对象的训练配置
│
├── Stage1_Visualization_UI/
│   ├── app.py                      # Streamlit 可视化界面
│   └── render_engine.py            # Gaussian 渲染接口
│
├── scene/                          # GaussianModel 与场景加载
├── gaussian_renderer/              # 3DGS 可微分渲染器
├── utils/                          # 通用工具函数
├── arguments/                      # 参数管理
└── submodules/                     # CUDA 扩展源码
```

## 2. 运行环境

开发环境：

- Linux
- NVIDIA RTX 4090
- PyTorch 2.1.2 + CUDA 12.1
- Conda 环境：`ed3dgs`
- 3D Gaussian Splatting 相关 CUDA 扩展
- NumPy、SciPy、OpenCV、Pillow、Streamlit 等 Python 依赖

激活环境：

```bash
conda activate ed3dgs
cd /root/autodl-tmp/GRACE-GS
```

> 独立部署时需要安装项目依赖，并编译相应 CUDA 扩展。`requirements.txt` 记录当前环境的 Python 依赖。

## 3. 数据准备

每个对象需要提供原始 RGB 图像、初始 Gaussian PLY 和相机内参。

例如：

```text
Mono_Data/
└── Fan/
    ├── Fan.png
    ├── Fan.ply
    └── camera_intrinsics.json
```

训练前，需要修改对应 INI 配置文件中的路径：

- `[Path]`：数据路径、初始 PLY、原始 RGB
- `[Camera Alignment]`：相机内参和标定输出目录
- `[Stage1 Pipeline]`：训练输出路径

## 4. Stage1 训练

训练流程：

```text
Single RGB Image
       |
       v
Initial Gaussian G0
       |
       v
Fixed-Camera Alignment
       |
       v
Gaussian Surface Refinement
       |
       v
Corrected Gaussian G0*
```

### 4.1 完整训练

执行相机标定和 Gaussian Refinement：

```bash
conda activate ed3dgs
cd /root/autodl-tmp/GRACE-GS

CUDA_VISIBLE_DEVICES=0 python -u Stage1/train_stage1.py \
  --config Stage1/fan_stage1.ini
```

其他对象只需要切换配置文件，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python -u Stage1/train_stage1.py \
  --config Stage1/lamp_stage1.ini
```

### 4.2 训练输出

输出目录由 INI 配置文件控制，例如：

```text
MonoEv/Stage1/Fan/round1/
├── alignment/
│   ├── fixed_camera.json
│   └── diagnostic_views/
│
└── refinement/
    ├── corrected_g0.ply
    └── diagnostics/
```

其中：

- `fixed_camera.json`：相机标定结果。
- `corrected_g0.ply`：修复后的 Gaussian 模型。
- `diagnostics/`：训练过程中的诊断与可视化结果。

## 5. 可视化 UI

启动 Streamlit：

```bash
conda activate ed3dgs
cd /root/autodl-tmp/GRACE-GS

streamlit run Stage1_Visualization_UI/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

浏览器访问服务器映射的 `8501` 端口。

### UI 功能

**Page 1 — Auto Detail Comparison**

- Original RGB / Original G0 / Corrected G0 对比
- 自动选择四个局部细节区域
- 5×3 科研对比图
- PSNR、SSIM、MAE、Mask IoU 等指标
- 支持导出对比图和指标表格

**Page 2 — Four-view Scientific Figure**

- Front / Left / Right / Back 四视角
- Original G0 与 Corrected G0 对比
- 支持导出科研对比图片

**Preset**

自动扫描：

```text
/root/autodl-tmp/MonoEv/Stage1/
```

读取各对象的训练结果，也支持手动指定文件路径。

## 6. 主要代码说明

| 文件 | 功能 |
|---|---|
| `Stage1/train_stage1.py` | Stage1 完整训练入口 |
| `Stage1/camera_alignment.py` | 固定内参虚拟相机标定 |
| `Stage1/gaussian_refinement.py` | RGB 监督的 Gaussian Refinement |
| `Stage1/gaussian_utils/` | 高斯几何处理、增密与修复工具 |
| `Stage1_Visualization_UI/app.py` | Streamlit 可视化入口 |
| `Stage1_Visualization_UI/render_engine.py` | 高斯模型加载及渲染 |
| `scene/` | GaussianModel 和场景管理 |
| `gaussian_renderer/` | Gaussian Splatting 渲染器 |
| `utils/` | 通用计算工具 |
| `submodules/` | CUDA 扩展相关源码 |

## 7. 注意事项

1. 运行前需要根据本地环境修改 INI 文件中的绝对路径。
2. Stage1 使用固定相机内参进行虚拟相机标定。
3. Gaussian Refinement 使用 RGB 和轮廓监督，不要求事件流作为静态修复输入。
4. 可视化 UI 只读取训练结果，不会修改原始 Gaussian 模型。
5. 单视角 RGB 指标不能直接证明未观测视角的几何准确性。
6. 独立打包后，需要检查 CUDA 扩展、第三方依赖以及代码中的硬编码路径。

## 8. MERCY 项目

GRACE-GS 是 MERCY 的静态高斯初始化模块。

**MERCY — Monocular Event-based Locally Rigid Motion-Coherent 4D Gaussian Reconstruction**

GitHub：

https://github.com/SuleynanAuir/MERCY-Monocular-Event-Locally-RigidMotion-Coherent-4DGS-Reconstruct
