# GRACE-GS

**Gaussian Reconstruction, Alignment, and Continuity Enhancement**

GRACE-GS 是 MERCY 项目的单目静态三维高斯初始化与优化模块。该模块以 **Hunyuan3D 生成并转换得到的初始三维高斯模型（G0）和对应的单张 RGB 图像**作为输入，通过固定内参虚拟相机标定、RGB 引导的 Gaussian Refinement 和表面连续性优化，对初始高斯模型进行几何与外观修正，并提供重建结果的可视化分析，最终生成优化后的高斯模型（G0*），为后续动态 4D Gaussian Reconstruction 提供静态初始化。

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
│   ├── app.py                      # Streamlit 可视化入口
│   └── render_engine.py            # Gaussian 渲染接口
│
├── scene/                          # Gaussian 模型与场景管理
├── gaussian_renderer/              # 3DGS 可微分渲染器
├── utils/                          # 通用工具函数
├── arguments/                      # 参数管理
├── submodules/                     # CUDA 扩展源码
├── third_party/
│   └── Hunyuan3D-2.1/              # Hunyuan3D Git Submodule
├── requirements.txt                # Python 依赖
└── README.md
```

## 2. 下载项目与安装环境

### 2.1 克隆仓库

本项目通过 Git Submodule 引用 Hunyuan3D 2.1，建议克隆时同步下载子模块：

```bash
git clone --recurse-submodules https://github.com/Kiki0719/GRACE-GS.git
cd GRACE-GS
```

如果已经克隆项目，可以执行：

```bash
git submodule update --init --recursive
```

Hunyuan3D 官方仓库：

[Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)

### 2.2 创建 Python 环境

推荐使用 Linux、支持 CUDA 的 NVIDIA GPU，以及与 CUDA 扩展兼容的 PyTorch 环境。

开发参考环境：

- Python 3.10
- PyTorch 2.1.2
- CUDA 12.1
- NVIDIA GPU

创建独立 Conda 环境：

```bash
conda create -n grace-gs python=3.10 -y
conda activate grace-gs
```

建议 Hunyuan3D 和 GRACE-GS 分别配置 Python 环境，避免依赖版本冲突。

### 2.3 安装 Python 依赖

根据本机 CUDA 环境安装兼容的 PyTorch，然后在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

如果 `requirements.txt` 包含平台相关依赖或本地安装路径，需要根据实际环境调整。

### 2.4 安装 CUDA 扩展

GRACE-GS 使用 3D Gaussian Splatting 的可微分渲染器及 KNN CUDA 扩展。

```bash
python -m pip install ./submodules/simple-knn
python -m pip install ./submodules/depth-diff-gaussian-rasterization
```

检查安装：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python -c "import diff_gaussian_rasterization; import simple_knn; print('CUDA extensions OK')"
```

## 3. Hunyuan3D 前置初始化

运行 GRACE-GS 之前，需要先通过 **Hunyuan3D 完成单张 RGB 图像的三维初始化**。

Hunyuan3D 负责生成初始三维几何与外观表示，GRACE-GS 在此基础上进行虚拟相机标定与 Gaussian Refinement。

### 3.1 Hunyuan3D 环境准备

进入 Hunyuan3D 子模块：

```bash
cd third_party/Hunyuan3D-2.1
```

根据 [Hunyuan3D 2.1 官方 README](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)，安装依赖、下载模型权重并运行单图三维重建。

Hunyuan3D 的具体推理命令及环境要求以官方仓库为准。

### 3.2 生成初始 Gaussian G0

使用 Hunyuan3D 对目标 RGB 图像进行三维初始化。

如果 Hunyuan3D 输出的是 Mesh，需要通过相应的 Gaussian 初始化或格式转换流程，将其转换为符合 3D Gaussian Splatting 格式的初始模型 `G0.ply`。

普通 Mesh PLY 不能直接作为 GRACE-GS 的 Gaussian PLY 输入。

最终应准备：

```text
datasets/
└── OBJECT_NAME/
    ├── OBJECT_NAME.png
    ├── OBJECT_NAME.ply
    └── camera_intrinsics.json
```

其中：

- `OBJECT_NAME.png`：参考 RGB 图像
- `OBJECT_NAME.ply`：初始 Gaussian G0
- `camera_intrinsics.json`：固定相机内参

### 3.3 完整技术流程

```text
Single RGB Image
       |
       v
Hunyuan3D Initialization
       |
       v
Initial 3D Model
       |
       v
Gaussian Initialization / Conversion
       |
       v
Initial Gaussian G0
       |
       v
GRACE-GS
       |
       ├── Fixed-Intrinsics Camera Alignment
       |
       └── RGB-guided Gaussian Refinement
       |
       v
Corrected Gaussian G0*
       |
       v
MERCY Dynamic 4DGS Reconstruction
```

Git Submodule 用于关联 Hunyuan3D 源码，不会自动下载模型权重或安装其运行环境。

## 4. GRACE-GS 配置文件

完成 Hunyuan3D 初始化后，返回 GRACE-GS 根目录：

```bash
cd /path/to/GRACE-GS
conda activate grace-gs
```

复制已有 INI 文件作为模板：

```bash
cp Stage1/fan_stage1.ini Stage1/custom_stage1.ini
```

根据自己的数据及输出目录修改以下配置。

### 4.1 数据路径

```ini
[Path]
source_path = /path/to/datasets/OBJECT_NAME
initial_gaussian_ply = /path/to/datasets/OBJECT_NAME/OBJECT_NAME.ply
t0_image_path = /path/to/datasets/OBJECT_NAME/OBJECT_NAME.png
t0_mask_path = None
camera_file = /path/to/outputs/OBJECT_NAME/round1/alignment/fixed_camera.json
model_path = /path/to/outputs/OBJECT_NAME/round1/refinement
diagnostics_path = /path/to/outputs/OBJECT_NAME/round1/refinement/diagnostics
```

### 4.2 相机标定

```ini
[Camera Alignment]
camera_intrinsics_path = /path/to/datasets/OBJECT_NAME/camera_intrinsics.json
camera_alignment_output = /path/to/outputs/OBJECT_NAME/round1/alignment
camera_prior_path = None
```

### 4.3 Stage1 输出目录

```ini
[Stage1 Pipeline]
stage1_pipeline_root = /path/to/outputs
stage1_round1_dir = /path/to/outputs/OBJECT_NAME/round1
stage1_refinement_output = /path/to/outputs/OBJECT_NAME/round1/refinement
stage1_final_dir = /path/to/outputs/OBJECT_NAME/final
```

以上仅展示主要路径配置。实际使用时应保留原始 INI 文件中的其他训练参数。

请将 `/path/to/...` 替换为实际文件路径，并检查配置文件中的其他绝对路径。

## 5. Stage1 训练

完成 Hunyuan3D 初始化、Gaussian G0 准备及 INI 配置后，即可执行 GRACE-GS 训练。

```bash
conda activate grace-gs
cd /path/to/GRACE-GS

CUDA_VISIBLE_DEVICES=0 python -u Stage1/train_stage1.py \
  --config Stage1/custom_stage1.ini
```

训练包含两个主要阶段：

1. **Camera Alignment**：利用固定相机内参和参考 RGB，估计与初始 Gaussian G0 匹配的虚拟相机位姿。
2. **Gaussian Refinement**：通过 RGB、SSIM、轮廓和几何约束优化 Gaussian 模型，改善外观一致性与表面覆盖。

### 5.1 训练输出

输出位置由 INI 文件决定，典型结构如下：

```text
outputs/
└── OBJECT_NAME/
    └── round1/
        ├── alignment/
        │   ├── fixed_camera.json
        │   └── diagnostic_views/
        │
        └── refinement/
            ├── corrected_g0.ply
            └── diagnostics/
```

主要文件：

- `fixed_camera.json`：固定内参虚拟相机标定结果
- `corrected_g0.ply`：优化后的 Gaussian G0*
- `diagnostics/`：训练过程诊断及可视化结果

## 6. 可视化 UI

GRACE-GS 提供基于 Streamlit 的高斯重建可视化界面。

启动命令：

```bash
conda activate grace-gs
cd /path/to/GRACE-GS

streamlit run Stage1_Visualization_UI/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

本地访问：

```text
http://localhost:8501
```

远程服务器用户应使用对应的端口映射地址。

### 6.1 UI 功能

**Page 1 — Auto Detail Comparison**

- Original RGB / Original G0 / Corrected G0 对比
- 自动选择四个局部细节区域
- 5×3 科研对比图
- PSNR、SSIM、MAE、Mask IoU 等重建指标
- 支持导出对比图和指标表格

**Page 2 — Four-view Scientific Figure**

- Front / Left / Right / Back 四视角
- Original G0 与 Corrected G0 对比
- 支持导出多视角科研图片

### 6.2 数据路径配置

UI 支持通过侧边栏指定：

- Original Gaussian PLY
- Corrected Gaussian PLY
- Fixed Camera JSON
- Stage1 INI
- Original RGB
- Target Mask（可选）

如果需要使用自动识别 Preset 功能，请修改：

```text
Stage1_Visualization_UI/app.py
```

根据自己的运行环境配置：

```python
preset_root = Path("/path/to/outputs")
data_root = Path("/path/to/datasets")
```

重新启动 Streamlit 后，即可自动识别输出目录下的对象文件夹。

## 7. 主要代码说明

| 文件 | 功能 |
|---|---|
| `Stage1/train_stage1.py` | Stage1 完整训练入口 |
| `Stage1/camera_alignment.py` | 固定内参虚拟相机标定 |
| `Stage1/gaussian_refinement.py` | RGB 引导的 Gaussian Refinement |
| `Stage1/gaussian_utils/` | 高斯几何处理、增密与表面修复工具 |
| `Stage1_Visualization_UI/app.py` | Streamlit 可视化入口 |
| `Stage1_Visualization_UI/render_engine.py` | Gaussian 模型加载与渲染 |
| `scene/` | GaussianModel 与场景管理 |
| `gaussian_renderer/` | 可微分 Gaussian Splatting 渲染 |
| `utils/` | 通用计算工具 |
| `submodules/` | CUDA 扩展源码 |
| `third_party/Hunyuan3D-2.1/` | Hunyuan3D 初始化子模块 |

## 8. 注意事项

1. 首次使用前，需要安装 Hunyuan3D 和 GRACE-GS 的对应依赖。
2. Hunyuan3D 的模型权重及推理环境应按照官方说明配置。
3. GRACE-GS 需要符合 3D Gaussian Splatting 格式的初始 G0，不能直接使用普通 Mesh PLY。
4. 运行前需要根据本机环境修改 INI 文件及 UI 中的数据、模型和输出路径。
5. Stage1 使用固定相机内参进行虚拟相机标定。
6. Gaussian Refinement 主要使用 RGB 和轮廓监督，静态阶段不要求事件流作为必需输入。

## 9. MERCY 项目

GRACE-GS 是 MERCY 的静态 Gaussian 初始化与优化模块。

**MERCY — Monocular Event-based Locally Rigid Motion-Coherent 4D Gaussian Reconstruction**

GitHub：

https://github.com/SuleynanAuir/MERCY-Monocular-Event-Locally-RigidMotion-Coherent-4DGS-Reconstruct

## Acknowledgements

本项目的单图三维初始化流程基于 [Tencent Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)。

Hunyuan3D 通过 Git Submodule 引用，相关第三方代码及模型的使用应遵守原始许可证和使用条款。
