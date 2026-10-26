# GRACE-GS

**Gaussian Reconstruction, Alignment, and Continuity Enhancement**

GRACE-GS 是 MERCY 项目的单目静态三维高斯初始化模块，主要实现固定内参相机标定、RGB 引导的 Gaussian Refinement，以及高斯重建结果的可视化分析。

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
└── requirements.txt                # Python 依赖
```

## 2. 下载项目与安装环境

### 2.1 克隆仓库

```bash
git clone https://github.com/Kiki0719/GRACE-GS.git
cd GRACE-GS
```

### 2.2 创建 Python 环境

推荐使用 Linux、支持 CUDA 的 NVIDIA GPU，以及与项目 CUDA 扩展兼容的 PyTorch 环境。

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

### 2.3 安装 requirements.txt

首先确保已安装与本机 CUDA 环境兼容的 PyTorch，然后在项目根目录安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

**注意：** 如果 `requirements.txt` 是由开发环境的 `pip freeze` 导出的，可能包含特定平台的依赖版本、本地安装路径或 CUDA 扩展。使用前请检查并根据自己的环境调整。

### 2.4 安装 CUDA 扩展

GRACE-GS 使用 3D Gaussian Splatting 的可微分渲染器和 KNN CUDA 扩展。

如果源码位于 `submodules/`，可以根据实际目录安装：

```bash
python -m pip install ./submodules/simple-knn
python -m pip install ./submodules/depth-diff-gaussian-rasterization
```

如果项目还包含其他 CUDA 扩展，应根据相应模块的安装说明完成编译。

安装后可检查：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization; import simple_knn; print('CUDA extensions OK')"
```

## 3. 数据准备

每个对象需要提供：

- 原始 RGB 图像
- 初始 3D Gaussian PLY（G0）
- 相机内参 JSON

推荐数据目录：

```text
datasets/
└── OBJECT_NAME/
    ├── OBJECT_NAME.png
    ├── OBJECT_NAME.ply
    └── camera_intrinsics.json
```

其中初始 PLY 需要采用项目支持的 Gaussian Splatting 模型格式，而非普通点云 PLY。

## 4. 配置文件

进入 `Stage1/`，选择或复制一个已有 INI 配置文件作为模板。

例如：

```bash
cp Stage1/fan_stage1.ini Stage1/custom_stage1.ini
```

根据自己的数据存储位置修改配置。

重点检查以下部分：

**[Path]**

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

**[Camera Alignment]**

```ini
[Camera Alignment]
camera_intrinsics_path = /path/to/datasets/OBJECT_NAME/camera_intrinsics.json
camera_alignment_output = /path/to/outputs/OBJECT_NAME/round1/alignment
camera_prior_path = None
```

**[Stage1 Pipeline]**

```ini
[Stage1 Pipeline]
stage1_pipeline_root = /path/to/outputs
stage1_round1_dir = /path/to/outputs/OBJECT_NAME/round1
stage1_refinement_output = /path/to/outputs/OBJECT_NAME/round1/refinement
stage1_final_dir = /path/to/outputs/OBJECT_NAME/final
```

以上配置仅展示需要重点修改的路径字段。实际使用时应保留原始 INI 中其他训练参数和配置项。

**请将所有 `/path/to/...` 替换为自己电脑或服务器上的真实绝对路径。**

另外，检查配置文件中是否有其他数据路径、模型路径或实验输出路径，避免沿用示例配置中的旧路径。

## 5. Stage1 训练

完整训练流程：

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

在项目根目录执行：

```bash
conda activate grace-gs
cd /path/to/GRACE-GS

CUDA_VISIBLE_DEVICES=0 python -u Stage1/train_stage1.py \
  --config Stage1/custom_stage1.ini
```

训练将依次执行：

1. **Camera Alignment**：根据固定相机内参和参考 RGB，对初始 G0 进行虚拟相机位姿估计。
2. **Gaussian Refinement**：使用 RGB、轮廓及几何约束对初始 Gaussian 模型进行优化。

### 训练输出

输出目录由 INI 配置决定，典型结构如下：

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

- `fixed_camera.json`：相机标定结果。
- `corrected_g0.ply`：修复后的 Gaussian 模型。
- `diagnostics/`：训练过程中的诊断输出。

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

本地运行时，通常可以访问：

```text
http://localhost:8501
```

远程服务器运行时，需要根据服务器配置访问对应的端口映射地址。

### UI 功能

**Page 1 — Auto Detail Comparison**

- Original RGB / Original G0 / Corrected G0 对比
- 自动选择四个局部细节区域
- 5×3 科研对比图
- PSNR、SSIM、MAE、Mask IoU 等指标
- 支持导出对比图及指标表格

**Page 2 — Four-view Scientific Figure**

- Front / Left / Right / Back 四视角
- Original G0 与 Corrected G0 对比
- 支持导出多视角科研图

### 数据路径设置

UI 支持通过侧边栏手动指定：

- Original Gaussian PLY
- Corrected Gaussian PLY
- Fixed Camera JSON
- Stage1 INI
- Original RGB
- Target Mask（可选）

当前 Preset 自动扫描目录需要根据自己的运行环境进行配置。

修改：

```text
Stage1_Visualization_UI/app.py
```

找到：

```python
preset_root = Path("/root/autodl-tmp/MonoEv/Stage1")
```

替换为自己的 Stage1 输出根目录，例如：

```python
preset_root = Path("/path/to/outputs")
```

修改后重新启动 Streamlit，即可自动识别该目录下的对象文件夹。

如果原始数据文件不在默认搜索目录，还需要调整 UI 中的 `data_root`，或者通过侧边栏手动指定文件路径。

## 7. 主要代码说明

| 文件 | 功能 |
|---|---|
| `Stage1/train_stage1.py` | 完整 Stage1 训练入口 |
| `Stage1/camera_alignment.py` | 固定内参虚拟相机标定 |
| `Stage1/gaussian_refinement.py` | RGB 引导的 Gaussian Refinement |
| `Stage1/gaussian_utils/` | 高斯增密、几何优化及表面修复工具 |
| `Stage1_Visualization_UI/app.py` | Streamlit 可视化入口 |
| `Stage1_Visualization_UI/render_engine.py` | Gaussian 模型加载与渲染 |
| `scene/` | GaussianModel 与场景管理 |
| `gaussian_renderer/` | 可微分 Gaussian Splatting 渲染 |
| `utils/` | 通用计算工具 |
| `submodules/` | CUDA 扩展源码 |

## 8. 注意事项

1. 首次运行前安装 `requirements.txt` 中的 Python 依赖，并正确配置 CUDA 扩展。
2. 所有数据路径、配置路径、模型路径和输出目录应根据实际环境修改。
3. Stage1 使用固定相机内参进行虚拟相机标定。
4. Gaussian Refinement 使用 RGB 和轮廓监督，静态初始化阶段不要求事件流作为必需输入。
5. 可视化 UI 仅用于读取和分析模型，不修改输入的 Gaussian PLY。
6. 单视角 RGB 重建指标不等同于真实三维几何精度。

## 9. MERCY 项目

GRACE-GS 是 MERCY 的静态高斯初始化模块。

**MERCY — Monocular Event-based Locally Rigid Motion-Coherent 4D Gaussian Reconstruction**

GitHub：

https://github.com/SuleynanAuir/MERCY-Monocular-Event-Locally-RigidMotion-Coherent-4DGS-Reconstruct
