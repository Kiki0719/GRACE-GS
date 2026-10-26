# GRACE-GS

**Gaussian Reconstruction, Alignment, and Continuity Enhancement**

GRACE-GS 是 MERCY 项目的单目静态三维高斯初始化与优化模块。项目首先使用 **Hunyuan3D 2.1** 从单张 RGB 图像生成初始三维 Mesh，再通过 **Mesh2Splat** 将 Mesh 转换为初始 3D Gaussian Splatting 模型（G0）。随后，GRACE-GS 利用初始 G0 和对应的 RGB 图像，通过固定内参虚拟相机标定、RGB 引导的 Gaussian Refinement 和表面连续性优化，对高斯模型进行几何与外观修正，最终生成优化后的 Gaussian 模型（G0*），为 MERCY 后续动态 4D Gaussian Reconstruction 提供静态初始化。

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
│
├── third_party/
│   ├── Hunyuan3D-2.1/              # RGB → Mesh (Git Submodule)
│   └── mesh2splat/                 # Mesh → Gaussian (Git Submodule)
│
├── requirements.txt                # Python 依赖
└── README.md
```

## 2. 下载项目与安装环境

### 2.1 克隆仓库

本项目通过 Git Submodule 引用 Hunyuan3D 2.1 和 Mesh2Splat。

建议使用以下命令，同时获取 GRACE-GS 及两个第三方项目的源码：

```bash
git clone --recurse-submodules https://github.com/Kiki0719/GRACE-GS.git
cd GRACE-GS
```

如果已经克隆 GRACE-GS，可以执行：

```bash
git submodule update --init --recursive
```

第三方项目：

- [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)：单张 RGB 图像到三维 Mesh
- [Mesh2Splat](https://github.com/electronicarts/mesh2splat)：三维 Mesh 到 3D Gaussian Splatting

### 2.2 创建 GRACE-GS Python 环境

推荐使用 Linux、支持 CUDA 的 NVIDIA GPU，以及与 CUDA 扩展兼容的 PyTorch 环境。

GRACE-GS 开发参考环境：

- Python 3.10
- PyTorch 2.1.2
- CUDA 12.1
- NVIDIA GPU

创建独立 Conda 环境：

```bash
conda create -n grace-gs python=3.10 -y
conda activate grace-gs
```

建议 **Hunyuan3D 与 GRACE-GS 分别配置 Python 环境**，避免 PyTorch、CUDA 扩展和其他依赖发生版本冲突。

Mesh2Splat 使用 C++、CMake 和图形渲染相关依赖，需单独构建。

### 2.3 安装 GRACE-GS Python 依赖

首先安装与本机 CUDA 环境兼容的 PyTorch，然后在 GRACE-GS 根目录执行：

```bash
python -m pip install -r requirements.txt
```

如果 `requirements.txt` 包含平台相关依赖、本地安装路径或 CUDA 扩展，需要根据实际环境进行调整。

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

## 3. Hunyuan3D + Mesh2Splat 静态初始化

在运行 GRACE-GS 之前，需要先完成两个初始化步骤：

1. **Hunyuan3D 2.1**：根据单张 RGB 图像生成带有三维几何和外观信息的 Mesh。
2. **Mesh2Splat**：将 Mesh 转换为兼容 3D Gaussian Splatting 的初始 Gaussian 模型 G0。

GRACE-GS 不直接执行这两个第三方程序，而是读取准备好的 Gaussian G0 进行后续优化。

### 3.1 Hunyuan3D 环境准备

进入 Hunyuan3D 子模块：

```bash
cd third_party/Hunyuan3D-2.1
```

参考官方安装与使用说明：

[Hunyuan3D 2.1 GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)

根据官方 README 安装环境、下载模型权重并运行单图三维生成流程。

Hunyuan3D 2.1 支持单图几何生成及纹理生成，具体推理方式以所引用的官方版本为准。

### 3.2 生成初始 Mesh

将目标 RGB 图像输入 Hunyuan3D，生成初始三维 Mesh。

推荐将结果保存为 `.glb` 格式，以便后续使用 Mesh2Splat。

例如：

```text
datasets/
└── OBJECT_NAME/
    ├── OBJECT_NAME.png
    └── OBJECT_NAME.glb
```

其中：

- `OBJECT_NAME.png`：原始参考 RGB 图像
- `OBJECT_NAME.glb`：Hunyuan3D 生成的三维 Mesh

如果原始输出不是 `.glb`，需要先转换为 Mesh2Splat 支持的格式。

### 3.3 Mesh2Splat 环境准备

Mesh2Splat 可以利用 Mesh 的几何、材质及纹理信息，生成 3D Gaussian Splatting 表示。

官方项目：

[Mesh2Splat GitHub](https://github.com/electronicarts/mesh2splat)

进入子模块：

```bash
cd third_party/mesh2splat
```

#### Linux 环境

根据官方 README，安装所需依赖：

```bash
sudo apt update

sudo apt install -y \
  build-essential cmake pkg-config git \
  libfreeimage-dev libglew-dev libglfw3-dev \
  libgl1-mesa-dev libxinerama-dev \
  libxcursor-dev libxi-dev libxxf86vm-dev
```

编译项目：

```bash
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Mesh2Splat 依赖兼容的图形环境。对于无图形桌面的远程服务器，可能需要额外配置 OpenGL 和显示环境。

Windows 用户可以参考 Mesh2Splat 官方 README 中的 Visual Studio 和 CMake 构建说明。

### 3.4 Mesh 转换为初始 Gaussian G0

使用 Mesh2Splat 加载 Hunyuan3D 生成的 `.glb` Mesh，并通过其转换功能生成 Gaussian Splatting 模型。

```text
Hunyuan3D Output
       |
       v
OBJECT_NAME.glb
       |
       v
Mesh2Splat
       |
       v
OBJECT_NAME.ply
       |
       v
Initial Gaussian G0
```

转换时可以根据对象表面细节调整 Gaussian 采样密度。

最终需要获得可由 GRACE-GS 读取的 Gaussian PLY，而不是普通 Mesh PLY。

需要注意：Mesh2Splat 的输出应检查 Gaussian 属性、坐标系及 PLY 数据格式是否与 GRACE-GS 的 GaussianModel 兼容。若二者的 SH、scale、rotation 或 opacity 等属性表示不同，需要先完成适配。

### 3.5 最终输入数据

完成 Hunyuan3D 与 Mesh2Splat 初始化后，推荐整理为：

```text
datasets/
└── OBJECT_NAME/
    ├── OBJECT_NAME.png
    ├── OBJECT_NAME.glb
    ├── OBJECT_NAME.ply
    └── camera_intrinsics.json
```

其中：

- `OBJECT_NAME.png`：原始 RGB 参考图像
- `OBJECT_NAME.glb`：Hunyuan3D 生成的初始 Mesh
- `OBJECT_NAME.ply`：Mesh2Splat 生成并完成兼容性检查的 Gaussian G0
- `camera_intrinsics.json`：固定相机内参

GRACE-GS 的 Stage1 主要使用：

1. RGB 图像
2. Gaussian G0
3. Camera Intrinsics

`.glb` Mesh 主要用于保留初始化过程中的中间结果。

### 3.6 完整技术流程

```text
                Single RGB Image
                        |
                        v
                Hunyuan3D 2.1
                 Shape + Texture
                        |
                        v
                  Initial Mesh
                 OBJECT_NAME.glb
                        |
                        v
                   Mesh2Splat
              Mesh-to-3DGS Conversion
                        |
                        v
               Initial Gaussian G0
                 OBJECT_NAME.ply
                        |
                        v
                    GRACE-GS
                        |
           +------------+------------+
           |                         |
           v                         |
   Fixed-Camera Alignment            |
   (Fixed Intrinsics)                |
           |                         |
           v                         |
    Optimized Camera                 |
           |                         |
           +------------+------------+
                        |
                        v
             Gaussian Refinement
           RGB / Silhouette / Geometry
                        |
                        v
              Corrected Gaussian G0*
                 corrected_g0.ply
                        |
                        v
               MERCY Dynamic 4DGS
```

Hunyuan3D 与 Mesh2Splat 通过 Git Submodule 引用源码，但模型下载、环境安装、Mesh 转换和格式兼容性检查仍需分别完成。

## 4. GRACE-GS 配置文件

完成 Hunyuan3D 和 Mesh2Splat 初始化后，返回 GRACE-GS 根目录：

```bash
cd /path/to/GRACE-GS
conda activate grace-gs
```

复制已有的 INI 配置文件作为模板：

```bash
cp Stage1/fan_stage1.ini Stage1/custom_stage1.ini
```

根据实际数据和输出目录修改以下配置。

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

以上仅展示主要路径配置。实际使用时，应保留原始 INI 文件中的其他训练参数。

请将 `/path/to/...` 替换为实际文件路径，并检查配置文件中的其他绝对路径。

## 5. Stage1 训练

完成 Hunyuan3D Mesh 初始化、Mesh2Splat Gaussian 转换及 INI 配置后，即可执行 GRACE-GS 训练。

```bash
conda activate grace-gs
cd /path/to/GRACE-GS

CUDA_VISIBLE_DEVICES=0 python -u Stage1/train_stage1.py \
  --config Stage1/custom_stage1.ini
```

训练包含两个主要阶段：

1. **Camera Alignment**：利用固定相机内参和参考 RGB，估计与初始 Gaussian G0 匹配的虚拟相机位姿。
2. **Gaussian Refinement**：通过 RGB、局部相似度、轮廓及几何约束优化 Gaussian 模型，改善外观一致性、表面覆盖及几何连续性。

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
- `diagnostics/`：训练过程诊断与可视化结果

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

远程服务器用户需要访问对应的端口映射地址。

### 6.1 UI 功能

**Page 1 — Auto Detail Comparison**

- Original RGB / Original G0 / Corrected G0 对比
- 自动选择四个局部细节区域
- 5×3 科研对比图
- PSNR、SSIM、MAE、Mask IoU 等重建指标
- 支持导出对比图及指标表格

**Page 2 — Four-view Scientific Figure**

- Front / Left / Right / Back 四视角
- Original G0 与 Corrected G0 对比
- 支持导出多视角科研图片

### 6.2 数据路径配置

UI 支持通过侧边栏手动指定：

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
| `third_party/Hunyuan3D-2.1/` | Hunyuan3D 单图三维初始化子模块 |
| `third_party/mesh2splat/` | Mesh2Splat Mesh-to-Gaussian 转换子模块 |

## 8. 注意事项

1. 首次使用前，需要分别安装 Hunyuan3D、Mesh2Splat 和 GRACE-GS 的对应依赖。
2. Hunyuan3D 的模型权重及推理环境应按照其官方说明配置。
3. Mesh2Splat 需要兼容的 CMake、C++ 编译器及图形环境。
4. GRACE-GS 需要符合 3D Gaussian Splatting 格式的初始 G0，不能直接使用普通 Mesh PLY。
5. 运行前需要根据本机环境修改 INI 文件及 UI 中的数据、模型和输出路径。
6. Stage1 使用固定相机内参进行虚拟相机标定。
7. Gaussian Refinement 主要使用 RGB 和轮廓监督，静态阶段不要求事件流作为必需输入。

## 9. MERCY 项目

GRACE-GS 是 MERCY 的静态 Gaussian 初始化与优化模块。

**MERCY — Monocular Event-based Locally Rigid Motion-Coherent 4D Gaussian Reconstruction**

GitHub：

https://github.com/SuleynanAuir/MERCY-Monocular-Event-Locally-RigidMotion-Coherent-4DGS-Reconstruct

## Acknowledgements

本项目的单图三维初始化流程使用了以下开源项目：

- **Hunyuan3D 2.1** — Tencent Hunyuan Team  
  https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1

- **Mesh2Splat** — Electronic Arts SEED  
  https://github.com/electronicarts/mesh2splat

上述项目均通过 Git Submodule 引用。第三方源码、模型及相关资源的使用应遵守各自的许可证和使用条款。
