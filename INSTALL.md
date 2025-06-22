# 🔧 DATASAM2GET 环境安装指南

在新电脑上快速部署DATASAM2GET项目环境的完整指南。

## 📋 安装准备

### 系统要求
- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 - 3.11 (推荐 3.10)
- **内存**: 8GB+ RAM (推荐 16GB+)
- **存储**: 10GB+ 可用空间
- **GPU**: NVIDIA GPU with CUDA 11.8+ (可选，CPU也可运行)

### 必备软件
1. **Anaconda 或 Miniconda** - [下载地址](https://docs.conda.io/en/latest/miniconda.html)
2. **Git** - [下载地址](https://git-scm.com/downloads)
3. **CUDA Toolkit** (GPU用户) - [下载地址](https://developer.nvidia.com/cuda-downloads)

## 🚀 快速安装 (推荐)

### 方法一：自动安装脚本

#### Linux/Mac 用户:
```bash
# 1. 克隆项目
git clone https://github.com/daftu1/mysam.git
cd mysam

# 2. 运行自动安装脚本
bash setup.sh

# 3. 启动项目
./start_ui.sh
```

#### Windows 用户:
```cmd
# 1. 克隆项目
git clone https://github.com/daftu1/mysam.git
cd mysam

# 2. 运行自动安装脚本
setup.bat

# 3. 启动项目
start_ui.bat
```

### 方法二：手动安装

#### 步骤 1: 创建Conda环境
```bash
# 从environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate datasam2get
```

#### 步骤 2: 安装Python依赖
```bash
# 安装额外的Python包
pip install -r requirements.txt
```

#### 步骤 3: 安装SAM2
```bash
# 下载SAM2源码
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2

# 安装SAM2
pip install -e .

# 下载预训练模型
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ../..
```

#### 步骤 4: 验证安装
```bash
python -c "
import torch
import ultralytics
import streamlit
import cv2
from sam2.build_sam import build_sam2_video_predictor
print('✅ 所有包安装成功!')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
```

## 🔧 自定义安装选项

### GPU/CPU版本选择

#### GPU版本 (推荐):
```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### CPU版本:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 模型选择

#### 基础模型 (2GB):
```bash
# 只下载基础模型
wget -O sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
```

#### 完整模型 (8GB):
```bash
# 下载所有模型
cd sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## 📁 配置文件说明

### `environment.yml`
Conda环境配置文件，包含：
- Python版本和基础包
- PyTorch GPU/CPU版本
- 计算机视觉相关包
- GUI和Web框架

### `requirements.txt`
额外的Python包，包含：
- YOLO和SAM2依赖
- Streamlit UI框架
- 数据处理工具
- 开发调试工具

### `setup.sh / setup.bat`
自动安装脚本，功能：
- 检查系统环境
- 创建Conda环境
- 安装所有依赖
- 下载预训练模型
- 生成启动脚本

## 🎯 启动项目

### 方式一：启动脚本 (推荐)
```bash
# Linux/Mac
./start_ui.sh

# Windows
start_ui.bat
```

### 方式二：手动启动
```bash
# 激活环境
conda activate datasam2get

# 数据标注工具
cd data_generation
streamlit run streamlit_app_enhanced.py --server.port 8501

# 模型训练
cd training_models
python run_training.py

# YOLO-SAM2演示
cd demo_apps
python run_yolo_sam2_ui.py
```

## 🔍 故障排除

### 常见问题

#### 1. CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装对应的PyTorch版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 2. SAM2安装失败
```bash
# 手动克隆并安装
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2
pip install -e .
```

#### 3. Streamlit启动失败
```bash
# 更新Streamlit
pip install --upgrade streamlit

# 检查端口占用
netstat -tuln | grep 8501
```

#### 4. 内存不足
```bash
# 减少模型大小
# 在environment.yml中使用更小的模型
# 或者增加虚拟内存
```

### 环境重置
```bash
# 完全删除环境重新安装
conda env remove -n datasam2get
conda env create -f environment.yml
```

## 📊 性能优化

### GPU优化
```bash
# 设置CUDA内存分配
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 启用混合精度
export TORCH_CUDNN_V8_API_ENABLED=1
```

### CPU优化
```bash
# 设置线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 🔄 更新指南

### 更新项目代码
```bash
git pull origin main
```

### 更新Python包
```bash
conda activate datasam2get
conda update --all
pip install --upgrade -r requirements.txt
```

### 更新SAM2
```bash
cd sam2
git pull origin main
pip install -e .
```

## 📞 技术支持

### 获取帮助
- 🐛 **Bug报告**: [GitHub Issues](https://github.com/daftu1/mysam/issues)
- 💡 **功能建议**: [GitHub Discussions](https://github.com/daftu1/mysam/discussions)
- 📖 **文档问题**: [README.md](README.md)

### 诊断信息
运行以下命令收集诊断信息：
```bash
python -c "
import sys, torch, ultralytics, cv2, streamlit
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
print(f'Ultralytics: {ultralytics.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'Streamlit: {streamlit.__version__}')
"
```

## 📦 打包部署

### 创建可分发环境
```bash
# 导出环境
conda env export > environment_full.yml

# 创建离线包
conda pack -n datasam2get
```

### Docker部署 (高级)
```dockerfile
# Dockerfile示例
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
COPY environment.yml .
RUN conda env create -f environment.yml
# ... 其他配置
```

---

## 🎉 安装完成

恭喜！现在你可以开始使用DATASAM2GET项目了：

1. **数据标注**: 访问 http://localhost:8501
2. **模型训练**: 在training_models/目录运行训练脚本
3. **视频分割**: 在demo_apps/目录测试YOLO-SAM2

**下一步**: 查看 [README.md](README.md) 了解详细使用方法。 