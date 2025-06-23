# 🍎 DATASAM2GET Mac版本安装指南

专为macOS系统优化的安装指南，解决GPU依赖和Linux特定命令的兼容性问题。

## 🚨 主要改动说明

相比Linux版本，Mac版本主要调整了以下内容：

### 1. 移除GPU/CUDA依赖
- ❌ 移除了 `pytorch-cuda=11.8`
- ✅ 使用CPU版本的PyTorch
- 🚀 支持Apple Silicon的MPS加速

### 2. 优化安装脚本
- 🔄 `wget` → `curl` (Mac原生支持)
- 🍎 自动检测Apple Silicon vs Intel
- 📦 优化依赖版本兼容性

### 3. 性能调整
- 💻 CPU处理模式
- ⚡ Apple Silicon MPS加速（如果可用）
- 🎯 推荐处理小规模数据

## 🚀 快速安装

### 方法一：使用Mac专用脚本（推荐）

```bash
# 1. 运行Mac版安装脚本
bash setup_mac.sh

# 2. 启动应用
./start_ui_mac.sh
```

### 方法二：手动安装

```bash
# 1. 创建Mac专用conda环境
conda env create -f environment_mac.yml

# 2. 激活环境
conda activate datasam2get

# 3. 安装Mac专用依赖
pip install -r requirements_mac.txt

# 4. 手动安装SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2 && pip install -e . && cd ..

# 5. 下载模型
mkdir -p sam2/checkpoints
curl -L -o sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
```

## 📋 系统要求

### 硬件要求
- **Mac**: macOS 10.15+ (Intel 或 Apple Silicon)
- **内存**: 8GB+ RAM（推荐16GB+）
- **存储**: 10GB+ 可用空间
- **网络**: 稳定的网络连接（下载模型）

### 软件要求
- **conda**: Anaconda或Miniconda
- **Git**: 用于下载SAM2源码
- **Python**: 3.8-3.11（推荐3.10）

## 🛠️ 安装故障排除

### 问题1：conda未找到
```bash
# 解决方案：安装Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Apple Silicon Mac用户
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

### 问题2：模型下载失败
```bash
# 方案1：使用代理或更换网络
export https_proxy=http://127.0.0.1:7890  # 如果有代理

# 方案2：手动下载
# 访问 https://github.com/facebookresearch/segment-anything-2
# 下载模型文件到 sam2/checkpoints/ 目录
```

### 问题3：PyQt5安装失败
```bash
# Apple Silicon Mac可能需要
brew install pyqt5
pip install PyQt5 --config-settings --global-option=build --global-option=--include-dirs=$(brew --prefix)/include
```

### 问题4：依赖冲突
```bash
# 清理并重新安装
conda env remove -n datasam2get
conda clean --all
bash setup_mac.sh
```

## ⚡ 性能优化建议

### Apple Silicon Mac用户
```python
# 在代码中启用MPS加速
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 使用Apple Silicon MPS加速")
else:
    device = torch.device("cpu")
    print("💻 使用CPU处理")
```

### Intel Mac用户
```bash
# 优化CPU性能
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 通用优化
- 📉 降低视频分辨率（720p → 480p）
- 🔢 减少处理帧数（100帧 → 50帧）
- 💾 使用SSD存储提升I/O性能

## 📊 性能预期

### Apple Silicon Mac (M1/M2/M3)
- **标注速度**: 20-30帧/分钟
- **训练速度**: 较慢，建议小数据集
- **推理速度**: 5-10 FPS

### Intel Mac
- **标注速度**: 10-15帧/分钟  
- **训练速度**: 较慢，推荐云端训练
- **推理速度**: 2-5 FPS

## 🎯 使用建议

### 数据标注
- 🎬 处理短视频（<1分钟）
- 🎯 选择关键帧标注而非全帧
- 💾 及时保存避免丢失

### 模型训练
- 📊 使用小数据集（<1000张）
- ☁️ 考虑使用云端GPU训练
- 🔄 启用断点续训功能

### 推理应用
- 📱 处理低分辨率视频
- ⏱️ 接受较慢的处理速度
- 🎯 专注功能验证而非性能

## 🆘 获取帮助

如果遇到问题：

1. **检查系统兼容性**
   ```bash
   system_profiler SPSoftwareDataType
   python --version
   conda --version
   ```

2. **收集诊断信息**
   ```bash
   python -c "
   import torch, platform, sys
   print(f'系统: {platform.system()} {platform.release()}')
   print(f'Python: {sys.version}')
   print(f'PyTorch: {torch.__version__}')
   print(f'MPS可用: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
   "
   ```

3. **常用命令**
   ```bash
   # 重置环境
   conda env remove -n datasam2get
   
   # 查看日志
   tail -f ~/.conda/envs/datasam2get/conda-meta/history
   
   # 测试基础功能
   python -c "import torch, cv2, streamlit; print('基础包导入成功')"
   ```

## ✅ 安装验证

安装完成后，运行以下命令验证：

```bash
# 1. 激活环境
conda activate datasam2get

# 2. 检查核心包
python -c "
import torch, ultralytics, streamlit, cv2
print('✅ 核心包安装成功')
print(f'PyTorch: {torch.__version__}')
print(f'设备: {\"MPS\" if torch.backends.mps.is_available() else \"CPU\"}')
"

# 3. 启动测试
./start_ui_mac.sh
```

## 🎉 开始使用

安装成功后，你可以：

1. **体验数据标注**: 选择选项1，访问 http://localhost:8501
2. **测试模型训练**: 选择选项2（准备好数据后）
3. **运行演示程序**: 选择选项3，测试YOLO-SAM2分割

**祝你使用愉快！** 🚀 