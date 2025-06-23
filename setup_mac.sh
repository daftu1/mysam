#!/bin/bash
# DATASAM2GET项目Mac版本环境自动安装脚本
# 作者: daftu1
# 运行方式: bash setup_mac.sh

set -e  # 遇到错误立即退出

echo "🚀 开始安装DATASAM2GET项目环境 (Mac版本)..."

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: conda未安装，请先安装Anaconda或Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ 检测到conda已安装"

# 检查是否为Apple Silicon Mac
if [[ $(uname -m) == "arm64" ]]; then
    echo "🍎 检测到Apple Silicon Mac (M1/M2/M3)"
    APPLE_SILICON=true
else
    echo "🖥️  检测到Intel Mac"
    APPLE_SILICON=false
fi

# 创建conda环境
echo "📦 创建conda环境: datasam2get"
if conda env list | grep -q "datasam2get"; then
    echo "⚠️  环境datasam2get已存在，是否删除重建? (y/N)"
    read -p "请输入选择: " choice
    if [[ $choice =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有环境..."
        conda env remove -n datasam2get -y
    else
        echo "🔄 使用现有环境"
    fi
fi

if ! conda env list | grep -q "datasam2get"; then
    echo "🏗️  创建新环境..."
    conda env create -f environment_mac.yml
fi

echo "✅ conda环境创建完成"

# 激活环境
echo "🔧 激活环境并安装额外依赖..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate datasam2get

# 安装Mac专用requirements
echo "📦 安装额外Python包 (Mac版本)..."
pip install -r requirements_mac.txt

# 检查PyTorch安装
echo "🔍 检查PyTorch安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CPU可用: {torch.backends.mkl.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS可用 (Apple Silicon加速): {torch.backends.mps.is_available()}')
else:
    print('MPS不可用 (非Apple Silicon或PyTorch版本过低)')
"

# 安装SAM2
echo "🎯 安装SAM2..."
SAM2_DIR="sam2"
if [ -d "$SAM2_DIR" ]; then
    echo "⚠️  SAM2目录已存在，是否重新安装? (y/N)"
    read -p "请输入选择: " choice
    if [[ $choice =~ ^[Yy]$ ]]; then
        rm -rf "$SAM2_DIR"
    else
        echo "🔄 使用现有SAM2安装"
    fi
fi

if [ ! -d "$SAM2_DIR" ]; then
    echo "📥 下载SAM2源码..."
    git clone https://github.com/facebookresearch/segment-anything-2.git sam2
    cd sam2
    
    echo "📦 安装SAM2..."
    pip install -e .
    
    echo "📥 下载SAM2模型..."
    mkdir -p checkpoints
    cd checkpoints
    
    # 下载SAM2.1模型 (使用curl，Mac更兼容)
    if [ ! -f "sam2.1_hiera_base_plus.pt" ]; then
        echo "⬇️  下载SAM2.1 Base Plus模型..."
        curl -L -o sam2.1_hiera_base_plus.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    fi
    
    # 可选：下载大模型（如果网络和存储允许）
    echo "是否下载Large模型？(需要额外4GB空间) (y/N)"
    read -p "请输入选择: " choice
    if [[ $choice =~ ^[Yy]$ ]] && [ ! -f "sam2.1_hiera_large.pt" ]; then
        echo "⬇️  下载SAM2.1 Large模型..."
        curl -L -o sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    fi
    
    cd ../..
fi

echo "✅ SAM2安装完成"

# 创建Mac版启动脚本
echo "📝 创建Mac版启动脚本..."
cat > start_ui_mac.sh << 'EOF'
#!/bin/bash
# DATASAM2GET项目Mac版启动脚本

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate datasam2get

echo "🎯 DATASAM2GET项目启动 (Mac版本)"
echo "选择要启动的应用:"
echo "1. 数据标注工具 (data_generation)"
echo "2. 模型训练工具 (training_models)"
echo "3. YOLO-SAM2演示 (demo_apps)"
echo "4. 查看系统状态"

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "🏷️  启动数据标注工具..."
        cd data_generation
        streamlit run streamlit_app_enhanced.py --server.port 8501
        ;;
    2)
        echo "🎓 启动模型训练..."
        cd training_models
        python run_training.py
        ;;
    3)
        echo "🎮 启动YOLO-SAM2演示..."
        cd demo_apps
        python run_yolo_sam2_ui.py
        ;;
    4)
        echo "🖥️  系统状态信息:"
        python -c "
import torch, platform
print(f'系统: {platform.system()} {platform.release()}')
print(f'处理器: {platform.processor()}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CPU线程数: {torch.get_num_threads()}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS加速可用 (Apple Silicon)')
else:
    print('ℹ️  使用CPU处理')
"
        ;;
    *)
        echo "❌ 无效选择"
        ;;
esac
EOF

chmod +x start_ui_mac.sh

# 安装验证
echo "🔍 验证安装..."
python -c "
import torch
import ultralytics
import streamlit
import cv2
import numpy as np
from PIL import Image
print('✅ 所有核心包导入成功!')
print(f'PyTorch版本: {torch.__version__}')
print(f'Ultralytics版本: {ultralytics.__version__}')
print(f'OpenCV版本: {cv2.__version__}')
print(f'Streamlit版本: {streamlit.__version__}')

# 检查Apple Silicon加速
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('🚀 Apple Silicon MPS加速可用')
else:
    print('💻 使用CPU处理 (正常)')
"

echo ""
echo "🎉 Mac版本安装完成!"
echo ""
echo "📋 使用说明:"
echo "1. 激活环境: conda activate datasam2get"
echo "2. 运行Mac版启动脚本: ./start_ui_mac.sh"
echo "3. 或者手动启动各个模块:"
echo "   - 数据标注: cd data_generation && streamlit run streamlit_app_enhanced.py --server.port 8501"
echo "   - 模型训练: cd training_models && python run_training.py"
echo "   - 演示应用: cd demo_apps && python run_yolo_sam2_ui.py"
echo ""
echo "⚠️  Mac版本注意事项:"
echo "   - 使用CPU处理，速度较GPU版本慢"
echo "   - Apple Silicon Mac支持MPS加速"
echo "   - 建议处理小规模数据或降低分辨率"
echo ""
echo "🔧 配置文件:"
echo "   - Mac环境: environment_mac.yml"
echo "   - Mac依赖: requirements_mac.txt"
echo "   - SAM2路径: ./sam2/"
echo ""
echo "�� 更多信息请查看 README.md" 