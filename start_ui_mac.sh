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
