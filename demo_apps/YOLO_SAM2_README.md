# 🎯 YOLO-SAM2 视频分割系统

结合YOLO11目标检测和SAM2精确分割的完整视频处理解决方案。

## ✨ 特色功能

- 🎯 **YOLO11检测**: 快速准确的目标检测
- 🎨 **SAM2分割**: 像素级精确分割
- 🎬 **视频处理**: 支持视频序列分割
- 📊 **实时预览**: 即时查看处理结果
- 💾 **多种导出**: 掩码、视频等格式
- 🖥️ **双界面**: Web UI + 命令行

## 🔄 工作流程

1. **YOLO检测** - 在参考帧检测目标物体
2. **中心点提取** - 自动提取检测框的中心点
3. **SAM2分割** - 基于中心点进行精确分割
4. **视频传播** - 将分割传播到整个视频序列
5. **结果导出** - 保存掩码和可视化结果

## 📋 系统要求

### 硬件要求
- GPU: NVIDIA GPU (推荐，支持CUDA)
- 内存: 8GB+ RAM
- 存储: 10GB+ 可用空间

### 软件依赖
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU版本)

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖包
pip install streamlit ultralytics torch opencv-python numpy pillow

# 确保SAM2已安装在 /home/zcx/sam2
# 确保模型文件存在:
# - /home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt
# - runs/detect/lajiao_detection_20250623_053550/weights/best.pt (可选)
```

### 2. 启动Web UI

```bash
# 方法1: 使用启动脚本 (推荐)
python run_yolo_sam2_ui.py

# 方法2: 直接启动Streamlit
streamlit run yolo_sam2_ui.py --server.port 8506
```

浏览器访问: http://localhost:8506

### 3. 命令行使用

```bash
# 基本用法
python yolo_sam2_demo.py --video your_video.mp4

# 完整参数
python yolo_sam2_demo.py \
    --video your_video.mp4 \
    --yolo runs/detect/lajiao_detection_20250623_053550/weights/best.pt \
    --sam2 /home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
    --conf 0.25 \
    --ref-frame 10 \
    --max-frames 100 \
    --output results
```

## 🖥️ Web UI 使用指南

### 侧边栏配置
- **YOLO模型**: 选择检测模型或使用训练好的模型
- **置信度阈值**: 调整检测敏感度 (0.1-1.0)
- **显示选项**: 控制检测框、掩码、中心点的显示

### 主界面操作
1. **上传视频** - 支持MP4、AVI、MOV等格式
2. **查看视频信息** - 显示帧数、帧率、时长
3. **开始处理** - 点击"开始YOLO-SAM2分割"
4. **选择参考帧** - 用滑块选择检测目标的帧
5. **查看检测结果** - 显示检测到的目标和置信度
6. **开始分割** - 点击"开始SAM2视频分割"
7. **预览结果** - 使用滑块浏览分割结果
8. **导出结果** - 保存掩码或生成视频

## 📊 输出格式

### 掩码文件
- 格式: PNG (黑白二值图像)
- 命名: `frame_XXXX_obj_Y.png`
- 位置: `output_dir/masks/`

### 可视化图像
- 格式: JPG (彩色叠加图像)
- 命名: `frame_XXXX.jpg`
- 位置: `output_dir/visualizations/`

### 分割视频
- 格式: MP4
- 内容: 带有分割掩码和检测框的视频
- 位置: `output_dir/segmented_video.mp4`

## ⚙️ 参数说明

### YOLO参数
- `--yolo`: YOLO模型路径
- `--conf`: 置信度阈值 (0.1-1.0)

### SAM2参数
- `--sam2`: SAM2模型路径
- `--ref-frame`: 参考帧索引 (用于初始检测)

### 处理参数
- `--max-frames`: 最大处理帧数
- `--output`: 输出目录

## 🔧 故障排除

### 常见问题

**1. 模型加载失败**
```
❌ SAM2模型未找到: /home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt
```
**解决**: 确保SAM2模型文件存在，或修改模型路径

**2. CUDA内存不足**
```
RuntimeError: CUDA out of memory
```
**解决**: 
- 减少`max_frames`参数
- 使用CPU版本: 在代码中设置`device = torch.device("cpu")`
- 关闭其他GPU程序

**3. 未检测到目标**
```
⚠️ 未检测到任何目标
```
**解决**:
- 降低置信度阈值 (如0.1)
- 选择不同的参考帧
- 检查YOLO模型是否适合目标类型

**4. 分割效果不佳**
```
分割掩码不准确
```
**解决**:
- 选择目标更清晰的参考帧
- 确保检测框准确包含目标
- 调整YOLO检测参数

### 性能优化

**GPU加速**
```python
# 确保CUDA可用
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```

**内存优化**
```python
# 在处理大视频时添加内存清理
torch.cuda.empty_cache()
```

## 📁 文件结构

```
yolo_sam2_system/
├── yolo_sam2_ui.py          # Streamlit Web UI
├── run_yolo_sam2_ui.py      # UI启动脚本
├── yolo_sam2_demo.py        # 命令行演示脚本
├── YOLO_SAM2_README.md      # 使用说明
├── temp_video_*.mp4         # 临时视频文件
├── frames_*/                # 临时帧目录
├── yolo_sam2_output_*/      # 输出结果目录
│   ├── masks/               # 分割掩码
│   ├── visualizations/      # 可视化图像
│   └── segmented_video.mp4  # 分割视频
└── runs/                    # YOLO训练结果
    └── detect/
        └── lajiao_detection_*/
            └── weights/
                └── best.pt  # 训练好的模型
```

## 🎯 应用场景

- **农业监测**: 作物分割和计数
- **工业检测**: 产品缺陷分割
- **医学影像**: 病灶区域分割
- **安防监控**: 目标跟踪和分割
- **体育分析**: 运动员动作分析

## 🔮 扩展功能

### 自定义YOLO模型
```python
# 训练自己的YOLO模型
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(data='your_dataset.yaml', epochs=100)
```

### 批量处理
```bash
# 处理多个视频
for video in *.mp4; do
    python yolo_sam2_demo.py --video "$video" --output "results_$(basename "$video" .mp4)"
done
```

### API集成
```python
# 集成到其他项目
from yolo_sam2_ui import load_yolo_model, load_sam2_model, yolo_detect, sam2_segment_video

yolo_model = load_yolo_model("your_model.pt")
sam2_model = load_sam2_model()
# ... 使用模型进行处理
```

## 📞 技术支持

- 🐛 **问题反馈**: 创建GitHub Issue
- 💡 **功能建议**: 提交Feature Request
- 📖 **文档更新**: 提交Pull Request

## 📄 许可证

本项目基于MIT许可证开源，详见LICENSE文件。

---

**🎉 享受YOLO-SAM2带来的精确视频分割体验！** 