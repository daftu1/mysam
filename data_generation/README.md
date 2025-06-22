# 📊 数据生成工具

此文件夹包含用于生成YOLO训练数据的工具和脚本。

## 📁 文件说明

### 核心工具
- **`streamlit_app_enhanced.py`** - SAM2视频标注工具（增强版）
  - 支持视频上传和帧提取
  - 智能点击标注和SAM2分割
  - 统一数据集管理
  - 自动生成YOLO格式标注

- **`convert_unified_dataset.py`** - 统一数据集转换工具
  - 将统一数据集转换为YOLO训练格式
  - 支持数据集合并
  - 自动划分训练/验证集

## 🚀 使用方法

### 1. 启动标注工具
```bash
cd data_generation
streamlit run streamlit_app_enhanced.py --server.port 8501
```

### 2. 视频标注流程
1. 上传视频文件
2. 选择起始和结束帧
3. 在关键帧上点击添加正/负点
4. 使用SAM2生成100帧掩码
5. 预览和修正分割结果
6. 输入标签并保存到统一数据集

### 3. 数据集转换
```bash
cd data_generation
python convert_unified_dataset.py
```

## 📊 输出数据格式

### 统一数据集结构
```
frames_{dataset_name}/          # 所有图像文件
├── video1_0001.jpg
├── video1_0002.jpg
├── video2_0001.jpg
└── ...

labels_{dataset_name}/          # 所有标注文件
├── video1_0001.txt
├── video1_0002.txt
├── video2_0001.txt
└── ...
```

### YOLO训练格式
```
yolo_dataset_{dataset_name}/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

## ✨ 特色功能

- 🎯 **智能标注**: 基于SAM2的半自动标注
- 🎬 **视频支持**: 批量处理视频帧
- 📱 **统一管理**: 多视频数据集统一管理
- 🔄 **格式转换**: 自动转换为YOLO格式
- 💾 **增量标注**: 支持数据集累积和扩展

## 🎨 应用场景

- 农业目标检测数据集制作
- 工业缺陷检测标注
- 医学影像分割标注
- 自动驾驶场景标注
- 任何需要精确分割标注的场景

---
**💡 提示**: 使用统一数据集管理可以避免数据分散，便于后续训练和管理。 