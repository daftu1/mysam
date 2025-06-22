# 🎯 YOLO11辣椒检测模型训练指南

## 📋 目录
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [开始训练](#开始训练)
- [训练监控](#训练监控)
- [模型评估](#模型评估)
- [模型部署](#模型部署)
- [常见问题](#常见问题)

## 🔧 环境准备

### 1. 安装依赖包
```bash
# 安装基础依赖
pip install -r requirements.txt

# 或者手动安装主要包
pip install ultralytics torch torchvision opencv-python PyYAML
```

### 2. 检查GPU环境（可选但推荐）
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

## 📁 数据准备

### 当前数据状态
- ✅ 图像文件: `frame_cache_4f2d1b14/` (90张辣椒图片)
- ✅ 标注文件: `yolo11_labels_4f2d1b14/` (YOLO格式标注)
- ✅ 类别文件: `yolo11_labels_4f2d1b14/classes.txt` (包含"lajiao"类别)

### 数据格式说明
```
yolo11_labels_4f2d1b14/
├── classes.txt          # 类别名称文件
├── 0000.txt             # 对应0000.jpg的标注
├── 0001.txt             # 对应0001.jpg的标注
└── ...

frame_cache_4f2d1b14/
├── 0000.jpg             # 图像文件
├── 0001.jpg             # 图像文件
└── ...
```

### 标注格式
每个.txt文件包含YOLO格式的标注：
```
class_id center_x center_y width height
0 0.515234 0.662500 0.189844 0.302778
```

## 🚀 开始训练

### 方法1: 一键训练（推荐）
```bash
python run_training.py
```

这个脚本会自动执行：
1. 检查依赖包
2. 验证数据文件
3. 准备YOLO数据集格式
4. 开始模型训练
5. 导出ONNX模型
6. 测试模型效果

### 方法2: 分步执行

#### 步骤1: 准备数据集
```bash
python prepare_dataset.py
```

#### 步骤2: 开始训练
```bash
python train_yolo11.py
```

### 方法3: 自定义训练参数
```python
from train_yolo11 import train_yolo11

# 自定义配置
config = {
    "data_config": "yolo_dataset/data.yaml",
    "model_size": "yolo11s",  # n, s, m, l, x
    "epochs": 200,
    "batch_size": 16,
    "img_size": 640,
    "device": "cuda",  # 或 "cpu"
    "project": "runs/detect",
    "name": "my_lajiao_model"
}

best_model, results = train_yolo11(**config)
```

## 📊 训练监控

### 1. 实时监控
训练过程中会显示：
- 当前轮数/总轮数
- 训练损失 (box_loss, cls_loss, dfl_loss)
- 验证指标 (mAP50, mAP50-95)
- 学习率变化
- GPU/CPU使用情况

### 2. TensorBoard监控
```bash
# 启动TensorBoard
tensorboard --logdir runs/detect

# 在浏览器中访问
http://localhost:6006
```

### 3. 训练结果文件
```
runs/detect/lajiao_detection_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt          # 最佳模型
│   ├── last.pt          # 最新模型
│   └── epoch_*.pt       # 定期保存的模型
├── results.png          # 训练曲线图
├── confusion_matrix.png # 混淆矩阵
├── val_batch*.jpg       # 验证结果可视化
└── args.yaml           # 训练参数
```

## 📈 模型评估

### 1. 验证指标说明
- **mAP50**: IoU阈值0.5时的平均精度
- **mAP50-95**: IoU阈值0.5-0.95的平均精度
- **Precision**: 精确率
- **Recall**: 召回率

### 2. 查看训练曲线
```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 加载训练结果
model = YOLO('runs/detect/lajiao_detection_*/weights/best.pt')
results = model.val()

print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
```

### 3. 测试模型
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/lajiao_detection_*/weights/best.pt')

# 预测单张图片
results = model.predict('test_image.jpg', save=True)

# 批量预测
results = model.predict('test_images/', save=True, conf=0.25)
```

## 🚀 模型部署

### 1. 导出ONNX格式
```python
from ultralytics import YOLO

model = YOLO('runs/detect/lajiao_detection_*/weights/best.pt')
model.export(format='onnx')  # 生成 .onnx 文件
```

### 2. 导出其他格式
```python
# TensorRT (需要安装TensorRT)
model.export(format='engine')

# CoreML (Mac部署)
model.export(format='coreml')

# TensorFlow Lite (移动端)
model.export(format='tflite')
```

### 3. 使用导出的模型
```python
import cv2
import numpy as np
from ultralytics import YOLO

# 使用ONNX模型
model = YOLO('runs/detect/lajiao_detection_*/weights/best.onnx')

# 预测
img = cv2.imread('test.jpg')
results = model(img)

# 绘制结果
annotated = results[0].plot()
cv2.imshow('Detection', annotated)
cv2.waitKey(0)
```

## ⚙️ 训练参数调优

### 1. 模型大小选择
- **yolo11n**: 最快，精度较低，适合快速测试
- **yolo11s**: 平衡速度和精度
- **yolo11m**: 中等大小，较好精度
- **yolo11l**: 大模型，高精度
- **yolo11x**: 最大模型，最高精度

### 2. 关键参数调整
```python
# 学习率相关
lr0=0.01,           # 初始学习率
lrf=0.01,           # 最终学习率比例
warmup_epochs=3.0,  # 预热轮数

# 数据增强
mosaic=1.0,         # 马赛克增强
mixup=0.0,          # 混合增强
fliplr=0.5,         # 左右翻转概率

# 训练策略
patience=50,        # 早停耐心值
batch_size=16,      # 批次大小
epochs=100,         # 训练轮数
```

### 3. 小数据集优化
```python
# 针对小数据集的配置
config = {
    "epochs": 200,           # 增加训练轮数
    "batch_size": 8,         # 减小批次大小
    "lr0": 0.001,           # 降低学习率
    "mosaic": 0.5,          # 减少数据增强
    "copy_paste": 0.1,      # 添加复制粘贴增强
    "mixup": 0.1,           # 添加混合增强
}
```

## ❓ 常见问题

### Q1: 训练时GPU内存不足
**解决方案:**
```python
# 减小批次大小
batch_size = 4  # 或更小

# 减小图像尺寸
img_size = 416  # 默认640

# 关闭图像缓存
cache = False
```

### Q2: 训练精度不高
**解决方案:**
1. 增加训练数据
2. 检查标注质量
3. 调整数据增强参数
4. 使用更大的模型
5. 增加训练轮数

### Q3: 训练过程中断
**解决方案:**
```python
# 恢复训练
model = YOLO('runs/detect/lajiao_detection_*/weights/last.pt')
model.train(resume=True)
```

### Q4: 验证精度波动大
**解决方案:**
1. 增加验证集大小
2. 使用更稳定的学习率调度
3. 增加早停耐心值

### Q5: 模型过拟合
**解决方案:**
```python
# 增加正则化
weight_decay = 0.001    # 权重衰减
dropout = 0.1          # Dropout
label_smoothing = 0.1  # 标签平滑
```

## 📞 技术支持

如果遇到问题，可以：
1. 查看训练日志文件
2. 检查数据格式是否正确
3. 验证环境配置
4. 参考YOLO官方文档: https://docs.ultralytics.com/

## 🎉 训练完成后的下一步

1. **模型评估**: 在测试集上评估模型性能
2. **错误分析**: 分析误检和漏检案例
3. **数据增强**: 根据错误分析结果补充训练数据
4. **模型优化**: 调整参数或尝试不同模型
5. **部署应用**: 将模型集成到实际应用中 