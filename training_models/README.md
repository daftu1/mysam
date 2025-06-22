# 🎓 模型训练工具

此文件夹包含YOLO11模型训练的完整工具链和脚本。

## 📁 文件说明

### 训练脚本
- **`prepare_dataset.py`** - 数据集准备工具
  - 数据集验证和预处理
  - 自动生成dataset.yaml配置
  - 数据集统计和分析

- **`train_yolo11.py`** - YOLO11训练脚本
  - 支持多种YOLO11模型（n/s/m/l/x）
  - 自动超参数优化
  - 训练过程监控

- **`run_training.py`** - 一键训练启动器
  - 整合数据准备和模型训练
  - 自动配置训练参数
  - 训练日志记录

- **`test_yolo.py`** - 模型测试和评估
  - 模型性能评估
  - 推理速度测试
  - 结果可视化

### 配置文件
- **`requirements.txt`** - Python依赖包列表
- **`TRAINING_GUIDE.md`** - 详细训练指南

## 🚀 使用方法

### 1. 环境准备
```bash
cd training_models
pip install -r requirements.txt
```

### 2. 数据集准备
```bash
python prepare_dataset.py --dataset_path ../data_generation/yolo_dataset_your_name
```

### 3. 开始训练
```bash
# 方法1: 使用一键训练脚本
python run_training.py

# 方法2: 直接训练
python train_yolo11.py --data dataset.yaml --epochs 100 --imgsz 640
```

### 4. 模型测试
```bash
python test_yolo.py --model runs/detect/train/weights/best.pt --source test_images/
```

## 📊 训练配置

### 支持的模型
- **YOLO11n** - 轻量级模型，快速推理
- **YOLO11s** - 小型模型，平衡性能
- **YOLO11m** - 中型模型，较高精度
- **YOLO11l** - 大型模型，高精度
- **YOLO11x** - 超大模型，最高精度

### 训练参数
```python
# 默认训练配置
epochs = 100           # 训练轮数
batch_size = 16        # 批次大小
imgsz = 640           # 输入图像尺寸
lr0 = 0.01            # 初始学习率
patience = 50         # 早停耐心值
save_period = 10      # 模型保存间隔
```

## 📈 训练监控

### TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir runs/detect/train
```

### 训练指标
- **mAP50** - IoU=0.5时的平均精度
- **mAP50-95** - IoU=0.5:0.95的平均精度
- **Precision** - 精确率
- **Recall** - 召回率
- **Loss** - 训练损失

## 🎯 输出结果

### 训练输出
```
runs/detect/train_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt          # 最佳模型
│   ├── last.pt          # 最后一轮模型
│   └── best.onnx        # ONNX格式模型
├── results.png          # 训练曲线图
├── confusion_matrix.png # 混淆矩阵
├── val_batch0_pred.jpg  # 验证预测结果
└── args.yaml           # 训练参数
```

### 模型性能示例
基于辣椒检测训练结果：
- **mAP50**: 99.5%
- **mAP50-95**: 99.5%
- **Precision**: 99.84%
- **Recall**: 100%

## ⚙️ 高级配置

### 自定义数据增强
```python
# 在train_yolo11.py中修改
augment_params = {
    'hsv_h': 0.015,      # 色调增强
    'hsv_s': 0.7,        # 饱和度增强
    'hsv_v': 0.4,        # 明度增强
    'degrees': 0.0,      # 旋转角度
    'translate': 0.1,    # 平移
    'scale': 0.5,        # 缩放
    'shear': 0.0,        # 剪切
    'perspective': 0.0,  # 透视变换
    'flipud': 0.0,       # 垂直翻转
    'fliplr': 0.5,       # 水平翻转
    'mosaic': 1.0,       # 马赛克增强
    'mixup': 0.0,        # 混合增强
}
```

### 多GPU训练
```bash
# 使用多GPU训练
python -m torch.distributed.run --nproc_per_node=2 train_yolo11.py --data dataset.yaml
```

## 🔧 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size
2. **训练不收敛**: 调整学习率和数据增强
3. **过拟合**: 增加数据或使用正则化
4. **欠拟合**: 增加模型复杂度或训练轮数

### 性能优化
- 使用混合精度训练 (`amp=True`)
- 启用编译优化 (`compile=True`)
- 调整工作进程数 (`workers=8`)

## 📚 相关资源

- [YOLO11官方文档](https://docs.ultralytics.com/)
- [训练最佳实践](https://docs.ultralytics.com/guides/training-tips/)
- [模型部署指南](https://docs.ultralytics.com/guides/deployment/)

---
**🎯 提示**: 训练前请确保数据集质量，良好的数据是模型成功的关键！ 