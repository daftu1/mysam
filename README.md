# 🎯 SAM2智能视频标注与训练系统

基于SAM2和YOLO11的完整视频目标检测与分割解决方案，从数据标注到模型训练再到应用部署的全流程工具链。

## 📁 项目结构

```
datasam2get/
├── 📊 data_generation/          # 数据生成工具
│   ├── streamlit_app_enhanced.py    # SAM2视频标注工具
│   ├── convert_unified_dataset.py  # 数据集转换工具
│   └── README.md                    # 数据生成说明
│
├── 🎓 training_models/          # 模型训练工具
│   ├── prepare_dataset.py          # 数据集准备
│   ├── train_yolo11.py             # YOLO11训练脚本
│   ├── run_training.py             # 一键训练启动器
│   ├── test_yolo.py                # 模型测试评估
│   ├── requirements.txt            # 训练依赖
│   ├── TRAINING_GUIDE.md           # 训练指南
│   └── README.md                   # 训练工具说明
│
├── 🎮 demo_apps/               # 演示应用
│   ├── yolo_sam2_ui.py            # YOLO-SAM2 Web UI
│   ├── run_yolo_sam2_ui.py        # UI启动脚本
│   ├── yolo_sam2_demo.py          # 命令行演示
│   ├── test_yolo_sam2.py          # 系统测试
│   ├── YOLO_SAM2_README.md        # 详细使用说明
│   ├── YOLO_SAM2_SUMMARY.md       # 系统总结
│   └── README.md                  # 演示应用说明
│
├── 📈 runs/                    # 训练输出结果
└── 📚 README.md               # 本文件
```

## 🚀 快速开始

### 1️⃣ 数据标注 (data_generation/)
使用SAM2进行智能视频标注，生成高质量训练数据
```bash
cd data_generation
streamlit run streamlit_app_enhanced.py --server.port 8501
```

### 2️⃣ 模型训练 (training_models/)
基于标注数据训练YOLO11检测模型
```bash
cd training_models
python run_training.py
```

### 3️⃣ 应用演示 (demo_apps/)
使用训练好的模型进行YOLO-SAM2视频分割
```bash
cd demo_apps
python run_yolo_sam2_ui.py
```

## 🔄 完整工作流程

### 第一阶段：数据标注
1. **视频上传** - 上传需要标注的视频
2. **智能标注** - 在关键帧点击添加正/负点
3. **SAM2分割** - 自动生成精确分割掩码
4. **批量传播** - 传播到100帧获得完整标注
5. **数据导出** - 生成YOLO格式训练数据

### 第二阶段：模型训练
1. **数据准备** - 验证和预处理标注数据
2. **模型训练** - 使用YOLO11进行目标检测训练
3. **性能评估** - 评估模型精度和性能
4. **模型导出** - 保存最佳模型用于部署

### 第三阶段：应用部署
1. **模型加载** - 加载训练好的YOLO和SAM2模型
2. **视频处理** - YOLO检测 + SAM2精确分割
3. **结果可视化** - 实时预览分割效果
4. **批量处理** - 支持大规模视频处理

## ✨ 核心特性

### 🎯 智能标注
- **半自动标注**: 结合人工点击和SAM2智能分割
- **视频传播**: 单帧标注自动传播到整个序列
- **统一管理**: 多视频数据集统一管理
- **格式转换**: 自动转换为YOLO训练格式

### 🎓 高效训练
- **一键训练**: 自动化的训练流程
- **性能监控**: TensorBoard可视化训练过程
- **模型优化**: 支持多种YOLO11模型规格
- **评估工具**: 完整的模型性能评估

### 🎮 易用部署
- **Web界面**: 直观的Streamlit界面
- **命令行工具**: 支持批量自动化处理
- **实时预览**: 即时查看分割效果
- **多格式导出**: 掩码、视频、可视化

## 📊 技术规格

### 支持的模型
- **检测模型**: YOLO11 (n/s/m/l/x)
- **分割模型**: SAM2 (sam2.1_hiera_base_plus)
- **训练框架**: Ultralytics YOLO
- **界面框架**: Streamlit

### 硬件要求
- **GPU**: NVIDIA GPU (推荐，支持CUDA)
- **内存**: 8GB+ RAM
- **存储**: 10GB+ 可用空间
- **Python**: 3.8+

### 性能表现
基于辣椒检测模型的测试结果：
- **检测精度**: mAP50=99.5%, mAP50-95=99.5%
- **分割质量**: 像素级精确分割
- **处理速度**: GPU加速，实时处理
- **支持格式**: MP4, AVI, MOV, MKV等

## 🎨 应用场景

### 农业领域
- 🌶️ **作物检测**: 辣椒、果实等农作物检测分割
- 🌱 **生长监测**: 植物生长状态分析
- 🐛 **病虫害检测**: 叶片病害自动识别

### 工业领域
- 🔧 **质量检测**: 产品缺陷检测分割
- 📦 **包装检测**: 包装完整性检查
- 🏭 **生产监控**: 生产线自动化监控

### 医学领域
- 🏥 **医学影像**: 病灶区域精确分割
- 🔬 **细胞分析**: 显微镜图像分析
- 📊 **诊断辅助**: 医学图像辅助诊断

### 其他应用
- 🚗 **自动驾驶**: 道路场景分割
- 📹 **视频分析**: 目标跟踪和分析
- 🏷️ **数据标注**: 自动生成高质量标注

## 🔧 环境配置

### 基础依赖
```bash
pip install streamlit ultralytics torch opencv-python numpy pillow
```

### SAM2环境
确保SAM2已安装在 `/home/zcx/sam2`，包含：
- 模型文件: `checkpoints/sam2.1_hiera_base_plus.pt`
- 配置文件: `configs/sam2.1/sam2.1_hiera_b+.yaml`

### CUDA支持
- CUDA 11.8+
- cuDNN 8.0+
- PyTorch 2.0+ (GPU版本)

## 📈 成功案例

### 辣椒检测项目
- **数据集**: 1000+ 标注图像
- **训练时间**: 2小时 (RTX 4060)
- **最终精度**: mAP50=99.5%
- **应用场景**: 农业自动化检测

### 处理能力
- **标注效率**: 100帧/分钟 (SAM2加速)
- **训练速度**: 100 epochs/2小时
- **推理速度**: 30+ FPS (GPU)
- **内存使用**: <8GB GPU显存

## 🔮 未来规划

- **实时处理**: 支持摄像头实时分割
- **移动端**: 开发移动应用版本
- **云部署**: 支持云端服务部署
- **API服务**: 提供REST API接口
- **多模态**: 支持更多数据类型

## 📞 技术支持

- 🐛 **问题反馈**: 创建GitHub Issue
- 💡 **功能建议**: 提交Feature Request
- 📖 **文档更新**: 提交Pull Request
- 🤝 **技术交流**: 欢迎技术讨论

## 📄 许可证

本项目基于MIT许可证开源，详见LICENSE文件。

---

## 🎉 立即开始

选择你需要的功能模块：

### 🔥 推荐路径
1. **新手用户** → `demo_apps/` → 体验YOLO-SAM2分割效果
2. **数据标注** → `data_generation/` → 制作训练数据
3. **模型训练** → `training_models/` → 训练自己的模型

### 🚀 快速体验
```bash
# 体验YOLO-SAM2分割
cd demo_apps && python run_yolo_sam2_ui.py

# 制作标注数据
cd data_generation && streamlit run streamlit_app_enhanced.py --server.port 8501

# 训练YOLO模型
cd training_models && python run_training.py
```

**🎯 开始你的AI视觉之旅吧！**
