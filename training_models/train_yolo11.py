#!/usr/bin/env python3
"""
YOLO11训练脚本
基于SAM2生成的标注数据训练YOLO11目标检测模型
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
from datetime import datetime

def train_yolo11(
    data_config="yolo_dataset/data.yaml",
    model_size="yolo11n",  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    img_size=640,
    device="auto",
    project="runs/detect",
    name="lajiao_detection",
    resume=False,
    save_period=10
):
    """
    训练YOLO11模型
    
    Args:
        data_config: 数据集配置文件路径
        model_size: 模型大小 (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图像大小
        device: 训练设备 (auto, cpu, cuda, 0, 1, 2, ...)
        project: 项目保存目录
        name: 实验名称
        resume: 是否恢复训练
        save_period: 保存模型的间隔轮数
    """
    
    print("🚀 开始YOLO11训练")
    print("=" * 50)
    
    # 检查数据配置文件
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"数据配置文件不存在: {data_config}")
    
    # 读取数据配置
    with open(data_config, 'r', encoding='utf-8') as f:
        data_info = yaml.safe_load(f)
    
    print(f"📊 数据集信息:")
    print(f"   路径: {data_info['path']}")
    print(f"   类别数: {data_info['nc']}")
    print(f"   类别: {data_info['names']}")
    
    # 检查设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔧 训练配置:")
    print(f"   模型: {model_size}")
    print(f"   设备: {device}")
    print(f"   轮数: {epochs}")
    print(f"   批次大小: {batch_size}")
    print(f"   图像大小: {img_size}")
    
    # 创建模型
    model = YOLO(f"{model_size}.pt")  # 加载预训练模型
    
    # 开始训练
    print("\n🎯 开始训练...")
    
    try:
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project,
            name=name,
            resume=resume,
            save_period=save_period,
            # 训练优化参数
            patience=50,          # 早停耐心值
            save=True,           # 保存训练检查点
            cache=False,         # 不缓存图像到内存
            # 数据增强参数
            hsv_h=0.015,         # 色调增强
            hsv_s=0.7,           # 饱和度增强
            hsv_v=0.4,           # 明度增强
            degrees=0.0,         # 旋转角度
            translate=0.1,       # 平移
            scale=0.5,           # 缩放
            shear=0.0,           # 剪切
            perspective=0.0,     # 透视变换
            flipud=0.0,          # 上下翻转
            fliplr=0.5,          # 左右翻转
            mosaic=1.0,          # 马赛克增强
            mixup=0.0,           # 混合增强
            copy_paste=0.0,      # 复制粘贴增强
            # 优化器参数
            optimizer='auto',    # 优化器 (SGD, Adam, AdamW, auto)
            lr0=0.01,           # 初始学习率
            lrf=0.01,           # 最终学习率 (lr0 * lrf)
            momentum=0.937,      # SGD动量/Adam beta1
            weight_decay=0.0005, # 权重衰减
            warmup_epochs=3.0,   # 预热轮数
            warmup_momentum=0.8, # 预热动量
            warmup_bias_lr=0.1,  # 预热偏置学习率
            # 验证参数
            val=True,           # 验证
            plots=True,         # 保存训练图表
            # 其他参数
            verbose=True,       # 详细输出
            seed=0,             # 随机种子
            deterministic=True, # 确定性训练
            single_cls=False,   # 单类训练
            rect=False,         # 矩形训练
            cos_lr=False,       # 余弦学习率调度
            close_mosaic=10,    # 关闭马赛克增强的轮数
            amp=True,           # 自动混合精度
            fraction=1.0,       # 使用数据集的比例
            profile=False,      # 性能分析
            freeze=None,        # 冻结层数
            # 多尺度训练
            multi_scale=False,  # 多尺度训练
            overlap_mask=True,  # 重叠掩码
            mask_ratio=4,       # 掩码比例
            dropout=0.0,        # 分类器dropout
            # 损失函数权重
            box=7.5,            # 框损失权重
            cls=0.5,            # 分类损失权重
            dfl=1.5,            # DFL损失权重
            pose=12.0,          # 姿态损失权重
            kobj=2.0,           # 关键点obj损失权重
            label_smoothing=0.0, # 标签平滑
            nbs=64,             # 标准批次大小
            # 数据加载
            workers=8,          # 数据加载器工作进程数
        )
        
        print("\n✅ 训练完成！")
        
        # 获取最佳模型路径
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        last_model_path = results.save_dir / 'weights' / 'last.pt'
        
        print(f"📁 模型保存位置:")
        print(f"   最佳模型: {best_model_path}")
        print(f"   最新模型: {last_model_path}")
        print(f"   训练日志: {results.save_dir}")
        
        # 验证模型
        print("\n🔍 开始模型验证...")
        val_results = model.val(data=data_config, device=device)
        
        print(f"\n📈 验证结果:")
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")
        
        return str(best_model_path), results
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {str(e)}")
        raise

def export_model(model_path, export_formats=['onnx', 'engine']):
    """
    导出训练好的模型到不同格式
    
    Args:
        model_path: 模型路径
        export_formats: 导出格式列表
    """
    print(f"\n📦 导出模型: {model_path}")
    
    model = YOLO(model_path)
    
    for fmt in export_formats:
        try:
            print(f"   导出 {fmt.upper()} 格式...")
            model.export(format=fmt)
            print(f"   ✅ {fmt.upper()} 导出成功")
        except Exception as e:
            print(f"   ❌ {fmt.upper()} 导出失败: {e}")

def test_model(model_path, test_image_dir, save_dir="test_results"):
    """
    测试训练好的模型
    
    Args:
        model_path: 模型路径
        test_image_dir: 测试图像目录
        save_dir: 结果保存目录
    """
    print(f"\n🧪 测试模型: {model_path}")
    
    model = YOLO(model_path)
    
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取测试图像
    test_images = list(Path(test_image_dir).glob("*.jpg"))
    
    if not test_images:
        print("❌ 未找到测试图像")
        return
    
    print(f"🖼️ 找到 {len(test_images)} 张测试图像")
    
    # 批量预测
    results = model.predict(
        source=test_image_dir,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name="predictions",
        conf=0.25,
        iou=0.45,
        show_labels=True,
        show_conf=True,
        line_width=2
    )
    
    print(f"✅ 测试完成，结果保存到: {save_dir}/predictions")

if __name__ == "__main__":
    # 训练参数
    config = {
        "data_config": "yolo_dataset/data.yaml",
        "model_size": "yolo11n",  # 轻量级模型，适合快速训练
        "epochs": 100,
        "batch_size": 16,
        "img_size": 640,
        "device": "auto",
        "project": "runs/detect",
        "name": f"lajiao_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "resume": False,
        "save_period": 10
    }
    
    print("🎯 YOLO11辣椒检测模型训练")
    print("=" * 50)
    
    # 检查数据集是否存在
    if not os.path.exists(config["data_config"]):
        print("❌ 数据集配置文件不存在，请先运行 prepare_dataset.py")
        exit(1)
    
    try:
        # 开始训练
        best_model, results = train_yolo11(**config)
        
        # 导出模型
        print("\n" + "=" * 50)
        export_model(best_model, ['onnx'])
        
        # 测试模型
        if os.path.exists("frame_cache_4f2d1b14"):
            print("\n" + "=" * 50)
            test_model(best_model, "frame_cache_4f2d1b14", "test_results")
        
        print("\n🎉 所有步骤完成！")
        print(f"🏆 最佳模型: {best_model}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc() 