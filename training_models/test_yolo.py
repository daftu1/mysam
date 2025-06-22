#!/usr/bin/env python3
"""
简单的YOLO11测试脚本
验证环境和基本功能
"""

from ultralytics import YOLO
import torch

def test_yolo_environment():
    """测试YOLO11环境"""
    print("🔍 测试YOLO11环境...")
    
    # 检查PyTorch和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 测试YOLO11模型加载
    try:
        print("\n📦 测试YOLO11模型加载...")
        model = YOLO('yolo11n.pt')  # 下载并加载预训练模型
        print("✅ YOLO11n模型加载成功")
        
        # 测试模型信息
        print(f"模型参数数量: {sum(p.numel() for p in model.model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"❌ YOLO11模型加载失败: {e}")
        return False

def test_data_format():
    """测试数据格式"""
    print("\n📊 测试数据格式...")
    
    import os
    from pathlib import Path
    
    # 检查数据集配置
    config_file = "yolo_dataset/data.yaml"
    if os.path.exists(config_file):
        print(f"✅ 数据配置文件存在: {config_file}")
        
        # 检查训练和验证目录
        train_dir = Path("yolo_dataset/train/images")
        val_dir = Path("yolo_dataset/val/images")
        
        if train_dir.exists():
            train_images = list(train_dir.glob("*.jpg"))
            print(f"✅ 训练图像: {len(train_images)} 张")
        else:
            print("❌ 训练图像目录不存在")
            
        if val_dir.exists():
            val_images = list(val_dir.glob("*.jpg"))
            print(f"✅ 验证图像: {len(val_images)} 张")
        else:
            print("❌ 验证图像目录不存在")
            
        return True
    else:
        print(f"❌ 数据配置文件不存在: {config_file}")
        return False

def quick_train_test():
    """快速训练测试"""
    print("\n🚀 快速训练测试...")
    
    try:
        model = YOLO('yolo11n.pt')
        
        # 进行1个epoch的训练测试
        results = model.train(
            data='yolo_dataset/data.yaml',
            epochs=1,
            batch=4,
            imgsz=640,
            device='auto',
            project='test_runs',
            name='quick_test',
            verbose=True
        )
        
        print("✅ 快速训练测试成功")
        print(f"训练结果保存在: {results.save_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 快速训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 YOLO11环境测试")
    print("=" * 50)
    
    # 测试环境
    env_ok = test_yolo_environment()
    
    # 测试数据格式
    data_ok = test_data_format()
    
    # 如果环境和数据都OK，进行快速训练测试
    if env_ok and data_ok:
        train_ok = quick_train_test()
        
        if train_ok:
            print("\n🎉 所有测试通过！可以开始正式训练")
        else:
            print("\n⚠️ 训练测试失败，请检查配置")
    else:
        print("\n❌ 环境或数据测试失败")
    
    print("\n" + "=" * 50) 