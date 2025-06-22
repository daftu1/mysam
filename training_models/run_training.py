#!/usr/bin/env python3
"""
一键训练脚本
自动执行数据准备 -> 模型训练 -> 模型测试的完整流程
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'ultralytics',
        'torch',
        'torchvision', 
        'yaml',
        'numpy',
        'opencv-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_data():
    """检查数据文件"""
    print("\n🔍 检查数据文件...")
    
    # 检查图像目录
    if not os.path.exists("frame_cache_4f2d1b14"):
        print("❌ 图像目录不存在: frame_cache_4f2d1b14")
        return False
    
    # 检查标注目录
    if not os.path.exists("yolo11_labels_4f2d1b14"):
        print("❌ 标注目录不存在: yolo11_labels_4f2d1b14")
        return False
    
    # 检查类别文件
    if not os.path.exists("yolo11_labels_4f2d1b14/classes.txt"):
        print("❌ 类别文件不存在: yolo11_labels_4f2d1b14/classes.txt")
        return False
    
    # 统计文件数量
    image_files = list(Path("frame_cache_4f2d1b14").glob("*.jpg"))
    label_files = list(Path("yolo11_labels_4f2d1b14").glob("*.txt"))
    label_files = [f for f in label_files if f.name != "classes.txt"]
    
    print(f"   📷 图像文件: {len(image_files)} 个")
    print(f"   🏷️ 标注文件: {len(label_files)} 个")
    
    if len(image_files) == 0:
        print("❌ 没有找到图像文件")
        return False
    
    if len(label_files) == 0:
        print("❌ 没有找到标注文件")
        return False
    
    # 检查文件对应关系
    valid_pairs = 0
    for img_file in image_files:
        label_file = Path("yolo11_labels_4f2d1b14") / f"{img_file.stem}.txt"
        if label_file.exists() and label_file.stat().st_size > 0:
            valid_pairs += 1
    
    print(f"   ✅ 有效的图像-标注对: {valid_pairs} 个")
    
    if valid_pairs < 10:
        print("⚠️ 有效数据对数量较少，可能影响训练效果")
    
    return True

def prepare_dataset():
    """准备数据集"""
    print("\n📁 准备数据集...")
    
    try:
        from prepare_dataset import prepare_yolo_dataset
        dataset_dir, config_file = prepare_yolo_dataset()
        print(f"✅ 数据集准备完成: {config_file}")
        return True
    except Exception as e:
        print(f"❌ 数据集准备失败: {e}")
        return False

def train_model():
    """训练模型"""
    print("\n🚀 开始训练模型...")
    
    try:
        from train_yolo11 import train_yolo11, export_model, test_model
        from datetime import datetime
        
        # 训练配置
        config = {
            "data_config": "yolo_dataset/data.yaml",
            "model_size": "yolo11n",  # 使用轻量级模型
            "epochs": 50,  # 减少训练轮数用于快速测试
            "batch_size": 8,  # 减少批次大小以适应小数据集
            "img_size": 640,
            "device": "auto",
            "project": "runs/detect",
            "name": f"lajiao_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "resume": False,
            "save_period": 10
        }
        
        # 开始训练
        best_model, results = train_yolo11(**config)
        
        # 导出模型
        print("\n📦 导出模型...")
        export_model(best_model, ['onnx'])
        
        # 测试模型
        print("\n🧪 测试模型...")
        test_model(best_model, "frame_cache_4f2d1b14", "test_results")
        
        print(f"\n🎉 训练完成！最佳模型: {best_model}")
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🎯 YOLO11辣椒检测模型一键训练")
    print("=" * 60)
    
    # 1. 检查依赖
    if not check_dependencies():
        return False
    
    # 2. 检查数据
    if not check_data():
        return False
    
    # 3. 准备数据集
    if not prepare_dataset():
        return False
    
    # 4. 训练模型
    if not train_model():
        return False
    
    print("\n" + "=" * 60)
    print("🎉 所有步骤完成！")
    print("\n📁 生成的文件:")
    print("   📊 数据集: yolo_dataset/")
    print("   🏆 训练结果: runs/detect/")
    print("   🧪 测试结果: test_results/")
    print("\n💡 使用建议:")
    print("   1. 查看训练曲线: runs/detect/*/results.png")
    print("   2. 查看验证结果: runs/detect/*/val_batch*.jpg")
    print("   3. 查看测试结果: test_results/predictions/")
    print("   4. 最佳模型位置: runs/detect/*/weights/best.pt")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ 训练流程完成！")
        else:
            print("\n❌ 训练流程失败！")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 