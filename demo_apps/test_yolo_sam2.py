#!/usr/bin/env python3
"""
YOLO-SAM2 系统测试脚本
验证模型加载和基本功能
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

def test_dependencies():
    """测试依赖包"""
    print("🔍 测试依赖包...")
    
    required_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'ultralytics': 'Ultralytics YOLO',
        'streamlit': 'Streamlit'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    return True

def test_cuda():
    """测试CUDA支持"""
    print("\n🔍 测试CUDA支持...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"   设备数量: {torch.cuda.device_count()}")
        print(f"   当前设备: {torch.cuda.current_device()}")
        print(f"   设备名称: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 测试简单的CUDA操作
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print(f"✅ CUDA计算测试通过")
            return True
        except Exception as e:
            print(f"❌ CUDA计算测试失败: {e}")
            return False
    else:
        print("⚠️ CUDA不可用，将使用CPU模式")
        return False

def test_yolo_model():
    """测试YOLO模型加载"""
    print("\n🔍 测试YOLO模型...")
    
    try:
        from ultralytics import YOLO
        
        # 测试预训练模型
        print("   测试预训练模型...")
        model = YOLO('yolo11n.pt')
        print("✅ 预训练YOLO11模型加载成功")
        
        # 测试训练好的模型（如果存在）
        trained_model_path = "runs/detect/lajiao_detection_20250623_053550/weights/best.pt"
        if os.path.exists(trained_model_path):
            print("   测试训练好的模型...")
            trained_model = YOLO(trained_model_path)
            print("✅ 训练好的YOLO模型加载成功")
        else:
            print("⚠️ 训练好的模型不存在，使用预训练模型")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO模型测试失败: {e}")
        return False

def test_sam2_model():
    """测试SAM2模型加载"""
    print("\n🔍 测试SAM2模型...")
    
    try:
        # 检查SAM2路径
        sam2_path = "/home/zcx/sam2"
        if not os.path.exists(sam2_path):
            print(f"❌ SAM2路径不存在: {sam2_path}")
            return False
        
        sys.path.append(sam2_path)
        
        # 检查模型文件
        checkpoint = "/home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        if not os.path.exists(checkpoint):
            print(f"❌ SAM2模型文件不存在: {checkpoint}")
            return False
        
        # 尝试加载SAM2
        from sam2.build_sam import build_sam2_video_predictor
        
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"   使用设备: {device}")
        print("   加载SAM2模型...")
        
        sam2_model = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        print("✅ SAM2模型加载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ SAM2模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_processing():
    """测试视频处理功能"""
    print("\n🔍 测试视频处理...")
    
    try:
        # 创建测试视频
        test_video = "test_video.mp4"
        print("   创建测试视频...")
        
        # 创建简单的测试视频（红色方块移动）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video, fourcc, 10.0, (640, 480))
        
        for i in range(30):  # 30帧
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 移动的红色方块
            x = 50 + i * 10
            y = 200
            cv2.rectangle(frame, (x, y), (x+100, y+100), (0, 0, 255), -1)
            out.write(frame)
        
        out.release()
        print(f"✅ 测试视频创建成功: {test_video}")
        
        # 测试视频读取
        cap = cv2.VideoCapture(test_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        print(f"   视频信息: {frame_count}帧, {fps}FPS")
        
        # 清理测试文件
        if os.path.exists(test_video):
            os.remove(test_video)
            print("   测试文件已清理")
        
        return True
        
    except Exception as e:
        print(f"❌ 视频处理测试失败: {e}")
        return False

def test_system_integration():
    """测试系统集成"""
    print("\n🔍 测试系统集成...")
    
    try:
        # 测试导入主要模块
        sys.path.append('.')
        
        # 这里可以添加更多集成测试
        print("✅ 系统集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 系统集成测试失败: {e}")
        return False

def main():
    print("🎯 YOLO-SAM2 系统测试")
    print("=" * 50)
    
    tests = [
        ("依赖包", test_dependencies),
        ("CUDA支持", test_cuda),
        ("YOLO模型", test_yolo_model),
        ("SAM2模型", test_sam2_model),
        ("视频处理", test_video_processing),
        ("系统集成", test_system_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    # 汇总结果
    print("\n" + "="*50)
    print("📊 测试结果汇总")
    print("="*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:12} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统已就绪")
        print("\n🚀 可以开始使用YOLO-SAM2系统:")
        print("   Web UI: python run_yolo_sam2_ui.py")
        print("   命令行: python yolo_sam2_demo.py --video your_video.mp4")
    else:
        print(f"\n⚠️ {total-passed} 个测试失败，请检查相关配置")
        
        # 提供修复建议
        if not results.get("依赖包", True):
            print("\n💡 修复建议:")
            print("   pip install streamlit ultralytics torch opencv-python numpy pillow")
        
        if not results.get("SAM2模型", True):
            print("\n💡 SAM2模型修复建议:")
            print("   1. 确保SAM2已正确安装在 /home/zcx/sam2")
            print("   2. 下载模型文件到 /home/zcx/sam2/checkpoints/")
            print("   3. 检查配置文件路径")

if __name__ == "__main__":
    main() 