#!/usr/bin/env python3
"""
YOLO-SAM2 UI 启动脚本
快速启动YOLO-SAM2视频分割界面
"""

import subprocess
import sys
import os

def check_dependencies():
    """检查必要的依赖"""
    required_packages = [
        'streamlit',
        'ultralytics', 
        'torch',
        'opencv-python',
        'numpy',
        'pillow'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ 缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_models():
    """检查模型文件"""
    print("\n🔍 检查模型文件...")
    
    # 检查YOLO模型
    yolo_model = "runs/detect/lajiao_detection_20250623_053550/weights/best.pt"
    if os.path.exists(yolo_model):
        print(f"✅ YOLO模型: {yolo_model}")
    else:
        print("⚠️ 训练好的YOLO模型未找到，将使用预训练模型")
    
    # 检查SAM2模型
    sam2_model = "/home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    if os.path.exists(sam2_model):
        print(f"✅ SAM2模型: {sam2_model}")
    else:
        print(f"❌ SAM2模型未找到: {sam2_model}")
        return False
    
    return True

def main():
    print("🎯 YOLO-SAM2 视频分割系统启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查模型
    if not check_models():
        print("\n❌ 模型检查失败，请确保SAM2模型已正确安装")
        return
    
    print("\n🚀 启动YOLO-SAM2 UI...")
    print("浏览器将自动打开 http://localhost:8506")
    print("按 Ctrl+C 停止服务")
    
    # 启动Streamlit应用
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "yolo_sam2_ui.py", 
            "--server.port", "8506",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 YOLO-SAM2 UI已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 