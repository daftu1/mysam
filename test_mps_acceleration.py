#!/usr/bin/env python3
"""
MPS加速性能测试脚本
测试Apple Silicon Mac的MPS加速效果
"""

import torch
import time
import numpy as np

def test_device_performance():
    """测试不同设备的性能"""
    print("🚀 Apple Silicon MPS加速性能测试")
    print("=" * 50)
    
    # 检测可用设备
    devices = []
    device_names = []
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
        device_names.append("🚀 Apple Silicon MPS")
    
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        device_names.append("🚀 CUDA GPU")
    
    devices.append(torch.device("cpu"))
    device_names.append("💻 CPU")
    
    # 测试不同大小的矩阵运算
    matrix_sizes = [512, 1024, 2048]
    
    print(f"检测到设备: {', '.join(device_names)}")
    print()
    
    for size in matrix_sizes:
        print(f"📊 测试矩阵大小: {size}x{size}")
        print("-" * 30)
        
        for device, name in zip(devices, device_names):
            try:
                # 预热
                if device.type != "cpu":
                    x = torch.randn(100, 100).to(device)
                    y = torch.randn(100, 100).to(device)
                    _ = torch.mm(x, y)
                    if device.type == "mps":
                        torch.mps.synchronize()
                    elif device.type == "cuda":
                        torch.cuda.synchronize()
                
                # 创建测试数据
                x = torch.randn(size, size).to(device)
                y = torch.randn(size, size).to(device)
                
                # 测试矩阵乘法
                start_time = time.time()
                for _ in range(10):  # 重复10次
                    z = torch.mm(x, y)
                
                # 等待计算完成
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                print(f"{name}: {avg_time:.4f}秒/次")
                
                # 清理内存
                del x, y, z
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"{name}: 错误 - {str(e)}")
        
        print()
    
    # 测试神经网络相关操作
    print("🧠 神经网络操作性能测试")
    print("-" * 30)
    
    for device, name in zip(devices, device_names):
        try:
            # 模拟卷积操作
            batch_size = 32
            channels = 64
            height, width = 224, 224
            
            x = torch.randn(batch_size, channels, height, width).to(device)
            conv = torch.nn.Conv2d(channels, 128, 3, padding=1).to(device)
            
            start_time = time.time()
            for _ in range(5):
                y = conv(x)
                y = torch.relu(y)
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            
            print(f"{name} 卷积操作: {avg_time:.4f}秒/次")
            
            # 清理内存
            del x, y, conv
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"{name} 卷积操作: 错误 - {str(e)}")
    
    print()
    print("✅ 性能测试完成！")
    print()
    print("💡 优化建议:")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("• 您的Mac支持MPS加速，比纯CPU快很多")
        print("• SAM2模型已自动启用MPS加速")
        print("• 推荐处理中等规模的视频数据")
    else:
        print("• 使用CPU处理，建议降低视频分辨率")
        print("• 考虑减少处理帧数以提升速度")

if __name__ == "__main__":
    test_device_performance() 