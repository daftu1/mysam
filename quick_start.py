#!/usr/bin/env python3
"""
SAM2智能视频标注与训练系统 - 快速启动脚本
提供菜单选择不同功能模块
"""

import os
import sys
import subprocess
import time

def print_banner():
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🎯 SAM2智能视频标注与训练系统                              ║
║                                                              ║
║    基于SAM2和YOLO11的完整视频目标检测与分割解决方案           ║
║    从数据标注到模型训练再到应用部署的全流程工具链            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """打印主菜单"""
    menu = """
🚀 请选择功能模块:

┌─────────────────────────────────────────────────────────────┐
│  1️⃣  数据标注工具 (data_generation)                         │
│      📊 使用SAM2进行智能视频标注，生成高质量训练数据        │
│      🌐 Web界面: http://localhost:8501                     │
│                                                             │
│  2️⃣  模型训练工具 (training_models)                         │
│      🎓 基于标注数据训练YOLO11检测模型                      │
│      📈 支持TensorBoard训练监控                             │
│                                                             │
│  3️⃣  演示应用 (demo_apps)                                   │
│      🎮 使用训练好的模型进行YOLO-SAM2视频分割               │
│      🌐 Web界面: http://localhost:8506                     │
│                                                             │
│  4️⃣  系统测试                                               │
│      🔧 测试环境配置和模型加载                              │
│                                                             │
│  5️⃣  查看项目结构                                           │
│      📁 显示完整的项目文件结构                              │
│                                                             │
│  0️⃣  退出系统                                               │
└─────────────────────────────────────────────────────────────┘
    """
    print(menu)

def run_data_generation():
    """启动数据标注工具"""
    print("\n🎯 启动数据标注工具...")
    print("=" * 50)
    print("📊 SAM2视频标注工具将在浏览器中打开")
    print("🌐 访问地址: http://localhost:8501")
    print("💡 使用说明: 上传视频 → 点击标注 → SAM2分割 → 导出数据")
    print("🔄 按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        os.chdir("data_generation")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app_enhanced.py", 
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 数据标注工具已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
    finally:
        os.chdir("..")

def run_model_training():
    """启动模型训练"""
    print("\n🎓 启动模型训练...")
    print("=" * 50)
    print("📈 YOLO11模型训练")
    print("💡 确保数据集已准备完成")
    print("🔄 训练过程将自动进行")
    print("=" * 50)
    
    try:
        os.chdir("training_models")
        subprocess.run([sys.executable, "run_training.py"])
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
    finally:
        os.chdir("..")
    
    input("\n按回车键返回主菜单...")

def run_demo_apps():
    """启动演示应用"""
    print("\n🎮 启动演示应用...")
    print("=" * 50)
    print("🎯 YOLO-SAM2视频分割系统将在浏览器中打开")
    print("🌐 访问地址: http://localhost:8506")
    print("💡 使用说明: 上传视频 → YOLO检测 → SAM2分割 → 查看结果")
    print("🔄 按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        os.chdir("demo_apps")
        subprocess.run([sys.executable, "run_yolo_sam2_ui.py"])
    except KeyboardInterrupt:
        print("\n👋 演示应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
    finally:
        os.chdir("..")

def run_system_test():
    """运行系统测试"""
    print("\n🔧 运行系统测试...")
    print("=" * 50)
    
    try:
        os.chdir("demo_apps")
        subprocess.run([sys.executable, "test_yolo_sam2.py"])
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
    finally:
        os.chdir("..")
    
    input("\n按回车键返回主菜单...")

def show_project_structure():
    """显示项目结构"""
    print("\n📁 项目文件结构:")
    print("=" * 50)
    
    structure = """
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
├── 📚 README.md               # 项目总览
└── 🚀 quick_start.py          # 本启动脚本
    """
    
    print(structure)
    
    # 显示各模块简介
    print("\n📋 模块功能简介:")
    print("=" * 50)
    print("🎯 数据标注: 智能视频标注，SAM2辅助分割，生成YOLO训练数据")
    print("🎓 模型训练: YOLO11训练，性能评估，模型导出")  
    print("🎮 演示应用: YOLO-SAM2分割，实时预览，批量处理")
    print("📈 训练结果: 包含您训练的99.5% mAP辣椒检测模型")
    
    input("\n按回车键返回主菜单...")

def check_environment():
    """检查环境配置"""
    print("🔍 检查系统环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}")
    else:
        print(f"⚠️ Python版本过低: {python_version.major}.{python_version.minor} (建议3.8+)")
    
    # 检查关键依赖
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
    except ImportError:
        print("❌ PyTorch未安装")
    
    # 检查SAM2模型
    sam2_model = "/home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    if os.path.exists(sam2_model):
        print("✅ SAM2模型已就绪")
    else:
        print("❌ SAM2模型未找到")
    
    # 检查训练好的YOLO模型
    yolo_model = "runs/detect/lajiao_detection_20250623_053550/weights/best.pt"
    if os.path.exists(yolo_model):
        print("✅ 训练好的YOLO模型已就绪")
    else:
        print("⚠️ 训练好的YOLO模型未找到，将使用预训练模型")
    
    print()

def main():
    """主函数"""
    while True:
        # 清屏
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # 显示横幅和菜单
        print_banner()
        check_environment()
        print_menu()
        
        # 获取用户选择
        try:
            choice = input("请输入选项 (0-5): ").strip()
            
            if choice == "1":
                run_data_generation()
            elif choice == "2":
                run_model_training()
            elif choice == "3":
                run_demo_apps()
            elif choice == "4":
                run_system_test()
            elif choice == "5":
                show_project_structure()
            elif choice == "0":
                print("\n👋 感谢使用SAM2智能视频标注与训练系统！")
                print("🎯 期待您的下次使用！")
                break
            else:
                print(f"\n❌ 无效选项: {choice}")
                print("请输入0-5之间的数字")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出系统")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main() 