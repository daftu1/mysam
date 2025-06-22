@echo off
REM DATASAM2GET项目环境自动安装脚本 (Windows)
REM 作者: daftu1
REM 运行方式: setup.bat

echo 🚀 开始安装DATASAM2GET项目环境...

REM 检查conda是否安装
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 错误: conda未安装，请先安装Anaconda或Miniconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✅ 检测到conda已安装

REM 创建conda环境
echo 📦 创建conda环境: datasam2get
conda env list | findstr "datasam2get" >nul
if %ERRORLEVEL% EQU 0 (
    echo ⚠️  环境datasam2get已存在，是否删除重建? (y/N^)
    set /p choice="请输入选择: "
    if /i "%choice%"=="y" (
        echo 🗑️  删除现有环境...
        conda env remove -n datasam2get -y
    ) else (
        echo 🔄 使用现有环境
    )
)

conda env list | findstr "datasam2get" >nul
if %ERRORLEVEL% NEQ 0 (
    echo 🏗️  创建新环境...
    conda env create -f environment.yml
)

echo ✅ conda环境创建完成

REM 激活环境
echo 🔧 激活环境并安装额外依赖...
call conda activate datasam2get

REM 安装requirements.txt中的额外包
echo 📦 安装额外Python包...
pip install -r requirements.txt

REM 检查GPU支持
echo 🖥️  检查GPU支持...
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"无\"}'); print(f'GPU数量: {torch.cuda.device_count()}')"

REM 安装SAM2
echo 🎯 安装SAM2...
if exist "sam2" (
    echo ⚠️  SAM2目录已存在，是否重新安装? (y/N^)
    set /p choice="请输入选择: "
    if /i "%choice%"=="y" (
        rmdir /s /q sam2
    ) else (
        echo 🔄 使用现有SAM2安装
        goto :skip_sam2
    )
)

if not exist "sam2" (
    echo 📥 下载SAM2源码...
    git clone https://github.com/facebookresearch/segment-anything-2.git sam2
    cd sam2
    
    echo 📦 安装SAM2...
    pip install -e .
    
    echo 📥 下载SAM2模型...
    if not exist "checkpoints" mkdir checkpoints
    cd checkpoints
    
    REM 下载SAM2.1模型
    if not exist "sam2.1_hiera_base_plus.pt" (
        echo ⬇️  下载SAM2.1 Base Plus模型...
        curl -L -o sam2.1_hiera_base_plus.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    )
    
    if not exist "sam2.1_hiera_large.pt" (
        echo ⬇️  下载SAM2.1 Large模型...
        curl -L -o sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    )
    
    cd ..\..
)

:skip_sam2
echo ✅ SAM2安装完成

REM 创建启动脚本
echo 📝 创建启动脚本...
(
echo @echo off
echo REM DATASAM2GET项目启动脚本
echo.
echo REM 激活conda环境
echo call conda activate datasam2get
echo.
echo echo 🎯 DATASAM2GET项目启动
echo echo 选择要启动的应用:
echo echo 1. 数据标注工具 (data_generation^)
echo echo 2. 模型训练工具 (training_models^)
echo echo 3. YOLO-SAM2演示 (demo_apps^)
echo echo 4. 查看GPU状态
echo.
echo set /p choice="请输入选择 (1-4^): "
echo.
echo if "%%choice%%"=="1" (
echo     echo 🏷️  启动数据标注工具...
echo     cd data_generation
echo     streamlit run streamlit_app_enhanced.py --server.port 8501
echo ^) else if "%%choice%%"=="2" (
echo     echo 🎓 启动模型训练...
echo     cd training_models
echo     python run_training.py
echo ^) else if "%%choice%%"=="3" (
echo     echo 🎮 启动YOLO-SAM2演示...
echo     cd demo_apps
echo     python run_yolo_sam2_ui.py
echo ^) else if "%%choice%%"=="4" (
echo     echo 🖥️  GPU状态信息:
echo     python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'^); print(f'GPU数量: {torch.cuda.device_count()}'^); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}'^) for i in range(torch.cuda.device_count()^)]"
echo ^) else (
echo     echo ❌ 无效选择
echo ^)
echo.
echo pause
) > start_ui.bat

REM 安装验证
echo 🔍 验证安装...
python -c "import torch; import ultralytics; import streamlit; import cv2; import numpy as np; from PIL import Image; print('✅ 所有核心包导入成功!'); print(f'PyTorch版本: {torch.__version__}'); print(f'Ultralytics版本: {ultralytics.__version__}'); print(f'OpenCV版本: {cv2.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo.
echo 🎉 安装完成!
echo.
echo 📋 使用说明:
echo 1. 激活环境: conda activate datasam2get
echo 2. 运行启动脚本: start_ui.bat
echo 3. 或者手动启动各个模块:
echo    - 数据标注: cd data_generation ^&^& streamlit run streamlit_app_enhanced.py --server.port 8501
echo    - 模型训练: cd training_models ^&^& python run_training.py
echo    - 演示应用: cd demo_apps ^&^& python run_yolo_sam2_ui.py
echo.
echo 🔧 配置文件:
echo    - Conda环境: environment.yml
echo    - Python依赖: requirements.txt
echo    - SAM2路径: .\sam2\
echo.
echo 📚 更多信息请查看 README.md

pause 