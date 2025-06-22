#!/usr/bin/env python3
"""
YOLO11数据集准备脚本
将SAM2生成的标注数据转换为YOLO11训练格式
"""

import os
import shutil
import random
from pathlib import Path
import yaml

def prepare_yolo_dataset(
    images_dir="frame_cache_4f2d1b14",
    labels_dir="yolo11_labels_4f2d1b14", 
    output_dir="yolo_dataset",
    train_ratio=0.8,
    val_ratio=0.2
):
    """
    准备YOLO11数据集
    
    Args:
        images_dir: 图像文件目录
        labels_dir: 标注文件目录
        output_dir: 输出数据集目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    
    # 创建输出目录结构
    dataset_path = Path(output_dir)
    dataset_path.mkdir(exist_ok=True)
    
    # 创建子目录
    for split in ['train', 'val']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.jpg"))
    
    # 过滤有对应标注文件的图像
    labels_path = Path(labels_dir)
    valid_files = []
    
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            # 检查标注文件是否为空
            if label_file.stat().st_size > 0:
                valid_files.append(img_file.stem)
    
    print(f"找到 {len(valid_files)} 个有效的图像-标注对")
    
    # 随机打乱文件列表
    random.shuffle(valid_files)
    
    # 计算分割点
    train_count = int(len(valid_files) * train_ratio)
    val_count = len(valid_files) - train_count
    
    train_files = valid_files[:train_count]
    val_files = valid_files[train_count:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 复制文件到对应目录
    def copy_files(file_list, split):
        for filename in file_list:
            # 复制图像文件
            src_img = images_path / f"{filename}.jpg"
            dst_img = dataset_path / split / 'images' / f"{filename}.jpg"
            shutil.copy2(src_img, dst_img)
            
            # 复制标注文件
            src_label = labels_path / f"{filename}.txt"
            dst_label = dataset_path / split / 'labels' / f"{filename}.txt"
            shutil.copy2(src_label, dst_label)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    # 读取类别信息
    classes_file = labels_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    else:
        classes = ['lajiao']  # 默认类别
    
    # 创建YOLO配置文件
    config = {
        'path': str(dataset_path.absolute()),  # 数据集根目录
        'train': 'train/images',  # 训练图像目录（相对路径）
        'val': 'val/images',      # 验证图像目录（相对路径）
        'nc': len(classes),       # 类别数量
        'names': classes          # 类别名称列表
    }
    
    config_file = dataset_path / "data.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ 数据集准备完成！")
    print(f"📁 数据集目录: {dataset_path.absolute()}")
    print(f"📄 配置文件: {config_file.absolute()}")
    print(f"🏷️ 类别数量: {len(classes)}")
    print(f"🏷️ 类别列表: {classes}")
    
    return str(dataset_path.absolute()), str(config_file.absolute())

if __name__ == "__main__":
    dataset_dir, config_file = prepare_yolo_dataset()
    print(f"\n🚀 可以开始训练了！")
    print(f"使用配置文件: {config_file}") 