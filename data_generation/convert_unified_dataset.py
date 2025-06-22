#!/usr/bin/env python3
"""
统一数据集转换脚本
将streamlit_app_enhanced.py生成的统一数据集转换为YOLO训练格式
"""

import os
import shutil
import random
from pathlib import Path
import yaml

def convert_unified_dataset(
    dataset_name="lajiao_dataset",
    output_dir="yolo_dataset",
    train_ratio=0.8,
    val_ratio=0.2
):
    """
    将统一数据集转换为YOLO训练格式
    
    Args:
        dataset_name: 统一数据集名称
        output_dir: 输出YOLO数据集目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    
    print(f"🔄 转换统一数据集: {dataset_name}")
    
    # 统一数据集路径
    unified_frame_dir = f"frames_{dataset_name}"
    unified_label_dir = f"labels_{dataset_name}"
    
    # 检查统一数据集是否存在
    if not os.path.exists(unified_frame_dir):
        print(f"❌ 图像目录不存在: {unified_frame_dir}")
        return False
    
    if not os.path.exists(unified_label_dir):
        print(f"❌ 标注目录不存在: {unified_label_dir}")
        return False
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 创建子目录
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(unified_frame_dir) if f.endswith('.jpg')]
    
    # 过滤有对应标注文件的图像
    valid_files = []
    for img_file in image_files:
        label_file = os.path.join(unified_label_dir, img_file.replace('.jpg', '.txt'))
        if os.path.exists(label_file):
            # 检查标注文件是否为空
            if os.path.getsize(label_file) > 0:
                valid_files.append(img_file.replace('.jpg', ''))
    
    print(f"📊 统计信息:")
    print(f"   总图像文件: {len(image_files)}")
    print(f"   有效图像-标注对: {len(valid_files)}")
    
    if len(valid_files) == 0:
        print("❌ 没有找到有效的图像-标注对")
        return False
    
    # 随机打乱文件列表
    random.shuffle(valid_files)
    
    # 计算分割点
    train_count = int(len(valid_files) * train_ratio)
    val_count = len(valid_files) - train_count
    
    train_files = valid_files[:train_count]
    val_files = valid_files[train_count:]
    
    print(f"📈 数据分割:")
    print(f"   训练集: {len(train_files)} 个文件")
    print(f"   验证集: {len(val_files)} 个文件")
    
    # 复制文件到对应目录
    def copy_files(file_list, split):
        copied_count = 0
        for filename in file_list:
            # 复制图像文件
            src_img = os.path.join(unified_frame_dir, f"{filename}.jpg")
            dst_img = output_path / split / 'images' / f"{filename}.jpg"
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                copied_count += 1
            
            # 复制标注文件
            src_label = os.path.join(unified_label_dir, f"{filename}.txt")
            dst_label = output_path / split / 'labels' / f"{filename}.txt"
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
        
        print(f"   {split}: 复制了 {copied_count} 个文件")
        return copied_count
    
    train_copied = copy_files(train_files, 'train')
    val_copied = copy_files(val_files, 'val')
    
    # 读取类别信息
    classes_file = os.path.join(unified_label_dir, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    else:
        classes = ['lajiao']  # 默认类别
        print("⚠️ 未找到classes.txt，使用默认类别")
    
    # 创建YOLO配置文件
    config = {
        'path': str(output_path.absolute()),  # 数据集根目录
        'train': 'train/images',  # 训练图像目录（相对路径）
        'val': 'val/images',      # 验证图像目录（相对路径）
        'nc': len(classes),       # 类别数量
        'names': classes          # 类别名称列表
    }
    
    config_file = output_path / "data.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ 数据集转换完成！")
    print(f"📁 输出目录: {output_path.absolute()}")
    print(f"📄 配置文件: {config_file.absolute()}")
    print(f"🏷️ 类别数量: {len(classes)}")
    print(f"🏷️ 类别列表: {classes}")
    print(f"📊 最终统计:")
    print(f"   训练集: {train_copied} 张图像")
    print(f"   验证集: {val_copied} 张图像")
    print(f"   总计: {train_copied + val_copied} 张图像")
    
    return True

def list_available_datasets():
    """列出所有可用的统一数据集"""
    print("📋 可用的统一数据集:")
    
    datasets = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('frames_'):
            dataset_name = item.replace('frames_', '')
            label_dir = f"labels_{dataset_name}"
            if os.path.exists(label_dir):
                # 统计文件数量
                frame_count = len([f for f in os.listdir(item) if f.endswith('.jpg')])
                label_count = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
                datasets.append((dataset_name, frame_count, label_count))
                print(f"   📁 {dataset_name}: {frame_count} 图像, {label_count} 标注")
    
    if not datasets:
        print("   ❌ 未找到任何统一数据集")
    
    return datasets

def merge_datasets(dataset_names, merged_name="merged_dataset"):
    """合并多个统一数据集"""
    print(f"🔗 合并数据集到: {merged_name}")
    
    merged_frame_dir = f"frames_{merged_name}"
    merged_label_dir = f"labels_{merged_name}"
    
    os.makedirs(merged_frame_dir, exist_ok=True)
    os.makedirs(merged_label_dir, exist_ok=True)
    
    all_classes = set()
    total_files = 0
    
    for dataset_name in dataset_names:
        frame_dir = f"frames_{dataset_name}"
        label_dir = f"labels_{dataset_name}"
        
        if not os.path.exists(frame_dir) or not os.path.exists(label_dir):
            print(f"⚠️ 跳过不存在的数据集: {dataset_name}")
            continue
        
        print(f"   📂 处理数据集: {dataset_name}")
        
        # 复制图像文件
        for img_file in os.listdir(frame_dir):
            if img_file.endswith('.jpg'):
                src_path = os.path.join(frame_dir, img_file)
                # 添加数据集前缀避免冲突
                dst_name = f"{dataset_name}_{img_file}"
                dst_path = os.path.join(merged_frame_dir, dst_name)
                shutil.copy2(src_path, dst_path)
        
        # 复制标注文件
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt') and label_file != 'classes.txt':
                src_path = os.path.join(label_dir, label_file)
                # 添加数据集前缀避免冲突
                dst_name = f"{dataset_name}_{label_file}"
                dst_path = os.path.join(merged_label_dir, dst_name)
                shutil.copy2(src_path, dst_path)
                total_files += 1
        
        # 收集类别信息
        classes_file = os.path.join(label_dir, "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
                all_classes.update(classes)
    
    # 保存合并后的类别文件
    merged_classes_file = os.path.join(merged_label_dir, "classes.txt")
    with open(merged_classes_file, 'w', encoding='utf-8') as f:
        for class_name in sorted(all_classes):
            f.write(f"{class_name}\n")
    
    print(f"✅ 数据集合并完成!")
    print(f"   📁 合并后目录: {merged_frame_dir}, {merged_label_dir}")
    print(f"   📊 总文件数: {total_files}")
    print(f"   🏷️ 类别数: {len(all_classes)}")
    print(f"   🏷️ 类别: {sorted(all_classes)}")
    
    return merged_name

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="统一数据集转换工具")
    parser.add_argument("--list", action="store_true", help="列出所有可用数据集")
    parser.add_argument("--convert", type=str, help="转换指定数据集")
    parser.add_argument("--merge", nargs="+", help="合并多个数据集")
    parser.add_argument("--output", type=str, default="yolo_dataset", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
    elif args.convert:
        success = convert_unified_dataset(
            dataset_name=args.convert,
            output_dir=args.output,
            train_ratio=args.train_ratio
        )
        if success:
            print(f"\n🚀 可以开始训练了！")
            print(f"使用命令: python train_yolo11.py")
    elif args.merge:
        merged_name = merge_datasets(args.merge)
        print(f"\n💡 接下来可以转换合并后的数据集:")
        print(f"python {__file__} --convert {merged_name}")
    else:
        # 默认行为：列出数据集并转换第一个
        datasets = list_available_datasets()
        if datasets:
            dataset_name = datasets[0][0]
            print(f"\n🎯 自动转换数据集: {dataset_name}")
            success = convert_unified_dataset(dataset_name)
            if success:
                print(f"\n🚀 可以开始训练了！")
                print(f"使用命令: python train_yolo11.py")
        else:
            print("\n❌ 没有找到可转换的数据集")
            print("💡 请先使用streamlit_app_enhanced.py生成统一数据集") 