#!/usr/bin/env python3
"""
YOLO-SAM2 命令行演示脚本
展示如何在命令行中使用YOLO+SAM2进行视频分割
"""

import cv2
import numpy as np
import torch
import os
import argparse
from ultralytics import YOLO
import sys

# 添加SAM2路径
sys.path.append('/home/zcx/sam2')
from sam2.build_sam import build_sam2_video_predictor

def load_models(yolo_path, sam2_checkpoint):
    """加载YOLO和SAM2模型"""
    print("🔄 加载模型...")
    
    # 加载YOLO
    yolo_model = YOLO(yolo_path)
    print(f"✅ YOLO模型加载完成: {yolo_path}")
    
    # 加载SAM2
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_model = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print(f"✅ SAM2模型加载完成: {sam2_checkpoint}")
    
    return yolo_model, sam2_model

def extract_frames(video_path, output_dir, max_frames=50):
    """提取视频帧"""
    print(f"🎬 提取视频帧: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    print(f"✅ 提取了 {frame_count} 帧到 {output_dir}")
    return frame_count

def yolo_detect(yolo_model, image_path, conf_threshold=0.25):
    """YOLO目标检测"""
    results = yolo_model.predict(image_path, conf=conf_threshold, verbose=False)
    
    detections = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': cls,
                'center': [int((x1+x2)/2), int((y1+y2)/2)]
            })
    
    return detections

def sam2_segment(sam2_model, frame_dir, detections, reference_frame=0):
    """SAM2视频分割"""
    print(f"🎨 开始SAM2分割，参考帧: {reference_frame}")
    
    # 初始化推理状态
    inference_state = sam2_model.init_state(video_path=frame_dir)
    
    # 为每个检测目标添加点
    for obj_id, detection in enumerate(detections, 1):
        center_x, center_y = detection['center']
        points = np.array([[center_x, center_y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        
        sam2_model.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=reference_frame,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        print(f"   添加目标 {obj_id}: 中心点 ({center_x}, {center_y})")
    
    # 传播分割
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {}
        for i, obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            video_segments[out_frame_idx][obj_id] = mask
    
    print(f"✅ 分割完成，处理了 {len(video_segments)} 帧")
    return video_segments

def save_results(frame_dir, video_segments, detections, output_dir):
    """保存分割结果"""
    print(f"💾 保存结果到: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # 颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]
    
    saved_count = 0
    
    for frame_idx, masks in video_segments.items():
        # 读取原始帧
        frame_path = os.path.join(frame_dir, f"{frame_idx:04d}.jpg")
        if not os.path.exists(frame_path):
            continue
        
        image = cv2.imread(frame_path)
        overlay = image.copy()
        
        # 保存每个目标的掩码
        for obj_id, mask in masks.items():
            if mask.sum() > 0:
                # 保存掩码
                mask_path = os.path.join(output_dir, "masks", f"frame_{frame_idx:04d}_obj_{obj_id}.png")
                mask_image = (mask * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask_image)
                
                # 在可视化图像上添加掩码
                color = colors[(obj_id-1) % len(colors)]
                overlay[mask > 0] = (overlay[mask > 0] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
                saved_count += 1
        
        # 添加检测框
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"Obj {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存可视化结果
        vis_path = os.path.join(output_dir, "visualizations", f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(vis_path, overlay)
    
    print(f"✅ 保存了 {saved_count} 个掩码和 {len(video_segments)} 个可视化图像")

def create_video(output_dir, output_video, fps=10):
    """创建分割结果视频"""
    print(f"🎬 创建分割视频: {output_video}")
    
    vis_dir = os.path.join(output_dir, "visualizations")
    if not os.path.exists(vis_dir):
        print("❌ 可视化目录不存在")
        return
    
    # 获取所有可视化图像
    vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.jpg')])
    if not vis_files:
        print("❌ 没有找到可视化图像")
        return
    
    # 读取第一张图像获取尺寸
    first_image = cv2.imread(os.path.join(vis_dir, vis_files[0]))
    height, width = first_image.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for vis_file in vis_files:
        frame = cv2.imread(os.path.join(vis_dir, vis_file))
        out.write(frame)
    
    out.release()
    print(f"✅ 视频创建完成: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="YOLO-SAM2 视频分割演示")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--yolo", default="runs/detect/lajiao_detection_20250623_053550/weights/best.pt", help="YOLO模型路径")
    parser.add_argument("--sam2", default="/home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="SAM2模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO置信度阈值")
    parser.add_argument("--ref-frame", type=int, default=0, help="参考帧索引")
    parser.add_argument("--max-frames", type=int, default=50, help="最大处理帧数")
    parser.add_argument("--output", default="yolo_sam2_results", help="输出目录")
    
    args = parser.parse_args()
    
    print("🎯 YOLO-SAM2 视频分割演示")
    print("=" * 50)
    print(f"输入视频: {args.video}")
    print(f"YOLO模型: {args.yolo}")
    print(f"SAM2模型: {args.sam2}")
    print(f"置信度阈值: {args.conf}")
    print(f"参考帧: {args.ref_frame}")
    print(f"最大帧数: {args.max_frames}")
    print(f"输出目录: {args.output}")
    print()
    
    # 检查输入文件
    if not os.path.exists(args.video):
        print(f"❌ 视频文件不存在: {args.video}")
        return
    
    if not os.path.exists(args.yolo):
        print(f"❌ YOLO模型不存在: {args.yolo}")
        return
    
    if not os.path.exists(args.sam2):
        print(f"❌ SAM2模型不存在: {args.sam2}")
        return
    
    try:
        # 1. 加载模型
        yolo_model, sam2_model = load_models(args.yolo, args.sam2)
        
        # 2. 提取帧
        frame_dir = "temp_frames"
        frame_count = extract_frames(args.video, frame_dir, args.max_frames)
        
        if frame_count == 0:
            print("❌ 未能提取到视频帧")
            return
        
        # 3. YOLO检测
        ref_frame_path = os.path.join(frame_dir, f"{args.ref_frame:04d}.jpg")
        if not os.path.exists(ref_frame_path):
            print(f"❌ 参考帧不存在: {ref_frame_path}")
            return
        
        print(f"🎯 在参考帧 {args.ref_frame} 进行YOLO检测...")
        detections = yolo_detect(yolo_model, ref_frame_path, args.conf)
        print(f"✅ 检测到 {len(detections)} 个目标")
        
        if len(detections) == 0:
            print("❌ 未检测到任何目标，请调整置信度阈值或选择其他参考帧")
            return
        
        # 显示检测结果
        for i, det in enumerate(detections):
            print(f"   目标 {i+1}: 置信度 {det['confidence']:.3f}, 中心点 {det['center']}")
        
        # 4. SAM2分割
        video_segments = sam2_segment(sam2_model, frame_dir, detections, args.ref_frame)
        
        if len(video_segments) == 0:
            print("❌ SAM2分割失败")
            return
        
        # 5. 保存结果
        save_results(frame_dir, video_segments, detections, args.output)
        
        # 6. 创建视频
        output_video = os.path.join(args.output, "segmented_video.mp4")
        create_video(args.output, output_video)
        
        print("\n🎉 处理完成！")
        print(f"📁 结果目录: {args.output}")
        print(f"🎬 分割视频: {output_video}")
        print(f"🖼️ 掩码文件: {args.output}/masks/")
        print(f"👁️ 可视化图像: {args.output}/visualizations/")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        if os.path.exists("temp_frames"):
            import shutil
            shutil.rmtree("temp_frames")
            print("🧹 临时文件已清理")

if __name__ == "__main__":
    main() 