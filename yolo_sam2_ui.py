#!/usr/bin/env python3
"""
YOLO-SAM2 视频分割UI
结合YOLO11目标检测和SAM2精确分割的完整解决方案
"""

import streamlit as st
import cv2
import numpy as np
import torch
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import uuid
from ultralytics import YOLO
import sys

# 添加SAM2路径
sys.path.append('/home/zcx/sam2')
from sam2.build_sam import build_sam2_video_predictor

# 页面配置
st.set_page_config(
    page_title="YOLO-SAM2 视频分割系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_yolo_model(model_path):
    """加载YOLO11模型"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"YOLO模型加载失败: {e}")
        return None

@st.cache_resource
def load_sam2_model():
    """加载SAM2模型"""
    try:
        checkpoint = "/home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam2_model = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        return sam2_model
    except Exception as e:
        st.error(f"SAM2模型加载失败: {e}")
        return None

def extract_frames(video_path, output_dir, max_frames=100):
    """提取视频帧"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        extracted_frames.append(frame_path)
        frame_count += 1
    
    cap.release()
    return extracted_frames, frame_count

def yolo_detect(model, image_path, conf_threshold=0.25):
    """使用YOLO检测目标"""
    results = model.predict(image_path, conf=conf_threshold, verbose=False)
    
    detections = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
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

def sam2_segment_video(sam2_model, frame_dir, detections, target_frame=0):
    """使用SAM2进行视频分割"""
    try:
        # 初始化SAM2推理状态
        inference_state = sam2_model.init_state(video_path=frame_dir)
        
        video_segments = {}
        
        # 为每个检测到的目标创建分割
        for obj_id, detection in enumerate(detections, 1):
            # 使用检测框的中心点作为正点
            center_x, center_y = detection['center']
            points = np.array([[center_x, center_y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # 正点
            
            # 在目标帧添加点
            _, _, _ = sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=target_frame,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
        
        # 传播到整个视频
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {}
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                video_segments[out_frame_idx][obj_id] = mask
        
        return video_segments, inference_state
        
    except Exception as e:
        st.error(f"SAM2分割失败: {e}")
        return {}, None

def visualize_results(image_path, detections, masks=None, show_boxes=True, show_masks=True):
    """可视化检测和分割结果"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image.copy()
    
    # 绘制分割掩码
    if show_masks and masks:
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        
        for obj_id, mask in masks.items():
            if mask.sum() > 0:
                color = colors[(obj_id-1) % len(colors)]
                overlay[mask > 0] = (overlay[mask > 0] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
    
    # 绘制检测框
    if show_boxes and detections:
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制置信度
            label = f"Object {i+1}: {conf:.2f}"
            cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制中心点
            center_x, center_y = detection['center']
            cv2.circle(overlay, (center_x, center_y), 5, (255, 0, 0), -1)
    
    return overlay

def main():
    st.title("🎯 YOLO-SAM2 视频分割系统")
    st.markdown("### 结合YOLO11检测和SAM2分割的智能视频处理系统")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 模型配置")
        
        # YOLO模型选择
        st.subheader("🎯 YOLO11 检测模型")
        
        # 检查是否有训练好的模型
        trained_model_path = "runs/detect/lajiao_detection_20250623_053550/weights/best.pt"
        if os.path.exists(trained_model_path):
            use_trained_model = st.checkbox("使用训练好的辣椒检测模型", value=True)
            if use_trained_model:
                yolo_model_path = trained_model_path
                st.success(f"✅ 使用训练好的模型: {yolo_model_path}")
            else:
                yolo_model_path = st.text_input("YOLO模型路径", value="yolo11n.pt")
        else:
            yolo_model_path = st.text_input("YOLO模型路径", value="yolo11n.pt")
        
        # 检测参数
        conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25, 0.05)
        
        st.divider()
        
        # SAM2模型状态
        st.subheader("🎨 SAM2 分割模型")
        if os.path.exists("/home/zcx/sam2/checkpoints/sam2.1_hiera_base_plus.pt"):
            st.success("✅ SAM2模型已就绪")
        else:
            st.error("❌ SAM2模型未找到")
        
        st.divider()
        
        # 显示选项
        st.subheader("🖼️ 显示选项")
        show_boxes = st.checkbox("显示检测框", value=True)
        show_masks = st.checkbox("显示分割掩码", value=True)
        show_centers = st.checkbox("显示中心点", value=True)
        
        st.divider()
        
        # 系统信息
        st.subheader("💻 系统信息")
        st.write(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 主界面
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 视频上传")
        uploaded_video = st.file_uploader(
            "选择视频文件", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="支持常见视频格式"
        )
        
        if uploaded_video:
            # 保存上传的视频
            temp_video_path = f"temp_video_{uuid.uuid4().hex[:8]}.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            
            st.video(temp_video_path)
            
            # 视频信息
            cap = cv2.VideoCapture(temp_video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            duration = frame_count / fps
            cap.release()
            
            st.write(f"📊 视频信息:")
            st.write(f"- 总帧数: {frame_count}")
            st.write(f"- 帧率: {fps} FPS")
            st.write(f"- 时长: {duration:.1f} 秒")
            
            # 处理按钮
            if st.button("🚀 开始YOLO-SAM2分割", type="primary"):
                with st.spinner("正在加载模型..."):
                    # 加载模型
                    yolo_model = load_yolo_model(yolo_model_path)
                    sam2_model = load_sam2_model()
                    
                    if yolo_model is None or sam2_model is None:
                        st.error("模型加载失败，请检查模型路径")
                        return
                
                with st.spinner("正在提取视频帧..."):
                    # 提取帧
                    frame_dir = f"frames_{uuid.uuid4().hex[:8]}"
                    frame_paths, total_frames = extract_frames(temp_video_path, frame_dir, max_frames=100)
                    st.success(f"✅ 提取了 {total_frames} 帧")
                
                # 选择参考帧进行检测
                reference_frame = st.slider(
                    "选择参考帧进行目标检测", 
                    0, min(total_frames-1, 99), 
                    min(10, total_frames//2)
                )
                
                if frame_paths:
                    # 在参考帧上进行YOLO检测
                    with st.spinner(f"正在检测第 {reference_frame} 帧的目标..."):
                        detections = yolo_detect(yolo_model, frame_paths[reference_frame], conf_threshold)
                        st.success(f"✅ 检测到 {len(detections)} 个目标")
                    
                    if detections:
                        # 显示检测结果
                        st.subheader(f"🎯 第 {reference_frame} 帧检测结果")
                        
                        # 可视化检测结果
                        detection_image = visualize_results(
                            frame_paths[reference_frame], 
                            detections, 
                            show_boxes=show_boxes
                        )
                        st.image(detection_image, caption=f"检测结果 - 第 {reference_frame} 帧")
                        
                        # 显示检测详情
                        for i, det in enumerate(detections):
                            st.write(f"目标 {i+1}: 置信度 {det['confidence']:.3f}, 中心点 {det['center']}")
                        
                        # 开始SAM2分割
                        if st.button("🎨 开始SAM2视频分割"):
                            with st.spinner("正在进行SAM2视频分割..."):
                                video_segments, inference_state = sam2_segment_video(
                                    sam2_model, frame_dir, detections, reference_frame
                                )
                                
                                if video_segments:
                                    st.success(f"✅ 分割完成！处理了 {len(video_segments)} 帧")
                                    
                                    # 保存结果到session state
                                    st.session_state['video_segments'] = video_segments
                                    st.session_state['frame_paths'] = frame_paths
                                    st.session_state['detections'] = detections
                                    st.session_state['frame_dir'] = frame_dir
                    else:
                        st.warning("⚠️ 未检测到任何目标，请调整置信度阈值或选择其他帧")
    
    with col2:
        st.subheader("🎨 分割结果预览")
        
        # 如果有分割结果，显示预览
        if 'video_segments' in st.session_state:
            video_segments = st.session_state['video_segments']
            frame_paths = st.session_state['frame_paths']
            detections = st.session_state['detections']
            
            # 帧选择器
            max_frame = min(len(frame_paths)-1, max(video_segments.keys()) if video_segments else 0)
            preview_frame = st.slider("预览帧", 0, max_frame, 0)
            
            if preview_frame in video_segments:
                masks = video_segments[preview_frame]
                
                # 可视化结果
                result_image = visualize_results(
                    frame_paths[preview_frame],
                    detections,
                    masks,
                    show_boxes=show_boxes,
                    show_masks=show_masks
                )
                
                st.image(result_image, caption=f"分割结果 - 第 {preview_frame} 帧")
                
                # 显示掩码统计
                st.write("📊 分割统计:")
                for obj_id, mask in masks.items():
                    pixel_count = mask.sum()
                    st.write(f"- 目标 {obj_id}: {pixel_count} 像素")
            else:
                st.info("该帧无分割结果")
            
            # 导出选项
            st.subheader("📤 导出选项")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("💾 保存分割掩码"):
                    output_dir = f"yolo_sam2_output_{uuid.uuid4().hex[:8]}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    saved_count = 0
                    for frame_idx, masks in video_segments.items():
                        if masks:
                            for obj_id, mask in masks.items():
                                if mask.sum() > 0:
                                    mask_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_obj_{obj_id}.png")
                                    mask_image = (mask * 255).astype(np.uint8)
                                    cv2.imwrite(mask_path, mask_image)
                                    saved_count += 1
                    
                    st.success(f"✅ 保存了 {saved_count} 个掩码到 {output_dir}")
            
            with col_b:
                if st.button("🎬 生成分割视频"):
                    output_video = f"segmented_video_{uuid.uuid4().hex[:8]}.mp4"
                    
                    # 获取第一帧尺寸
                    first_frame = cv2.imread(frame_paths[0])
                    height, width = first_frame.shape[:2]
                    
                    # 创建视频写入器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video, fourcc, 10.0, (width, height))
                    
                    for i, frame_path in enumerate(frame_paths):
                        if i in video_segments:
                            # 有分割结果的帧
                            result_frame = visualize_results(
                                frame_path,
                                detections,
                                video_segments[i],
                                show_boxes=show_boxes,
                                show_masks=show_masks
                            )
                            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                        else:
                            # 原始帧
                            result_frame = cv2.imread(frame_path)
                        
                        out.write(result_frame)
                    
                    out.release()
                    st.success(f"✅ 分割视频已保存: {output_video}")
                    
                    # 显示生成的视频
                    st.video(output_video)
        else:
            st.info("👆 请先上传视频并完成检测分割")
            
            # 显示示例
            st.markdown("""
            ### 🔄 处理流程
            1. **上传视频** - 支持MP4、AVI等格式
            2. **YOLO检测** - 在参考帧检测目标
            3. **SAM2分割** - 基于检测结果进行精确分割
            4. **结果预览** - 查看分割效果
            5. **导出结果** - 保存掩码或生成视频
            
            ### ✨ 特色功能
            - 🎯 **YOLO11检测**: 快速准确的目标检测
            - 🎨 **SAM2分割**: 像素级精确分割
            - 🎬 **视频处理**: 支持视频序列分割
            - 📊 **实时预览**: 即时查看处理结果
            - 💾 **多种导出**: 掩码、视频等格式
            """)

    # 清理临时文件
    if st.sidebar.button("🧹 清理临时文件"):
        # 清理临时视频和帧目录
        for item in os.listdir('.'):
            if item.startswith('temp_video_') or item.startswith('frames_'):
                if os.path.isfile(item):
                    os.remove(item)
                elif os.path.isdir(item):
                    shutil.rmtree(item)
        
        # 清理session state
        for key in ['video_segments', 'frame_paths', 'detections', 'frame_dir']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ 临时文件已清理")
        st.rerun()

if __name__ == "__main__":
    main() 