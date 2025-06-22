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
from ultralytics import YOLO, SAM

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
def load_sam2_model(model_name="sam2.1_b.pt"):
    """加载SAM2模型 - 使用ultralytics的SAM"""
    try:
        sam2_model = SAM(model_name)
        return sam2_model
    except Exception as e:
        st.error(f"SAM2模型加载失败: {e}")
        return None

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

def sam2_segment_with_prompts(sam2_model, image_path, detections):
    """使用SAM2对单张图像进行分割（基于检测结果提供提示）"""
    try:
        if not detections:
            return {}
        
        # 准备提示点（使用检测框的中心点）
        points = []
        labels = []
        
        for detection in detections:
            center_x, center_y = detection['center']
            points.append([center_x, center_y])
            labels.append(1)  # 正点
        
        # 使用点提示进行分割
        results = sam2_model.predict(
            image_path, 
            points=points, 
            labels=labels,
            verbose=False
        )
        
        # 提取掩码
        masks = {}
        if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            for i, mask in enumerate(results[0].masks.data):
                mask_np = mask.cpu().numpy().astype(bool)
                masks[i+1] = mask_np
        
        return masks
        
    except Exception as e:
        st.error(f"SAM2分割失败: {e}")
        return {}

# 注意：移除了sam2_segment_video函数，因为ultralytics的SAM2
# 不支持真正的视频时间一致性分割，只能逐帧独立处理

def process_frame_with_yolo_sam2(yolo_model, sam2_model, frame_path, conf_threshold=0.25):
    """结合YOLO检测和SAM2分割处理单帧"""
    # YOLO检测
    detections = yolo_detect(yolo_model, frame_path, conf_threshold)
    
    # SAM2分割
    masks = {}
    if detections:
        masks = sam2_segment_with_prompts(sam2_model, frame_path, detections)
    
    return detections, masks

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

def extract_frames(video_path, output_dir, max_frames=None):
    """提取视频帧"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 如果设置了最大帧数限制，则检查
        if max_frames is not None and frame_count >= max_frames:
            break
        
        frame_path = os.path.join(output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        extracted_frames.append(frame_path)
        frame_count += 1
    
    cap.release()
    return extracted_frames, frame_count

def main():
    st.title("🎯 YOLO-SAM2 视频分割系统")
    st.markdown("### 结合YOLO11检测和SAM2分割的智能视频处理系统")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 模型配置")
        
        # YOLO模型选择
        st.subheader("🎯 YOLO11 检测模型")
        
        # 直接使用训练好的辣椒检测模型
        trained_model_path = "/home/zcx/datasam2get/runs/detect/lajiao_detection_20250623_053550/weights/best.pt"
        yolo_model_path = trained_model_path  # 不再需要用户选择
        if os.path.exists(trained_model_path):
            st.success(f"✅ 使用辣椒专用检测模型")
            st.info(f"🌶️ 模型路径: {yolo_model_path}")
        else:
            st.error("❌ 辣椒检测模型未找到")
            yolo_model_path = "yolo11n.pt"
            st.warning("⚠️ 回退到预训练模型: yolo11n.pt")
        
        # 检测参数
        conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25, 0.05)
        
        st.divider()
        
        # SAM2模型选择
        st.subheader("🎨 SAM2 分割模型")
        
        # 默认使用SAM2.1-B (平衡型)
        sam2_model_name = "sam2.1_b.pt"
        st.success(f"✅ 使用SAM2.1-B模型 (平衡型)")
        
        # 高级选项 - 可选择其他模型
        with st.expander("🔧 高级选项 - 更换SAM2模型"):
            sam2_model_options = {
                "SAM2.1-B (平衡型，推荐)": "sam2.1_b.pt",
                "SAM2.1-L (大型，高精度)": "sam2.1_l.pt", 
                "SAM2.1-S (小型，快速)": "sam2.1_s.pt",
                "SAM2.1-T (最小，最快)": "sam2.1_t.pt"
            }
            
            selected_sam2 = st.selectbox(
                "选择SAM2模型", 
                options=list(sam2_model_options.keys()),
                index=0
            )
            sam2_model_name = sam2_model_options[selected_sam2]
            st.info(f"切换为: {selected_sam2}")
        
        # 处理模式
        st.subheader("🔄 处理模式")
        
        # 说明ultralytics SAM2的限制
        st.info("ℹ️ 注意：ultralytics的SAM2不支持真正的视频时间一致性分割，只能逐帧独立处理")
        
        # 只提供逐帧处理模式，因为直接视频分割实际上也是逐帧的
        st.write("**逐帧处理 (YOLO+SAM2) 🌶️** - 推荐模式")
        st.write("- 使用你训练的辣椒检测模型")
        st.write("- 对每一帧进行YOLO检测 + SAM2分割")
        st.write("- 虽然不是真正的视频分割，但结果更精确")
        
        processing_mode = "逐帧处理 (YOLO+SAM2) 🌶️"
        
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
            if st.button("🚀 开始YOLO-SAM2视频分割", type="primary"):
                with st.spinner("正在加载模型..."):
                    # 加载YOLO和SAM2模型
                    yolo_model = load_yolo_model(yolo_model_path)
                    if yolo_model is None:
                        st.error("YOLO模型加载失败，请检查模型路径")
                        return
                    
                    sam2_model = load_sam2_model(sam2_model_name)
                    if sam2_model is None:
                        st.error("SAM2模型加载失败")
                        return
                    
                    st.success(f"✅ 模型加载成功")
                
                # 逐帧处理模式
                    with st.spinner("正在提取视频帧..."):
                        # 提取帧
                        frame_dir = f"frames_{uuid.uuid4().hex[:8]}"
                        
                        # 先获取视频总帧数用于进度显示
                        cap_info = cv2.VideoCapture(temp_video_path)
                        total_video_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap_info.release()
                        
                        st.info(f"🎬 视频总帧数: {total_video_frames}，开始提取所有帧...")
                        
                        frame_paths, total_frames = extract_frames(temp_video_path, frame_dir)
                        st.success(f"✅ 提取了 {total_frames} 帧")
                    
                    # 选择参考帧进行检测
                    reference_frame = st.slider(
                        "选择参考帧进行目标检测", 
                        0, total_frames-1, 
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
                            
                            # 开始逐帧处理
                            if st.button("🎨 开始逐帧YOLO-SAM2处理"):
                                with st.spinner("正在进行逐帧YOLO-SAM2处理..."):
                                    video_segments = {}
                                    all_detections = {}
                                    
                                    # 显示处理信息
                                    st.info(f"📊 开始处理 {len(frame_paths)} 帧，这可能需要一些时间...")
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    for i, frame_path in enumerate(frame_paths):
                                        # 更新状态
                                        status_text.text(f"正在处理第 {i+1}/{len(frame_paths)} 帧...")
                                        
                                        # 处理每一帧
                                        frame_detections, frame_masks = process_frame_with_yolo_sam2(
                                            yolo_model, sam2_model, frame_path, conf_threshold
                                        )
                                        
                                        if frame_masks:
                                            video_segments[i] = frame_masks
                                        if frame_detections:
                                            all_detections[i] = frame_detections
                                        
                                        # 更新进度条
                                        progress_bar.progress((i + 1) / len(frame_paths))
                                    
                                    # 清除状态文本
                                    status_text.empty()
                                    
                                    if video_segments:
                                        st.success(f"✅ 逐帧处理完成！成功处理了 {len(video_segments)} 帧")
                                        st.info(f"📈 检测统计: 共在 {len(all_detections)} 帧中发现目标")
                                        
                                        # 保存结果到session state
                                        st.session_state['video_segments'] = video_segments
                                        st.session_state['frame_paths'] = frame_paths
                                        st.session_state['all_detections'] = all_detections
                                        st.session_state['frame_dir'] = frame_dir
                        else:
                            st.warning("⚠️ 未检测到任何目标，请调整置信度阈值或选择其他帧")
    
    with col2:
        st.subheader("🎨 分割结果预览")
        
        # 如果有分割结果，显示预览
        if 'video_segments' in st.session_state:
            video_segments = st.session_state['video_segments']
            frame_paths = st.session_state['frame_paths']
            all_detections = st.session_state.get('all_detections', {})
            
            # 帧选择器
            max_frame = len(frame_paths)-1 if frame_paths else 0
            if video_segments:
                max_frame = max(max_frame, max(video_segments.keys()))
            preview_frame = st.slider("预览帧", 0, max_frame, 0)
            
            if preview_frame in video_segments:
                masks = video_segments[preview_frame]
                detections = all_detections.get(preview_frame, [])
                
                # 可视化结果
                result_image = visualize_results(
                    frame_paths[preview_frame],
                    detections,
                    masks,
                    show_boxes=show_boxes,
                    show_masks=show_masks
                )
                
                st.image(result_image, caption=f"YOLO+SAM2分割结果 - 第 {preview_frame} 帧")
                
                # 显示掩码统计
                st.write("📊 分割统计:")
                for obj_id, mask in masks.items():
                    pixel_count = mask.sum()
                    st.write(f"- 辣椒目标 {obj_id}: {pixel_count} 像素")
                
                # 显示检测信息
                if detections:
                    st.write("🎯 检测信息:")
                    for i, det in enumerate(detections):
                        st.write(f"- 检测 {i+1}: 置信度 {det['confidence']:.3f}")
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
                    st.info("🔄 视频生成功能开发中...")
                    
        else:
            st.info("👆 请先上传视频并完成检测分割")
            
            # 显示示例
            st.markdown(f"""
            ### 🔄 处理流程
            
            **YOLO+SAM2逐帧处理 🌶️**
            1. 上传视频并提取所有帧
            2. 使用你的辣椒检测模型找到目标
            3. SAM2基于检测结果进行精确分割
            4. 查看结果并导出
            
            ### ⚠️ 重要说明
            - **ultralytics的SAM2不支持真正的视频时间一致性分割**
            - 只能对每一帧独立进行分割处理
            - 如需真正的视频分割，需要使用Meta原版SAM2
            
            ### ✨ 功能特性
            - 🌶️ **专用检测**: 使用你训练的辣椒检测模型
            - 🎯 **精确分割**: YOLO检测+SAM2像素级分割
            - 🎬 **完整处理**: 处理视频所有帧，不限制数量
            - 📊 **实时进度**: 显示处理进度和统计信息
            - 💾 **结果导出**: 支持掩码和视频导出
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
        keys_to_clear = ['video_segments', 'frame_paths', 'all_detections', 'frame_dir']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ 临时文件已清理")
        st.rerun()

if __name__ == "__main__":
    main() 