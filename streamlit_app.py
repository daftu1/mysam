import os
# 移除CPU强制设置，改为GPU版本
# os.environ["TORCHINDUCTOR_DISABLE"] = "1"
# os.environ["TORCH_COMPILE_DISABLE"] = "1"

import uuid
import shutil
import numpy as np
import cv2
import streamlit as st

st.set_page_config(layout="wide")
from PIL import Image
import torch
from moviepy import VideoFileClip
from torchvision.ops import masks_to_boxes
from sam2.build_sam import build_sam2_video_predictor
from streamlit_image_coordinates import streamlit_image_coordinates
import contextlib

# 初始化模型
@st.cache_resource
def load_sam2_model():
    checkpoint = os.path.join(os.path.expanduser("~"), "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")
    model_cfg = os.path.join("configs", "sam2.1", "sam2.1_hiera_b+.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 改为GPU版本
    return build_sam2_video_predictor(model_cfg, checkpoint, device=device, vos_optimized=True)

sam2_model = load_sam2_model()

st.title("🎬 SAM2 智能视频标注工具 & YOLO11格式导出")

VIDEO_DIR = "video_segments"
os.makedirs(VIDEO_DIR, exist_ok=True)

# 上传视频
uploaded_video = st.file_uploader("📁 上传完整原始视频", type=["mp4"])
if uploaded_video:
    original_path = "temp_uploaded.mp4"
    with open(original_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(original_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count // fps
    st.video(original_path)

    st.subheader("✂️ 视频裁剪")
    start_time = st.slider("起始时间（秒）", 0, duration - 1, 0)
    end_time = st.slider("结束时间（秒）", start_time + 1, duration, start_time + 5)

    if st.button("裁剪并保存片段"):
        clip = VideoFileClip(original_path).subclip(start_time, end_time)
        session_id = uuid.uuid4().hex[:8]
        segment_path = os.path.join(VIDEO_DIR, f"{session_id}.mp4")
        clip.write_videofile(segment_path, codec="libx264")
        st.success(f"✅ 视频裁剪完成: {segment_path}")
        st.session_state["segment_path"] = segment_path
        st.session_state["session_id"] = session_id
        cap.release()
        shutil.move(original_path, f"{original_path}.bak")

# 视频拆帧 + 加载状态
session_id = st.session_state.get("session_id", None)
segment_path = st.session_state.get("segment_path", None)

if session_id and segment_path:
    FRAME_DIR = f"frame_cache_{session_id}"
    os.makedirs(FRAME_DIR, exist_ok=True)

    if not os.listdir(FRAME_DIR):
        cap = cv2.VideoCapture(segment_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(FRAME_DIR, f"{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()
        st.success(f"✅ 共提取 {frame_idx} 帧至缓存目录 {FRAME_DIR}")

    frame_files = sorted(os.listdir(FRAME_DIR))
    
    # 添加工作模式选择
    st.subheader("🔧 工作模式选择")
    work_mode = st.radio("选择工作模式", ["初始标注", "预览100帧掩码", "修正特定帧"], horizontal=True)
    
    if work_mode == "初始标注":
        frame_index = st.session_state.get("frame_index", 0)
        current_frame_path = os.path.join(FRAME_DIR, frame_files[frame_index])
        current_img = Image.open(current_frame_path)

        frame_np = np.array(current_img.convert("RGB"))
        if "overlay_map" in st.session_state and frame_index in st.session_state["overlay_map"]:
            preview_img = st.session_state["overlay_map"][frame_index]
        else:
            preview_img = frame_np.copy()

        points = st.session_state.get("points", {}).get(frame_index, [])
        for x, y, l in points:
            color = (0, 255, 0) if l == 1 else (0, 0, 255)
            cv2.circle(preview_img, (int(x), int(y)), 5, color, -1)

        # 左图右控
        col1, col2 = st.columns([3, 1])

        with col1:
            # 预览帧并点击打点
            click = streamlit_image_coordinates(preview_img, key=f"frame_{frame_index}_{st.session_state.get('refresh_flag', False)}")
            st.image(preview_img, caption=f"点击帧添加点，当前帧: {frame_index}")
            if click:
                if "points" not in st.session_state:
                    st.session_state["points"] = {}
                if frame_index not in st.session_state["points"]:
                    st.session_state["points"][frame_index] = []
                # 记录当前点击坐标，等待用户选择正/负点
                st.session_state["last_click"] = (click["x"], click["y"])

        with col2:
            st.markdown("### 🎞️ 当前帧控制")
            frame_index = st.slider("帧位置", 0, len(frame_files) - 1, value=frame_index, key="frame_index")
            st.write(f"当前帧编号：**{frame_index}**")

            # 标签管理
            st.subheader("🏷️ 标签输入与确认")
            if "label_history" not in st.session_state:
                st.session_state["label_history"] = []
            label_input = st.text_input("✏️ 输入标签名", value="", placeholder="e.g. 猪肉")
            if label_input:
                suggestions = [l for l in st.session_state["label_history"] if l.startswith(label_input.lower())]
                if suggestions:
                    st.markdown("🔍 自动补全建议：" + ", ".join(suggestions[:5]))

            if st.button("✅ 确定标签"):
                label = label_input.strip().lower()
                if label:
                    if label not in st.session_state["label_history"]:
                        st.session_state["label_history"].append(label)
                    st.session_state["current_label"] = label
                    st.success(f"✅ 当前使用标签：`{label}`")
                else:
                    st.warning("⚠️ 标签不能为空")

            label = st.session_state.get("current_label", None)

            # 新增：保留点/去除点按钮，点击后将上次点击的点加入points
            if "last_click" in st.session_state:
                if st.button("保留点(正点)"):
                    if frame_index not in st.session_state["points"]:
                        st.session_state["points"][frame_index] = []
                    st.session_state["points"][frame_index].append((*st.session_state["last_click"], 1))
                    st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)
                    del st.session_state["last_click"]
                if st.button("去除点(负点)"):
                    if frame_index not in st.session_state["points"]:
                        st.session_state["points"][frame_index] = []
                    st.session_state["points"][frame_index].append((*st.session_state["last_click"], 0))
                    st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)
                    del st.session_state["last_click"]

            if st.button("🧹 清除当前帧所有点"):
                if "points" in st.session_state and frame_index in st.session_state["points"]:
                    st.session_state["points"][frame_index] = []
                    if "overlay_map" in st.session_state and frame_index in st.session_state["overlay_map"]:
                        del st.session_state["overlay_map"][frame_index]
                    st.success("✅ 当前帧点清除完毕")
                    st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)

            if st.button("🧼 清除所有帧的点"):
                st.session_state["points"] = {}
                if "overlay_map" in st.session_state:
                    st.session_state["overlay_map"] = {}
                st.success("✅ 所有帧点清除完毕")
                st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)

            if st.button("👁️ 预览当前帧标注"):
                if not points:
                    st.warning("⚠️ 当前帧无点击点")
                else:
                    pts = []
                    lbls = []
                    for x, y, l in points:
                        pts.append([x, y])
                        lbls.append(l)
                    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                        inference_state = sam2_model.init_state(segment_path)
                        frame_idx, obj_ids, masks = sam2_model.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_index,
                            obj_id=0,
                            points=pts,
                            labels=lbls,
                            clear_old_points=True,
                            normalize_coords=False
                        )
                        mask = masks[0, 0].cpu().numpy().astype(np.uint8)
                        if mask.sum() > 0:  # 修复：检查掩码是否有前景
                            box = masks_to_boxes(torch.tensor(mask[None]))[0].int().tolist()
                            x1, y1, x2, y2 = box
                            overlay = frame_np.copy()
                            overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([128, 128, 255]) * 0.5).astype(np.uint8)
                            for x, y, l in points:
                                cv2.circle(overlay, (int(x), int(y)), 5, (0,255,0) if l==1 else (0,0,255), -1)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                            st.image(overlay, caption=f"标签: {label} | 点数: {len(points)} | BBox: {box}")
                            if label is not None and label in st.session_state["label_history"]:
                                label_id = st.session_state["label_history"].index(label)
                                save_dir = f"yolo_labels_{session_id}"
                                os.makedirs(save_dir, exist_ok=True)
                                h, w = mask.shape
                                yolo_line = f"{label_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
                                label_file = os.path.join(save_dir, frame_files[frame_index].replace(".jpg", ".txt"))
                                with open(label_file, "w") as f:
                                    f.write(yolo_line)
                                st.success(f"✅ 单帧标签保存成功: {label_file}")
                        else:
                            st.warning("⚠️ 未检测到有效掩码")

            if st.button("⚡ 生成100帧掩码"):
                ref_points = st.session_state["points"].get(frame_index, [])
                if not ref_points or not label:
                    st.warning("⚠️ 首帧未打点或标签未设置")
                else:
                    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                        inference_state = sam2_model.init_state(segment_path)
                        pts = [[p[0], p[1]] for p in ref_points]
                        lbls = [p[2] for p in ref_points]
                        sam2_model.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=0,
                            obj_id=0,
                            points=pts,
                            labels=lbls,
                            clear_old_points=True,
                            normalize_coords=False
                        )
                        # 批量传播
                        video_segments = {}
                        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(inference_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                        st.session_state["video_segments"] = video_segments
                        st.session_state["inference_state"] = inference_state
                        st.success("✅ 100帧掩码生成完成，请切换到'预览100帧掩码'模式查看")

    elif work_mode == "预览100帧掩码":
        if "video_segments" not in st.session_state:
            st.warning("⚠️ 请先在'初始标注'模式下生成100帧掩码")
        else:
            st.subheader("🎞️ 100帧掩码预览")
            preview_frame_idx = st.slider("预览帧位置", 0, min(len(frame_files)-1, 99), 0, key="preview_frame_idx")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                frame_path = os.path.join(FRAME_DIR, frame_files[preview_frame_idx])
                img = np.array(Image.open(frame_path).convert("RGB"))
                
                video_segments = st.session_state["video_segments"]
                mask = video_segments.get(preview_frame_idx, {}).get(0, None)
                
                if mask is not None and mask.sum() > 0:
                    overlay = img.copy()
                    overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([128,128,255]) * 0.5).astype(np.uint8)
                    box = masks_to_boxes(torch.tensor(mask[None]))[0].int().tolist()
                    x1, y1, x2, y2 = box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                    st.image(overlay, caption=f"帧 {preview_frame_idx} - 掩码预览")
                else:
                    st.image(img, caption=f"帧 {preview_frame_idx} - 无掩码")
            
            with col2:
                st.markdown("### 🔧 操作选项")
                if st.button("选择此帧进行修正"):
                    st.session_state["refine_frame_idx"] = preview_frame_idx
                    st.success(f"✅ 已选择帧 {preview_frame_idx} 进行修正")
                
                label = st.session_state.get("current_label", "unknown")
                if st.button("📤 导出YOLO11格式"):
                    save_dir = f"yolo11_labels_{session_id}"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 保存classes.txt
                    with open(os.path.join(save_dir, "classes.txt"), "w") as f:
                        for i, label_name in enumerate(st.session_state.get("label_history", [])):
                            f.write(f"{label_name}\n")
                    
                    # 保存每帧的标注
                    exported_count = 0
                    for i, frame_file in enumerate(frame_files[:100]):  # 限制100帧
                        mask = video_segments.get(i, {}).get(0, None)
                        if mask is not None and mask.sum() > 0:
                            box = masks_to_boxes(torch.tensor(mask[None]))[0].int().tolist()
                            x1, y1, x2, y2 = box
                            h, w = mask.shape
                            label_id = st.session_state["label_history"].index(label) if label in st.session_state["label_history"] else 0
                            yolo_line = f"{label_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
                            label_file = os.path.join(save_dir, frame_file.replace(".jpg", ".txt"))
                            with open(label_file, "w") as f:
                                f.write(yolo_line)
                            exported_count += 1
                    
                    st.success(f"✅ YOLO11格式导出完成！共导出 {exported_count} 帧标注到 {save_dir}")

    elif work_mode == "修正特定帧":
        if "refine_frame_idx" not in st.session_state:
            st.warning("⚠️ 请先在'预览100帧掩码'模式下选择要修正的帧")
        else:
            refine_frame_idx = st.session_state["refine_frame_idx"]
            st.subheader(f"🔧 修正帧 {refine_frame_idx}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                frame_path = os.path.join(FRAME_DIR, frame_files[refine_frame_idx])
                img = np.array(Image.open(frame_path).convert("RGB"))
                
                # 显示当前掩码
                video_segments = st.session_state.get("video_segments", {})
                mask = video_segments.get(refine_frame_idx, {}).get(0, None)
                
                if mask is not None and mask.sum() > 0:
                    overlay = img.copy()
                    overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([128,128,255]) * 0.5).astype(np.uint8)
                    
                    # 显示修正点
                    refine_points = st.session_state.get("refine_points", [])
                    for x, y, l in refine_points:
                        color = (0, 255, 0) if l == 1 else (0, 0, 255)
                        cv2.circle(overlay, (int(x), int(y)), 8, color, -1)
                    
                    click = streamlit_image_coordinates(overlay, key=f"refine_{refine_frame_idx}")
                    if click:
                        st.session_state["last_refine_click"] = (click["x"], click["y"])
                    
                    st.image(overlay, caption=f"修正帧 {refine_frame_idx} - 点击添加修正点")
                else:
                    st.image(img, caption=f"帧 {refine_frame_idx} - 无掩码")
            
            with col2:
                st.markdown("### 🔧 修正操作")
                
                if "last_refine_click" in st.session_state:
                    if st.button("添加正点修正"):
                        if "refine_points" not in st.session_state:
                            st.session_state["refine_points"] = []
                        st.session_state["refine_points"].append((*st.session_state["last_refine_click"], 1))
                        del st.session_state["last_refine_click"]
                    
                    if st.button("添加负点修正"):
                        if "refine_points" not in st.session_state:
                            st.session_state["refine_points"] = []
                        st.session_state["refine_points"].append((*st.session_state["last_refine_click"], 0))
                        del st.session_state["last_refine_click"]
                
                if st.button("应用修正并重新传播"):
                    refine_points = st.session_state.get("refine_points", [])
                    if refine_points:
                        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                            inference_state = st.session_state["inference_state"]
                            pts = [[p[0], p[1]] for p in refine_points]
                            lbls = [p[2] for p in refine_points]
                            
                            # 在指定帧添加修正点
                            sam2_model.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=refine_frame_idx,
                                obj_id=0,
                                points=pts,
                                labels=lbls,
                                clear_old_points=False,
                                normalize_coords=False
                            )
                            
                            # 重新传播
                            video_segments = {}
                            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(inference_state):
                                video_segments[out_frame_idx] = {
                                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                    for i, out_obj_id in enumerate(out_obj_ids)
                                }
                            
                            st.session_state["video_segments"] = video_segments
                            st.session_state["refine_points"] = []
                            st.success("✅ 修正完成并重新传播！")
                    else:
                        st.warning("⚠️ 请先添加修正点")
                
                if st.button("清除修正点"):
                    st.session_state["refine_points"] = []
                    st.success("✅ 修正点已清除")
