import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil

from sam2.build_sam import build_sam2_video_predictor

# 设置设备
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print("\n⚠️ MPS支持仍在开发中，建议使用CUDA。")

# 加载模型
sam2_checkpoint = os.path.join(os.path.expanduser("~"), "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")
model_cfg = os.path.join("configs", "sam2.1", "sam2.1_hiera_b+.yaml")
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# 视频帧目录（仅处理前20帧）
video_dir = "/home/zcx/datasam2get/frame_cache_5c652146"
temp_video_dir = "/home/zcx/datasam2get/frame_cache_subset"
os.makedirs(temp_video_dir, exist_ok=True)

frame_names = sorted([
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
], key=lambda p: int(os.path.splitext(p)[0]))[:20]

for name in frame_names:
    shutil.copy(os.path.join(video_dir, name), os.path.join(temp_video_dir, name))

# 初始化推理状态
frame_idx = 0
ann_frame_idx = 0
ann_obj_id = 1
inference_state = predictor.init_state(video_path=temp_video_dir)

# ========== Streamlit UI ==========
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")
st.title("🖱️ 视频帧点击标注系统 + 掩码预览")

# 初始化点击点
if "click_points" not in st.session_state:
    st.session_state.click_points = []
if "click_labels" not in st.session_state:
    st.session_state.click_labels = []
if "mask_preview" not in st.session_state:
    st.session_state.mask_preview = None
if "frame_image" not in st.session_state:
    st.session_state.frame_image = None

uploaded_frame = st.file_uploader("上传一帧图像（jpg/png）", type=["jpg", "jpeg", "png"])
if uploaded_frame:
    file_bytes = np.asarray(bytearray(uploaded_frame.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    st.session_state.frame_image = frame_rgb.copy()

    label_mode = st.radio("请选择点击模式", ["保留（正点）", "去除（负点）"])
    click = streamlit_image_coordinates(image, key="annotator")

    if click:
        point = [click["x"], click["y"]]
        label = 1 if label_mode == "保留（正点）" else 0
        st.session_state.click_points.append(point)
        st.session_state.click_labels.append(label)

    st.markdown("### 当前点击点:")
    st.write(st.session_state.click_points)
    st.write("### 标签:")
    st.write(st.session_state.click_labels)

    if st.button("🚀 应用到 Predictor 并生成掩码"):
        if predictor is None or inference_state is None:
            st.warning("⚠️ predictor 或 inference_state 未初始化")
        else:
            points = np.array(st.session_state.click_points, dtype=np.float32)
            labels = np.array(st.session_state.click_labels, dtype=np.int32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            h, w = st.session_state.frame_image.shape[:2]
            mask_resized = cv2.resize(mask, (w, h))
            colored_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(st.session_state.frame_image, 0.6, colored_mask, 0.4, 0)

            st.session_state.mask_preview = overlay
            st.success("🎉 掩码已生成并应用！")


    if st.session_state.mask_preview is not None:
        st.image(st.session_state.mask_preview, caption="掩码预览", channels="RGB", use_column_width=True)

    if st.button("🗑️ 清除所有点击点"):
        st.session_state.click_points = []
        st.session_state.click_labels = []
        st.session_state.mask_preview = None
