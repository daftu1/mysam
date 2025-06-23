import os
# 禁用torch编译器和inductor来避免CUDA图形错误
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

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

# 智能设备检测函数
def get_device():
    """智能检测最佳可用设备：MPS > CUDA > CPU"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_autocast_device():
    """获取用于autocast的设备类型字符串"""
    device = get_device()
    if device.type == "mps":
        return "cpu"  # MPS在autocast中使用cpu模式
    else:
        return device.type

# 边界框生成函数
def get_tight_bbox(mask):
    """
    基于像素分布的紧密边界框
    """
    if mask.sum() == 0:
        return None
    
    # 确保掩码是2D
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    if len(mask.shape) != 2:
        return None
    
    # 找到所有前景像素的位置
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0:
        return None
    
    # 计算边界框
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    return [x_min, y_min, x_max + 1, y_max + 1]

def get_contour_bbox(mask):
    """
    基于轮廓的更精确边界框
    """
    if mask.sum() == 0:
        return None
    
    # 确保掩码是2D且为uint8类型
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    if len(mask.shape) != 2:
        return None
    
    # 确保掩码是二值的uint8类型
    mask_uint8 = (mask > 0).astype(np.uint8)
    
    # 找到轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return [x, y, x + w, y + h]

def get_bbox_by_method(mask, method="轮廓边界框"):
    """
    根据选择的方法生成边界框
    """
    if mask.sum() == 0:
        return None
    
    # 确保掩码是2D
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    if len(mask.shape) != 2:
        return None
    
    if method == "轮廓边界框":
        box = get_contour_bbox(mask)
        if box is None:
            box = get_tight_bbox(mask)
        return box
    elif method == "紧密边界框":
        return get_tight_bbox(mask)
    elif method == "传统方法":
        try:
            # 确保掩码是2D且为布尔类型
            mask_2d = (mask > 0).astype(bool)
            box = masks_to_boxes(torch.tensor(mask_2d[None]))[0].int().tolist()
            return box
        except:
            return get_tight_bbox(mask)
    else:
        return get_contour_bbox(mask)

# 初始化模型
@st.cache_resource
def load_sam2_model():
    checkpoint = os.path.join(os.path.expanduser("~"), "mysam", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    # 智能设备检测：优先使用MPS加速 (Apple Silicon)，然后CUDA，最后CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 使用Apple Silicon MPS加速")
        st.success("🚀 正在使用Apple Silicon MPS加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 使用CUDA GPU加速")
        st.success("🚀 正在使用CUDA GPU加速")
    else:
        device = torch.device("cpu")
        print("💻 使用CPU处理")
        st.info("💻 正在使用CPU处理")
    
    return build_sam2_video_predictor(model_cfg, checkpoint, device=device, vos_optimized=True)

sam2_model = load_sam2_model()

st.title("🎬 SAM2 智能视频标注工具 & YOLO11格式导出")

# 侧边栏：数据集管理
with st.sidebar:
    st.header("📁 数据集管理")
    
    # 数据集名称设置
    dataset_name = st.text_input(
        "数据集名称", 
        value=st.session_state.get("dataset_name", "lajiao_dataset"),
        help="所有视频的标注数据将统一保存到这个数据集中"
    )
    st.session_state["dataset_name"] = dataset_name
    
    # 显示当前数据集统计
    unified_frame_dir = f"frames_{dataset_name}"
    unified_label_dir = f"labels_{dataset_name}"
    
    if os.path.exists(unified_frame_dir):
        frame_count = len([f for f in os.listdir(unified_frame_dir) if f.endswith('.jpg')])
        st.metric("📷 总图像数", frame_count)
    else:
        st.metric("📷 总图像数", 0)
    
    if os.path.exists(unified_label_dir):
        label_count = len([f for f in os.listdir(unified_label_dir) if f.endswith('.txt')])
        st.metric("🏷️ 总标注数", label_count)
    else:
        st.metric("🏷️ 总标注数", 0)
    
    # 数据集操作按钮
    if st.button("🔄 刷新统计"):
        st.rerun()
    
    if st.button("📦 导出当前数据集"):
        if os.path.exists(unified_frame_dir) and os.path.exists(unified_label_dir):
            import zipfile
            zip_path = f"{dataset_name}_export.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # 添加图像文件
                for file in os.listdir(unified_frame_dir):
                    if file.endswith('.jpg'):
                        zipf.write(os.path.join(unified_frame_dir, file), f"images/{file}")
                # 添加标注文件
                for file in os.listdir(unified_label_dir):
                    if file.endswith('.txt'):
                        zipf.write(os.path.join(unified_label_dir, file), f"labels/{file}")
            st.success(f"✅ 数据集已导出为: {zip_path}")
        else:
            st.warning("⚠️ 数据集为空，无法导出")
    
    st.divider()
    
    # 边界框算法选择
    st.header("⚙️ 算法配置")
    bbox_method = st.selectbox(
        "边界框算法",
        ["传统方法", "轮廓边界框", "紧密边界框"],
        help="传统方法：最小外接矩形，效果最好\\n轮廓边界框：基于物体轮廓\\n紧密边界框：基于像素分布"
    )
    st.session_state["bbox_method"] = bbox_method

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
    
    # 防止除零错误
    if fps == 0:
        fps = 30  # 默认帧率
        st.warning("⚠️ 无法获取视频帧率，使用默认值30fps")
    
    duration = frame_count // fps
    st.video(original_path)

    st.subheader("✂️ 视频裁剪")
    start_time = st.slider("起始时间（秒）", 0, duration - 1, 0)
    end_time = st.slider("结束时间（秒）", start_time + 1, duration, start_time + 5)

    if st.button("裁剪并保存片段"):
        clip = VideoFileClip(original_path).subclipped(start_time, end_time)
        
        # 使用统一的数据集名称，而不是随机session_id
        dataset_name = st.session_state.get("dataset_name", "lajiao_dataset")
        session_id = uuid.uuid4().hex[:8]  # 只用于视频文件命名
        segment_path = os.path.join(VIDEO_DIR, f"{session_id}.mp4")
        clip.write_videofile(segment_path, codec="libx264")
        st.success(f"✅ 视频裁剪完成: {segment_path}")
        st.session_state["segment_path"] = segment_path
        st.session_state["current_session_id"] = session_id  # 当前视频的ID
        cap.release()
        shutil.move(original_path, f"{original_path}.bak")

# 视频拆帧 + 加载状态
current_session_id = st.session_state.get("current_session_id", None)
segment_path = st.session_state.get("segment_path", None)
dataset_name = st.session_state.get("dataset_name", "lajiao_dataset")

# 使用统一的目录名称
UNIFIED_FRAME_DIR = f"frames_{dataset_name}"
UNIFIED_LABEL_DIR = f"labels_{dataset_name}"

if current_session_id and segment_path:
    # 创建统一的数据目录
    os.makedirs(UNIFIED_FRAME_DIR, exist_ok=True)
    os.makedirs(UNIFIED_LABEL_DIR, exist_ok=True)
    
    # 为当前视频创建临时帧目录
    FRAME_DIR = f"frame_cache_{current_session_id}"
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

            # 新增：保留点/去除点按钮
            if "last_click" in st.session_state:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("保留点(正点)"):
                        if frame_index not in st.session_state["points"]:
                            st.session_state["points"][frame_index] = []
                        st.session_state["points"][frame_index].append((*st.session_state["last_click"], 1))
                        st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)
                        del st.session_state["last_click"]
                        st.rerun()
                with col_b:
                    if st.button("去除点(负点)"):
                        if frame_index not in st.session_state["points"]:
                            st.session_state["points"][frame_index] = []
                        st.session_state["points"][frame_index].append((*st.session_state["last_click"], 0))
                        st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)
                        del st.session_state["last_click"]
                        st.rerun()

            if st.button("🧹 清除当前帧所有点"):
                if "points" in st.session_state and frame_index in st.session_state["points"]:
                    st.session_state["points"][frame_index] = []
                st.success("✅ 当前帧点清除完毕")
                st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)

            if st.button("🧼 清除所有帧的点"):
                st.session_state["points"] = {}
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
                    
                    # 添加调试信息
                    st.write(f"🔍 调试信息：点击点数量: {len(points)}")
                    st.write(f"🔍 点坐标: {pts}")
                    st.write(f"🔍 点标签: {lbls}")
                    
                    try:
                        # 清理GPU/MPS内存
                        device = get_device()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps":
                            torch.mps.empty_cache()
                        
                        with torch.autocast(get_autocast_device()):
                            # 限制视频长度为100帧来节省内存
                            temp_video_path = f"temp_video_100frames_{current_session_id}.mp4"
                            if not os.path.exists(temp_video_path):
                                # 创建只有前100帧的临时视频
                                cap = cv2.VideoCapture(segment_path)
                                fps = int(cap.get(cv2.CAP_PROP_FPS))
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                                
                                frame_count = 0
                                while frame_count < 100:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    out.write(frame)
                                    frame_count += 1
                                
                                cap.release()
                                out.release()
                                st.write(f"✅ 创建100帧临时视频: {frame_count} 帧")
                            
                            # 使用帧目录而不是视频文件，就像GPU.py中一样
                            inference_state = sam2_model.init_state(video_path=FRAME_DIR)
                            st.write("✅ 推理状态初始化成功")
                            
                            _, _, mask_logits = sam2_model.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=frame_index,
                                obj_id=1,  # 使用obj_id=1，就像GPU.py中一样
                                points=np.array(pts, dtype=np.float32),
                                labels=np.array(lbls, dtype=np.int32),
                            )
                            st.write(f"✅ SAM2预测完成，掩码形状: {mask_logits.shape}")
                            
                            # 直接使用GPU.py中的逻辑：(mask_logits[0] > 0)
                            binary_mask = (mask_logits[0] > 0).cpu().numpy().squeeze()
                            st.write(f"🔍 掩码统计: 形状={binary_mask.shape}, 前景像素={binary_mask.sum()}")
                            
                            # 立即清理GPU/MPS内存
                            del mask_logits
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            elif device.type == "mps":
                                torch.mps.empty_cache()
                            
                            if binary_mask.sum() > 0:
                                # 使用用户选择的边界框算法，传统方法效果最好
                                bbox_method = st.session_state.get("bbox_method", "传统方法")
                                box = get_bbox_by_method(binary_mask, bbox_method)
                                
                                if box is not None:
                                    x1, y1, x2, y2 = box
                                    overlay = frame_np.copy()
                                    overlay[binary_mask == 1] = (overlay[binary_mask == 1] * 0.5 + np.array([128, 128, 255]) * 0.5).astype(np.uint8)
                                    for x, y, l in points:
                                        cv2.circle(overlay, (int(x), int(y)), 5, (0,255,0) if l==1 else (0,0,255), -1)
                                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                                    st.image(overlay, caption=f"标签: {label} | 点数: {len(points)} | BBox: {box} | 方法: {bbox_method}")
                                    
                                    if label is not None and label in st.session_state["label_history"]:
                                        # 保存到统一的数据集目录
                                        label_id = st.session_state["label_history"].index(label)
                                        h, w = binary_mask.shape
                                        yolo_line = f"{label_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
                                        
                                        # 生成唯一的文件名（避免重复）
                                        base_name = frame_files[frame_index].replace(".jpg", "")
                                        unique_name = f"{current_session_id}_{base_name}"
                                        
                                        # 保存图像到统一目录
                                        src_img_path = os.path.join(FRAME_DIR, frame_files[frame_index])
                                        dst_img_path = os.path.join(UNIFIED_FRAME_DIR, f"{unique_name}.jpg")
                                        shutil.copy2(src_img_path, dst_img_path)
                                        
                                        # 保存标注到统一目录
                                        label_file = os.path.join(UNIFIED_LABEL_DIR, f"{unique_name}.txt")
                                        with open(label_file, "w") as f:
                                            f.write(yolo_line)
                                        
                                        # 更新类别文件
                                        classes_file = os.path.join(UNIFIED_LABEL_DIR, "classes.txt")
                                        with open(classes_file, "w") as f:
                                            for label_name in st.session_state["label_history"]:
                                                f.write(f"{label_name}\n")
                                        
                                        st.success(f"✅ 单帧数据已添加到数据集: {unique_name}")
                                else:
                                    st.warning("⚠️ 无法生成有效边界框")
                            else:
                                st.warning("⚠️ 掩码为空，可能是点击位置不合适或模型预测失败")
                                st.write("💡 建议：尝试点击目标物体的中心区域，或添加更多正点")
                    
                    except Exception as e:
                        st.error(f"❌ 预测过程出错: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            if st.button("⚡ 生成100帧掩码"):
                ref_points = st.session_state["points"].get(frame_index, [])
                if not ref_points or not label:
                    st.warning("⚠️ 当前帧未打点或标签未设置")
                else:
                    with st.spinner("正在生成100帧掩码..."):
                        try:
                            # 清理GPU/MPS内存
                            device = get_device()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            elif device.type == "mps":
                                torch.mps.empty_cache()
                                
                            with torch.autocast(get_autocast_device()):
                                # 使用帧目录，就像GPU.py中一样
                                inference_state = sam2_model.init_state(video_path=FRAME_DIR)
                                pts = [[p[0], p[1]] for p in ref_points]
                                lbls = [p[2] for p in ref_points]
                                
                                st.write(f"🔍 使用帧 {frame_index} 作为参考帧")
                                st.write(f"🔍 参考点: {pts}, 标签: {lbls}")
                                
                                # 使用GPU.py中的逻辑
                                _, _, _ = sam2_model.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=frame_index,
                                    obj_id=1,  # 使用obj_id=1
                                    points=np.array(pts, dtype=np.float32),
                                    labels=np.array(lbls, dtype=np.int32),
                                )
                                
                                # 批量传播，使用GPU.py中的逻辑
                                video_segments = {}
                                for i, obj_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
                                    video_segments[i] = {
                                        obj_id: (mask_logits[j] > 0).cpu().numpy()
                                        for j, obj_id in enumerate(obj_ids)
                                    }
                                    # 立即清理每帧的GPU/MPS内存
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                    elif device.type == "mps":
                                        torch.mps.empty_cache()
                                    # 限制只处理前100帧
                                    if i >= 99:
                                        break
                                
                                # 统计生成的掩码
                                valid_masks = sum(1 for frame_data in video_segments.values() 
                                                for mask in frame_data.values() if mask.sum() > 0)
                                st.write(f"✅ 成功生成 {valid_masks} 个有效掩码")
                                
                                st.session_state["video_segments"] = video_segments
                                st.session_state["inference_state"] = inference_state
                                st.session_state["reference_frame"] = frame_index
                        
                        except Exception as e:
                            st.error(f"❌ 生成100帧掩码时出错: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    st.success("✅ 100帧掩码生成完成，请切换到'预览100帧掩码'模式查看")

    elif work_mode == "预览100帧掩码":
        if "video_segments" not in st.session_state:
            st.warning("⚠️ 请先在'初始标注'模式下生成100帧掩码")
        else:
            st.subheader("🎞️ 100帧掩码预览")
            max_frames = min(len(frame_files), 100)
            preview_frame_idx = st.slider("预览帧位置", 0, max_frames-1, 0, key="preview_frame_idx")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 确保每次都重新加载帧
                frame_path = os.path.join(FRAME_DIR, frame_files[preview_frame_idx])
                img = np.array(Image.open(frame_path).convert("RGB"))
                
                video_segments = st.session_state["video_segments"]
                mask = video_segments.get(preview_frame_idx, {}).get(1, None)  # 使用obj_id=1
                
                # 添加调试信息
                st.write(f"🔍 当前预览帧: {preview_frame_idx}, 帧文件: {frame_files[preview_frame_idx]}")
                st.write(f"🔍 掩码状态: {'有掩码' if mask is not None and mask.sum() > 0 else '无掩码'}")
                
                if mask is not None and mask.sum() > 0:
                    overlay = img.copy()
                    
                    # 确保掩码形状与图像匹配
                    mask_valid = True
                    if len(mask.shape) == 3:
                        mask = mask.squeeze()
                    if len(mask.shape) == 1:
                        # 如果掩码是1D，需要重新reshape
                        h, w = img.shape[:2]
                        if mask.size == h * w:
                            mask = mask.reshape(h, w)
                        else:
                            st.error(f"掩码大小不匹配: 掩码大小={mask.size}, 图像大小={h}x{w}")
                            mask_valid = False
                    
                    if mask_valid and len(mask.shape) == 2:
                        # 确保掩码是二值的
                        mask = (mask > 0).astype(np.uint8)
                        
                        # 检查掩码和图像形状是否匹配
                        if mask.shape[:2] == img.shape[:2]:
                            overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([128,128,255]) * 0.5).astype(np.uint8)
                            
                            # 使用更精确的边界框算法，传统方法效果最好
                            bbox_method = st.session_state.get("bbox_method", "传统方法")
                            box = get_bbox_by_method(mask, bbox_method)
                            
                            if box is not None:
                                x1, y1, x2, y2 = box
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                                st.image(overlay, caption=f"帧 {preview_frame_idx} - 掩码预览 | BBox: {box}")
                            else:
                                st.image(overlay, caption=f"帧 {preview_frame_idx} - 掩码预览 (无边界框)")
                        else:
                            st.error(f"掩码形状 {mask.shape} 与图像形状 {img.shape[:2]} 不匹配")
                            st.image(img, caption=f"帧 {preview_frame_idx} - 掩码形状错误")
                    else:
                        st.image(img, caption=f"帧 {preview_frame_idx} - 掩码无效")
                else:
                    st.image(img, caption=f"帧 {preview_frame_idx} - 无掩码")
            
            with col2:
                st.markdown("### 🔧 操作选项")
                if st.button("选择此帧进行修正"):
                    st.session_state["refine_frame_idx"] = preview_frame_idx
                    st.success(f"✅ 已选择帧 {preview_frame_idx} 进行修正")
                
                label = st.session_state.get("current_label", "unknown")
                if st.button("📤 添加到统一数据集"):
                    if label not in st.session_state.get("label_history", []):
                        st.warning("⚠️ 请先设置有效标签")
                    else:
                        # 批量添加到统一数据集
                        exported_count = 0
                        for i in range(max_frames):
                            mask = video_segments.get(i, {}).get(1, None)  # 使用obj_id=1
                            if mask is not None and mask.sum() > 0:
                                # 使用更精确的边界框算法，传统方法效果最好
                                bbox_method = st.session_state.get("bbox_method", "传统方法")
                                box = get_bbox_by_method(mask, bbox_method)
                                
                                if box is not None:
                                    x1, y1, x2, y2 = box
                                    # 确保掩码是2D的
                                    if len(mask.shape) == 3:
                                        mask = mask.squeeze()
                                    if len(mask.shape) == 2:
                                        h, w = mask.shape
                                    else:
                                        h, w = img.shape[:2]  # 使用图像尺寸作为后备
                                    
                                    label_id = st.session_state["label_history"].index(label)
                                    yolo_line = f"{label_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
                                    
                                    # 生成唯一的文件名
                                    base_name = frame_files[i].replace(".jpg", "")
                                    unique_name = f"{current_session_id}_{base_name}"
                                    
                                    # 保存图像到统一目录
                                    src_img_path = os.path.join(FRAME_DIR, frame_files[i])
                                    dst_img_path = os.path.join(UNIFIED_FRAME_DIR, f"{unique_name}.jpg")
                                    shutil.copy2(src_img_path, dst_img_path)
                                    
                                    # 保存标注到统一目录
                                    label_file = os.path.join(UNIFIED_LABEL_DIR, f"{unique_name}.txt")
                                    with open(label_file, "w") as f:
                                        f.write(yolo_line)
                                    
                                    exported_count += 1
                        
                        # 更新类别文件
                        classes_file = os.path.join(UNIFIED_LABEL_DIR, "classes.txt")
                        with open(classes_file, "w") as f:
                            for label_name in st.session_state["label_history"]:
                                f.write(f"{label_name}\n")
                        
                        st.success(f"✅ 已添加 {exported_count} 帧数据到统一数据集 '{dataset_name}'")

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
                mask = video_segments.get(refine_frame_idx, {}).get(1, None)  # 使用obj_id=1
                
                if mask is not None and mask.sum() > 0:
                    overlay = img.copy()
                    
                    # 确保掩码形状与图像匹配
                    if len(mask.shape) == 3:
                        mask = mask.squeeze()
                    if len(mask.shape) == 1:
                        # 如果掩码是1D，需要重新reshape
                        h, w = img.shape[:2]
                        if mask.size == h * w:
                            mask = mask.reshape(h, w)
                        else:
                            st.error(f"掩码大小不匹配: 掩码大小={mask.size}, 图像大小={h}x{w}")
                            st.image(img, caption=f"帧 {refine_frame_idx} - 掩码形状错误")
                    
                    # 确保掩码是二值的
                    mask = (mask > 0).astype(np.uint8)
                    
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
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("添加正点修正"):
                            if "refine_points" not in st.session_state:
                                st.session_state["refine_points"] = []
                            st.session_state["refine_points"].append((*st.session_state["last_refine_click"], 1))
                            del st.session_state["last_refine_click"]
                            st.rerun()
                    
                    with col_b:
                        if st.button("添加负点修正"):
                            if "refine_points" not in st.session_state:
                                st.session_state["refine_points"] = []
                            st.session_state["refine_points"].append((*st.session_state["last_refine_click"], 0))
                            del st.session_state["last_refine_click"]
                            st.rerun()
                
                if st.button("应用修正并重新传播"):
                    refine_points = st.session_state.get("refine_points", [])
                    if refine_points:
                        with st.spinner("正在应用修正并重新传播..."):
                            with torch.autocast(get_autocast_device()):
                                inference_state = st.session_state["inference_state"]
                                pts = [[p[0], p[1]] for p in refine_points]
                                lbls = [p[2] for p in refine_points]
                                
                                # 在指定帧添加修正点
                                sam2_model.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=refine_frame_idx,
                                    obj_id=1,  # 使用obj_id=1
                                    points=pts,
                                    labels=lbls,
                                )
                                
                                # 重新传播
                                video_segments = {}
                                for i, obj_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
                                    video_segments[i] = {
                                        obj_id: (mask_logits[j] > 0).cpu().numpy()
                                        for j, obj_id in enumerate(obj_ids)
                                    }
                                    # 限制只处理前100帧
                                    if i >= 99:
                                        break
                                
                                st.session_state["video_segments"] = video_segments
                                st.session_state["refine_points"] = []
                        st.success("✅ 修正完成并重新传播！")
                    else:
                        st.warning("⚠️ 请先添加修正点")
                
                if st.button("清除修正点"):
                    st.session_state["refine_points"] = []
                    st.success("✅ 修正点已清除")
                    st.rerun()

# 侧边栏显示当前状态
with st.sidebar:
    st.header("📊 当前状态")
    if current_session_id:
        st.write(f"会话ID: {current_session_id}")
        device = get_device()
        if device.type == "mps":
            st.write("设备: 🚀 Apple Silicon MPS")
        elif device.type == "cuda":
            st.write("设备: 🚀 CUDA GPU")
        else:
            st.write("设备: 💻 CPU")
        
        # GPU内存监控
        if device.type == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
            st.write(f"GPU总内存: {gpu_memory:.1f} GB")
            st.write(f"GPU已分配: {gpu_allocated:.1f} GB")
            st.write(f"GPU缓存: {gpu_cached:.1f} GB")
            
            if st.button("🧹 清理GPU内存"):
                torch.cuda.empty_cache()
                st.success("✅ GPU内存已清理")
                st.rerun()
        
        # MPS内存监控（Apple Silicon）
        elif device.type == "mps":
            if st.button("🧹 清理MPS内存"):
                torch.mps.empty_cache()
                st.success("✅ MPS内存已清理")
                st.rerun()
        
        if "video_segments" in st.session_state:
            total_masks = sum(1 for frame_data in st.session_state["video_segments"].values() 
                            for mask in frame_data.values() if mask.sum() > 0)
            st.write(f"有效掩码数: {total_masks}")
        if "label_history" in st.session_state:
            st.write("标签历史:")
            for i, label in enumerate(st.session_state["label_history"]):
                st.write(f"{i}: {label}")
    
    # 设置GPU内存管理
    st.header("⚙️ 内存设置")
    if st.button("设置GPU内存优化"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        st.success("✅ 已设置GPU内存优化")
        st.write("重启应用生效") 