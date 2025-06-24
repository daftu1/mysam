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

st.title("🎬 SAM2 多目标专用视频标注工具 & YOLO11格式导出")

# ===== 多目标数据结构管理 =====
def init_multi_target_data():
    """初始化多目标数据结构"""
    if "multi_target_data" not in st.session_state:
        st.session_state["multi_target_data"] = {}
    if "next_obj_id" not in st.session_state:
        st.session_state["next_obj_id"] = 1
    if "current_active_label" not in st.session_state:
        st.session_state["current_active_label"] = None
    if "current_active_object" not in st.session_state:
        st.session_state["current_active_object"] = None

def add_new_label(label_name):
    """添加新标签"""
    if label_name and label_name not in st.session_state["multi_target_data"]:
        st.session_state["multi_target_data"][label_name] = {"objects": {}}
        return True
    return False

def add_new_object_to_label(label_name):
    """为指定标签添加新目标"""
    if label_name in st.session_state["multi_target_data"]:
        obj_id = st.session_state["next_obj_id"]
        st.session_state["multi_target_data"][label_name]["objects"][obj_id] = {
            "ann_obj_id": obj_id,
            "points": {},  # {frame_idx: [(x, y, label), ...]}
            "masks": {},   # 存储生成的掩码
        }
        st.session_state["next_obj_id"] += 1
        return obj_id
    return None

def get_current_active_object():
    """获取当前激活的目标数据"""
    active_label = st.session_state["current_active_label"]
    active_obj = st.session_state["current_active_object"]
    
    if (active_label and active_obj and 
        active_label in st.session_state["multi_target_data"] and
        active_obj in st.session_state["multi_target_data"][active_label]["objects"]):
        return st.session_state["multi_target_data"][active_label]["objects"][active_obj]
    return None

def add_point_to_current_object(frame_idx, x, y, label):
    """为当前激活的目标添加点"""
    obj_data = get_current_active_object()
    if obj_data is not None:
        if frame_idx not in obj_data["points"]:
            obj_data["points"][frame_idx] = []
        obj_data["points"][frame_idx].append((x, y, label))
        return True
    return False

# 初始化多目标数据
init_multi_target_data()

def preview_all_targets_only(frame_index, frame_np, frame_files, current_session_id):
    """只预览当前帧所有目标的分割效果，不保存数据"""
    multi_target_data = st.session_state.get("multi_target_data", {})
    FRAME_DIR = f"frame_cache_{current_session_id}"
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    overlay = frame_np.copy()
    
    try:
        device = get_device()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        
        with torch.autocast(get_autocast_device()):
            inference_state = sam2_model.init_state(video_path=FRAME_DIR)
            
            color_idx = 0
            processed_targets = 0
            for label_name, label_data in multi_target_data.items():
                for obj_id, obj_data in label_data["objects"].items():
                    if frame_index in obj_data["points"] and obj_data["points"][frame_index]:
                        obj_points = obj_data["points"][frame_index]
                        pts = [[p[0], p[1]] for p in obj_points]
                        lbls = [p[2] for p in obj_points]
                        
                        # 为每个目标生成掩码，使用正确的obj_id
                        _, _, mask_logits = sam2_model.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_index,
                            obj_id=obj_id,
                            points=np.array(pts, dtype=np.float32),
                            labels=np.array(lbls, dtype=np.int32),
                        )
                        
                        binary_mask = (mask_logits[0] > 0).cpu().numpy().squeeze()
                        
                        if binary_mask.sum() > 0:
                            # 为每个目标使用不同颜色
                            target_color = colors[color_idx % len(colors)]
                            overlay[binary_mask == 1] = (overlay[binary_mask == 1] * 0.6 + np.array(target_color) * 0.4).astype(np.uint8)
                            
                            # 绘制边界框
                            bbox_method = st.session_state.get("bbox_method", "传统方法")
                            box = get_bbox_by_method(binary_mask, bbox_method)
                            
                            if box is not None:
                                x1, y1, x2, y2 = box
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), target_color, 2)
                                # 标注目标信息
                                cv2.putText(overlay, f"{label_name}-{obj_id}", (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)
                            
                            processed_targets += 1
                        
                        # 绘制标注点
                        for i, (x, y, l) in enumerate(obj_points):
                            color = colors[color_idx % len(colors)]
                            radius = 10 if l == 1 else 5
                            cv2.circle(overlay, (int(x), int(y)), radius, color, -1)
                            cv2.putText(overlay, str(obj_id), (int(x)+10, int(y)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        color_idx += 1
                        
                        # 清理内存
                        del mask_logits
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps":
                            torch.mps.empty_cache()
            
            st.image(overlay, caption=f"全部目标预览 - 已处理{processed_targets}个目标")
            st.info(f"👁️ 预览完成！显示了{processed_targets}个目标的分割效果")
    
    except Exception as e:
        st.error(f"❌ 预览失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def save_all_targets_in_frame(frame_index, frame_np, frame_files, current_session_id):
    """保存当前帧所有目标的数据"""
    multi_target_data = st.session_state.get("multi_target_data", {})
    FRAME_DIR = f"frame_cache_{current_session_id}"
    UNIFIED_FRAME_DIR = f"frames_{st.session_state.get('dataset_name', 'lajiao_dataset')}"
    UNIFIED_LABEL_DIR = f"labels_{st.session_state.get('dataset_name', 'lajiao_dataset')}"
    
    all_labels = list(multi_target_data.keys())
    saved_count = 0
    
    try:
        device = get_device()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        
        with torch.autocast(get_autocast_device()):
            inference_state = sam2_model.init_state(video_path=FRAME_DIR)
            
            for label_name, label_data in multi_target_data.items():
                for obj_id, obj_data in label_data["objects"].items():
                    if frame_index in obj_data["points"] and obj_data["points"][frame_index]:
                        obj_points = obj_data["points"][frame_index]
                        pts = [[p[0], p[1]] for p in obj_points]
                        lbls = [p[2] for p in obj_points]
                        
                        # 为每个目标生成掩码
                        _, _, mask_logits = sam2_model.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_index,
                            obj_id=obj_id,
                            points=np.array(pts, dtype=np.float32),
                            labels=np.array(lbls, dtype=np.int32),
                        )
                        
                        binary_mask = (mask_logits[0] > 0).cpu().numpy().squeeze()
                        
                        if binary_mask.sum() > 0:
                            # 计算边界框
                            bbox_method = st.session_state.get("bbox_method", "传统方法")
                            box = get_bbox_by_method(binary_mask, bbox_method)
                            
                            if box is not None:
                                x1, y1, x2, y2 = box
                                
                                # 保存数据
                                label_id = all_labels.index(label_name)
                                h, w = binary_mask.shape
                                yolo_line = f"{label_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
                                
                                base_name = frame_files[frame_index].replace(".jpg", "")
                                unique_name = f"{current_session_id}_{base_name}_obj{obj_id}"
                                
                                # 保存图像和标注
                                src_img_path = os.path.join(FRAME_DIR, frame_files[frame_index])
                                dst_img_path = os.path.join(UNIFIED_FRAME_DIR, f"{unique_name}.jpg")
                                shutil.copy2(src_img_path, dst_img_path)
                                
                                label_file = os.path.join(UNIFIED_LABEL_DIR, f"{unique_name}.txt")
                                with open(label_file, "w") as f:
                                    f.write(yolo_line)
                                
                                saved_count += 1
                        
                        # 清理内存
                        del mask_logits
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps":
                            torch.mps.empty_cache()
            
            # 更新类别文件
            if saved_count > 0:
                classes_file = os.path.join(UNIFIED_LABEL_DIR, "classes.txt")
                with open(classes_file, "w") as f:
                    for label_name in all_labels:
                        f.write(f"{label_name}\n")
            
            st.success(f"💾 保存完成！已保存{saved_count}个目标数据")
    
    except Exception as e:
        st.error(f"❌ 保存失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def generate_100_frames_all_targets():
    """一键生成100帧全部目标的掩码"""
    multi_target_data = st.session_state.get("multi_target_data", {})
    current_session_id = st.session_state.get("current_session_id")
    FRAME_DIR = f"frame_cache_{current_session_id}"
    
    # 找到有标注点的目标
    targets_with_points = {}
    for label_name, label_data in multi_target_data.items():
        for obj_id, obj_data in label_data["objects"].items():
            # 使用实际的ann_obj_id作为键
            ann_obj_id = obj_data["ann_obj_id"]
            for frame_idx, points in obj_data["points"].items():
                if points:  # 有标注点
                    if ann_obj_id not in targets_with_points:
                        targets_with_points[ann_obj_id] = {
                            "label": label_name,
                            "frame": frame_idx,
                            "points": points,
                            "original_obj_id": obj_id  # 保存原始obj_id用于调试
                        }
                    break  # 只需要第一个有点的帧作为参考
    
    if not targets_with_points:
        st.warning("⚠️ 没有找到有标注点的目标")
        return None
    
    st.write(f"🔍 找到{len(targets_with_points)}个目标有标注点")
    
    try:
        device = get_device()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        
        with torch.autocast(get_autocast_device()):
            inference_state = sam2_model.init_state(video_path=FRAME_DIR)
            
            # 为每个目标添加参考点
            for ann_obj_id, target_info in targets_with_points.items():
                ref_frame = target_info["frame"]
                ref_points = target_info["points"]
                pts = [[p[0], p[1]] for p in ref_points]
                lbls = [p[2] for p in ref_points]
                
                st.write(f"🎯 处理目标ann_obj_id={ann_obj_id}({target_info['label']}) - 参考帧{ref_frame}")
                
                sam2_model.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ref_frame,
                    obj_id=ann_obj_id,  # 使用ann_obj_id
                    points=np.array(pts, dtype=np.float32),
                    labels=np.array(lbls, dtype=np.int32),
                )
            
            # 批量传播所有目标
            video_segments = {}
            for i, obj_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
                # 确保obj_ids和mask_logits的对应关系正确
                frame_masks = {}
                st.write(f"🔍 帧{i}: obj_ids={obj_ids}, mask数量={len(mask_logits)}")
                
                for j, obj_id in enumerate(obj_ids):
                    if j < len(mask_logits):
                        mask = (mask_logits[j] > 0).cpu().numpy()
                        frame_masks[obj_id] = mask
                        st.write(f"  • 目标{obj_id}: 掩码大小={mask.sum()}")
                    else:
                        st.warning(f"  ⚠️ 目标{obj_id}: 索引{j}超出mask_logits范围")
                
                video_segments[i] = frame_masks
                
                # 立即清理内存
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                
                # 限制处理100帧
                if i >= 99:
                    break
            
            # 统计结果
            total_masks = sum(1 for frame_data in video_segments.values() 
                            for mask in frame_data.values() if mask.sum() > 0)
            
            st.write(f"✅ 成功生成{total_masks}个有效掩码")
            
            # 保存结果到session state
            st.session_state["video_segments"] = video_segments
            st.session_state["inference_state"] = inference_state
            st.session_state["multi_target_reference"] = targets_with_points
            
            # 打印保存的参考信息用于调试
            st.write("🔍 保存的多目标参考信息:")
            for ann_obj_id, info in targets_with_points.items():
                st.write(f"  • ann_obj_id={ann_obj_id}: 标签={info['label']}, 参考帧={info['frame']}, 原始obj_id={info.get('original_obj_id', '未知')}")
            
            return video_segments
    
    except Exception as e:
        st.error(f"❌ 生成100帧掩码失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

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

    st.divider()
    
    # ===== 多目标管理界面 =====
    st.header("🏷️ 多目标管理")
    
    # 显示当前多目标数据结构
    multi_target_data = st.session_state["multi_target_data"]
    active_label = st.session_state["current_active_label"]
    active_obj = st.session_state["current_active_object"]
    
    if not multi_target_data:
        st.info("💡 添加第一个标签开始多目标标注")
    
    # 显示所有标签和目标
    for label_name in multi_target_data.keys():
        label_objects = multi_target_data[label_name]["objects"]
        
        # 标签行
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            is_active_label = (active_label == label_name)
            label_style = "🟢" if is_active_label else "⚪"
            st.write(f"{label_style} **{label_name}**")
        
        with col2:
            # 添加新目标
            if st.button("➕", key=f"add_obj_{label_name}", help=f"为{label_name}添加新目标"):
                new_obj_id = add_new_object_to_label(label_name)
                if new_obj_id:
                    st.session_state["current_active_label"] = label_name
                    st.session_state["current_active_object"] = new_obj_id
                    st.success(f"✅ 已为'{label_name}'添加目标{new_obj_id}")
                    st.rerun()
        
        with col3:
            # 删除标签
            if st.button("🗑️", key=f"del_label_{label_name}", help=f"删除标签{label_name}"):
                if label_name in st.session_state["multi_target_data"]:
                    del st.session_state["multi_target_data"][label_name]
                    if st.session_state["current_active_label"] == label_name:
                        st.session_state["current_active_label"] = None
                        st.session_state["current_active_object"] = None
                    st.warning(f"⚠️ 已删除标签'{label_name}'")
                    st.rerun()
        
        # 显示此标签下的所有目标
        if label_objects:
            for obj_id, obj_data in label_objects.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    is_active_obj = (active_obj == obj_id and active_label == label_name)
                    obj_style = "🔴" if is_active_obj else "⭕"
                    point_count = sum(len(points) for points in obj_data["points"].values())
                    st.write(f"    {obj_style} 目标{obj_id} ({point_count}点)")
                
                with col2:
                    # 激活此目标
                    if st.button("📍", key=f"activate_{label_name}_{obj_id}", help=f"激活目标{obj_id}"):
                        st.session_state["current_active_label"] = label_name
                        st.session_state["current_active_object"] = obj_id
                        st.success(f"✅ 已激活'{label_name}'的目标{obj_id}")
                        st.rerun()
                
                with col3:
                    # 删除此目标
                    if st.button("❌", key=f"del_obj_{label_name}_{obj_id}", help=f"删除目标{obj_id}"):
                        if obj_id in st.session_state["multi_target_data"][label_name]["objects"]:
                            del st.session_state["multi_target_data"][label_name]["objects"][obj_id]
                            if st.session_state["current_active_object"] == obj_id:
                                st.session_state["current_active_object"] = None
                            st.warning(f"⚠️ 已删除目标{obj_id}")
                            st.rerun()
        
        st.write("")  # 间距
    
    # 添加新标签
    st.subheader("➕ 添加新标签")
    new_label_name = st.text_input("标签名称", placeholder="例如: niurou, tudou", key="new_label_input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 添加标签", key="add_new_label"):
            if new_label_name.strip():
                if add_new_label(new_label_name.strip().lower()):
                    st.success(f"✅ 已添加标签'{new_label_name}'")
                    st.rerun()
                else:
                    st.error("❌ 标签已存在")
            else:
                st.warning("⚠️ 请输入标签名称")
    
    with col2:
        if st.button("🔄 重置所有目标", key="reset_multi_targets"):
            st.session_state["multi_target_data"] = {}
            st.session_state["next_obj_id"] = 1
            st.session_state["current_active_label"] = None
            st.session_state["current_active_object"] = None
            st.success("✅ 已重置所有目标")
            st.rerun()
    
    # 当前激活状态
    st.subheader("🎯 当前激活状态")
    if active_label and active_obj:
        st.success(f"📍 标签: **{active_label}**")
        st.success(f"🎯 目标: **{active_obj}** (ID: {active_obj})")
        
        obj_data = get_current_active_object()
        if obj_data:
            total_points = sum(len(points) for points in obj_data["points"].values())
            st.info(f"📊 已标注点数: {total_points}")
    else:
        st.warning("⚠️ 请激活一个目标开始标注")

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

        # 显示多目标的点
        multi_target_data = st.session_state.get("multi_target_data", {})
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        color_idx = 0
        
        for label_name, label_data in multi_target_data.items():
            for obj_id, obj_data in label_data["objects"].items():
                if frame_index in obj_data["points"]:
                    obj_points = obj_data["points"][frame_index]
                    color = colors[color_idx % len(colors)]
                    
                    for x, y, l in obj_points:
                        # 正点用大圆圈，负点用小圆圈
                        radius = 8 if l == 1 else 4
                        cv2.circle(preview_img, (int(x), int(y)), radius, color, -1)
                        # 在点旁边显示目标ID
                        cv2.putText(preview_img, f"{label_name}-{obj_id}", (int(x)+10, int(y)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                color_idx += 1

        # 左图右控
        col1, col2 = st.columns([3, 1])

        with col1:
            # 预览帧并点击打点 - 图像尺寸优化
            img_height, img_width = preview_img.shape[:2]
            max_width = 800
            scale = 1.0
            if img_width > max_width:
                scale = max_width / img_width
                new_width = max_width
                new_height = int(img_height * scale)
                preview_img_display = cv2.resize(preview_img, (new_width, new_height))
            else:
                preview_img_display = preview_img
            
            click = streamlit_image_coordinates(preview_img_display, key=f"frame_{frame_index}_{st.session_state.get('refresh_flag', False)}")
            if click:
                # 坐标映射回原始尺寸
                click_x = click["x"] / scale
                click_y = click["y"] / scale
                
                # 检查是否有激活的多目标
                active_label = st.session_state.get("current_active_label")
                active_obj = st.session_state.get("current_active_object")
                
                if active_label and active_obj:
                    # 多目标模式
                    st.session_state["last_multi_click"] = (click_x, click_y)
                    st.session_state["last_multi_label"] = active_label
                    st.session_state["last_multi_obj"] = active_obj
                else:
                    st.warning("⚠️ 请先在侧边栏激活要标注的标签和目标")

        with col2:
            st.markdown("### 🎞️ 当前帧控制")
            frame_index = st.slider("帧位置", 0, len(frame_files) - 1, value=frame_index, key="frame_index")
            st.write(f"当前帧编号：**{frame_index}**")



            # 多目标模式点击处理
            if "last_multi_click" in st.session_state:
                st.subheader("🎯 多目标点击处理")
                multi_label = st.session_state["last_multi_label"]
                multi_obj = st.session_state["last_multi_obj"]
                st.write(f"标签: **{multi_label}** | 目标: **{multi_obj}**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("保留点(正点)", key="multi_positive"):
                        if add_point_to_current_object(frame_index, *st.session_state["last_multi_click"], 1):
                            st.success(f"✅ 已为'{multi_label}'的目标{multi_obj}添加正点")
                        st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)
                        del st.session_state["last_multi_click"]
                        del st.session_state["last_multi_label"]
                        del st.session_state["last_multi_obj"]
                        st.rerun()
                with col_b:
                    if st.button("去除点(负点)", key="multi_negative"):
                        if add_point_to_current_object(frame_index, *st.session_state["last_multi_click"], 0):
                            st.success(f"✅ 已为'{multi_label}'的目标{multi_obj}添加负点")
                        st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)
                        del st.session_state["last_multi_click"]
                        del st.session_state["last_multi_label"]
                        del st.session_state["last_multi_obj"]
                        st.rerun()
                st.divider()


            
            # ===== 多目标预览功能 =====
            st.subheader("🎯 多目标预览")
            
            # 统计当前帧的目标
            multi_target_data = st.session_state.get("multi_target_data", {})
            frame_targets = []
            for label_name, label_data in multi_target_data.items():
                for obj_id, obj_data in label_data["objects"].items():
                    if frame_index in obj_data["points"] and obj_data["points"][frame_index]:
                        frame_targets.append((label_name, obj_id, len(obj_data["points"][frame_index])))
            
            if frame_targets:
                st.write(f"📊 当前帧有{len(frame_targets)}个目标有标注")
                for label_name, obj_id, point_count in frame_targets:
                    st.write(f"  • {label_name}-{obj_id}: {point_count}个点")
                
                col_preview, col_save = st.columns(2)
                with col_preview:
                    if st.button("👁️ 预览全部目标"):
                        preview_all_targets_only(frame_index, frame_np, frame_files, current_session_id)
                with col_save:
                    if st.button("💾 保存全部目标"):
                        save_all_targets_in_frame(frame_index, frame_np, frame_files, current_session_id)
            else:
                st.info("💡 当前帧没有目标标注点")

            if st.button("🧹 清除当前帧所有目标的点"):
                # 清除当前帧的所有多目标点
                multi_target_data = st.session_state.get("multi_target_data", {})
                cleared_count = 0
                for label_name, label_data in multi_target_data.items():
                    for obj_id, obj_data in label_data["objects"].items():
                        if frame_index in obj_data["points"]:
                            obj_data["points"][frame_index] = []
                            cleared_count += 1
                st.success(f"✅ 已清除当前帧 {cleared_count} 个目标的点")
                st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)

            if st.button("🧼 清除所有目标的点"):
                # 清除所有帧的所有多目标点
                multi_target_data = st.session_state.get("multi_target_data", {})
                cleared_count = 0
                for label_name, label_data in multi_target_data.items():
                    for obj_id, obj_data in label_data["objects"].items():
                        obj_data["points"] = {}
                        obj_data["masks"] = {}
                        cleared_count += 1
                st.success(f"✅ 已清除所有 {cleared_count} 个目标的点和掩码")
                st.session_state["refresh_flag"] = not st.session_state.get("refresh_flag", False)




            
            # ===== 多目标100帧生成功能 =====
            st.divider()
            st.subheader("🚀 多目标批量生成")
            
            # 统计有标注点的目标
            multi_target_data = st.session_state.get("multi_target_data", {})
            targets_with_annotations = 0
            target_info = []
            for label_name, label_data in multi_target_data.items():
                for obj_id, obj_data in label_data["objects"].items():
                    if any(obj_data["points"].values()):
                        targets_with_annotations += 1
                        # 找到第一个有点的帧作为参考帧
                        ref_frame = None
                        for frame_idx, points in obj_data["points"].items():
                            if points:
                                ref_frame = frame_idx
                                break
                        if ref_frame is not None:
                            target_info.append((label_name, obj_id, ref_frame, len(obj_data["points"][ref_frame])))
            
            if targets_with_annotations > 0:
                st.write(f"📊 发现{targets_with_annotations}个目标有标注点:")
                for label_name, obj_id, ref_frame, point_count in target_info:
                    st.write(f"  • {label_name}-{obj_id}: 参考帧{ref_frame}, {point_count}个点")
                
                if st.button("⚡ 一键生成全部目标100帧掩码"):
                    with st.spinner(f"正在为{targets_with_annotations}个目标生成100帧掩码..."):
                        video_segments = generate_100_frames_all_targets()
                        if video_segments:
                            st.success(f"✅ 全部目标100帧掩码生成完成！")
                            st.balloons()
            else:
                st.info("💡 请先为至少一个目标添加标注点")



    elif work_mode == "预览100帧掩码":
        if "video_segments" not in st.session_state:
            st.warning("⚠️ 请先在'初始标注'模式下生成100帧掩码")
        else:
            # 检查是否有多目标参考信息
            multi_target_ref = st.session_state.get("multi_target_reference", {})
            if multi_target_ref:
                st.subheader(f"🎞️ 多目标100帧掩码预览 ({len(multi_target_ref)}个目标)")
                target_names = [f"目标{obj_id}({info['label']})" for obj_id, info in multi_target_ref.items()]
                st.write("🎯 包含目标:", ", ".join(target_names))
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
                frame_masks = video_segments.get(preview_frame_idx, {})
                
                # 添加调试信息
                st.write(f"🔍 当前预览帧: {preview_frame_idx}, 帧文件: {frame_files[preview_frame_idx]}")
                st.write(f"🔍 发现{len(frame_masks)}个目标掩码")
                for obj_id in frame_masks.keys():
                    mask_valid = frame_masks[obj_id] is not None and frame_masks[obj_id].sum() > 0
                    st.write(f"  • 目标{obj_id}: {'有效' if mask_valid else '无效'}")
                
                # 多目标模式显示
                if len(frame_masks) > 0:
                    overlay = img.copy()
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    valid_targets = 0
                    
                    # 打印调试信息：显示每个obj_id的映射
                    st.write(f"🔍 预览调试 - 帧{preview_frame_idx}的所有目标:")
                    for obj_id, mask in frame_masks.items():
                        mask_size = mask.sum() if mask is not None else 0
                        st.write(f"  • obj_id={obj_id}: 掩码像素数={mask_size}")
                    
                    # 按obj_id排序以确保一致的颜色分配
                    sorted_targets = sorted(frame_masks.items(), key=lambda x: x[0])
                    
                    for i, (obj_id, mask) in enumerate(sorted_targets):
                        if mask is not None and mask.sum() > 0:
                            # 确保掩码格式正确
                            if len(mask.shape) == 3:
                                mask = mask.squeeze()
                            if len(mask.shape) == 2 and mask.shape[:2] == img.shape[:2]:
                                mask = (mask > 0).astype(np.uint8)
                                
                                # 为每个目标使用不同颜色（基于obj_id确保一致性）
                                target_color = colors[obj_id % len(colors)]
                                overlay[mask == 1] = (overlay[mask == 1] * 0.7 + np.array(target_color) * 0.3).astype(np.uint8)
                                
                                # 绘制边界框
                                bbox_method = st.session_state.get("bbox_method", "传统方法")
                                box = get_bbox_by_method(mask, bbox_method)
                                
                                if box is not None:
                                    x1, y1, x2, y2 = box
                                    cv2.rectangle(overlay, (x1, y1), (x2, y2), target_color, 2)
                                    # 添加标签（使用ann_obj_id）
                                    if multi_target_ref and obj_id in multi_target_ref:
                                        label_name = multi_target_ref[obj_id]["label"]
                                        cv2.putText(overlay, f"{label_name}-{obj_id}", (x1, y1-10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)
                                    else:
                                        cv2.putText(overlay, f"ann_obj_id{obj_id}", (x1, y1-10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)
                                
                                valid_targets += 1
                                st.write(f"  ✅ 成功渲染obj_id={obj_id}, 颜色索引={obj_id % len(colors)}")
                            else:
                                st.warning(f"  ⚠️ obj_id={obj_id}: 掩码尺寸不匹配 - 掩码:{mask.shape}, 图像:{img.shape[:2]}")
                        else:
                            st.info(f"  ℹ️ obj_id={obj_id}: 掩码为空或无效")
                    
                    if valid_targets > 0:
                        caption = f"帧 {preview_frame_idx} - 多目标预览 ({valid_targets}个目标)" if len(frame_masks) > 1 else f"帧 {preview_frame_idx} - 目标预览"
                        st.image(overlay, caption=caption)
                    else:
                        st.image(img, caption=f"帧 {preview_frame_idx} - 无有效掩码")
                else:
                    st.image(img, caption=f"帧 {preview_frame_idx} - 无掩码")
            
            with col2:
                st.markdown("### 🔧 操作选项")
                if st.button("选择此帧进行修正"):
                    st.session_state["refine_frame_idx"] = preview_frame_idx
                    st.success(f"✅ 已选择帧 {preview_frame_idx} 进行修正")
                
                if st.button("📤 批量添加到统一数据集"):
                    # 检查是否有多目标参考信息
                    multi_target_ref = st.session_state.get("multi_target_reference", {})
                    
                    if multi_target_ref:
                        # 多目标模式：批量添加所有目标
                        all_labels = list(st.session_state.get("multi_target_data", {}).keys())
                        exported_count = 0
                        
                        for i in range(max_frames):
                            frame_masks = video_segments.get(i, {})
                            
                            for ann_obj_id, mask in frame_masks.items():
                                if mask is not None and mask.sum() > 0:
                                    # 使用边界框算法
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
                                            h, w = img.shape[:2]
                                        
                                        # 获取对应的标签（使用ann_obj_id）
                                        if ann_obj_id in multi_target_ref:
                                            label_name = multi_target_ref[ann_obj_id]["label"]
                                            label_id = all_labels.index(label_name)
                                            yolo_line = f"{label_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
                                            
                                            # 生成唯一的文件名（使用ann_obj_id）
                                            base_name = frame_files[i].replace(".jpg", "")
                                            unique_name = f"{current_session_id}_{base_name}_ann{ann_obj_id}"
                                            
                                            # 保存图像到统一目录
                                            src_img_path = os.path.join(FRAME_DIR, frame_files[i])
                                            dst_img_path = os.path.join(UNIFIED_FRAME_DIR, f"{unique_name}.jpg")
                                            shutil.copy2(src_img_path, dst_img_path)
                                            
                                            # 保存标注到统一目录
                                            label_file = os.path.join(UNIFIED_LABEL_DIR, f"{unique_name}.txt")
                                            with open(label_file, "w") as f:
                                                f.write(yolo_line)
                                            
                                            exported_count += 1
                                        else:
                                            st.warning(f"⚠️ 未找到ann_obj_id={ann_obj_id}的标签映射")
                        
                        # 更新类别文件
                        if exported_count > 0:
                            classes_file = os.path.join(UNIFIED_LABEL_DIR, "classes.txt")
                            with open(classes_file, "w") as f:
                                for label_name in all_labels:
                                    f.write(f"{label_name}\n")
                        
                        st.success(f"✅ 多目标模式：已添加 {exported_count} 个目标数据到统一数据集")
                    else:
                        st.warning("⚠️ 没有找到多目标参考信息，请使用多目标模式生成100帧掩码")

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
        
        # 多目标数据统计
        multi_target_data = st.session_state.get("multi_target_data", {})
        if multi_target_data:
            st.write("多目标统计:")
            total_labels = len(multi_target_data)
            total_objects = sum(len(label_data["objects"]) for label_data in multi_target_data.values())
            total_points = 0
            total_masks = 0
            for label_data in multi_target_data.values():
                for obj_data in label_data["objects"].values():
                    total_points += sum(len(points) for points in obj_data["points"].values())
                    total_masks += len(obj_data["masks"])
            
            st.write(f"标签数量: {total_labels}")
            st.write(f"目标数量: {total_objects}")
            st.write(f"标注点数: {total_points}")
            st.write(f"生成掩码: {total_masks}")
    
    # 设置GPU内存管理
    st.header("⚙️ 内存设置")
    if st.button("设置GPU内存优化"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        st.success("✅ 已设置GPU内存优化")
        st.write("重启应用生效") 