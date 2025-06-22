import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, uuid, cv2, shutil, numpy as np, torch
from PIL import Image, ImageTk
from moviepy import VideoFileClip
from torchvision.ops import masks_to_boxes
from sam2.build_sam import build_sam2_video_predictor
import contextlib
import threading
import sys

class VideoAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("视频标注工具")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.session_id = None
        self.segment_path = None
        self.frame_files = []
        self.current_frame_index = 0
        self.points = {}
        self.label_history = []
        self.current_label = None
        self.overlay_map = {}
        self.click_mode = "保留点"  # 或 "去除点"
        
        # 初始化SAM2模型
        self.init_sam2_model()
        
        # 创建界面
        self.create_widgets()
        
    def init_sam2_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_file = "configs/sam2.1/sam2.1_hiera_b+.yaml"  # 使用SAM2.1 base+模型
        checkpoint_path = os.path.join(os.path.expanduser("~"), "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")
        
        if not os.path.exists(checkpoint_path):
            messagebox.showerror("错误", f"找不到模型文件：{checkpoint_path}\n请确保模型文件已下载到正确位置。")
            sys.exit(1)
            
        self.sam2_model = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
            vos_optimized=True  # 使用视频优化模式
        )
        
    def create_widgets(self):
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板 - 视频预览和标注
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 视频预览区域
        self.preview_canvas = tk.Canvas(self.left_panel, bg='black')
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind('<Button-1>', self.on_canvas_click)
        
        # 右侧面板 - 控制按钮
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # 上传视频按钮
        self.upload_btn = ttk.Button(self.right_panel, text="上传视频", command=self.upload_video)
        self.upload_btn.pack(fill=tk.X, pady=5)
        
        # 视频裁剪控制
        self.crop_frame = ttk.LabelFrame(self.right_panel, text="视频裁剪")
        self.crop_frame.pack(fill=tk.X, pady=5)
        
        self.start_time_var = tk.IntVar(value=0)
        self.end_time_var = tk.IntVar(value=5)
        
        ttk.Label(self.crop_frame, text="起始时间(秒):").pack()
        self.start_time_scale = ttk.Scale(self.crop_frame, from_=0, to=100, 
                                        variable=self.start_time_var, orient=tk.HORIZONTAL)
        self.start_time_scale.pack(fill=tk.X)
        
        ttk.Label(self.crop_frame, text="结束时间(秒):").pack()
        self.end_time_scale = ttk.Scale(self.crop_frame, from_=0, to=100, 
                                      variable=self.end_time_var, orient=tk.HORIZONTAL)
        self.end_time_scale.pack(fill=tk.X)
        
        self.crop_btn = ttk.Button(self.crop_frame, text="裁剪视频", command=self.crop_video)
        self.crop_btn.pack(fill=tk.X, pady=5)
        
        # 帧控制
        self.frame_control = ttk.LabelFrame(self.right_panel, text="帧控制")
        self.frame_control.pack(fill=tk.X, pady=5)
        
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(self.frame_control, from_=0, to=100, 
                                   variable=self.frame_var, orient=tk.HORIZONTAL,
                                   command=self.on_frame_change)
        self.frame_scale.pack(fill=tk.X)
        
        # 点击模式选择
        self.click_mode_var = tk.StringVar(value="保留点")
        ttk.Radiobutton(self.frame_control, text="保留点", variable=self.click_mode_var, 
                       value="保留点").pack()
        ttk.Radiobutton(self.frame_control, text="去除点", variable=self.click_mode_var, 
                       value="去除点").pack()
        
        # 标签管理
        self.label_frame = ttk.LabelFrame(self.right_panel, text="标签管理")
        self.label_frame.pack(fill=tk.X, pady=5)
        
        self.label_entry = ttk.Entry(self.label_frame)
        self.label_entry.pack(fill=tk.X, pady=5)
        
        self.confirm_label_btn = ttk.Button(self.label_frame, text="确认标签", 
                                          command=self.confirm_label)
        self.confirm_label_btn.pack(fill=tk.X)
        
        # 操作按钮
        self.action_frame = ttk.LabelFrame(self.right_panel, text="操作")
        self.action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.action_frame, text="预览当前帧", 
                  command=self.preview_current_frame).pack(fill=tk.X, pady=2)
        ttk.Button(self.action_frame, text="清除当前帧点", 
                  command=self.clear_current_frame).pack(fill=tk.X, pady=2)
        ttk.Button(self.action_frame, text="清除所有帧点", 
                  command=self.clear_all_frames).pack(fill=tk.X, pady=2)
        ttk.Button(self.action_frame, text="自动掩码传播", 
                  command=self.propagate_masks).pack(fill=tk.X, pady=2)
        
    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            self.original_path = "temp_uploaded.mp4"
            shutil.copy(file_path, self.original_path)
            
            # 获取视频信息
            cap = cv2.VideoCapture(self.original_path)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.duration = self.frame_count // self.fps
            cap.release()
            
            # 更新裁剪控制
            self.start_time_scale.config(to=self.duration-1)
            self.end_time_scale.config(to=self.duration)
            
            messagebox.showinfo("成功", "视频上传成功！")
            
    def crop_video(self):
        if not hasattr(self, 'original_path'):
            messagebox.showerror("错误", "请先上传视频！")
            return
            
        start_time = self.start_time_var.get()
        end_time = self.end_time_var.get()
        
        if end_time <= start_time:
            messagebox.showerror("错误", "结束时间必须大于开始时间！")
            return
            
        try:
            clip = VideoFileClip(self.original_path).subclip(start_time, end_time)
            self.session_id = uuid.uuid4().hex[:8]
            self.segment_path = os.path.join("video_segments", f"{self.session_id}.mp4")
            os.makedirs("video_segments", exist_ok=True)
            
            clip.write_videofile(self.segment_path, codec="libx264")
            self.extract_frames()
            messagebox.showinfo("成功", f"视频裁剪完成: {self.segment_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"裁剪失败: {str(e)}")
            
    def extract_frames(self):
        if not self.segment_path:
            return
            
        self.frame_dir = f"frame_cache_{self.session_id}"
        os.makedirs(self.frame_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.segment_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(self.frame_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()
        
        self.frame_files = sorted(os.listdir(self.frame_dir))
        self.frame_scale.config(to=len(self.frame_files)-1)
        self.update_preview()
        
    def update_preview(self):
        if not self.frame_files:
            return
            
        frame_path = os.path.join(self.frame_dir, self.frame_files[self.current_frame_index])
        img = Image.open(frame_path)
        
        # 调整图像大小以适应画布
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:  # 确保画布已经创建
            img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        # 如果有标注点，绘制它们
        if self.current_frame_index in self.points:
            draw = ImageDraw.Draw(img)
            for x, y, l in self.points[self.current_frame_index]:
                color = (0, 255, 0) if l == 1 else (0, 0, 255)
                draw.ellipse([x-5, y-5, x+5, y+5], fill=color)
        
        # 如果有预览图，显示它
        if self.current_frame_index in self.overlay_map:
            img = Image.fromarray(self.overlay_map[self.current_frame_index])
        
        self.photo = ImageTk.PhotoImage(img)
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
    def on_canvas_click(self, event):
        if not self.frame_files:
            return
            
        x, y = event.x, event.y
        label = 1 if self.click_mode_var.get() == "保留点" else 0
        
        if self.current_frame_index not in self.points:
            self.points[self.current_frame_index] = []
        self.points[self.current_frame_index].append((x, y, label))
        
        self.update_preview()
        
    def on_frame_change(self, value):
        self.current_frame_index = int(float(value))
        self.update_preview()
        
    def confirm_label(self):
        label = self.label_entry.get().strip().lower()
        if not label:
            messagebox.showwarning("警告", "标签不能为空！")
            return
            
        if label not in self.label_history:
            self.label_history.append(label)
        self.current_label = label
        messagebox.showinfo("成功", f"当前使用标签：{label}")
        
    def preview_current_frame(self):
        if not self.points.get(self.current_frame_index):
            messagebox.showwarning("警告", "当前帧无点击点！")
            return
            
        if not self.current_label:
            messagebox.showwarning("警告", "请先设置标签！")
            return
            
        # 使用SAM2进行预测
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            inference_state = self.sam2_model.init_state(self.segment_path)
            pts = [[p[0], p[1]] for p in self.points[self.current_frame_index]]
            lbls = [p[2] for p in self.points[self.current_frame_index]]
            
            frame_idx, obj_ids, masks = self.sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=self.current_frame_index,
                obj_id=0,
                points=pts,
                labels=lbls,
                clear_old_points=True,
                normalize_coords=False
            )
            
            mask = masks[0, 0].cpu().numpy().astype(np.uint8)
            box = masks_to_boxes(torch.tensor(mask[None]))[0].int().tolist()
            
            # 生成预览图
            frame_path = os.path.join(self.frame_dir, self.frame_files[self.current_frame_index])
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            overlay = img.copy()
            overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([128,128,255]) * 0.5).astype(np.uint8)
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
            
            self.overlay_map[self.current_frame_index] = overlay
            self.update_preview()
            
            # 保存YOLO格式标注
            label_id = self.label_history.index(self.current_label)
            save_dir = f"yolo_labels_{self.session_id}"
            os.makedirs(save_dir, exist_ok=True)
            
            h, w = mask.shape
            yolo_line = f"{label_id} {(box[0]+box[2])/2/w:.6f} {(box[1]+box[3])/2/h:.6f} {(box[2]-box[0])/w:.6f} {(box[3]-box[1])/h:.6f}\n"
            label_file = os.path.join(save_dir, self.frame_files[self.current_frame_index].replace(".jpg", ".txt"))
            with open(label_file, "w") as f:
                f.write(yolo_line)
                
            messagebox.showinfo("成功", f"单帧标签保存成功: {label_file}")
            
    def clear_current_frame(self):
        if self.current_frame_index in self.points:
            del self.points[self.current_frame_index]
        if self.current_frame_index in self.overlay_map:
            del self.overlay_map[self.current_frame_index]
        self.update_preview()
        
    def clear_all_frames(self):
        self.points.clear()
        self.overlay_map.clear()
        self.update_preview()
        
    def propagate_masks(self):
        if not self.points.get(self.current_frame_index):
            messagebox.showwarning("警告", "首帧未打点！")
            return
            
        if not self.current_label:
            messagebox.showwarning("警告", "请先设置标签！")
            return
            
        # 在新线程中执行传播
        threading.Thread(target=self._propagate_masks_thread, daemon=True).start()
        
    def _propagate_masks_thread(self):
        try:
            save_dir = f"yolo_labels_{self.session_id}"
            os.makedirs(save_dir, exist_ok=True)
            label_id = self.label_history.index(self.current_label)
            
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                inference_state = self.sam2_model.init_state(self.segment_path)
                pts = [[p[0], p[1]] for p in self.points[self.current_frame_index]]
                lbls = [p[2] for p in self.points[self.current_frame_index]]
                
                # 在第一帧添加点
                frame_idx, obj_ids, masks = self.sam2_model.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=0,
                    points=pts,
                    labels=lbls,
                    clear_old_points=True,
                    normalize_coords=False
                )
                
                # 传播到所有帧
                for i in range(1, len(self.frame_files)):
                    frame_idx, obj_ids, masks = self.sam2_model.propagate_to_next_frame(
                        inference_state=inference_state,
                        frame_idx=i-1
                    )
                    
                    mask = masks[0, 0].cpu().numpy().astype(np.uint8)
                    box = masks_to_boxes(torch.tensor(mask[None]))[0].int().tolist()
                    
                    # 保存YOLO格式标注
                    h, w = mask.shape
                    yolo_line = f"{label_id} {(box[0]+box[2])/2/w:.6f} {(box[1]+box[3])/2/h:.6f} {(box[2]-box[0])/w:.6f} {(box[3]-box[1])/h:.6f}\n"
                    label_file = os.path.join(save_dir, self.frame_files[i].replace(".jpg", ".txt"))
                    with open(label_file, "w") as f:
                        f.write(yolo_line)
                    
                    # 生成预览图
                    frame_path = os.path.join(self.frame_dir, self.frame_files[i])
                    img = cv2.imread(frame_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    overlay = img.copy()
                    overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([128,128,255]) * 0.5).astype(np.uint8)
                    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
                    self.overlay_map[i] = overlay
                    
                    # 更新预览
                    self.root.after(0, self.update_preview)
                    
            messagebox.showinfo("成功", "所有帧标签与图像已自动生成，可逐帧预览")
            
        except Exception as e:
            messagebox.showerror("错误", f"传播失败: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotator(root)
    root.mainloop() 