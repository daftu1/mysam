import sys, os, numpy as np, cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QSpinBox, QCheckBox, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from moviepy import VideoFileClip
from PIL import Image
import torch
from sam2.build_sam import build_sam2_video_predictor

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = build_sam2_video_predictor(config, checkpoint, device=device)

class Sam2Annotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 标注系统")

        self.original_video_path = None
        self.frame_dir = "./frames"
        self.frame_idx = 0
        self.click_points = []
        self.click_labels = []
        self.inference_state = None
        self.image = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 添加可滚动图像显示区域
        self.image_label = QLabel("图像显示区域")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        layout.addWidget(self.scroll_area)

        # 上传视频与剪裁控制
        button_layout = QHBoxLayout()
        self.upload_button = QPushButton("上传视频")
        self.upload_button.clicked.connect(self.load_video)
        self.clip_checkbox = QCheckBox("是否剪裁")
        self.start_spin = QSpinBox(); self.end_spin = QSpinBox()
        self.start_spin.setPrefix("起始: "); self.end_spin.setPrefix("结束: ")
        self.start_spin.setMaximum(9999); self.end_spin.setMaximum(9999)
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.clip_checkbox)
        button_layout.addWidget(self.start_spin)
        button_layout.addWidget(self.end_spin)
        layout.addLayout(button_layout)

        # 点击标注与掩码控制
        action_layout = QHBoxLayout()
        self.positive_btn = QPushButton("正点")
        self.negative_btn = QPushButton("负点")
        self.mask_btn = QPushButton("生成掩码")
        self.clear_btn = QPushButton("清除点")
        self.positive_btn.clicked.connect(lambda: self.set_click_label(1))
        self.negative_btn.clicked.connect(lambda: self.set_click_label(0))
        self.mask_btn.clicked.connect(self.generate_mask)
        self.clear_btn.clicked.connect(self.clear_points)
        action_layout.addWidget(self.positive_btn)
        action_layout.addWidget(self.negative_btn)
        action_layout.addWidget(self.mask_btn)
        action_layout.addWidget(self.clear_btn)
        layout.addLayout(action_layout)

        self.setLayout(layout)

        # 图像点击事件绑定
        self.image_label.mousePressEvent = self.get_mouse_click

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)")
        if not file_path:
            return
        self.original_video_path = file_path
        os.makedirs(self.frame_dir, exist_ok=True)

        if self.clip_checkbox.isChecked():
            start, end = self.start_spin.value(), self.end_spin.value()
            print(f"剪裁视频: {start} 到 {end}")
            clip = VideoFileClip(self.original_video_path).subclipped(start, end)
        else:
            print("不剪裁，使用完整视频")
            clip = VideoFileClip(self.original_video_path)

        for i, frame in enumerate(clip.iter_frames(fps=10)):
            frame_path = os.path.join(self.frame_dir, f"{i}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_idx = 0
        self.load_frame()

        self.inference_state = predictor.init_state(video_path=self.frame_dir)
        print("加载并初始化完成")

    def load_frame(self):
        frame_path = os.path.join(self.frame_dir, f"{self.frame_idx}.jpg")
        if not os.path.exists(frame_path):
            print("帧不存在:", frame_path)
            return
        frame = cv2.imread(frame_path)
        self.image = frame.copy()
        self.update_display()

    def update_display(self):
        display = self.image.copy()
        for (x, y), label in zip(self.click_points, self.click_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display, (x, y), 8, color, -1)

        h, w = display.shape[:2]
        self.image_label.setFixedSize(w, h)
        qimage = QImage(display.data, w, h, display.strides[0], QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        self.adjustSize()

    def set_click_label(self, label_val):
        self.current_label = label_val

    def get_mouse_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        scaled_x = int(x * self.image.shape[1] / self.image_label.width())
        scaled_y = int(y * self.image.shape[0] / self.image_label.height())
        self.click_points.append((scaled_x, scaled_y))
        self.click_labels.append(self.current_label)
        print(f"添加点: ({scaled_x}, {scaled_y}), 标签: {self.current_label}")
        self.update_display()

    def clear_points(self):
        self.click_points = []
        self.click_labels = []
        self.load_frame()

    def generate_mask(self):
        if self.inference_state is None:
            print("请先加载视频")
            return
        points = np.array(self.click_points, dtype=np.float32)
        labels = np.array(self.click_labels, dtype=np.int32)
        _, _, mask_logits = predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )
        mask = (mask_logits[0] > 0).cpu().numpy().squeeze()
        colored = self.image.copy()
        green_overlay = np.zeros_like(colored)
        green_overlay[mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(colored, 0.7, green_overlay, 0.3, 0)
        self.image = blended
        self.update_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Sam2Annotator()
    window.show()
    sys.exit(app.exec_())
