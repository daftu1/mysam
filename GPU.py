import sys, os, numpy as np, cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QSpinBox, QCheckBox, QScrollArea, QDialog, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtCore import Qt
from moviepy import VideoFileClip
import torch
from sam2.build_sam import build_sam2_video_predictor

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = build_sam2_video_predictor(config, checkpoint, device=device)

class PreviewDialog(QDialog):
    def __init__(self, masks, frame_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("100帧掩码预览与修改")
        self.masks = masks
        self.frame_dir = frame_dir
        self.selected_frame_idx = None

        layout = QGridLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout()

        for i, (frame_idx, frame_masks) in enumerate(masks.items()):
            img_path = os.path.join(frame_dir, f"{frame_idx}.jpg")
            img = cv2.imread(img_path)
            if frame_masks:
                for _, mask in frame_masks.items():
                    try:
                        mask = np.squeeze(mask)
                        if mask.ndim != 2:
                            continue
                        overlay = np.zeros_like(img)
                        overlay[mask > 0] = [0, 255, 0]
                        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                    except Exception as e:
                        print(f"帧 {frame_idx} 掩码出错: {e}")


            label = QLabel()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            qimg = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(160, 90, Qt.KeepAspectRatio)
            label.setPixmap(pix)
            label.mousePressEvent = self.make_select_handler(frame_idx)
            grid.addWidget(label, i // 5, i % 5)

        container.setLayout(grid)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def make_select_handler(self, idx):
        def handler(event):
            self.selected_frame_idx = idx
            self.accept()
        return handler

class Sam2Annotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 标注系统")
        self.frame_dir = "./frames"
        self.frame_idx = 0
        self.click_points = []
        self.click_labels = []
        self.inference_state = None
        self.image = None
        self.video_segments = {}
        self.current_label = 1
        self.original_video_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll)

        btns = QHBoxLayout()
        self.upload_button = QPushButton("上传视频")
        self.upload_button.clicked.connect(self.load_video)
        self.clip_checkbox = QCheckBox("是否剪裁")
        self.start_spin = QSpinBox(); self.end_spin = QSpinBox()
        self.start_spin.setPrefix("起始: "); self.end_spin.setPrefix("结束: ")
        self.start_spin.setMaximum(9999); self.end_spin.setMaximum(9999)
        btns.addWidget(self.upload_button)
        btns.addWidget(self.clip_checkbox)
        btns.addWidget(self.start_spin)
        btns.addWidget(self.end_spin)
        layout.addLayout(btns)

        actions = QHBoxLayout()
        self.positive_btn = QPushButton("正点")
        self.negative_btn = QPushButton("负点")
        self.mask_btn = QPushButton("生成掩码")
        self.propagate_btn = QPushButton("生成100帧并预览")
        self.clear_btn = QPushButton("清除点")
        self.positive_btn.clicked.connect(lambda: self.set_click_label(1))
        self.negative_btn.clicked.connect(lambda: self.set_click_label(0))
        self.mask_btn.clicked.connect(self.generate_mask)
        self.propagate_btn.clicked.connect(self.propagate_masks)
        self.clear_btn.clicked.connect(self.clear_points)
        actions.addWidget(self.positive_btn)
        actions.addWidget(self.negative_btn)
        actions.addWidget(self.mask_btn)
        actions.addWidget(self.propagate_btn)
        actions.addWidget(self.clear_btn)
        layout.addLayout(actions)

        self.setLayout(layout)
        self.image_label.mousePressEvent = self.get_mouse_click

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)")
        if not file_path: return
        self.original_video_path = file_path
        os.makedirs(self.frame_dir, exist_ok=True)

        if self.clip_checkbox.isChecked():
            start, end = self.start_spin.value(), self.end_spin.value()
            clip = VideoFileClip(self.original_video_path).subclipped(start, end)
        else:
            clip = VideoFileClip(self.original_video_path)

        for i, frame in enumerate(clip.iter_frames(fps=10)):
            frame_path = os.path.join(self.frame_dir, f"{i}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_idx = 0
        self.load_frame()
        self.inference_state = predictor.init_state(video_path=self.frame_dir)
        print("视频加载完成")

    def load_frame(self):
        frame_path = os.path.join(self.frame_dir, f"{self.frame_idx}.jpg")
        if not os.path.exists(frame_path): return
        frame = cv2.imread(frame_path)
        self.image = frame.copy()
        self.update_display()

    def update_display(self):
        if self.image is None: return
        display = self.image.copy()
        for (x, y), label in zip(self.click_points, self.click_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display, (x, y), 8, color, -1)

        # 添加左上角尺寸文字
        h, w = display.shape[:2]
        cv2.putText(display, f"{w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2)

        qimg = QImage(display.data, w, h, display.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()

    def set_click_label(self, label_val):
        self.current_label = label_val

    def get_mouse_click(self, event):
        x = int(event.pos().x() * self.image.shape[1] / self.image_label.width())
        y = int(event.pos().y() * self.image.shape[0] / self.image_label.height())
        self.click_points.append((x, y))
        self.click_labels.append(self.current_label)
        self.update_display()

    def clear_points(self):
        self.click_points = []
        self.click_labels = []
        self.load_frame()

    def generate_mask(self):
        if self.inference_state is None: return
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
        overlay = np.zeros_like(colored)
        overlay[mask > 0] = [0, 255, 0]
        self.image = cv2.addWeighted(colored, 0.7, overlay, 0.3, 0)
        self.update_display()

    def propagate_masks(self):
        if self.inference_state is None: return
        points = np.array(self.click_points, dtype=np.float32)
        labels = np.array(self.click_labels, dtype=np.int32)
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )
        self.video_segments = {}
        for i, obj_ids, mask_logits in predictor.propagate_in_video(self.inference_state):
            self.video_segments[i] = {
                obj_id: (mask_logits[j] > 0).cpu().numpy()
                for j, obj_id in enumerate(obj_ids)
            }
        dialog = PreviewDialog(self.video_segments, self.frame_dir)
        if dialog.exec_() == QDialog.Accepted:
            self.frame_idx = dialog.selected_frame_idx
            self.load_frame()
            self.click_points = []
            self.click_labels = []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Sam2Annotator()
    window.show()
    sys.exit(app.exec_())
