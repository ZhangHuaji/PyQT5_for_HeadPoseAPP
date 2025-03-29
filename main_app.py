import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QPushButton, QLabel, QVBoxLayout, QWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QImage, QPixmap

# 导入你已有的模块
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

class VideoThread(QThread):
    """工作线程：负责打开摄像头或视频文件，读取帧并做人脸检测、关键点检测、姿态解算，再把结果发送给主线程。"""
    change_pixmap_signal = pyqtSignal(np.ndarray)  # 发射处理好的帧（用numpy数组表示）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = False
        self.cap = None
        self.source = 0  # 默认摄像头
        self.face_detector = None
        self.mark_detector = None
        self.pose_estimator = None

    def set_source_camera(self, cam_index=0):
        self.source = cam_index

    def set_source_video(self, filepath):
        self.source = filepath

    def set_detectors(self, face_detector, mark_detector, pose_estimator):
        self.face_detector = face_detector
        self.mark_detector = mark_detector
        #self.pose_estimator = pose_estimator

    def run(self):
        """线程入口函数"""
        # 打开视频源
        if isinstance(self.source, str):
            # 文件路径
            self.cap = cv2.VideoCapture(self.source)
        else:
            # 摄像头索引
            self.cap = cv2.VideoCapture(self.source)

        self._run_flag = True

        if not self.cap.isOpened():
            print("Failed to open video source:", self.source)
            return

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 确保 PoseEstimator 使用实际的帧宽度和高度
        self.pose_estimator = PoseEstimator(frame_height, frame_width)

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 如果是摄像头，可以镜像翻转
            if not isinstance(self.source, str):
                frame = cv2.flip(frame, 1)  # 修正为 cv2.flip(frame, 1)

            # 在这里调用你的检测 + 姿态估算逻辑
            if self.face_detector and self.mark_detector and self.pose_estimator:
                # 参考你的 main.py, 这里大概是:
                faces, _ = self.face_detector.detect(frame, 0.7)
                print(f"Detected {len(faces)} faces")

                if len(faces) > 0:
                    # refine 前先把原始坐标打印
                    for i, box in enumerate(faces):
                        print(f"Raw face {i} => (x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}, score={box[4]})")
                    face = refine(faces, frame.shape[1], frame.shape[0], 0.15)[0]
                    x1, y1, x2, y2 = face[:4].astype(int)
                    patch = frame[y1:y2, x1:x2]

                    # 检测关键点
                    marks = self.mark_detector.detect([patch])[0].reshape([68, 2])
                    print("Marks shape:", marks.shape)

                    marks *= (x2 - x1)
                    print("Marks shape222:", marks.shape)
                    # 映射回全图
                    marks[:, 0] += x1
                    marks[:, 1] += y1

                    for i in range(68):
                        x = int(marks[i, 0])
                        y = int(marks[i, 1])
                        # 画一个小圆点
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                        # 也可以在旁边加一个序号，方便对照
                        cv2.putText(frame, str(i), (x + 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.imshow("Debug Landmarks", frame)
                    cv2.waitKey(1)  # 保证窗口能刷新

                    # 姿态解算
                    pose = self.pose_estimator.solve(marks)
                    print("Pose:", pose)

                    # 可视化：在frame上画立方体
                    self.pose_estimator.visualize(frame, pose, color=(0,255,0))
                    print("绘制成功！")
                    # 画关键点
                    self.mark_detector.visualize(frame, marks, color=(0, 255, 0))
                    print("Marks shape:", marks.shape)  # (68, 2)

                    self.pose_estimator.draw_axes(frame, pose)

            # 将处理后的图像发送给主线程
            self.change_pixmap_signal.emit(frame)

        # 退出循环，释放资源
        self.cap.release()

    def stop(self):
        """停止线程"""
        self._run_flag = False
        self.quit()
        self.wait()


class MainWindow(QMainWindow):
    """主界面"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Head Pose Estimation with PyQt5")

        # 设置主窗口初始大小
        self.resize(640, 480)

        # === 布局 ===
        self.label_video = QLabel("Video feed")
        self.label_video.setScaledContents(True)
        self.button_open_cam = QPushButton("打开摄像头")
        self.button_open_file = QPushButton("打开视频文件")

        layout = QVBoxLayout()
        layout.addWidget(self.label_video)
        layout.addWidget(self.button_open_cam)
        layout.addWidget(self.button_open_file)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # === 线程与检测器初始化 ===
        self.thread = VideoThread()
        self.face_detector = FaceDetector("assets/face_detector.onnx")
        self.mark_detector = MarkDetector("assets/face_landmarks.onnx")
        # 这里不需要初始化 PoseEstimator，因为在 VideoThread 中会根据实际分辨率初始化
        # self.pose_estimator = PoseEstimator(640, 480)

        # 传递给工作线程
        self.thread.set_detectors(self.face_detector, self.mark_detector, None)

        # === 信号槽 ===
        self.button_open_cam.clicked.connect(self.open_camera)
        self.button_open_file.clicked.connect(self.open_file_dialog)
        self.thread.change_pixmap_signal.connect(self.update_image)

    def open_camera(self):
        """打开摄像头"""
        # 如果线程已经在跑，先停掉
        if self.thread.isRunning():
            self.thread.stop()

        self.thread.set_source_camera(0)
        self.thread.start()

    def open_file_dialog(self):
        """打开本地视频文件"""
        from PyQt5.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if filepath:
            if self.thread.isRunning():
                self.thread.stop()

            self.thread.set_source_video(filepath)
            self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """将工作线程发回的图像显示到 QLabel"""
        # cv_img是BGR格式，需要先转换为RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.label_video.width(), self.label_video.height(), Qt.KeepAspectRatio)
        self.label_video.setPixmap(QPixmap.fromImage(p))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
