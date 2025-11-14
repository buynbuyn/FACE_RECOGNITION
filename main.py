import sys
import cv2  # Import OpenCV
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog, QTextEdit,
    QToolButton, QSizePolicy, QFrame, QStackedWidget
)
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtCore import Qt, QTimer


class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Manager")
        self.setMinimumSize(1200, 700)  # Tăng chiều cao 1 chút

        # Biến cho camera
        self.camera = None
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self.update_webcam_frame)

        try:
            self.setStyleSheet(open("style.qss", "r", encoding="utf-8").read())
        except FileNotFoundError:
            print("Không tìm thấy file style.qss! Đảm bảo nó ở cùng thư mục.")

        main_layout = QHBoxLayout(self)

        # Khởi tạo 2 panel
        left_widget = self.init_left_panel()
        right_widget = self.init_right_panel()

        # Add panels to main layout
        main_layout.addWidget(left_widget, 3)  # Panel trái, tỉ lệ 3
        main_layout.addWidget(right_widget, 7)  # Panel phải, tỉ lệ 7 (lớn hơn)

        self.setLayout(main_layout)

    # ========================================================================
    # KHỞI TẠO PANEL BÊN TRÁI
    # ========================================================================
    def init_left_panel(self):
        left_card = QGroupBox(" Thêm Người Mới")
        left_layout = QVBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nhập họ tên")
        self.dob_input = QLineEdit()
        self.dob_input.setPlaceholderText("dd/mm/yyyy")
        self.major_input = QLineEdit()
        self.major_input.setPlaceholderText("Ví dụ: Công nghệ thông tin")
        self.course_input = QLineEdit()
        self.course_input.setPlaceholderText("Ví dụ: K16")

        upload_btn = QPushButton("Chọn Folder Ảnh")
        upload_btn.setObjectName("uploadButton")  # Đặt tên riêng
        upload_btn.clicked.connect(self.select_folder)

        left_layout.addWidget(QLabel("Họ và Tên"))
        left_layout.addWidget(self.name_input)
        left_layout.addStretch(1)  # Tự động giãn cách

        left_layout.addWidget(QLabel("Ngày Sinh"))
        left_layout.addWidget(self.dob_input)
        left_layout.addStretch(1)

        left_layout.addWidget(QLabel("Ngành Học"))
        left_layout.addWidget(self.major_input)
        left_layout.addStretch(1)

        left_layout.addWidget(QLabel("Khóa"))
        left_layout.addWidget(self.course_input)
        left_layout.addStretch(2)  # Giãn cách lớn hơn

        # Gạch ngang
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line.setObjectName("separator")
        left_layout.addWidget(separator_line)

        left_layout.addStretch(2)

        left_layout.addWidget(QLabel("Upload Ảnh Người Mới"))
        left_layout.addWidget(upload_btn)
        left_layout.addStretch(2)

        ready_label = QLabel("● Model đã sẵn sàng nhận diện")
        ready_label.setObjectName("status_ready")
        left_layout.addWidget(ready_label)

        description_label = QLabel(
            "Hệ thống đã có dữ liệu training sẵn. "
            "Bạn có thể test ngay hoặc train lại để cập nhật model."
        )
        description_label.setObjectName("status_description")
        description_label.setWordWrap(True)
        left_layout.addWidget(description_label)

        left_layout.addStretch(1)  # Giãn cách nhỏ ở cuối

        left_card.setLayout(left_layout)
        return left_card

    # ========================================================================
    # KHỞI TẠO PANEL BÊN PHẢI
    # ========================================================================
    def init_right_panel(self):
        right_card = QGroupBox(" Điều Khiển Hệ Thống")
        right_layout = QVBoxLayout()

        button_row = QHBoxLayout()

        # Nút Train
        self.train_btn = QToolButton()
        self.train_btn.setText("Train Model\nHuấn luyện lại AI")
        self.train_btn.setObjectName("trainButton")
        self.train_btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.train_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        train_icon_pixmap = QPixmap("img/Container.png")  # Icon của bạn
        if not train_icon_pixmap.isNull():
            self.train_btn.setIcon(QIcon(train_icon_pixmap))
            self.train_btn.setIconSize(train_icon_pixmap.size() * 0.3)  # 50%
        else:
            print("Không tìm thấy 'img/Container.png' cho nút Train.")

        # Nút Test
        self.test_btn = QToolButton()
        self.test_btn.setText("Test Webcam\nNhận diện ngay")
        self.test_btn.setObjectName("testButton")
        self.test_btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.test_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        test_icon_pixmap = QPixmap("img/webcam.png")  # Icon của bạn
        if not test_icon_pixmap.isNull():
            self.test_btn.setIcon(QIcon(test_icon_pixmap))
            self.test_btn.setIconSize(test_icon_pixmap.size() * 0.3)  # 50%
        else:
            print("Không tìm thấy 'img/webcam.png' cho nút Test.")

        # Kết nối nút Test
        self.test_btn.clicked.connect(self.toggle_webcam)
        # Nút Train sẽ tắt webcam
        self.train_btn.clicked.connect(self.stop_webcam_and_reset)

        button_row.addWidget(self.train_btn)
        button_row.addWidget(self.test_btn)
        right_layout.addLayout(button_row)

        # --- TẠO QSTACKEDWIDGET ĐỂ CHUYỂN ĐỔI GIAO DIỆN ---
        self.main_stack = QStackedWidget()

        # Page 0: Giao diện "Sẵn sàng" (cái não)
        idle_widget = self.create_idle_widget()
        self.main_stack.addWidget(idle_widget)

        # Page 1: Giao diện "Webcam"
        webcam_widget = self.create_webcam_widget()
        self.main_stack.addWidget(webcam_widget)

        right_layout.addWidget(self.main_stack)
        # --------------------------------------------------

        right_card.setLayout(right_layout)
        return right_card

    # --------------------------------------------------------------------
    # Helper: Tạo Giao diện "Sẵn Sàng" (Page 0)
    def create_idle_widget(self):
        idle_widget = QWidget()
        idle_layout = QVBoxLayout(idle_widget)
        idle_layout.setAlignment(Qt.AlignCenter)

        status_icon = QLabel("🧠")
        status_icon.setObjectName("idleIcon")
        status_icon.setAlignment(Qt.AlignCenter)

        status_text_main = QLabel("Model đã sẵn sàng")
        status_text_main.setObjectName("statusTextMain")
        status_text_main.setAlignment(Qt.AlignCenter)

        status_text_sub = QLabel("Nhấn Test Webcam để bắt đầu nhận diện")
        status_text_sub.setObjectName("statusTextSub")
        status_text_sub.setAlignment(Qt.AlignCenter)

        idle_layout.addStretch()
        idle_layout.addWidget(status_icon)
        idle_layout.addWidget(status_text_main)
        idle_layout.addWidget(status_text_sub)
        idle_layout.addStretch()

        return idle_widget

    # --------------------------------------------------------------------
    # Helper: Tạo Giao diện "Webcam" (Page 1)
    def create_webcam_widget(self):
        webcam_widget = QWidget()
        webcam_layout = QHBoxLayout(webcam_widget)

        # 1. Cửa sổ Webcam (bên trái)
        self.webcam_label = QLabel("Nhấn 'Test Webcam' để bắt đầu")
        self.webcam_label.setObjectName("webcam_display")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(400, 300)  # Kích thước tối thiểu
        webcam_layout.addWidget(self.webcam_label, 7)  # Tỉ lệ 7

        # 2. Sidebar (bên phải)
        sidebar_layout = QVBoxLayout()

        # 2a. Thẻ Kết quả
        result_card = self.create_result_card()
        sidebar_layout.addWidget(result_card)

        # 2b. Thẻ Hướng dẫn
        guide_card = self.create_guide_card()
        sidebar_layout.addWidget(guide_card)

        webcam_layout.addLayout(sidebar_layout, 3)  # Tỉ lệ 3

        return webcam_widget

    # Helper: Tạo thẻ "Kết Quả"
    def create_result_card(self):
        card = QGroupBox(" ● Kết Quả Nhận Diện")
        card.setObjectName("resultCard")
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignCenter)

        self.result_icon = QLabel("👤")
        self.result_icon.setObjectName("resultIcon")
        self.result_icon.setAlignment(Qt.AlignCenter)

        self.result_text_main = QLabel("Không phát hiện khuôn mặt")
        self.result_text_main.setObjectName("resultTextMain")
        self.result_text_main.setAlignment(Qt.AlignCenter)

        self.result_text_sub = QLabel("Vui lòng đứng trước camera")
        self.result_text_sub.setObjectName("resultTextSub")
        self.result_text_sub.setAlignment(Qt.AlignCenter)

        layout.addStretch(1)
        layout.addWidget(self.result_icon)
        layout.addWidget(self.result_text_main)
        layout.addWidget(self.result_text_sub)
        layout.addStretch(2)

        return card

    # Helper: Tạo thẻ "Hướng Dẫn"
    def create_guide_card(self):
        card = QGroupBox(" ⓘ Hướng Dẫn Sử Dụng")
        card.setObjectName("guideCard")
        layout = QVBoxLayout(card)

        instructions = [
            "Nhìn thẳng vào camera",
            "Đảm bảo đủ ánh sáng",
            "Giữ khuôn mặt trong khung",
            "Không đeo khẩu trang"
        ]

        layout.addSpacing(10)  # Khoảng cách từ tiêu đề

        for text in instructions:
            label = QLabel(f"• {text}")
            label.setProperty("class", "guideItem")  # Đặt class để CSS
            label.setWordWrap(True)
            layout.addWidget(label)

        layout.addStretch()  # Đẩy mọi thứ lên trên
        return card

    # ========================================================================
    # CHỨC NĂNG WEBCAM
    # ========================================================================

    def toggle_webcam(self):
        if self.webcam_timer.isActive():
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.camera = cv2.VideoCapture(0)  # Mở camera
        if not self.camera.isOpened():
            self.webcam_label.setText("Lỗi: Không thể mở camera.")
            self.camera = None
            return

        self.webcam_timer.start(30)  # Chạy timer (khoảng 33 FPS)
        self.test_btn.setText("Stop Webcam\nDừng nhận diện")
        self.main_stack.setCurrentIndex(1)  # Chuyển sang giao diện webcam

    def stop_webcam(self):
        self.webcam_timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None

        self.test_btn.setText("Test Webcam\nNhận diện ngay")
        self.webcam_label.setText("Đã tắt camera.")
        self.webcam_label.setPixmap(QPixmap())  # Xóa hình ảnh

    def stop_webcam_and_reset(self):
        self.stop_webcam()
        self.main_stack.setCurrentIndex(0)  # Về giao diện "Sẵn sàng"

    def update_webcam_frame(self):
        if not self.camera:
            return

        ret, frame = self.camera.read()
        if not ret:
            self.webcam_label.setText("Lỗi: Mất kết nối camera.")
            self.stop_webcam()
            return

        # 1. Xử lý ảnh (lật và đổi màu)
        frame = cv2.flip(frame, 1)  # Lật ngang
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. (CHƯA LÀM) Nhận diện khuôn mặt
        # ... code nhận diện của bạn sẽ ở đây ...
        # Dựa vào kết quả, bạn sẽ cập nhật 2 dòng text
        # self.result_text_main.setText("Đã nhận diện: ABC")
        # self.result_text_sub.setText("MSSV: 123456")

        # 3. Chuyển đổi sang QPixmap
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_pixmap = QPixmap.fromImage(qt_image)

        # 4. Hiển thị ảnh (scale cho vừa)
        scaled_pixmap = qt_pixmap.scaled(
            self.webcam_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.webcam_label.setPixmap(scaled_pixmap)

    # ========================================================================
    # CÁC CHỨC NĂNG KHÁC
    # ========================================================================

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn Folder Ảnh")
        if folder:
            print("Đã chọn folder:", folder)

    # Đảm bảo tắt camera khi đóng cửa sổ
    def closeEvent(self, event):
        self.stop_webcam()
        event.accept()


# ========================================================================
# CHẠY ỨNG DỤNG
# ========================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec())