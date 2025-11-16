import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog, QTextEdit,
    QToolButton, QSizePolicy, QFrame, QStackedWidget,
    QScrollArea
)
from PySide6.QtGui import QIcon, QPixmap, QImage, QMovie
from PySide6.QtCore import Qt, QTimer, QSize


class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Manager")
        self.setMinimumSize(1200, 700)

        self.camera = None
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self.update_webcam_frame)

        try:
            self.setStyleSheet(open("style.qss", "r", encoding="utf-8").read())
        except FileNotFoundError:
            print("Không tìm thấy file style.qss! Đảm bảo nó ở cùng thư mục.")

        main_layout = QHBoxLayout(self)

        left_widget = self.init_left_panel()
        right_widget = self.init_right_panel()

        main_layout.addWidget(left_widget, 3)
        main_layout.addWidget(right_widget, 7)

        self.setLayout(main_layout)

    # ========================================================================
    # PANEL BÊN TRÁI (KHÔNG THAY ĐỔI)
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
        upload_btn.setObjectName("uploadButton")
        upload_btn.clicked.connect(self.select_folder)

        left_layout.addWidget(QLabel("Họ và Tên"))
        left_layout.addWidget(self.name_input)
        left_layout.addStretch(1)

        left_layout.addWidget(QLabel("Ngày Sinh"))
        left_layout.addWidget(self.dob_input)
        left_layout.addStretch(1)

        left_layout.addWidget(QLabel("Ngành Học"))
        left_layout.addWidget(self.major_input)
        left_layout.addStretch(1)

        left_layout.addWidget(QLabel("Khóa"))
        left_layout.addWidget(self.course_input)
        left_layout.addStretch(2)

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

        left_layout.addStretch(1)

        left_card.setLayout(left_layout)
        return left_card

    # ========================================================================
    # KHỞI TẠO PANEL BÊN PHẢI (ĐÃ CHỈNH SỬA)
    # ========================================================================
    def init_right_panel(self):
        right_card = QGroupBox(" Điều Khiển Hệ Thống")
        # Layout chính của QGroupBox (chỉ chứa QScrollArea)
        card_main_layout = QVBoxLayout(right_card)
        card_main_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Tạo ScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)  # Bỏ viền
        scroll_area.setObjectName("resultsScrollArea")  # Để style QSS

        # 2. Tạo Widget chứa nội dung
        scroll_content_widget = QWidget()
        # --- SỬA LỖI: THÊM DÒNG NÀY ---
        scroll_content_widget.setObjectName("scrollContent")
        # --- KẾT THÚC SỬA LỖI ---

        # Đây là layout chính cho tất cả nội dung
        right_layout = QVBoxLayout(scroll_content_widget)

        # 3. Tạo các nút (như cũ)
        button_row = QHBoxLayout()

        self.train_btn = QToolButton()
        self.train_btn.setText("Train Model\nHuấn luyện lại AI")
        self.train_btn.setObjectName("trainButton")
        self.train_btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.train_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        train_icon_pixmap = QPixmap("img/Container.png")
        if not train_icon_pixmap.isNull():
            self.train_btn.setIcon(QIcon(train_icon_pixmap))
            self.train_btn.setIconSize(train_icon_pixmap.size() * 0.3)
        else:
            print("Không tìm thấy 'img/Container.png'.")

        self.test_btn = QToolButton()
        self.test_btn.setText("Test Webcam\nNhận diện ngay")
        self.test_btn.setObjectName("testButton")
        self.test_btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.test_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        test_icon_pixmap = QPixmap("img/webcam_icon.png")
        if not test_icon_pixmap.isNull():
            self.test_btn.setIcon(QIcon(test_icon_pixmap))
            self.test_btn.setIconSize(test_icon_pixmap.size() * 0.3)
        else:
            print("Không tìm thấy 'img/webcam_icon.png'.")

        # Kết nối nút
        self.test_btn.clicked.connect(self.start_webcam_mode)
        self.train_btn.clicked.connect(self.start_training_process)

        button_row.addWidget(self.train_btn)
        button_row.addWidget(self.test_btn)
        right_layout.addLayout(button_row)  # Thêm nút vào layout

        # 4. Tạo QStackedWidget (như cũ, nhưng thêm trang 2)
        self.main_stack = QStackedWidget()
        self.main_stack.setObjectName("mainStack")

        # Page 0: Giao diện "Sẵn sàng" (cái não)
        idle_widget = self.create_idle_widget()
        self.main_stack.addWidget(idle_widget)

        # Page 1: Giao diện "Webcam"
        webcam_widget = self.create_webcam_widget()
        self.main_stack.addWidget(webcam_widget)

        # Page 2: Giao diện "Loading" (MỚI)
        loading_widget = self.create_loading_widget()
        self.main_stack.addWidget(loading_widget)

        right_layout.addWidget(self.main_stack,1)  # Thêm stack vào layout

        # 5. Tạo Widget "Kết Quả Training" (MỚI)
        self.results_widget = self.create_results_widget()
        self.results_widget.setVisible(False)  # Ẩn đi lúc đầu
        right_layout.addWidget(self.results_widget)  # Thêm vào layout


        # 6. Gắn nội dung vào ScrollArea
        scroll_area.setWidget(scroll_content_widget)
        card_main_layout.addWidget(scroll_area)  # Thêm scroll vào layout thẻ

        return right_card

    # ========================================================================
    # HELPER WIDGETS (CÁC GIAO DIỆN CON)
    # ========================================================================

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
        self.webcam_label = QLabel("Nhấn 'Test Webcam' để bắt đầu")
        self.webcam_label.setObjectName("webcam_display")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(400, 300)
        webcam_layout.addWidget(self.webcam_label, 7)
        sidebar_layout = QVBoxLayout()
        result_card = self.create_result_card()
        sidebar_layout.addWidget(result_card)
        guide_card = self.create_guide_card()
        sidebar_layout.addWidget(guide_card)
        webcam_layout.addLayout(sidebar_layout, 3)
        return webcam_widget

    # Helper: Tạo thẻ "Kết Quả" (cho webcam)
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

    # Helper: Tạo thẻ "Hướng Dẫn" (cho webcam)
    def create_guide_card(self):
        card = QGroupBox(" ⓘ Hướng Dẫn Sử Dụng")
        card.setObjectName("guideCard")
        layout = QVBoxLayout(card)
        instructions = [
            "Nhìn thẳng vào camera", "Đảm bảo đủ ánh sáng",
            "Giữ khuôn mặt trong khung", "Không đeo khẩu trang"
        ]
        layout.addSpacing(10)
        for text in instructions:
            label = QLabel(f"• {text}")
            label.setProperty("class", "guideItem")
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch()
        return card

    # --------------------------------------------------------------------
    # Helper: Tạo Giao diện "Loading" (Page 2 - MỚI)
    def create_loading_widget(self):
        loading_widget = QWidget()
        loading_layout = QVBoxLayout(loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)

        # GIF (bạn cần có file "loading.gif")
        self.loading_label = QLabel()
        self.loading_label.setObjectName("loadingIcon")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_movie = QMovie("loading.gif")  # <-- TẢI FILE GIF
        self.loading_movie.setScaledSize(QSize(100, 100))  # Đặt kích thước
        self.loading_label.setMovie(self.loading_movie)

        loading_text_main = QLabel("Đang training model...")
        loading_text_main.setObjectName("loadingTextMain")  # Để style QSS
        loading_text_main.setAlignment(Qt.AlignCenter)

        loading_text_sub = QLabel("Đang cập nhật dữ liệu mới")
        loading_text_sub.setObjectName("loadingTextSub")  # Để style QSS
        loading_text_sub.setAlignment(Qt.AlignCenter)

        loading_layout.addStretch()
        loading_layout.addWidget(self.loading_label)
        loading_layout.addWidget(loading_text_main)
        loading_layout.addWidget(loading_text_sub)
        loading_layout.addStretch()

        return loading_widget

    # --------------------------------------------------------------------
    # Helper: Tạo Giao diện "Kết Quả Training" (Widget ẩn - MỚI)
    def create_results_widget(self):
        results_group = QGroupBox("Kết Quả Training")
        results_group.setObjectName("resultsCard")  # Tên để style
        results_layout = QVBoxLayout(results_group)

        # 1. Các thẻ chỉ số (Accuracy, Precision...)
        stats_group = QGroupBox()  # Groupbox trong suốt
        stats_group.setObjectName("transparentGroup")
        stats_layout = QHBoxLayout(stats_group)

        # Placeholder cho 4 thẻ
        stats_layout.addWidget(self.create_stat_box("Độ Chính Xác", "96.5%", "statBoxGreen"))
        stats_layout.addWidget(self.create_stat_box("Precision", "96.5%", "statBoxBlue"))
        stats_layout.addWidget(self.create_stat_box("Recall", "95.8%", "statBoxRed"))
        stats_layout.addWidget(self.create_stat_box("F1-Score", "96.1%", "statBoxYellow"))

        # 2. Các biểu đồ đường
        charts_group = QGroupBox()
        charts_group.setObjectName("transparentGroup")
        charts_layout = QHBoxLayout(charts_group)

        # Placeholder cho 2 biểu đồ
        chart1 = QLabel("Biểu Đồ Độ Chính Xác (Placeholder)")
        chart1.setMinimumHeight(250)
        chart1.setAlignment(Qt.AlignCenter)
        chart1.setObjectName("chartPlaceholder")

        chart2 = QLabel("Biểu Đồ Loss (Placeholder)")
        chart2.setMinimumHeight(250)
        chart2.setAlignment(Qt.AlignCenter)
        chart2.setObjectName("chartPlaceholder")

        charts_layout.addWidget(chart1)
        charts_layout.addWidget(chart2)

        # 3. Biểu đồ cột
        barchart_group = QGroupBox("Các Chỉ Số Đánh Giá")
        barchart_group.setObjectName("transparentGroup")
        barchart_layout = QVBoxLayout(barchart_group)

        barchart = QLabel("Biểu Đồ Cột (Placeholder)")
        barchart.setMinimumHeight(250)
        barchart.setAlignment(Qt.AlignCenter)
        barchart.setObjectName("chartPlaceholder")
        barchart_layout.addWidget(barchart)

        # 4. Thanh trạng thái
        status_bar = QFrame()
        status_bar.setObjectName("trainingSuccessBar")  # Để style
        status_layout = QHBoxLayout(status_bar)

        status_icon = QLabel("✅")  # Icon
        status_icon.setObjectName("successIcon")

        status_text = QLabel("Training hoàn tất thành công!\nModel đã đạt độ chính xác 96.5%...")
        status_text.setObjectName("successText")
        status_text.setWordWrap(True)

        status_layout.addWidget(status_icon)
        status_layout.addWidget(status_text, 1)

        # Thêm tất cả vào layout chính
        results_layout.addWidget(stats_group)
        results_layout.addWidget(charts_group)
        results_layout.addWidget(barchart_group)
        results_layout.addWidget(status_bar)

        return results_group

    # Helper: Tạo 1 thẻ chỉ số
    def create_stat_box(self, title, value, object_name):
        box = QGroupBox(title)
        box.setObjectName(object_name)  # Để style QSS (vd: statBoxGreen)
        layout = QVBoxLayout(box)

        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(value_label)
        return box

    # ========================================================================
    # CHỨC NĂNG (ĐÃ CHỈNH SỬA)
    # ========================================================================

    # Hàm reset chung
    def reset_to_idle(self):
        self.stop_webcam()  # Tắt cam (nếu đang chạy)
        self.results_widget.setVisible(False)  # Ẩn kết quả
        self.main_stack.setCurrentIndex(0)  # Về màn hình "Sẵn sàng"

    # --------------------------------------------------------------------
    # Chức năng Training (MỚI)
    def start_training_process(self):
        self.reset_to_idle()  # Reset về trạng thái ban đầu
        self.main_stack.setCurrentIndex(2)  # Chuyển sang màn hình Loading
        self.loading_movie.start()  # Bắt đầu xoay

        # Cập nhật nút
        self.train_btn.setText("Đang Train...\nHuấn luyện lại AI")
        self.train_btn.setEnabled(False)  # Tắt nút Train
        self.test_btn.setEnabled(False)  # Tắt nút Test

        # Mô phỏng training trong 4 giây
        QTimer.singleShot(4000, self.on_training_complete)

    def on_training_complete(self):
        self.loading_movie.stop()  # Dừng xoay
        self.main_stack.setCurrentIndex(0)  # Quay về màn hình "Sẵn sàng"

        # HIỆN KẾT QUẢ
        self.results_widget.setVisible(True)

        # Khôi phục các nút
        self.train_btn.setText("Train Model\nHuấn luyện lại AI")
        self.train_btn.setEnabled(True)
        self.test_btn.setEnabled(True)

    # --------------------------------------------------------------------
    # Chức năng Webcam (Chỉnh sửa)
    def start_webcam_mode(self):
        if self.webcam_timer.isActive():
            self.reset_to_idle()  # Nếu đang chạy thì reset
        else:
            self.reset_to_idle()  # Reset trước
            self.start_webcam()  # Mới bắt đầu

    def start_webcam(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.webcam_label.setText("Lỗi: Không thể mở camera.")
            self.camera = None
            return

        self.webcam_timer.start(30)
        self.test_btn.setText("Stop Webcam\nDừng nhận diện")
        self.main_stack.setCurrentIndex(1)  # Chuyển sang giao diện webcam

    def stop_webcam(self):
        self.webcam_timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None

        self.test_btn.setText("Test Webcam\nNhận diện ngay")
        self.webcam_label.setText("Đã tắt camera.")
        self.webcam_label.setPixmap(QPixmap())

    def update_webcam_frame(self):
        if not self.camera:
            return
        ret, frame = self.camera.read()
        if not ret:
            self.webcam_label.setText("Lỗi: Mất kết nối camera.")
            self.stop_webcam()
            return
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ... (code nhận diện của bạn sẽ ở đây) ...
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = qt_pixmap.scaled(
            self.webcam_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.webcam_label.setPixmap(scaled_pixmap)

    # --------------------------------------------------------------------
    # Chức năng khác
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn Folder Ảnh")
        if folder:
            print("Đã chọn folder:", folder)

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