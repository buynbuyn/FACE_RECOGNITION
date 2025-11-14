import cv2
import os
import numpy as np
import json
from utils import calculate_lbp, extract_lbp_features

# Đường dẫn
dataset_path = "dataset"
cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
features_path = "features.npy"
labels_path = "labels.npy"
label_map_path = "labels.json"
info_path = "info.json"

print("📂 Thư mục đang chạy:", os.getcwd())

# Khởi tạo bộ phát hiện khuôn mặt
face_detector = cv2.CascadeClassifier(cascade_path)
if face_detector.empty():
    print("❌ Không thể load Haar Cascade.")
    exit()

features_list = []
labels_list = []
label_map = {}
current_label = 0

# Duyệt qua từng thư mục người dùng
for person_name in sorted(os.listdir(dataset_path)):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"\n🔍 Đang xử lý: {person_name}")
    label_map[current_label] = person_name
    face_count = 0

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path)

        if img is None:
            print(f"⚠️ Không đọc được ảnh: {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rect = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces_rect) == 0:
            print(f"⚠️ Không tìm thấy khuôn mặt trong ảnh: {image_name}")
            continue

        for (x, y, w, h) in faces_rect:
            face = gray[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face, (64, 64))
                lbp_img = calculate_lbp(face_resized)
                features = extract_lbp_features(lbp_img)

                features_list.append(features)
                labels_list.append(current_label)
                face_count += 1
            except Exception as e:
                print(f"❌ Lỗi xử lý ảnh {image_name}: {e}")

    print(f"✅ {person_name} → label = {current_label} ({face_count} khuôn mặt)")
    current_label += 1

# Lưu đặc trưng và nhãn
if features_list:
    np.save(features_path, np.array(features_list))
    np.save(labels_path, np.array(labels_list))
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
    print("\n✅ Đã trích xuất đặc trưng LBP và lưu dữ liệu huấn luyện!")
else:
    print("\n❌ Không tìm thấy khuôn mặt nào để huấn luyện.")
    exit()

# Kiểm tra info.json
if os.path.exists(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        info_map = json.load(f)
else:
    info_map = {}

# Báo người chưa có thông tin
missing_info = []
for label, name in label_map.items():
    if name not in info_map:
        missing_info.append(name)

if missing_info:
    print("\n⚠️ Những người chưa có thông tin trong info.json:")
    for name in missing_info:
        print(f" - {name}")
    print("👉 Vui lòng nhập thông tin qua UI trước khi huấn luyện hoàn chỉnh.")
else:
    print("📄 Tất cả người dùng đã có thông tin trong info.json.")