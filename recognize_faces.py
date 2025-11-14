import cv2
import numpy as np
import json
from utils import calculate_lbp, extract_lbp_features

# Load dữ liệu huấn luyện
features = np.load("features.npy")
labels = np.load("labels.npy")
with open("labels.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
with open("info.json", "r", encoding="utf-8") as f:
    info_map = json.load(f)

# Kiểm tra dữ liệu huấn luyện
if len(features) == 0 or len(labels) == 0:
    print("Dữ liệu huấn luyện trống. Vui lòng huấn luyện trước.")
    exit()

# Load mô hình Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

# Hàm dự đoán người với ngưỡng khoảng cách
def predict(test_vector, threshold=150):
    min_dist = float("inf")
    predicted_label = -1
    for i, vec in enumerate(features):
        dist = np.linalg.norm(test_vector - vec)
        if dist < min_dist:
            min_dist = dist
            predicted_label = labels[i]
    if min_dist > threshold:
        return "Unknown", min_dist
    return label_map.get(str(predicted_label), "Unknown"), min_dist

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Lật ngang (gương)
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        lbp_img = calculate_lbp(face_resized)
        test_vector = extract_lbp_features(lbp_img)
        name, dist = predict(test_vector)

        # Vẽ khung
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        if name != "Unknown" and name in info_map:
            info = info_map[name]
            ho_ten = info.get("ten", name)
            nam_sinh = info.get("nam_sinh", "")
            gioi_tinh = info.get("gioi_tinh", "")
            nganh = info.get("nganh", "")
            khoa = info.get("khoa", "")
            sdt = info.get("sdt", "")

            # Hiển thị thông tin
            cv2.putText(frame, f"{ho_ten}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"{gioi_tinh}, NS: {nam_sinh}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"{nganh} - Khóa {khoa}", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"SDT: {sdt}", (x, y + h + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:  # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()