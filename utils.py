import numpy as np
import cv2

# Tính toán LBP cho ảnh xám
def calculate_lbp(gray_img):
    lbp_img = np.zeros_like(gray_img)
    for i in range(1, gray_img.shape[0] - 1):
        for j in range(1, gray_img.shape[1] - 1):
            center = gray_img[i, j]
            binary = ''
            binary += '1' if gray_img[i-1, j-1] >= center else '0'
            binary += '1' if gray_img[i-1, j] >= center else '0'
            binary += '1' if gray_img[i-1, j+1] >= center else '0'
            binary += '1' if gray_img[i, j+1] >= center else '0'
            binary += '1' if gray_img[i+1, j+1] >= center else '0'
            binary += '1' if gray_img[i+1, j] >= center else '0'
            binary += '1' if gray_img[i+1, j-1] >= center else '0'
            binary += '1' if gray_img[i, j-1] >= center else '0'
            lbp_img[i, j] = int(binary, 2)
    return lbp_img

# Trích xuất đặc trưng từ ảnh LBP bằng cách chia lưới và tạo histogram
def extract_lbp_features(lbp_img, grid_size=(8, 8)):
    h, w = lbp_img.shape
    gh, gw = grid_size
    dh, dw = h // gh, w // gw
    features = []
    for i in range(gh):
        for j in range(gw):
            cell = lbp_img[i*dh:(i+1)*dh, j*dw:(j+1)*dw]
            hist, _ = np.histogram(cell, bins=256, range=(0, 256))
            features.extend(hist)
    return np.array(features)