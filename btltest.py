import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# 1. Load và tiền xử lý dữ liệu
def load_data(folder_path, label):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))  # Resize ảnh về kích thước 64x64
            data.append(img_resized.flatten())  # Biến ảnh thành vector
            labels.append(label)
    return np.array(data), np.array(labels)

start_time = time.time()

# Đường dẫn đến thư mục ảnh
frog_folder = "frog"
grasshopper_folder = "grasshopper"
mouse_folder = "rat"

frog_data, frog_labels = load_data(frog_folder, 0)  # 0: ếch
grasshopper_data, grasshopper_labels = load_data(grasshopper_folder, 1)  # 1: châu chấu
mouse_data, mouse_labels = load_data(mouse_folder, 2)  # 2: chuột

# Gộp tất cả dữ liệu
X = np.vstack((frog_data, grasshopper_data, mouse_data))
y = np.hstack((frog_labels, grasshopper_labels, mouse_labels))

# 2. Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)  # K = 5
knn.fit(X_train, y_train)

# 4. Kiểm tra và dự đoán
y_pred = knn.predict(X_test)
print("Độ chính xác:", accuracy_score(y_test, y_pred))

# Dự đoán loài và nguy hại
def predict_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (64, 64))
    img_vector = img_resized.flatten().reshape(1, -1)
    label = knn.predict(img_vector)[0]
    if label == 0:
        return "Ếch - Không nguy hại"
    elif label == 1:
        return "Châu chấu - Nguy hại"
    elif label == 2:
        return "Chuột - Nguy hại"

# Ví dụ
print(predict_image("frog/frog_11.jpeg"))

total_time = time.time() - start_time
print(f"Tổng thời gian chạy chương trình: {total_time:.2f} giây")
