import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import time

# Hàm trích xuất đặc trưng từ ResNet
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().cpu().numpy()

# Load dữ liệu và trích xuất đặc trưng
def load_data(folder_path, label):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file_name)
        if img_path.endswith(('jpg', 'png', 'jpeg')):
            features = extract_features(img_path)
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

# Hàm dự đoán loài và nguy hại
def predict_image(image_path):
    start = time.time()
    features = extract_features(image_path).reshape(1, -1)
    label = knn.predict(features)[0]
    end = time.time()
    if label == 0:
        return "Ếch - Không nguy hại"
    elif label == 1:
        return "Châu chấu - Nguy hại"
    elif label == 2:
        return "Chuột - Nguy hại"

# Kiểm tra xem GPU có sẵn hay không, nếu không thì dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sử dụng weights thay vì pretrained
weights = ResNet50_Weights.IMAGENET1K_V1
resnet = resnet50(weights=weights).to(device)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Bỏ lớp cuối
resnet.eval()

# Hàm tiền xử lý ảnh
transform = transforms.Compose([
    transforms.ToPILImage(),  # Chuyển từ OpenCV numpy array sang PIL Image
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Đo thời gian bắt đầu
start_time = time.time()

# Đường dẫn đến thư mục ảnh
frog_folder = "frog"
grasshopper_folder = "grasshopper"
rat_folder = "rat"

# Load dữ liệu và đo thời gian
print("Đang load dữ liệu...")
frog_data, frog_labels = load_data(frog_folder, 0)
grasshopper_data, grasshopper_labels = load_data(grasshopper_folder, 1)
mouse_data, mouse_labels = load_data(rat_folder, 2)

# Gộp tất cả dữ liệu
X = np.vstack((frog_data, grasshopper_data, mouse_data))
y = np.hstack((frog_labels, grasshopper_labels, mouse_labels))

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Kiểm tra và dự đoán
y_pred = knn.predict(X_test)
print("Độ chính xác:", round(accuracy_score(y_test, y_pred), 2))

# Ví dụ
print(predict_image("rat/rat_11.jpeg"))

# Tổng thời gian
total_time = time.time() - start_time
print(f"Tổng thời gian chạy chương trình: {total_time:.2f} giây")
