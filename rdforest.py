import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

# 1. Chuẩn bị mô hình ResNet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 với trọng số từ ImageNet
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Loại bỏ lớp cuối cùng
resnet.eval()

# Hàm tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hàm trích xuất đặc trưng từ ResNet50
def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().cpu().numpy()

# 2. Chuẩn bị dữ liệu
def load_images_with_labels(folder):
    data = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is not None:
            # Gắn nhãn dựa trên tên file
            if "frog" in filename.lower():
                label = 0  # Ếch không gây hại
            elif "rat" in filename.lower():
                label = 1  # Chuột gây hại
            elif "grasshopper" in filename.lower():
                label = 2  # Châu chấu gây hại
            else:
                continue  # Bỏ qua file không hợp lệ
            
            # Trích xuất đặc trưng
            features = extract_features(filepath)
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

# Đường dẫn tới folder chứa tất cả ảnh
data_folder = "D:/random/data"

# Load dữ liệu và gắn nhãn
X, y = load_images_with_labels(data_folder)

# 3. Chia dữ liệu thành tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

# 6. Dự đoán trên ảnh mới
def predict_image(image_path):
    features = extract_features(image_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    if prediction == 0:
        return "Ếch - Không nguy hại"
    elif prediction == 1:
        return "Chuột - Nguy hại"
    elif prediction == 2:
        return "Châu chấu - Nguy hại"

# Ví dụ dự đoán một ảnh mới
new_image_path = "D:/random/data/frog_319.jpeg"  # Đường dẫn tới ảnh cần dự đoán
result = predict_image(new_image_path)
print("Kết quả dự đoán:", result)
