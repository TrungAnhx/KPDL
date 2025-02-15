import os
import numpy as np
import cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

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
def predict_image(image_path, model):
    features = extract_features(image_path).reshape(1, -1)
    label = model.predict(features)[0]
    if label == 0:
        return "Ếch - Không nguy hại"
    elif label == 1:
        return "Châu chấu - Nguy hại"
    elif label == 2:
        return "Chuột - Nguy hại"

# Huấn luyện mô hình KNN
def train_knn(source_folder):
    frog_folder = os.path.join(source_folder, "frog")
    grasshopper_folder = os.path.join(source_folder, "grasshopper")
    rat_folder = os.path.join(source_folder, "rat")

    frog_data, frog_labels = load_data(frog_folder, 0)
    grasshopper_data, grasshopper_labels = load_data(grasshopper_folder, 1)
    mouse_data, mouse_labels = load_data(rat_folder, 2)

    X = np.vstack((frog_data, grasshopper_data, mouse_data))
    y = np.hstack((frog_labels, grasshopper_labels, mouse_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)
    print("Độ chính xác KNN:", acc)

    return knn, acc

# Huấn luyện mô hình Random Forest
def train_rf(source_folder):
    frog_folder = os.path.join(source_folder, "frog")
    grasshopper_folder = os.path.join(source_folder, "grasshopper")
    rat_folder = os.path.join(source_folder, "rat")

    frog_data, frog_labels = load_data(frog_folder, 0)
    grasshopper_data, grasshopper_labels = load_data(grasshopper_folder, 1)
    mouse_data, mouse_labels = load_data(rat_folder, 2)

    X = np.vstack((frog_data, grasshopper_data, mouse_data))
    y = np.hstack((frog_labels, grasshopper_labels, mouse_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)
    print("Độ chính xác Random Forest:", acc)

    return rf, acc

# Huấn luyện mô hình Naive Bayes
def train_nb(source_folder):
    frog_folder = os.path.join(source_folder, "frog")
    grasshopper_folder = os.path.join(source_folder, "grasshopper")
    rat_folder = os.path.join(source_folder, "rat")

    frog_data, frog_labels = load_data(frog_folder, 0)
    grasshopper_data, grasshopper_labels = load_data(grasshopper_folder, 1)
    mouse_data, mouse_labels = load_data(rat_folder, 2)

    X = np.vstack((frog_data, grasshopper_data, mouse_data))
    y = np.hstack((frog_labels, grasshopper_labels, mouse_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)
    print("Độ chính xác Naive Bayes:", acc)

    return nb, acc

class Worker(QThread):
    finished = pyqtSignal(object)  # emit kết quả (str)

    def __init__(self, source_folder, target_image, selected_algorithm):
        super().__init__()
        self.source_folder = source_folder
        self.target_image = target_image
        self.selected_algorithm = selected_algorithm

    def run(self):
        if self.selected_algorithm == 0:
            model, acc = train_knn(self.source_folder)
        elif self.selected_algorithm == 1:
            model, acc = train_rf(self.source_folder)
        elif self.selected_algorithm == 2:
            model, acc = train_nb(self.source_folder)
        else:
            self.finished.emit("Invalid algorithm selected")
            return

        prediction = predict_image(self.target_image, model)
        self.finished.emit(f"Độ chính xác: {acc} | {prediction}")

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 482)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Button to run
        self.pushButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(270, 390, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(18)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton {\n"
                                      "    border: 2px solid black;\n"
                                      "    border-radius: 20px;\n"
                                      "    background-color: rgb(252, 251, 231);\n"
                                      "}\n"
                                      "QPushButton:hover {\n"
                                      "    background-color: rgb(190, 247, 223);\n"
                                      "}\n"
                                      "QPushButton:pressed {\n"
                                      "    background-color: rgb(252, 251, 231);\n"
                                      "}")
        self.pushButton.setObjectName("pushButton")

        # Background
        self.background = QtWidgets.QLabel(parent=self.centralwidget)
        self.background.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.background.setStyleSheet("border-image: url(C:/Users/TrungAnhx/Documents/Python stuff/KPDL/dd.png);")
        self.background.setText("")
        self.background.setObjectName("background")

        # ComboBox for selecting algorithm
        self.comboBox = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(290, 260, 220, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.comboBox.setFont(font)
        self.comboBox.setStyleSheet("background-color: rgb(252, 251, 231);")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Thuật toán 1: KNN")
        self.comboBox.addItem("Thuật toán 2: Random Forest")
        self.comboBox.addItem("Thuật toán 3: Naive Bayes")

        # Label for algorithm selection
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 260, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setStyleSheet("border: 2px solid black;\n"
                                 "border-radius: 20px;\n"
                                 "background-color: rgb(65, 196, 172);")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")

        # Button to browse source folder
        self.pushButton_2 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(500, 180, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("QPushButton {\n"
                                        "    border: 2px solid black;\n"
                                        "    border-radius: 20px;\n"
                                        "    background-color: rgb(252, 251, 231);\n"
                                        "}\n"
                                        "QPushButton:hover {\n"
                                        "    background-color: rgb(190, 247, 223);\n"
                                        "}\n"
                                        "QPushButton:pressed {\n"
                                        "    background-color: rgb(252, 251, 231);\n"
                                        "}")
        self.pushButton_2.setObjectName("pushButton_2")

        # QLabel to display the source folder path
        self.label_source = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_source.setGeometry(QtCore.QRect(125, 180, 380, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.label_source.setFont(font)
        self.label_source.setStyleSheet("    border: 2px solid black;\n"
                                        "    border-radius: 20px;\n"
                                        "    background-color: rgb(252, 251, 231);")
        self.label_source.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_source.setObjectName("label_source")

        # Label for "Source Folder"
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(120, 180, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("border: 2px solid black;\n"
                                   "border-radius: 20px;\n"
                                   "background-color: rgb(65, 196, 172);")
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3.setObjectName("label_3")

        # Button to browse target image
        self.pushButton_4 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(520, 220, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("QPushButton {\n"
                                        "    border: 2px solid black;\n"
                                        "    border-radius: 20px;\n"
                                        "    background-color: rgb(252, 251, 231);\n"
                                        "}\n"
                                        "QPushButton:hover {\n"
                                        "    background-color: rgb(190, 247, 223);\n"
                                        "}\n"
                                        "QPushButton:pressed {\n"
                                        "    background-color: rgb(252, 251, 231);\n"
                                        "}")
        self.pushButton_4.setObjectName("pushButton_4")

        # QLabel to display the target image path
        self.label_target = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_target.setGeometry(QtCore.QRect(125, 220, 401, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.label_target.setFont(font)
        self.label_target.setStyleSheet("    border: 2px solid black;\n"
                                        "    border-radius: 20px;\n"
                                        "    background-color: rgb(252, 251, 231);")
        self.label_target.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_target.setObjectName("label_target")

        # Label cho "Target Image"
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(80, 220, 50, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("border: 2px solid black;\n"
                                   "border-radius: 20px;\n"
                                   "background-color: rgb(65, 196, 172);")
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label_4")

        # New label to display status and results
        self.resultLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.resultLabel.setGeometry(QtCore.QRect(150, 450, 341, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.resultLabel.setFont(font)
        self.resultLabel.setStyleSheet("border: 2px solid black; border-radius: 20px; background-color: rgb(252, 251, 231); padding: 0px; margin: 0px;")
        self.resultLabel.setContentsMargins(0, 0, 0, 0)
        self.resultLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.resultLabel.setObjectName("resultLabel")
        self.resultLabel.setText("")  # Ban đầu để trống

        self.background.raise_()
        self.pushButton.raise_()
        self.comboBox.raise_()
        self.label.raise_()
        self.pushButton_2.raise_()
        self.label_source.raise_()
        self.pushButton_4.raise_()
        self.label_target.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.resultLabel.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Connect buttons to functions
        self.pushButton_2.clicked.connect(self.select_source_folder)
        self.pushButton_4.clicked.connect(self.select_target_image)
        self.pushButton.clicked.connect(self.run_algorithm)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Phân loại động vật"))
        self.pushButton.setText(_translate("MainWindow", "Chạy"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Thuật toán 1: KNN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Thuật toán 2: Random Forest"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Thuật toán 3: Naive Bayes"))
        self.label.setText(_translate("MainWindow", "Thuật toán sử dụng"))
        self.pushButton_2.setText(_translate("MainWindow", "..."))
        self.pushButton_4.setText(_translate("MainWindow", "..."))
        self.label_3.setText(_translate("MainWindow", "Nguồn ảnh"))
        self.label_4.setText(_translate("MainWindow", "Ảnh"))

    def select_source_folder(self):
        folder_dialog = QtWidgets.QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(None, "Chọn thư mục nguồn")
        if folder_path:
            self.label_source.setText(folder_path)

    def select_target_image(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(None, "Chọn ảnh đích", "", "Images (*.png *.xpm *.jpg *.bmp *.jpeg)")
        if file_path:
            self.label_target.setText(file_path)

    def run_algorithm(self):
        source_folder = self.label_source.text()
        target_image = self.label_target.text()
        selected_algorithm = self.comboBox.currentIndex()

        # Cập nhật giao diện trạng thái "Đang load dữ liệu..."
        self.resultLabel.setText("Đang load dữ liệu...")
        QtWidgets.QApplication.processEvents()

        # Tạo worker và kết nối tín hiệu finished để cập nhật kết quả
        self.worker = Worker(source_folder, target_image, selected_algorithm)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self, result):
        # Cập nhật kết quả từ worker
        self.resultLabel.setText(result)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setFixedSize(MainWindow.size())
    MainWindow.show()
    sys.exit(app.exec())