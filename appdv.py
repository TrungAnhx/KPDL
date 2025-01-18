import os
from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
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

        # ComboBox for selecting algorithm (no border around)
        self.comboBox = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(330, 260, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.comboBox.setFont(font)
        self.comboBox.setStyleSheet("background-color: rgb(252, 251, 231);")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        # Label for algorithm selection
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 260, 151, 31))
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
        self.pushButton_2.setGeometry(QtCore.QRect(470, 180, 31, 31))
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
        self.label_source.setGeometry(QtCore.QRect(220, 180, 251, 31))
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
        self.label_3.setGeometry(QtCore.QRect(150, 180, 71, 31))
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
        self.pushButton_4.setGeometry(QtCore.QRect(470, 220, 31, 31))
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
        self.label_target.setGeometry(QtCore.QRect(220, 220, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.label_target.setFont(font)
        self.label_target.setStyleSheet("    border: 2px solid black;\n"
                                        "    border-radius: 20px;\n"
                                        "    background-color: rgb(252, 251, 231);")
        self.label_target.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_target.setObjectName("label_target")

        # Label for "Target Image"
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(180, 220, 41, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("border: 2px solid black;\n"
                                   "border-radius: 20px;\n"
                                   "background-color: rgb(65, 196, 172);")
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label_4")

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
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Connect buttons to functions
        self.pushButton_2.clicked.connect(self.select_source_folder)
        self.pushButton_4.clicked.connect(self.select_target_image)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Chạy"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Thuật toán 1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Thuật toán 2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Thuật toán 3"))
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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
