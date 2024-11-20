import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Tải bộ dữ liệu MNIST và huấn luyện mô hình
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

def predict_digit(image_array):
    """Dự đoán chữ số từ ảnh đã được tiền xử lý"""
    image_array = image_array.reshape(1, -1)  # Chuyển thành vector 1D
    prediction = knn.predict(image_array)
    return prediction[0]

# Hàm xử lý ảnh khi người dùng tải lên
def process_image(image_path):
    """Chuyển ảnh thành định dạng 28x28 và tiền xử lý"""
    img = Image.open(image_path).convert("L")  # Chuyển sang ảnh xám
    img = img.resize((28, 28))  # Resize về kích thước 28x28
    img = np.array(img)  # Chuyển thành mảng numpy
    img = 255 - img  # Invert màu (MNIST là chữ số màu trắng trên nền đen)
    img = img / 255.0  # Chuẩn hóa ảnh (từ 0-255 sang 0-1)
    return img

# Hàm dự đoán khi người dùng nhập ảnh
def predict_from_image(image):
    """Nhận diện chữ số từ ảnh"""
    image_array = process_image(image)
    prediction = predict_digit(image_array)
    return prediction

# Tạo giao diện GUI cho người dùng
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        
        # Nút để tải ảnh
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
        # Label để hiển thị kết quả dự đoán
        self.result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14))
        self.result_label.pack(pady=20)
        
        # Label để hiển thị ảnh
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def upload_image(self):
        """Hàm tải ảnh từ máy tính"""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                # Dự đoán và hiển thị ảnh
                prediction = predict_from_image(file_path)
                self.result_label.config(text=f"Prediction: {prediction}")
                img = Image.open(file_path)
                img.thumbnail((200, 200))
                img = ImageTk.PhotoImage(img)
                self.image_label.config(image=img)
                self.image_label.image = img
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {e}")

# Khởi tạo giao diện và chạy ứng dụng
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
