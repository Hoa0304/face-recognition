import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os
from sklearn.preprocessing import LabelEncoder

# Phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đọc dữ liệu khuôn mặt
def prepare_data(data_path):
    faces = []
    labels = []
    
    # Đọc tất cả các ảnh trong thư mục
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                img = cv2.imread(file_path)  # Đọc ảnh RGB thay vì xám
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh RGB sang ảnh xám
                    # Phát hiện khuôn mặt trong ảnh
                    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    
                    for (x, y, w, h) in faces_rect:
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (100, 100))  # Resize để dễ xử lý
                        faces.append(face)  # Giữ ảnh không flatten để mô hình CNN xử lý tốt hơn
                        labels.append(label)  # Lưu nhãn cho khuôn mặt

    return np.array(faces), np.array(labels)

# Cập nhật đường dẫn đến bộ dữ liệu khuôn mặt
data_path = r'D:\lab\python\lab5\face_dataset'  # Đảm bảo rằng bạn cập nhật đúng đường dẫn thư mục
faces, labels = prepare_data(data_path)

# Kiểm tra xem bạn có đủ dữ liệu không
if len(faces) == 0:
    raise ValueError("Không có dữ liệu khuôn mặt trong thư mục. Hãy kiểm tra lại dữ liệu đầu vào!")

# Chuyển đổi nhãn thành số (bạn có thể cần mã hóa nhãn nếu chúng là chuỗi)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labels)

# Chia dữ liệu thành X_train và y_train
X_train = faces  # Dữ liệu đầu vào
y_train = y_train  # Dữ liệu đầu ra

# Đảm bảo X_train có dạng phù hợp (Dữ liệu phải có 3 chiều: samples x height x width x channels)
X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)  # Chuyển đổi thành dạng phù hợp cho CNN

# Xây dựng mô hình CNN (Convolutional Neural Network)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Cấu hình kích thước video tùy chỉnh
width = 640  # Chiều rộng mong muốn của cửa sổ và video
height = 480  # Chiều cao mong muốn của cửa sổ và video

# Mở camera và thiết lập độ phân giải
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Đặt chiều rộng
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Đặt chiều cao

# Thiết lập kích thước cửa sổ hiển thị
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", width, height)

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    
    if not ret:
        break

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces_rect:
        # Cắt khuôn mặt từ ảnh
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))  # Resize khuôn mặt về kích thước cố định
        face_resized = face_resized.reshape(1, 100, 100, 1)  # Chuyển đổi thành dạng phù hợp cho mô hình

        # Dự đoán khuôn mặt
        prediction = model.predict(face_resized)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])  # Chuyển dự đoán thành nhãn

        # Kiểm tra xác suất dự đoán (ngưỡng tự chọn)
        confidence = np.max(prediction)  # Lấy xác suất cao nhất của mô hình
        threshold = 0.5  # Ngưỡng, bạn có thể thay đổi ngưỡng này nếu cần

        if confidence < threshold:
            predicted_label = ["Person"]  # Nếu xác suất thấp hơn ngưỡng, hiển thị "Person"
        
        # Vẽ hình chữ nhật quanh khuôn mặt và dán nhãn
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Vẽ hình chữ nhật
        cv2.putText(frame, predicted_label[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Dán nhãn
    
    # Hiển thị frame
    cv2.imshow('Face Recognition', frame)
    
    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
