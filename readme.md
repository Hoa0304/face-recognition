ứng dụng nhận diện khuôn mặt sử dụng mạng nơ-ron tích chập (CNN) với thư viện TensorFlow và OpenCV. Dưới đây là giải thích chi tiết về từng phần trong mã:

### 1. **Nhập các thư viện cần thiết**
   - **`cv2`**: Thư viện OpenCV để xử lý hình ảnh và video.
   - **`numpy`**: Dùng để xử lý mảng (array) cho dữ liệu đầu vào và đầu ra.
   - **`tensorflow`**: Dùng để xây dựng và huấn luyện mô hình học sâu (Deep Learning).
   - **`os`**: Để làm việc với các file và thư mục trên hệ thống.

### 2. **Phát hiện khuôn mặt**
   - **`face_cascade`**: Đây là một đối tượng Cascade Classifier từ OpenCV được sử dụng để phát hiện khuôn mặt trong ảnh. Mô hình này dựa trên thuật toán Haar Cascades. Cụ thể, file `haarcascade_frontalface_default.xml` được sử dụng để phát hiện khuôn mặt.
   
### 3. **Chuẩn bị dữ liệu**
   - **`prepare_data(data_path)`**: Hàm này nhận vào một đường dẫn thư mục chứa dữ liệu ảnh khuôn mặt. Trong thư mục này, mỗi thư mục con sẽ chứa ảnh của một người và tên thư mục đó sẽ là nhãn cho người đó (ví dụ, `person1`, `person2`).
   
   - **`cv2.imread(file_path)`**: Đọc ảnh từ đường dẫn.
   - **`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`**: Chuyển ảnh từ không gian màu RGB sang ảnh xám (grayscale) để giảm độ phức tạp tính toán.
   - **`face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)`**: Phát hiện khuôn mặt trong ảnh xám. Hàm này trả về các tọa độ của các khuôn mặt trong ảnh (x, y, w, h).
   - **`faces.append(face)`**: Lưu các khuôn mặt vào danh sách `faces`.
   - **`labels.append(label)`**: Lưu nhãn (tên người) vào danh sách `labels`.
   - Kết quả trả về là một mảng numpy chứa ảnh khuôn mặt và nhãn của từng khuôn mặt.

### 4. **Chuyển đổi nhãn thành số**
   - **`LabelEncoder`** từ thư viện `sklearn.preprocessing`: Dùng để mã hóa các nhãn (chuỗi như `person1`, `person2`,...) thành các số nguyên.
   - **`y_train = label_encoder.fit_transform(labels)`**: Chuyển đổi nhãn chuỗi thành số nguyên.

### 5. **Tiền xử lý dữ liệu đầu vào**
   - **`X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)`**: Chuyển đổi ảnh khuôn mặt thành định dạng phù hợp cho mô hình CNN. Mỗi ảnh có kích thước 100x100 pixel và có một kênh (ảnh xám), nên dữ liệu có dạng `(số lượng ảnh, 100, 100, 1)`.

### 6. **Xây dựng mô hình CNN**
   - **`Sequential`**: Dùng để tạo mô hình học sâu tuần tự.
   - **`Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1))`**: Thêm lớp tích chập (convolutional layer) với 32 bộ lọc kích thước 3x3 và sử dụng hàm kích hoạt ReLU. Đây là lớp đầu vào của mô hình.
   - **`MaxPooling2D(pool_size=(2, 2))`**: Thêm lớp giảm kích thước (pooling layer) với kích thước cửa sổ 2x2 để giảm kích thước không gian của dữ liệu.
   - **`Flatten()`**: Biến dữ liệu từ dạng 2D thành 1D, chuẩn bị cho lớp fully connected.
   - **`Dense(128, activation='relu')`**: Thêm một lớp fully connected (dense layer) với 128 nơ-ron và hàm kích hoạt ReLU.
   - **`Dense(len(np.unique(y_train)), activation='softmax')`**: Lớp cuối cùng với số nơ-ron bằng với số lớp nhãn và hàm kích hoạt softmax để phân loại nhiều lớp.
   
   - **`model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])`**: Biên dịch mô hình với tối ưu hóa Adam, hàm mất mát sparse categorical crossentropy (vì nhãn là số nguyên) và theo dõi độ chính xác.

### 7. **Huấn luyện mô hình**
   - **`model.fit(X_train, y_train, epochs=10, batch_size=32)`**: Huấn luyện mô hình với dữ liệu `X_train` và nhãn `y_train` trong 10 epoch (lượt huấn luyện).

### 8. **Mở camera và nhận diện khuôn mặt**
   - **`cap = cv2.VideoCapture(0)`**: Mở camera (ID = 0 là camera mặc định).
   - Trong vòng lặp `while`, các frame từ camera được đọc và chuyển thành ảnh xám. Sau đó, hàm `detectMultiScale` được sử dụng để phát hiện khuôn mặt.
   
   - **`face_resized.reshape(1, 100, 100, 1)`**: Khuôn mặt được cắt từ ảnh và thay đổi kích thước thành 100x100, rồi chuyển thành dạng phù hợp cho mô hình CNN (1 ảnh, kích thước 100x100, 1 kênh).
   - **`prediction = model.predict(face_resized)`**: Mô hình dự đoán nhãn của khuôn mặt.
   - **`predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])`**: Chuyển dự đoán thành nhãn người tương ứng.
   
   - **`confidence = np.max(prediction)`**: Lấy xác suất cao nhất từ mô hình.
   - **`if confidence < threshold:`**: Nếu xác suất thấp hơn ngưỡng, gán nhãn "Person".

### 9. **Vẽ kết quả lên ảnh**
   - **`cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)`**: Vẽ hình chữ nhật quanh khuôn mặt phát hiện.
   - **`cv2.putText(frame, predicted_label[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)`**: Dán nhãn dự đoán vào ảnh gần khuôn mặt.

### 10. **Thoát chương trình**
   - **`cv2.imshow('Face Recognition', frame)`**: Hiển thị ảnh kết quả trong cửa sổ.
   - **`if cv2.waitKey(1) & 0xFF == ord('q'):`**: Nếu người dùng nhấn phím 'q', thoát khỏi vòng lặp.
   - **`cap.release()`**: Giải phóng tài nguyên camera.
   - **`cv2.destroyAllWindows()`**: Đóng tất cả các cửa sổ OpenCV.

### Tổng kết:
Đoạn mã trên thực hiện quá trình nhận diện khuôn mặt trực tiếp từ video. Mô hình CNN được huấn luyện trên các khuôn mặt trong bộ dữ liệu và có thể nhận diện các khuôn mặt trong video camera, dán nhãn và xác suất nhận diện lên từng khuôn mặt.