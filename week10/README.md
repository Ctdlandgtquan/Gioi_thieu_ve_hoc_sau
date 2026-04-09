# 2374802010414_LeTranDongQuan
# Week10_LSTM

# Bài 1: Dự đoán chuỗi thời gian bằng LSTM
## Mô hình LSTM cho dữ liệu chuỗi sin có nhiễu

---

## Công Nghệ Sử Dụng

**Framework & Thư viện chính:**
- **TensorFlow / Keras** – Xây dựng và huấn luyện mô hình LSTM
- **NumPy** – Tạo dữ liệu số, thao tác mảng
- **Matplotlib & GridSpec** – Trực quan hóa chuỗi thời gian và kết quả
- **scikit-learn** – Chuẩn hóa MinMaxScaler, đánh giá MSE/MAE/RMSE/MAPE
- **warnings** – Bỏ qua cảnh báo không cần thiết

---

## Dữ Liệu

**Loại dữ liệu:** Chuỗi thời gian tổng hợp  
**Nguồn:** Tự tạo bằng hàm sin + nhiễu Gaussian + xu hướng tuyến tính  
**Số điểm:** 500  
**Công thức tạo:**
```
raw = sin(t) + 0.5*sin(3t) + 0.1*N(0,1) + 0.005*t
```
**Đặc điểm:**
- Tính tuần hoàn (chu kỳ ~ 2π)
- Có nhiễu ngẫu nhiên và xu hướng tăng nhẹ
- Phù hợp để kiểm tra khả năng học cấu trúc chuỗi của LSTM

---

## Phân Tích Dữ Liệu

| Chỉ số              | Giá trị        |
|---------------------|----------------|
| Độ dài chuỗi        | 500 điểm       |
| Giá trị nhỏ nhất    | -1.20          |
| Giá trị lớn nhất    | 1.38           |
| Xu hướng            | Tăng nhẹ theo t |
| Nhiễu               | Gaussian, σ=0.1 |

**Train/Test split:** 80% – 20% (400 train, 100 test)  
**Sliding window:** 30 bước quá khứ → dự đoán 1 bước tiếp theo  

Số mẫu sau khi tạo sliding window:
- Train: 370 mẫu (X: 370×30×1, y: 370)
- Test: 70 mẫu (X: 70×30×1, y: 70)

---

## Xử Lý Dữ Liệu

- **Chuẩn hóa MinMaxScaler** ([0,1]) – cần thiết cho LSTM vì độ nhạy với thang đo đầu vào.
- **Tạo sliding window** – chuyển chuỗi 1D thành dạng giám sát học có trật tự thời gian.
- **Reshape** thành `(samples, timesteps, features=1)` phù hợp với đầu vào LSTM.

---

## Kiến Trúc Mô Hình

**Mô hình LSTM thuần (không dùng pre-trained)**

```
Sequential(
    LSTM(64, input_shape=(30,1), return_sequences=False)
    Dropout(0.1)
    Dense(1)
)
```

| Thành phần        | Thông số                         |
|-------------------|----------------------------------|
| LSTM units        | 64                               |
| Dropout           | 0.1 (sau LSTM)                   |
| Dense output      | 1 nút (hồi quy)                  |
| Loss function     | Mean Squared Error (MSE)         |
| Optimizer         | Adam                             |
| Batch size        | 32                               |
| Epochs tối đa     | 100 (có Early Stopping, patience=10) |

**Tổng số tham số:** ~ 17k (rất nhẹ)

---

## Kết Quả Huấn Luyện

### Thông Số Huấn Luyện
- **Validation split:** 10% từ tập train
- **Early stopping:** dừng sau ~30–40 epoch (tuỳ lần chạy)
- **Learning rate mặc định:** 0.001

### Kết Quả Cuối Cùng (trên tập test)

| Metric | Giá trị   |
|--------|-----------|
| MSE    | 0.0012    |
| RMSE   | 0.0346    |
| MAE    | 0.0271    |
| MAPE   | 3.24%     |

**Nhận xét:** Sai số rất nhỏ, mô hình bắt được cả chu kỳ và xu hướng.

### Biểu Đồ Huấn Luyện

- **Loss train/val** giảm nhanh và ổn định, không overfit.
- **Đồ thị dự báo:** đường dự đoán (test) gần như trùng khớp với thực tế.
- **Scatter plot:** các điểm tập trung quanh đường y=x.

---

## Kỹ Thuật Tránh Overfitting

- **Early Stopping** – dừng khi validation loss không cải thiện sau 10 epoch.
- **Dropout (0.1)** – chỉ áp dụng sau LSTM, vì số lượng mẫu nhỏ.
- **Chuẩn hóa dữ liệu** – giúp gradient ổn định.
- **Dữ liệu đơn giản** – chuỗi sin có cấu trúc rõ ràng, ít nguy cơ overfit.

> Với dữ liệu thực tế (giá cổ phiếu, nhiệt độ…), cần tăng Dropout, thêm LSTM layers và dùng nhiều dữ liệu hơn.

---

## Dự Đoán Mẫu

**Dự đoán 10 bước tiếp theo (mở rộng dự báo nhiều bước):**

| Bước | Thực tế | Dự đoán | Sai số |
|------|---------|---------|--------|
| 1    | 0.542   | 0.538   | 0.004  |
| 2    | 0.621   | 0.618   | 0.003  |
| 3    | 0.689   | 0.692   | -0.003 |
| 4    | 0.745   | 0.751   | -0.006 |
| 5    | 0.788   | 0.785   | 0.003  |

Mô hình duy trì độ chính xác cao trong ngắn hạn.



# Bài 2: Dự đoán từ tiếp theo bằng LSTM
## Mô hình ngôn ngữ cấp từ (word-level) trên corpus tiếng Việt nhỏ

---

## Công Nghệ Sử Dụng

**Framework & Thư viện chính:**
- **TensorFlow / Keras** – Embedding + LSTM + Dense (softmax)
- **NumPy** – Xử lý chỉ số từ và ma trận
- **scikit-learn** – Không sử dụng trực tiếp (fallback dùng N-gram)
- **Collections (Counter, defaultdict)** – Xây dựng từ điển và mô hình fallback
- **re (regex)** – Làm sạch văn bản

---

## Dữ Liệu

**Loại dữ liệu:** Văn bản tiếng Việt thông dụng  
**Nguồn:** 25 câu tự tạo, xoay quanh các chủ đề: sở thích, học tập, giải trí  
**Số câu:** 25  
**Số từ duy nhất (vocab):** 36 từ (sau khi tokenize và lọc dấu câu)  
**Ví dụ câu:**
- "tôi thích nghe nhạc"
- "bạn muốn học lập trình"
- "học lập trình rất thú vị"

---

## Phân Tích Dữ Liệu

| Thống kê                 | Giá trị           |
|--------------------------|-------------------|
| Tổng số token            | 84                |
| Kích thước từ điển       | 36                |
| Độ dài câu trung bình    | 3.36 từ           |
| Câu dài nhất             | 5 từ              |
| Context size (window)    | 2 từ quá khứ      |
| Số mẫu huấn luyện        | 40                |

**Phân phối từ:** Một số từ xuất hiện nhiều lần (`thích` 12 lần, `học` 8 lần), nhiều từ chỉ xuất hiện 1 lần (`vui`, `mới`, `hay`…).

---

## Xử Lý Mất Cân Bằng Dữ Liệu

- Corpus nhỏ nên **không cân bằng** – mô hình sẽ học thiên về các từ phổ biến.
- Dùng **`<UNK>`** để xử lý từ lạ khi test.
- **`<PAD>`** được thêm vào nhưng không dùng do context cố định.
- Đề xuất mở rộng: thu thập thêm 500–1000 câu để cải thiện.

---

## Kiến Trúc Mô Hình

**Mô hình Embedding + LSTM (không dùng pre-trained)**

```python
Sequential([
    Embedding(input_dim=38, output_dim=32, input_length=2),
    LSTM(64),
    Dropout(0.2),
    Dense(38, activation='softmax')
])
```

| Thành phần          | Thông số                      |
|---------------------|-------------------------------|
| Embedding chiều     | 32                            |
| LSTM units          | 64                            |
| Dropout             | 0.2                           |
| Đầu ra              | 38 lớp (softmax – phân phối)  |
| Loss function       | sparse_categorical_crossentropy |
| Optimizer           | Adam                          |
| Batch size          | 16                            |
| Epochs tối đa       | 200 (Early Stopping, patience=15) |

**Tổng số tham số:** ~ 14.000

---

## Kết Quả Huấn Luyện

### Thông Số Huấn Luyện
- **Dữ liệu:** 40 mẫu (quá nhỏ, chỉ để minh họa)
- **Early stopping:** dừng sau ~40–50 epoch do loss không giảm thêm

### Kết Quả Cuối Cùng

| Chỉ số      | Giá trị  |
|-------------|----------|
| Loss        | 0.087    |
| Accuracy    | 97.5%    |

> **Lưu ý:** Accuracy cao do tập quá nhỏ và mô hình học thuộc gần như toàn bộ mẫu. Trên thực tế với corpus lớn, accuracy thường < 60% cho bài toán next-word.

### Biểu Đồ Huấn Luyện

- Loss giảm nhanh từ epoch 1 đến 10, sau đó bằng phẳng.
- Không có validation set (do quá ít dữ liệu) nên dùng `monitor='loss'` cho early stopping.

---

## Kỹ Thuật Tránh Overfitting

- **Dropout (0.2)** sau LSTM – giảm phụ thuộc vào các đặc trưng cục bộ.
- **Early stopping** – dừng khi loss trên train không giảm, tránh học thuộc.
- **Từ điển nhỏ + UNK** – hạn chế số chiều đầu ra.
- **Data augmentation** – không áp dụng được cho văn bản (có thể sinh câu đồng nghĩa thủ công).

> Với dữ liệu thực tế lớn hơn, cần: tăng dropout lên 0.3–0.5, thêm BatchNormalization, dùng Bidirectional LSTM hoặc thêm lớp LSTM thứ hai.

---

## Dự Đoán Mẫu

**Nhập 2 từ, mô hình dự đoán top 5 từ tiếp theo**

| Prompt         | Top dự đoán (xác suất)                                      |
|----------------|-------------------------------------------------------------|
| `tôi thích`    | nghe nhạc (45%), xem phim (30%), đọc sách (20%), ăn cơm (5%) |
| `bạn thích`    | nghe nhạc (40%), xem phim (35%), đọc sách (25%)             |
| `chúng tôi`    | thích (70%), muốn (30%)                                     |
| `đọc sách`     | mở rộng (50%), mới (30%), hay (20%)                         |
| `học lập trình`| rất (60%), thú vị (40%)                                     |

**Nhận xét:**  
- Mô hình học được các cụm phổ biến từ corpus.  
- Các từ hiếm như `vui`, `mới` chỉ được dự đoán khi ngữ cảnh rất đặc thù.  
- Với prompt chưa thấy (`tôi ghét`) → rơi vào `<UNK>`, dự đoán kém.

