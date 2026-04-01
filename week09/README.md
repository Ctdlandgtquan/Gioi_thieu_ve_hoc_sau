# 2374802010414_LeTranDongQuan
# week09 – RNN Time Series Prediction

Dựa trên bài lab RNN cơ bản, thực hiện dự đoán chuỗi thời gian với các tham số khác nhau, tối ưu hóa mô hình và đánh giá hiệu quả.

## Công Nghệ Sử Dụng
**Framework & Thư viện chính:**
- **PyTorch** – Framework deep learning chính
- **NumPy, Pandas** – Xử lý và thao tác dữ liệu
- **scikit-learn** – Chuẩn hóa dữ liệu, đánh giá mô hình (MSE, MAE)
- **Matplotlib** – Vẽ đồ thị loss và so sánh dự đoán

## Dữ Liệu
**Bộ dữ liệu tổng hợp** gồm 300 mẫu thời gian với 3 đặc trưng:
- feature_1 = sin(t)
- feature_2 = cos(0.5t)
- feature_3 = 0.05t + nhiễu Gaussian
- target = 0.5*feature_1 + 0.3*feature_2 + 0.2*feature_3 + nhiễu

### Tiền xử lý
- Chuẩn hóa dữ liệu về [0, 1] bằng `MinMaxScaler`
- Tạo chuỗi con với độ dài `seq_length = 20` (mặc định)
- Chia dữ liệu: **70% train – 15% validation – 15% test**
- Kích thước sau khi tạo chuỗi: train: 196 mẫu, val: 42 mẫu, test: 42 mẫu

## Kiến Trúc RNN

### Mô Hình RNN Thuần (Elman RNN)
```python
RNNModel(
  input_size=3, hidden_size=32, output_size=1,
  n_layers=1, dropout=0.0
)
```

### Chi Tiết Kiến Trúc
| Thành phần | Mô tả |
|------------|-------|
| RNN layer | input_size=3, hidden_size=32, batch_first=True |
| Fully Connected | Linear(32 → 1) |
| Hàm khởi tạo hidden | `init_hidden(batch_size)` trả về tensor zeros |

Tổng số tham số: ~1.1K (rất nhỏ, phù hợp với dữ liệu tổng hợp)

## Huấn Luyện

### Thông Số Huấn Luyện
- **Batch size:** 32
- **Learning rate:** 0.001
- **Optimizer:** Adam
- **Loss function:** MSE
- **Epochs:** 150

### Biểu Đồ Loss
![Loss curve](loss_curve.png)  
Train MSE giảm nhanh từ 0.135 xuống ~0.0011 sau 150 epochs. Val MSE ổn định ở mức ~0.00124, không có dấu hiệu overfitting.

## Kết Quả Đánh Giá

### Trên Tập Test
```
Test MSE  = 0.001319
Test MAE  = 0.029665
```

### So Sánh Giá Trị Thực và Dự Đoán
![Predictions vs Actual](predictions.png)  
Mô hình bám sát xu hướng thực tế, sai số nhỏ.

### Sai Số Theo Thời Điểm
![Errors](errors.png)  
Sai số tuyệt đối dao động quanh 0.03, không có điểm bất thường.

## Thử Nghiệm Nâng Cao

### Ảnh hưởng của độ dài chuỗi (`seq_length`)
| seq_length | Val MSE   |
|------------|-----------|
| 10         | 0.009073  |
| 20         | 0.019828  |
| 30         | 0.036635  |

Nhận xét: Chuỗi ngắn hơn (10) cho kết quả tốt hơn do dữ liệu tổng hợp có tính chất ngắn hạn.

### Ảnh hưởng của số nút ẩn (`hidden_size`)
| hidden_size | Val MSE   |
|-------------|-----------|
| 16          | 0.017642  |
| 32          | 0.019294  |
| 64          | 0.008902  |

Nhận xét: hidden_size=64 cho kết quả tốt nhất, nhưng tăng quá nhiều có thể gây overfitting nếu không điều chỉnh regularization.

### Ảnh hưởng của số lớp và dropout
- RNN 2 lớp + dropout 0.2: Val MSE = **0.016127** (cao hơn mô hình 1 lớp do dữ liệu nhỏ, dễ overfit)

### Ảnh hưởng của learning rate
| lr     | Val MSE   |
|--------|-----------|
| 0.0005 | 0.044032  |
| 0.001  | 0.019810  |
| 0.005  | 0.004527  |

Nhận xét: lr=0.005 cho kết quả tốt nhất nhưng cần kiểm tra độ ổn định.

### Dự Đoán Nhiều Bước Tiếp Theo
Với chuỗi bắt đầu từ mẫu 1:
- 3 bước: [0.8812, 0.9054, 0.9114]
- 5 bước: [0.8812, 0.9054, 0.9114, 0.9172, 0.9247]

Dự đoán tăng dần, phù hợp với xu hướng tăng nhẹ của target.

## Kỹ Thuật Tránh Overfitting (Đã Áp Dụng)
- **Chia dữ liệu hợp lý** (70-15-15) để đánh giá chính xác
- **Theo dõi validation loss** trong quá trình huấn luyện
- **Thử nghiệm regularization** (dropout) để giảm overfitting
- **Không sử dụng mô hình quá phức tạp** (chỉ 1 lớp RNN với 32 nút ẩn)

