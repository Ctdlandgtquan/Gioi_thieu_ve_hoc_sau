# 2374802010414_LeTranDongQuan
# bài tập tuần 7
Dựa trên bài lab đã học ở tuần trước, hãy thay đổi thành các

tập dữ liệu khác như:

- Cat and dog

- CIFAR-10

- PlantVillage

Xử dụng các phương pháp phân tích dữ liệu, cân bằng dữ

liệu,..

Cố gắng thay đổi các tham số sao cho độ chính xác phải lớn

hơn 90% và tránh trường hợp overfitting.
Không sử dụng các mô hình re-train hoặc các mô hình như

Resnet, Convnext tiny....... 
cnn thuần
deadline 21/3


## 🛠️ Công Nghệ Sử Dụng
**Framework & Thư viện chính:**
- **PyTorch** - Framework deep learning chính
- **TorchVision** - Xử lý ảnh và augmentation
- **Matplotlib & Seaborn** - Trực quan hóa dữ liệu và kết quả
- **NumPy** - Thao tác mảng và tính toán số học
- **scikit-learn** - Đánh giá mô hình (confusion matrix, classification report)
- **PIL & OpenCV** - Xử lý ảnh cơ bản
- **tqdm** - Hiển thị tiến trình huấn luyện

## 📊 Dữ Liệu
**Bộ dữ liệu PlantVillage** gồm ảnh lá cây khỏe mạnh và bị bệnh.

### Phân Tích Dữ Liệu
```
Tổng số ảnh: 20,638
Số lớp: 15
Phân phối dữ liệu:
  - Lớp nhiều nhất: Tomato__Tomato_YellowLeaf__Curl_Virus - 3,208 ảnh
  - Lớp ít nhất: Potato___healthy - 152 ảnh
  - Tỉ lệ mất cân bằng: 21.11:1
```

### Xử Lý Mất Cân Bằng Dữ Liệu
- Sử dụng **WeightedRandomSampler** để cân bằng dữ liệu trong quá trình huấn luyện
- Phân chia train/val: 80% - 20% (16,510 train / 4,128 val)
- Data augmentation mạnh cho tập train

## 🏗️ Kiến Trúc CNN

### Mô Hình CNN Thuần (Không dùng pretrained)
```python
PlantCNN(
  - 4 blocks tích chập (Conv2D + BatchNorm + ReLU + MaxPool + Dropout)
  - 3 fully connected layers
  - Tổng số tham số: ~9.7 triệu
)
```

### Chi Tiết Kiến Trúc
| Block | Layers | Output Size | Dropout |
|-------|--------|-------------|---------|
| Block 1 | Conv(3→32) + Conv(32→32) + Pool | 64x64 | 0.25 |
| Block 2 | Conv(32→64) + Conv(64→64) + Pool | 32x32 | 0.25 |
| Block 3 | Conv(64→128) + Conv(128→128) + Pool | 16x16 | 0.25 |
| Block 4 | Conv(128→256) + Conv(256→256) + Pool | 8x8 | 0.25 |
| FC Layers | 256*8*8 → 512 → 256 → 15 | - | 0.5 |

## 📈 Kết Quả Huấn Luyện

### Thông Số Huấn Luyện
- **Batch size:** 32
- **Learning rate:** 0.001
- **Optimizer:** Adam (weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss function:** CrossEntropyLoss
- **Epochs:** 50 (dừng sớm khi đạt target)

### Kết Quả Chi Tiết
```
Epoch 1: Train Acc: 66.11% | Val Acc: 83.41%
Epoch 2: Train Acc: 69.12% | Val Acc: 86.34%
Epoch 3: Train Acc: 72.43% | Val Acc: 83.87%
Epoch 4: Train Acc: 73.68% | Val Acc: 88.15%
Epoch 5: Train Acc: 74.40% | Val Acc: 89.61%
Epoch 6: Train Acc: 75.69% | Val Acc: 89.17%
Epoch 7: Train Acc: 77.42% | Val Acc: 90.29% ✅
```

### Kết Quả Cuối Cùng
```
Best validation accuracy: 90.29%
Dừng huấn luyện tại epoch 7 (đạt mục tiêu 90%)
```

### Biểu Đồ Huấn Luyện
- **Accuracy:** Tăng đều qua các epoch, không có dấu hiệu overfitting
- **Loss:** Giảm dần và ổn định

## 🧪 Kỹ Thuật Tránh Overfitting
1. **Data Augmentation** mạnh:
   - RandomRotation(30°)
   - RandomHorizontalFlip (p=0.5)
   - RandomVerticalFlip (p=0.1)
   - RandomAffine (translate=0.2)
   - ColorJitter (brightness, contrast, saturation)

2. **Regularization:**
   - Dropout (0.25 ở các lớp conv, 0.5 ở FC layers)
   - Weight decay (1e-4) trong optimizer
   - Batch Normalization sau mỗi lớp conv

3. **Early Stopping:**
   - Patience = 5 epochs
   - Dừng khi val_acc đạt 90%

4. **Cân bằng dữ liệu:**
   - WeightedRandomSampler cho tập train

## 📊 Dự Đoán Mẫu
Mô hình được thử nghiệm trên 5 ảnh từ tập validation, kết quả dự đoán chính xác với độ tin cậy cao.

## 📝 Kết Luận
- Mô hình CNN tự xây dựng đạt độ chính xác **90.29%** trên tập validation
- Các kỹ thuật cân bằng dữ liệu và regularization giúp tránh overfitting
- Chỉ sau 7 epoch đã đạt được mục tiêu đề ra
- Mô hình có khả năng tổng quát hóa tốt

