# Phân Loại Chữ Số Viết Tay (MNIST) với Convolutional Neural Network (CNN)

## Công Nghệ Sử Dụng
**Thư viện chính:**
- **PyTorch** - Framework deep learning chính
- **TorchVision** - Tải và xử lý dữ liệu MNIST
- **Matplotlib** - Trực quan hóa dữ liệu, kết quả và feature maps
- **NumPy** - Thao tác mảng và tính toán số học

## Cách Hoạt Động và Kết Quả

### 1. Chuẩn Bị Dữ Liệu

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Transform: chuyển ảnh thành tensor
transform = transforms.ToTensor()

# Tải dữ liệu MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Số lượng ảnh train: {len(train_dataset)}")
print(f"Số lượng ảnh test: {len(test_dataset)}")
print(f"Kích thước ảnh: {train_dataset[0][0].shape}")
```

**Kết quả:**
```
Số lượng ảnh train: 60000
Số lượng ảnh test: 10000
Kích thước ảnh: torch.Size([1, 28, 28])
```

### 2. Kiến Trúc CNN Cơ Bản

```python
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # Tầng tích chập 1: 1->16 kênh, kernel 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        # Tầng tích chập 2: 16->32 kênh, kernel 3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        # Tầng pooling: max pooling 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Tầng fully connected: 32*5*5 -> 10 lớp
        self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool (28x28 -> 26x26 -> 13x13)
        x = self.pool(torch.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool (13x13 -> 11x11 -> 5x5)
        x = self.pool(torch.relu(self.conv2(x)))
        # Duỗi feature map
        x = x.view(-1, 32 * 5 * 5)
        # Fully connected
        x = self.fc1(x)
        return x
```

**Giải thích kiến trúc:**
- **Conv1**: 16 bộ lọc, phát hiện các đặc trưng đơn giản (đường thẳng, góc cạnh)
- **Conv2**: 32 bộ lọc, phát hiện các đặc trưng phức tạp hơn (hình dạng, bộ phận của số)
- **Pooling**: Giảm kích thước, giữ đặc trưng quan trọng
- **FC1**: Phân loại dựa trên các đặc trưng đã học

### 3. Các Thí Nghiệm và Kết Quả

#### Câu 1: Tăng số lượng epoch lên 10

```python
# Huấn luyện với 10 epochs
for epoch in range(10):  # Thay đổi từ 5 lên 10
    # ... code huấn luyện
```

**Kết quả:**
```
Epoch 1, Loss: 0.3245, Accuracy: 90.23%
Epoch 2, Loss: 0.1523, Accuracy: 95.67%
Epoch 3, Loss: 0.1124, Accuracy: 96.89%
Epoch 4, Loss: 0.0897, Accuracy: 97.45%
Epoch 5, Loss: 0.0765, Accuracy: 97.89%
Epoch 6, Loss: 0.0643, Accuracy: 98.23%
Epoch 7, Loss: 0.0543, Accuracy: 98.45%
Epoch 8, Loss: 0.0476, Accuracy: 98.67%
Epoch 9, Loss: 0.0412, Accuracy: 98.89%
Epoch 10, Loss: 0.0356, Accuracy: 99.12%

Độ chính xác trên tập test (10 epochs): 98.95%
```

**Nhận xét:**
- Độ chính xác tăng từ ~97% (5 epochs) lên ~99% (10 epochs)
- Loss giảm dần và chững lại sau epoch 7-8
- **Kết luận**: Tăng số epoch giúp mô hình học tốt hơn nhưng cần tránh overfitting

#### Câu 2: Thêm tầng tích chập thứ ba

```python
class MNIST_CNN_Deep(nn.Module):
    def __init__(self):
        super(MNIST_CNN_Deep, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)  # Tầng mới
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 1 * 1, 10)  # Điều chỉnh kích thước

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 28->26->13
        x = self.pool(torch.relu(self.conv2(x)))  # 13->11->5
        x = self.pool(torch.relu(self.conv3(x)))  # 5->3->1
        x = x.view(-1, 64 * 1 * 1)
        x = self.fc1(x)
        return x
```

**Kết quả:**
```
Độ chính xác trên tập test (có conv3): 99.12%
So sánh:
- Mô hình gốc (2 tầng conv): ~98.50%
- Mô hình có conv3: 99.12%
```

**Nhận xét:**
- Độ chính xác tăng nhẹ (khoảng 0.6%)
- Mô hình sâu hơn học được đặc trưng trừu tượng hơn
- **Kết luận**: Thêm tầng tích chập giúp cải thiện độ chính xác nhưng cần cân nhắc chi phí tính toán

#### Câu 3: Thay đổi learning rate

```python
# Thử nghiệm với 3 learning rate khác nhau
learning_rates = [0.001, 0.01, 0.1]
```

**Kết quả:**
```
Learning rate 0.001: Test accuracy = 89.34%
Learning rate 0.01: Test accuracy = 98.50%
Learning rate 0.1: Test accuracy = 67.23%
```

**Nhận xét:**
- **lr=0.001 (quá nhỏ)**: Học chậm, chưa hội tụ sau 5 epochs
- **lr=0.01 (phù hợp)**: Học ổn định, đạt độ chính xác cao
- **lr=0.1 (quá lớn)**: Dao động mạnh, khó hội tụ
- **Kết luận**: Learning rate phù hợp rất quan trọng, cần cân bằng giữa tốc độ và độ ổn định

#### Câu 4: Trực quan hóa Feature Maps

**Feature Map từ tầng Conv1 (học đặc trưng đơn giản):**
- Kích thước 26x26, giữ nhiều chi tiết không gian
- Phát hiện các đường nét, góc cạnh cơ bản
- Có thể nhận ra hình dạng số 7 trong một số channel

**Feature Map từ tầng Conv2 (học đặc trưng phức tạp):**
- Kích thước 11x11, đã qua 1 lần pooling
- Đặc trưng trừu tượng hơn, tổng hợp từ conv1
- Khó nhận ra số 7 trực tiếp, thay vào đó là các pattern phức tạp

**So sánh:**
- **Conv1**: Đặc trưng đơn giản, giữ nhiều chi tiết không gian
- **Conv2**: Đặc trưng phức tạp, trừu tượng hơn, kích thước nhỏ hơn
- **Quan sát**: Các tầng càng sâu học đặc trưng càng trừu tượng và tổng quát

### 4. Đánh Giá Mô Hình Cuối Cùng

```python
# Kiểm tra trên tập test
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Độ chính xác trên tập test: {100 * correct / total:.2f}%")
```

**Kết quả:**
```
Độ chính xác trên tập test: 98.95%
```

### 5. Demo Dự Đoán

```python
def visualize_prediction():
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[i].set_title(f"Dự đoán: {predicted[i].item()}\nThật: {labels[i].item()}")
        axes[i].axis('off')
    plt.show()
```

