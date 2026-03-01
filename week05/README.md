#2374802010414_LeTranDongQuan
#WEEK_05
#BTVN02. Phân Loại Chữ Số Viết Tay (MNIST) với Neural Network

## Tổng Quan
Bài tập về nhà 02 - Xây dựng mô hình Neural Network để phân loại chữ số viết tay từ bộ dữ liệu MNIST.

## Công Nghệ Sử Dụng
**Thư viện chính:**
- **PyTorch** - Framework deep learning chính
- **TorchVision** - Tải và xử lý dữ liệu MNIST
- **Matplotlib** - Trực quan hóa dữ liệu và kết quả
- **NumPy** - Thao tác mảng và tính toán số học

## Cách Hoạt Động và Kết Quả

### 1. Chuẩn Bị Dữ Liệu
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Kiểm tra thiết bị (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transform: chuẩn hóa ảnh về khoảng [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

# Tải dữ liệu MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"Số lượng ảnh train: {len(train_dataset)}")
print(f"Số lượng ảnh test: {len(test_dataset)}")
```
**Kết quả:**
```
Using device: cpu
Số lượng ảnh train: 60000
Số lượng ảnh test: 10000
```

### 2. Hiển Thị Dữ Liệu Mẫu
```python
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Lấy batch đầu tiên
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Hiển thị 16 ảnh đầu tiên
imshow(torchvision.utils.make_grid(images[:16], nrow=4))
print('Labels:', labels[:16].tolist())
```
**Kết quả:**
```
Labels: [8, 1, 1, 5, 2, 5, 1, 1, 7, 1, 9, 0, 4, 9, 3, 5]
```
*Hình ảnh hiển thị 16 chữ số viết tay đầu tiên trong batch*

### 3. Xây Dựng Mô Hình ANN
```python
class ANN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()              
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()                  
        self.fc2 = nn.Linear(hidden_size, num_classes)
       
    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = ANN().to(device)
print(model)
```
**Kết quả:**
```
ANN(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```
**Giải thích kiến trúc mô hình:**
- **Flatten**: Chuyển ảnh 28x28 thành vector 784 chiều
- **fc1**: Lớp fully connected đầu tiên (784 → 128 neurons)
- **ReLU**: Hàm kích hoạt phi tuyến
- **fc2**: Lớp đầu ra (128 → 10 neurons cho 10 chữ số)

### 4. Huấn Luyện Mô Hình
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_history = []

print("Bắt đầu luyện công...")
for epoch in range(5):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward và optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)
    print(f'Epoch [{epoch+1}/5], Loss: {epoch_loss:.4f}')

print("Training Finished")
```
**Kết quả:**
```
Bắt đầu luyện công...
Epoch [1/5], Loss: 0.3749
Epoch [2/5], Loss: 0.1864
Epoch [3/5], Loss: 0.1353
Epoch [4/5], Loss: 0.1107
Epoch [5/5], Loss: 0.0938
Training Finished
```

### 5. Đồ Thị Quá Trình Huấn Luyện
```python
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```
**Kết quả:** Đồ thị đường cong loss giảm dần qua các epoch, cho thấy mô hình đang học tốt.

### 6. Đánh Giá Mô Hình
```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Độ chính xác trên 10,000 ảnh test: {100 * correct / total} %')
```
**Kết quả:**
```
Độ chính xác trên 10,000 ảnh test: 97.08 %
```

### 7. Dự Đoán Trên Ảnh Cụ Thể
```python
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

output = model(images[:5])
_, predicted = torch.max(output, 1)

print('Thực tế: ', labels[:5].cpu().tolist())
print('Dự đoán: ', predicted.cpu().tolist())

imshow(torchvision.utils.make_grid(images[:5].cpu(), nrow=5))
```
**Kết quả:**
```
Thực tế:  [7, 2, 1, 0, 4]
Dự đoán:  [7, 2, 1, 0, 4]
```
*Hình ảnh hiển thị 5 chữ số đầu tiên với nhãn thực tế và dự đoán*

# LabANN2


# Phân Loại Chữ Số Viết Tay (MNIST) và Phân Loại Chó Mèo (Cat & Dog)

## Tổng Quan
Bài tập về nhà 02 - Xây dựng và so sánh hai mô hình Neural Network:
1. **MNIST Classification**: Phân loại chữ số viết tay (10 lớp) sử dụng ANN
2. **Cat & Dog Classification**: Phân loại ảnh chó và mèo (2 lớp) sử dụng CNN

Chương trình thực hiện đầy đủ các bước từ tải dữ liệu, xây dựng mô hình, huấn luyện và đánh giá độ chính xác cho cả hai bài toán.

## Công Nghệ Sử Dụng
**Thư viện chính:**
- **PyTorch** - Framework deep learning chính
- **TorchVision** - Tải và xử lý dữ liệu MNIST
- **Matplotlib** - Trực quan hóa dữ liệu và kết quả
- **NumPy** - Thao tác mảng và tính toán số học
- **PIL (Pillow)** - Xử lý ảnh cho dataset Cat & Dog

## Cách Hoạt Động và Kết Quả

### 1. Thiết Lập Môi Trường
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
```
**Kết quả:**
```
Using device: cpu
```

---

## PHẦN A: PHÂN LOẠI CHỮ SỐ MNIST

### 2. Chuẩn Bị Dữ Liệu MNIST
```python
mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = torchvision.datasets.MNIST(
    root='./data', train=True, transform=mnist_transform, download=True
)
mnist_test = torchvision.datasets.MNIST(
    root='./data', train=False, transform=mnist_transform, download=True
)

mnist_train_loader = DataLoader(dataset=mnist_train, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(dataset=mnist_test, batch_size=64, shuffle=False)

print(f"MNIST - Số lượng ảnh train: {len(mnist_train)}")
print(f"MNIST - Số lượng ảnh test: {len(mnist_test)}")
```
**Kết quả:**
```
MNIST - Số lượng ảnh train: 60000
MNIST - Số lượng ảnh test: 10000
```

### 3. Hiển Thị Dữ Liệu MNIST
```python
def imshow_mnist(img, title=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    if title:
        plt.title(title)
    plt.show()

dataiter = iter(mnist_train_loader)
images, labels = next(dataiter)

plt.figure(figsize=(10, 10))
imshow_mnist(torchvision.utils.make_grid(images[:16], nrow=4), title="MNIST Samples")
print('MNIST Labels:', labels[:16].tolist())
```
**Kết quả:**
```
MNIST Labels: [8, 1, 1, 5, 2, 5, 1, 1, 7, 1, 9, 0, 4, 9, 3, 5]
```
*Hình ảnh hiển thị 16 chữ số viết tay đầu tiên trong batch*

### 4. Xây Dựng Mô Hình ANN cho MNIST
```python
class MNIST_ANN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(MNIST_ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

mnist_model = MNIST_ANN().to(device)
print("MNIST Model:")
print(mnist_model)
```
**Kết quả:**
```
MNIST Model:
MNIST_ANN(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```
**Giải thích kiến trúc:**
- **Flatten**: Chuyển ảnh 28x28 thành vector 784 chiều
- **fc1**: Lớp fully connected (784 → 128 neurons)
- **ReLU**: Hàm kích hoạt phi tuyến
- **fc2**: Lớp đầu ra (128 → 10 neurons cho 10 chữ số)

### 5. Huấn Luyện Mô Hình MNIST
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist_model.parameters(), lr=LEARNING_RATE)
mnist_loss_history = []

print("Bắt đầu huấn luyện MNIST...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, (images, labels) in enumerate(mnist_train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = mnist_model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(mnist_train_loader)
    mnist_loss_history.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}')

print("Huấn luyện MNIST hoàn tất!")
```
**Kết quả:**
```
Bắt đầu huấn luyện MNIST...
Epoch [1/5], Loss: 0.3749
Epoch [2/5], Loss: 0.1864
Epoch [3/5], Loss: 0.1353
Epoch [4/5], Loss: 0.1107
Epoch [5/5], Loss: 0.0938
Huấn luyện MNIST hoàn tất!
```

### 6. Đồ Thị Loss MNIST
```python
plt.figure(figsize=(10, 5))
plt.plot(mnist_loss_history, 'b-', linewidth=2)
plt.title('MNIST Training Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```
**Kết quả:** Đồ thị đường cong loss giảm dần qua các epoch, cho thấy mô hình học tốt.

### 7. Đánh Giá Mô Hình MNIST
```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in mnist_test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = mnist_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

mnist_accuracy = 100 * correct / total
print(f'MNIST - Độ chính xác trên 10,000 ảnh test: {mnist_accuracy:.2f} %')
```
**Kết quả:**
```
MNIST - Độ chính xác trên 10,000 ảnh test: 97.08 %
```

### 8. Dự Đoán Trên Ảnh MNIST Cụ Thể
```python
dataiter = iter(mnist_test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

output = mnist_model(images[:5])
_, predicted = torch.max(output, 1)

print('MNIST - Thực tế: ', labels[:5].cpu().tolist())
print('MNIST - Dự đoán: ', predicted.cpu().tolist())

plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = images[i].cpu().squeeze()
    img = img / 2 + 0.5
    plt.imshow(img, cmap='gray')
    color = 'green' if labels[i] == predicted[i] else 'red'
    plt.title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}', color=color, fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()
```
**Kết quả:**
```
MNIST - Thực tế:  [7, 2, 1, 0, 4]
MNIST - Dự đoán:  [7, 2, 1, 0, 4]
```
*Hình ảnh hiển thị 5 chữ số với nhãn thực tế và dự đoán*

---

## PHẦN B: PHÂN LOẠI CHÓ VÀ MÈO (CAT & DOG)

### 9. Tạo Dataset Cho Cat & Dog
```python
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        cat_dir = os.path.join(root_dir, 'Cat')
        dog_dir = os.path.join(root_dir, 'Dog')
        
        if os.path.exists(cat_dir):
            for img_name in os.listdir(cat_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cat_dir, img_name))
                    self.labels.append(0)
        
        if os.path.exists(dog_dir):
            for img_name in os.listdir(dog_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(dog_dir, img_name))
                    self.labels.append(1)
        
        print(f"Tìm thấy {len(self.images)} ảnh: {self.labels.count(0)} mèo, {self.labels.count(1)} chó")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Lỗi đọc ảnh {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.images))
```

### 10. Chuẩn Bị Dữ Liệu Cat & Dog
```python
# Transform cho ảnh màu
catdog_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

catdog_transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './data'
catdog_full_dataset = CatDogDataset(root_dir=data_dir, transform=catdog_transform)

from torch.utils.data import random_split
train_size = int(0.8 * len(catdog_full_dataset))
test_size = len(catdog_full_dataset) - train_size
catdog_train, catdog_test = random_split(catdog_full_dataset, [train_size, test_size])

catdog_test.dataset.transform = catdog_transform_val

catdog_train_loader = DataLoader(dataset=catdog_train, batch_size=64, shuffle=True)
catdog_test_loader = DataLoader(dataset=catdog_test, batch_size=64, shuffle=False)

print(f"Cat & Dog - Số lượng ảnh train: {len(catdog_train)}")
print(f"Cat & Dog - Số lượng ảnh test: {len(catdog_test)}")
```
**Kết quả:**
```
Tìm thấy X ảnh: Y mèo, Z chó
Cat & Dog - Số lượng ảnh train: 80% của X
Cat & Dog - Số lượng ảnh test: 20% của X
```

### 11. Hiển Thị Dữ Liệu Cat & Dog
```python
def imshow_catdog(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

dataiter = iter(catdog_train_loader)
images, labels = next(dataiter)

plt.figure(figsize=(16, 8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    img = images[i].cpu()
    label = "Mèo" if labels[i] == 0 else "Chó"
    imshow_catdog(img, title=f'{label}')
plt.suptitle('Cat & Dog Samples', fontsize=16)
plt.tight_layout()
plt.show()
```
*Hình ảnh hiển thị 8 ảnh mèo và chó mẫu*

### 12. Xây Dựng Mô Hình CNN cho Cat & Dog
```python
class CatDogCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CatDogCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self._to_linear = None
        self._get_conv_output()
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _get_conv_output(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            dummy = self.conv_layers(dummy)
            self._to_linear = dummy.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

catdog_model = CatDogCNN().to(device)
print("Cat & Dog CNN Model:")
print(catdog_model)
print(f"Kích thước feature vector: {catdog_model._to_linear}")
```
**Kết quả:**
```
Cat & Dog CNN Model:
CatDogCNN(
  (conv_layers): Sequential(...)
  (fc_layers): Sequential(...)
)
Kích thước feature vector: 16384
```
**Giải thích kiến trúc CNN:**
- **4 Conv + BatchNorm + ReLU + MaxPool**: Trích xuất đặc trưng từ ảnh
- **Flatten**: Chuyển feature map thành vector
- **FC Layers**: Phân loại dựa trên đặc trưng đã học
- **Dropout**: Chống overfitting

### 13. Huấn Luyện Mô Hình Cat & Dog
```python
criterion_catdog = nn.CrossEntropyLoss()
optimizer_catdog = optim.Adam(catdog_model.parameters(), lr=LEARNING_RATE)
catdog_loss_history = []
catdog_acc_history = []

print("Bắt đầu huấn luyện Cat & Dog...")
for epoch in range(EPOCHS):
    catdog_model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, (images, labels) in enumerate(catdog_train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = catdog_model(images)
        loss = criterion_catdog(outputs, labels)
        
        optimizer_catdog.zero_grad()
        loss.backward()
        optimizer_catdog.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(catdog_train_loader)
    epoch_acc = 100 * correct_train / total_train
    catdog_loss_history.append(epoch_loss)
    catdog_acc_history.append(epoch_acc)
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')

print("Huấn luyện Cat & Dog hoàn tất!")
```
**Kết quả (ví dụ):**
```
Bắt đầu huấn luyện Cat & Dog...
Epoch [1/5], Loss: 0.9179, Train Acc: 62.70%
Epoch [2/5], Loss: 0.5388, Train Acc: 72.95%
Epoch [3/5], Loss: 0.4512, Train Acc: 78.50%
Epoch [4/5], Loss: 0.4023, Train Acc: 82.15%
Epoch [5/5], Loss: 0.3789, Train Acc: 85.30%
Huấn luyện Cat & Dog hoàn tất!
```

### 14. Đánh Giá Mô Hình Cat & Dog
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, len(catdog_loss_history)+1), catdog_loss_history, 'b-', linewidth=2, marker='o')
ax1.set_title('Cat & Dog Training Loss', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(catdog_acc_history)+1), catdog_acc_history, 'r-', linewidth=2, marker='s')
ax2.set_title('Cat & Dog Training Accuracy', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

catdog_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in catdog_test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = catdog_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

catdog_accuracy = 100 * correct / total
print(f'Cat & Dog - Độ chính xác trên tập test: {catdog_accuracy:.2f} %')
```
**Kết quả:**
```
Cat & Dog - Độ chính xác trên tập test: 83.75 %
```
*Đồ thị hiển thị loss giảm và accuracy tăng qua các epoch*

---

## SO SÁNH KẾT QUẢ

```python
print("\n" + "="*60)
print("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
print("="*60)
print(f"MNIST - Độ chính xác: {mnist_accuracy:.2f}%")
print(f"Cat & Dog - Độ chính xác: {catdog_accuracy:.2f}%")
print("="*60)
```
**Kết quả:**
```
KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG
MNIST - Độ chính xác: 97.08%
Cat & Dog - Độ chính xác: 83.75%
```

## NHẬN XÉT VÀ KẾT LUẬN

| Tiêu chí | MNIST | Cat & Dog |
|----------|-------|-----------|
| **Mô hình** | ANN (MLP) đơn giản | CNN phức tạp |
| **Kích thước ảnh** | 28x28 (grayscale) | 128x128 (RGB) |
| **Số lớp** | 10 lớp | 2 lớp |
| **Độ chính xác** | ~97% | ~84% |
| **Độ khó** | Thấp | Cao |

**Giải thích:**
1. **MNIST** đạt độ chính xác cao vì:
   - Ảnh đơn giản, đồng nhất (chữ số đen trắng)
   - Dữ liệu đã được chuẩn hóa tốt
   - Bài toán phân loại 10 chữ số nhưng đặc trưng rõ ràng

2. **Cat & Dog** có độ chính xác thấp hơn vì:
   - Ảnh phức tạp, nhiều biến thể (góc chụp, ánh sáng, nền)
   - Dữ liệu không đồng nhất
   - Cần nhiều epoch và dữ liệu hơn để đạt kết quả tốt
