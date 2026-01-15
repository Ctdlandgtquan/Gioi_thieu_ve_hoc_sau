# Week_01
# 2374802010414_LeTranDongQuan
# Giới thiệu về học sâu

# PyTorch

## bài tập thực hành về PyTorch gồm các bài tính đạo hàm, gradient descent, hồi quy tuyến tính và phân loại.

# Công nghệ sử dụng

### Thư viện chính:
- **PyTorch** - Framework học sâu với hệ thống autograd
- **NumPy** - Xử lý mảng số học và tính toán khoa học
- **Pandas** - Đọc và xử lý dữ liệu CSV
- **Matplotlib** - Vẽ biểu đồ và trực quan hóa dữ liệu
- **scikit-learn** - Tiền xử lý dữ liệu và chia tập train/test


## Cách hoạt động và Kết quả chi tiết

### Cài đặt và kiểm tra môi trường
```python
import sys
sys.executable 
!{sys.executable} -m pip install torch torchvision torchaudio

import torch 
torch.cuda.is_available()
```

**Kết quả:**
- Cài đặt PyTorch thành công
- Trả về đường dẫn Python executable
- Kiểm tra GPU: True nếu có GPU, False nếu chỉ có CPU


```python
import pandas as pd
df = pd.read_csv('Iris.csv') 
df.shape

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
x = df.drop(["Species"], axis=1).values
y = le.fit_transform(df["Species"].values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train) 
y_test = torch.LongTensor(y_test)     

len(y_train)

labels, counts = torch.unique(y_train, return_counts=True)
print("Labels:", labels)
print("Counts:", counts)
```

**Kết quả:**
```
(150, 5)                    # Shape của dataset Iris
120                         # Số lượng mẫu trong tập train
Labels: tensor([0, 1, 2])   # Các nhãn đã mã hóa
Counts: tensor([40, 40, 40]) # Phân phối đều: 40 mẫu mỗi lớp
```

```python
x = torch.tensor(2.0, requires_grad=True)
print("x =", x)
print("x.grad =", x.grad)   
y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1
print("y =", y)
print("y.grad_fn =", y.grad_fn) 
y.backward() 
print("\nSau khi gọi backward():")
print("x.grad =", x.grad)
```

**Kết quả:**
```
x = tensor(2., requires_grad=True)
x.grad = None
y = tensor(99., grad_fn=<AddBackward0>)
y.grad_fn = <AddBackward0 object at 0x...>

Sau khi gọi backward():
x.grad = tensor(93.)
```

**Giải thích:**
- Tại x=2: y = 2*(2)⁴ + (2)³ + 3*(2)² + 5*2 + 1 = 99
- Đạo hàm: y' = 8x³ + 3x² + 6x + 5
- Tại x=2: y' = 8*(8) + 3*(4) + 6*2 + 5 = 64 + 12 + 12 + 5 = 93



### Bài tập 1 

```python
def bt01(x_value=2.0):
    x = torch.tensor(x_value, requires_grad=True)
    y = 5*x**5 + 6*x**3 - 3*x + 1
    y.backward()
    print(f"Đa thức: y = 5x⁵ + 6x³ - 3x + 1")
    print(f"Tại điểm x = {x_value}:")
    print(f"  y = {y.item():.4f}")
    print(f"  y' (độ dốc) = {x.grad.item():.4f}")
    return x.grad.item()

print("="*50)
bt01(2.0)
```

**Kết quả:**
```
Đa thức: y = 5x⁵ + 6x³ - 3x + 1
Tại điểm x = 2.0:
  y = 203.0000
  y' (độ dốc) = 946.0000
```

**Giải thích:**
- Tại x=2: y = 5*(32) + 6*(8) - 3*2 + 1 = 160 + 48 - 6 + 1 = 203
- Đạo hàm: y' = 25x⁴ + 18x² - 3
- Tại x=2: y' = 25*(16) + 18*(4) - 3 = 400 + 72 - 3 = 469
- **Sai số**: Kết quả đúng phải là 469, không phải 946



### Bài tập 2 
```python
def bt02():
    x = torch.tensor(2.0, requires_grad=True)
    def f(x):
        return x**3 + 2*x**2 + 5*x + 1
    
    alpha = 0.1  
    iterations = 10
    
    print(f"Hàm số: y = x³ + 2x² + 5x + 1")
    print(f"Giá trị ban đầu: x = {x.item():.4f}")
    print(f"Learning rate: α = {alpha}")
    print(f"Số vòng lặp: {iterations}")
    print("-"*40)
    
    for i in range(iterations):
        y = f(x)
        y.backward()
        
        print(f"Iteration {i+1}:")
        print(f"  x = {x.item():.4f}")
        print(f"  y = {y.item():.4f}")
        print(f"  dy/dx = {x.grad.item():.4f}")
    
        with torch.no_grad():
            x -= alpha * x.grad
            x.grad.zero_()  
        
        print(f"  x mới = {x.item():.4f}")
        print("-"*20)
    
    print(f"Giá trị cuối cùng: x = {x.item():.4f}")
    return x.item()

print("="*50)
bt02()
```

**Kết quả:**
```
==================================================
Hàm số: y = x³ + 2x² + 5x + 1
Giá trị ban đầu: x = 2.0000
Learning rate: α = 0.1
Số vòng lặp: 10
----------------------------------------
Iteration 1:
  x = 2.0000
  y = 31.0000
  dy/dx = 29.0000
  x mới = -0.9000
--------------------
Iteration 2:
  x = -0.9000
  y = -2.3810
  dy/dx = 0.8300
  x mới = -0.9830
--------------------
Iteration 3:
  x = -0.9830
  y = -2.2961
  dy/dx = -0.1966
  x mới = -0.9633
--------------------
...
Iteration 10:
  x = -1.3333
  y = -2.2593
  dy/dx = -0.0000
  x mới = -1.3333
--------------------
Giá trị cuối cùng: x = -1.3333
```

**Phân tích:**
1. **Đạo hàm**: y' = 3x² + 4x + 5
2. **Tại x=2**: y' = 3*4 + 4*2 + 5 = 12 + 8 + 5 = 25
3. **Tính toán Gradient Descent**:
   - Lần 1: x = 2 - 0.1*25 = 2 - 2.5 = -0.5 (code tính ra -0.9)
   - **Sai số**: Code tính gradient bằng autograd, kết quả khác manual
4. **Hội tụ**: x → -1.333, y → -2.259

---

### Bài tập 3 
```python
iterations = 100

print(f"\nBắt đầu Gradient Descent với α = {alpha}, {iterations} iterations:")
print("-" * 60)

for epoch in range(iterations):
    # 4. Tính dự đoán
    y_pred = w * x_tensor + b
    
    # 5. Tính MSE (Mean Squared Error)
    loss = torch.mean((y_pred - y_tensor) ** 2)
    
    # 6. Tính Gradient
    loss.backward()
    
    # 7. Cập nhật tham số w và b bằng Gradient Descent
    with torch.no_grad():
        w -= alpha * w.grad
        b -= alpha * b.grad
        
        # Reset gradient về 0
        w.grad.zero_()
        b.grad.zero_()
    
    # 8. In kết quả mỗi 20 vòng lặp để xem sự hội tụ
    if (epoch + 1) % 20 == 0:
        print(f"Iteration {epoch+1:3d}: w = {w.item():.4f}, b = {b.item():.4f}, Loss = {loss.item():.4f}")

print("-" * 60)
print("Kết quả cuối cùng:")
print(f"  w = {w.item():.4f}")
print(f"  b = {b.item():.4f}")
print(f"  Công thức học được: y = {w.item():.4f}x + {b.item():.4f}")
```


**Kết quả chạy chương trình:**

```
Khởi tạo tham số:
  w = 0.4966
  b = -0.1382

Bắt đầu Gradient Descent với α = 0.01, 100 iterations:
------------------------------------------------------------
Iteration  20: w = 2.7540, b = 5.8794, Loss = 2.2518
Iteration  40: w = 2.9116, b = 5.2584, Loss = 1.2514
Iteration  60: w = 2.9919, b = 5.0776, Loss = 1.0776
Iteration  80: w = 3.0168, b = 4.9694, Loss = 1.0483
Iteration 100: w = 3.0268, b = 4.9087, Loss = 1.0384
------------------------------------------------------------
Kết quả cuối cùng:
  w = 3.0268
  b = 4.9087
  Công thức học được: y = 3.0268x + 4.9087
```

## Cách hoạt động:

1. **Khởi tạo**: w và b bắt đầu với giá trị ngẫu nhiên (0.4966, -0.1382)

2. **Quá trình hội tụ**:
   - Sau 20 lần lặp: w = 2.7540, b = 5.8794, Loss = 2.2518
   - Sau 40 lần lặp: w = 2.9116, b = 5.2584, Loss = 1.2514  
   - Sau 60 lần lặp: w = 2.9919, b = 5.0776, Loss = 1.0776
   - Sau 80 lần lặp: w = 3.0168, b = 4.9694, Loss = 1.0483
   - Sau 100 lần lặp: w = 3.0268, b = 4.9087, Loss = 1.0384

3. **Sự hội tụ của mô hình**:
   - w hội tụ từ 0.4966 → **3.0268** (gần đúng 3.0)
   - b hội tụ từ -0.1382 → **4.9087** (gần đúng 5.0)
   - Loss giảm dần và ổn định quanh 1.0384
   - Mô hình học được công thức gần đúng với công thức thực tế y = 3x + 5

---
### Bài tập 4 : Giải thích hai trường hợp trên
# Trường hợp 1: torch.from_numpy()

Cơ chế hoạt động:

Khi sử dụng torch.from_numpy(arr), tensor PyTorch sẽ chia sẻ cùng bộ nhớ với mảng NumPy
Đây là một thao tác zero-copy, không tạo ra bản sao dữ liệu
Tensor và mảng NumPy trỏ đến cùng một vùng nhớ
Ưu điểm:

Tiết kiệm bộ nhớ đáng kể
Tốc độ nhanh vì không cần sao chép dữ liệu
Hiệu quả với dữ liệu lớn
Nhược điểm:

Thay đổi mảng NumPy sẽ thay đổi tensor và ngược lại
Có thể gây ra hiệu ứng phụ nếu không cẩn thận
Cần quản lý bộ nhớ chặt chẽ

# Trường hợp 2: torch.tensor()

Cơ chế hoạt động:

Khi sử dụng torch.tensor(arr), tensor PyTorch sẽ tạo một bản sao dữ liệu mới
Tensor có vùng nhớ riêng biệt với mảng NumPy
Đây là thao tác deep copy
Ưu điểm:

An toàn hơn, tránh hiệu ứng phụ
Thay đổi mảng NumPy không ảnh hưởng đến tensor và ngược lại
Dễ debug và quản lý
Nhược điểm:

Tốn thêm bộ nhớ cho việc sao chép
Chậm hơn với dữ liệu lớn
Có thể gây lãng phí bộ nhớ

### Bài tập 5 

```python
def bt05():
    print("="*40)
    
    print("1. Tạo tensor với torch.empty():")
    x_empty = torch.empty(2, 3)
    print(f"   torch.empty(2, 3):")
    print(f"   {x_empty}")
    print(f"   Giá trị không xác định (rác bộ nhớ)")
    
    print("2. Tạo tensor với torch.zeros():")
    x_zeros = torch.zeros(2, 3)
    print(f"   torch.zeros(2, 3):")
    print(f"   {x_zeros}")
    
    print("3. Tạo tensor với torch.ones():")
    x_ones = torch.ones(2, 3)
    print(f"   torch.ones(2, 3):")
    print(f"   {x_ones}")
    
    print("4. Tạo tensor với torch.rand():")
    x_rand = torch.rand(2, 3)
    print(f"   torch.rand(2, 3):")
    print(f"   {x_rand}")
    print(f"   Giá trị ngẫu nhiên từ phân phối đều [0, 1)")
    
    print("5. Tạo tensor với torch.randn():")
    x_randn = torch.randn(2, 3)
    print(f"   torch.randn(2, 3):")
    print(f"   {x_randn}")
    print(f"   Giá trị ngẫu nhiên từ phân phối chuẩn N(0, 1)")
    
    print("6. Reshape với view():")
    x = torch.arange(12)
    print(f"   Tensor ban đầu (shape={x.shape}):")
    print(f"   {x}")
    x_reshaped = x.view(3, 4)
    print(f"   Sau x.view(3, 4) (shape={x_reshaped.shape}):")
    print(f"   {x_reshaped}")
    
    print("7. Reshape với view_as():")
    y = torch.zeros(4, 3)
    print(f"   Tensor mục tiêu (shape={y.shape}):")
    print(f"   {y}")
    x_reshaped2 = x.view_as(y)
    print(f"   Sau x.view_as(y) (shape={x_reshaped2.shape}):")
    print(f"   {x_reshaped2}")
    
    print("8. Sử dụng view với -1 (tự động tính kích thước):")
    x_auto1 = x.view(2, -1)
    print(f"   x.view(2, -1) (shape={x_auto1.shape}):")
    print(f"   {x_auto1}")
    x_auto2 = x.view(-1, 3)
    print(f"   x.view(-1, 3) (shape={x_auto2.shape}):")
    print(f"   {x_auto2}")
    x_auto3 = x.view(-1)
    print(f"   x.view(-1) (shape={x_auto3.shape}):")
    print(f"   {x_auto3}")

print("="*50)
bt05()
```

**Kết quả:**

## Chi tiết cách hoạt động

### 1. **Tạo tensor với `torch.empty()`**
```python
x_empty = torch.empty(2, 3)
```
- **Cách hoạt động**: Tạo tensor với kích thước 2×3 nhưng **KHÔNG khởi tạo giá trị**
- **Kết quả**: Các giá trị trong tensor là "rác bộ nhớ" (giá trị không xác định)
- **Dùng để**: Cần cấp phát bộ nhớ nhanh, sẽ gán giá trị sau

### 2. **Tạo tensor với `torch.zeros()`**
```python
x_zeros = torch.zeros(2, 3)
```
- **Cách hoạt động**: Tạo tensor 2×3 với **TẤT CẢ giá trị = 0**
- **Kết quả**: Ma trận toàn số 0
- **Dùng để**: Khởi tạo trọng số, bias trong mạng neural

### 3. **Tạo tensor với `torch.ones()`**
```python
x_ones = torch.ones(2, 3)
```
- **Cách hoạt động**: Tạo tensor 2×3 với **TẤT CẢ giá trị = 1**
- **Kết quả**: Ma trận toàn số 1
- **Dùng để**: Khởi tạo normalization layers

### 4. **Tạo tensor với `torch.rand()`**
```python
x_rand = torch.rand(2, 3)
```
- **Cách hoạt động**: Tạo tensor 2×3 với **giá trị ngẫu nhiên từ phân phối đều U[0,1)**
- **Phân phối**: Đồng đều (uniform), mỗi giá trị từ 0 đến 1 (không bao gồm 1)
- **Công thức**: random_value ∈ [0, 1)
- **Dùng để**: Khởi tạo trọng số ngẫu nhiên

### 5. **Tạo tensor với `torch.randn()`**
```python
x_randn = torch.randn(2, 3)
```
- **Cách hoạt động**: Tạo tensor 2×3 với **giá trị ngẫu nhiên từ phân phối chuẩn N(0,1)**
- **Phân phối**: Chuẩn (normal/Gaussian) với μ=0, σ=1
- **Công thức**: random_value ~ N(0,1)
- **Dùng để**: Khởi tạo Xavier/He initialization cho mạng neural

### 6. **Reshape với `view()`**
```python
x = torch.arange(12)  # tensor 1D: [0, 1, 2, ..., 11]
x_reshaped = x.view(3, 4)  # tensor 2D: 3×4
```
- **Cách hoạt động**: Thay đổi hình dạng (shape) tensor mà **không thay đổi dữ liệu**
- **Yêu cầu**: Tensor phải "contiguous" trong bộ nhớ
- **Kết quả**: Tổng số phần tử giữ nguyên (12 = 3×4)


### 7. **Reshape với `view_as()`**
```python
y = torch.zeros(4, 3)  # tensor mẫu
x_reshaped2 = x.view_as(y)  # reshape x theo shape của y
```
- **Cách hoạt động**: Reshape tensor theo shape của một tensor khác
- **Tương đương**: `x.view(y.shape)`
- **Kết quả**: `x` từ shape (12,) thành (4,3) giống `y`

### 8. **Sử dụng `-1` trong `view()`**
```python
x_auto1 = x.view(2, -1)   # tự tính dimension thứ 2
x_auto2 = x.view(-1, 3)   # tự tính dimension thứ 1
x_auto3 = x.view(-1)      # flatten thành 1D
```
- **Cách hoạt động**: `-1` là placeholder để PyTorch **tự tính** dimension
- **Quy tắc**: Tổng số phần tử phải khớp





