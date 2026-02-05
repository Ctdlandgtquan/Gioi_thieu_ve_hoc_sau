# 2374802010414_Lê Trần Đông Quân_buổi 04
# ANN
# bài tập về nhà 
# Chứng minh bước 1 và bước 2
###. Lan truyền ngược (Backpropagation)
- **Quy trình**: Sửa sai từ cuối về đầu:  
  1. Tính lỗi ở đầu ra: Với dự đoán $\hat{y} = 0.65$ và thực tế $y = 1$, lỗi là $y - \hat{y} = 0.35$ (trực giác đơn giản, thực tế dùng gradient).  
  2. Quay lại lớp ẩn: Xác định lỗi do trọng số nào gây ra bằng đạo hàm (gradient).  
  3. Điều chỉnh: Giảm trọng số nếu làm $z$ quá nhỏ, tăng nếu quá lớn.  
- **Công thức?**: Dựa trên **quy tắc chuỗi (chain rule)** trong vi tích phân. Gradient của mất mát $L$ theo trọng số $w$ được tính qua các lớp:  
  - Bước 1: Gradient của $L$ theo $\hat{y}$:  
    $$
    \frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}
    $$
  - Bước 2: Gradient của $\hat{y}$ theo $z$ (với Sigmoid):  
    $$
    \frac{\partial \hat{y}}{\partial z} = \hat{y} \cdot (1 - \hat{y})
    $$
  - Bước 3: Gradient của $z$ theo $w$ (với $z = w \cdot x + b$):  
    $$
    \frac{\partial z}{\partial w} = x
    $$
  - Tổng hợp:  
    $$
    \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
    $$
  - Cập nhật trọng số:  
    $$
    w = w - \alpha \cdot \frac{\partial L}{\partial w}
    $$
    ($\alpha$ là tốc độ học, ví dụ $0.01$).  
- **Ví dụ số**: Nếu $w = 0.5$, gradient $\frac{\partial L}{\partial w} \approx 0.35$ (giả định đơn giản), thì:  
  $$
  w = 0.5 - 0.01 \cdot 0.35 = 0.4965
  $$
- **Tại sao dùng lan truyền ngược?**: Để lần sau dự đoán chính xác hơn bằng cách điều chỉnh trọng số dựa trên lỗi.

# bài làm: 

## **1. Thiết lập bài toán**

```python
z = w * x + b

y_hat = 1 / (1 + exp(-z))

L = - ( y * log(y_hat) + (1 - y) * log(1 - y_hat) )
```

---

## **2. Chứng minh Bước 1: dL/dy_hat**

Từ hàm loss:

```python
L = - ( y * log(y_hat) + (1 - y) * log(1 - y_hat) )
```

Lấy đạo hàm theo `y_hat`:

```python
dL_dyhat = - y / y_hat + (1 - y) / (1 - y_hat)
```

Giải thích:

```python
d/d(y_hat)[ -y * log(y_hat) ] = - y / y_hat

d/d(y_hat)[ - (1 - y) * log(1 - y_hat) ] = (1 - y) / (1 - y_hat)
```

---

## **3. Chứng minh Bước 2: dy_hat/dz**

Sigmoid:

```python
y_hat = 1 / (1 + exp(-z))
```

Lấy đạo hàm:

```python
dyhat_dz = exp(-z) / (1 + exp(-z))**2
```

Viết lại:

```python
dyhat_dz = (1 / (1 + exp(-z))) * (exp(-z) / (1 + exp(-z)))
```

Nhận xét:

```python
1 / (1 + exp(-z)) = y_hat
exp(-z) / (1 + exp(-z)) = 1 - y_hat
```

Suy ra:

```python
dyhat_dz = y_hat * (1 - y_hat)
```



