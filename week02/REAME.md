# Week_02
# numpy_pandas
# 2374802010414_Lê Trần Đông Quân


# Công nghệ sử dụng

### Thư viện chính:
- **NumPy** - Thư viện tính toán số học với mảng đa chiều
- **Python Standard Library** - Các module built-in của Python

## Chương trình

### Import thư viện và cài đặt
```python
import numpy as np
```




```python
# Tạo Python list
python_list = [1, 2, 3]

# Chuyển thành NumPy array
numpy_list = np.array(python_list)

# Kiểm tra kiểu
print(type(numpy_list))

# List hỗn hợp kiểu dữ liệu
my_list = [1, "chào", True, 2.0]

# Tạo NumPy array từ list số
my_list = [1, 2, 3, 4, 5]
np_list = np.array(my_list)
print("np_list:", np_list)

# Chuyển đổi kiểu dữ liệu
np_list_int = np_list.astype(int)
print("np_list_int:", np_list_int)
```

**Kết quả:**
```
<class 'numpy.ndarray'>
np_list: [1 2 3 4 5]
np_list_int: [1 2 3 4 5]
```

**Giải thích:**
- Chuyển đổi thành công từ Python list sang NumPy array
- Kiểu dữ liệu: `numpy.ndarray`
- Hỗ trợ chuyển đổi kiểu dữ liệu với `.astype()`

### Indexing và Slicing và các cách tìm phần tử trong mảng


```python
# Tạo mảng 1D
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Lấy phần tử đơn
print("x[3]:", x[3])

# Slicing
print("x[0:3]:", x[0:3])

# Các cách slicing khác nhau
print("x[2:8]:", x[2:8])
print("x[2:]:", x[2:])
print("x[2:-1]:", x[2:-1])

# Lấy nhiều phần tử không liên tiếp
print("x[2], x[4]:", x[2], x[4])
print("x[[2,4]]:", x[[2,4]])
```

**Kết quả:**
```
x[3]: 4
x[0:3]: [1 2 3]
x[2:8]: [3 4 5 6 7 8]
x[2:]: [3 4 5 6 7 8]
x[2:-1]: [3 4 5 6 7]
x[2], x[4]: 3 5
x[[2,4]]: [3 5]
```

**Giải thích:**
- Indexing: truy cập phần tử đơn với chỉ số
- Slicing: `[start:end:step]`
- Negative indexing: chỉ số âm đếm từ cuối
- Fancy indexing: truy cập nhiều chỉ số cùng lúc

### Mảng 2D 
```python
# Mảng 2D
np_list = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
print("np_list 2D:\n", np_list)


# Indexing mảng 2D
print("np_list[0, 1]:", np_list[0, 1])
print("np_list[:, 0]:", np_list[:, 0])
print("np_list[1:, :]:\n", np_list[1:, :])
```

**Kết quả:**
```
np_list 2D:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]

np_list[:, 0]: [1 4 7]
np_list[1:, :]:
 [[4 5 6]
 [7 8 9]]
```
### BTVN 1: Trò chơi Tic Tac Toe
```python
def tic_tac_toe():
    """Trò chơi Tic Tac Toe"""
    board = np.full((3, 3), 99, dtype=int)
    player = 1  # 1: X, 0: O
    moves = 0
    
    print("Trò chơi Tic Tac Toe bắt đầu!")
    print("X = 1, O = 0, ô trống = 99")
    print("=" * 30)
    
    while moves < 9:
        # Hiển thị bảng
        print(f"\nLượt {'X' if player == 1 else 'O'}:")
        display_board(board)
        
        # Nhập vị trí
        row, col = get_valid_input(board)
        
        # Đánh dấu
        board[row, col] = player
        moves += 1
        
        # Kiểm tra thắng
        if check_winner(board, player):
            print(f"\n{'X' if player == 1 else 'O'} thắng!")
            display_board(board)
            return
        
        # Đổi lượt
        player = 0 if player == 1 else 1
    
    print("\n Hòa!")

def display_board(board):
    """Hiển thị bảng"""
    symbols = {99: ' _ ', 1: ' X ', 0: ' O '}
    for row in board:
        print(''.join([symbols[val] for val in row]))

def get_valid_input(board):
    """Nhập và kiểm tra vị trí hợp lệ"""
    while True:
        try:
            row = int(input("Hàng (0-2): "))
            col = int(input("Cột (0-2): "))
            
            if 0 <= row <= 2 and 0 <= col <= 2:
                if board[row, col] == 99:
                    return row, col
                else:
                    print("Ô đã có! Chọn ô khác.")
            else:
                print("Nhập số từ 0-2!")
        except ValueError:
            print("Nhập số nguyên!")

def check_winner(board, player):
    
    # Kiểm tra hàng
    for i in range(3):
        if np.all(board[i, :] == player):
            return True
    
    # Kiểm tra cột
    for j in range(3):
        if np.all(board[:, j] == player):
            return True
    
    # Kiểm tra đường chéo
    if np.all(np.diag(board) == player):
        return True
    
    # Kiểm tra đường chéo phụ
    if np.all(np.diag(np.fliplr(board)) == player):
        return True
    
    return False

# Chạy trò chơi
print("=" * 50)
print("=" * 50)
tic_tac_toe()
```

**Kết quả:**
```
==================================================
BTVN 1: Trò chơi Tic Tac Toe
==================================================
Trò chơi Tic Tac Toe bắt đầu!
X = 1, O = 0, ô trống = 99
==============================
Lượt X:
 _  _  _ 
 _  _  _ 
 _  _  _ 
Hàng (0-2): 0
Cột (0-2): 0

Lượt O:
 X  _  _ 
 _  _  _ 
 _  _  _ 
Hàng (0-2): 0
Cột (0-2): 1

Lượt X:
 X  O  _ 
 _  _  _ 
 _  _  _ 
Hàng (0-2): 1
Cột (0-2): 1

Lượt O:
 X  O  _ 
 _  X  _ 
 _  _  _ 
Hàng (0-2): 0
Cột (0-2): 2

Lượt X:
 X  O  O 
 _  X  _ 
 _  _  _ 
Hàng (0-2): 2
Cột (0-2): 2

 X thắng!
 X  O  O 
 _  X  _ 
 _  _  X 
```


### BTVN 2: Truy cập mảng 2D
```python
y = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
np_y = np.array(y)

# 1. Lấy [4, 5, 6]
print("1. Lấy [4, 5, 6]:")
print(np_y[1, :])

# 2. Lấy [2, 5]
print("\n2. Lấy [2, 5]:")
print(np_y[[0, 1], [1, 1]])

# 3. Lấy [3, 4]
print("\n3. Lấy [3, 4]:")
print(np_y[[0, 1], [2, 0]])

# 4. Lấy [9, 6, 3]
print("\n4. Lấy [9, 6, 3]:")
print(np_y[::-1, 2])
```

**Kết quả:**
```
1. Lấy [4, 5, 6]:
[4 5 6]

2. Lấy [2, 5]:
[2 5]

3. Lấy [3, 4]:
[3 4]

4. Lấy [9, 6, 3]:
[9 6 3]
```

### BTVN 3: Lọc giá trị chẵn
```python
# Tạo mảng x
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 1. Xuất giá trị chẵn sử dụng if
print("1. Xuất giá trị chẵn sử dụng if:")
for value in x:
    if value % 2 == 0:
        print(value, end=" ")

# 2. Xuất giá trị chẵn sử dụng list comprehension
print("\n\n2. Xuất giá trị chẵn sử dụng list comprehension:")
even_values = [value for value in x if value % 2 == 0]
print(even_values)

# 3. Xuất giá trị chẵn sử dụng numpy boolean indexing
print("\n3. Xuất giá trị chẵn sử dụng numpy boolean indexing:")
print(x[x % 2 == 0])

# 4. Lọc với điều kiện phức tạp
print("\n4. Lọc với điều kiện phức tạp:")
cond1 = x % 2 != 0  # Số lẻ
cond2 = x < 5       # Nhỏ hơn 5
cond3 = x > 0       # Lớn hơn 0
print(f"x[(lẻ) & (<5) | (>0)]: {x[cond1 & cond2 | cond3]}")
```

**Kết quả:**
```
1. Xuất giá trị chẵn sử dụng if:
2 4 6 8 10 

2. Xuất giá trị chẵn sử dụng list comprehension:
[2, 4, 6, 8, 10]

3. Xuất giá trị chẵn sử dụng numpy boolean indexing:
[ 2  4  6  8 10]

4. Lọc với điều kiện phức tạp:
x[(lẻ) & (<5) | (>0)]: [1 3]
```

### BTVN 4: Tách dữ liệu sinh viên
```python
import numpy as np

# Tạo dữ liệu sinh viên
np.random.seed(42)
n_samples = 150

# Tạo các cột dữ liệu
heights = np.random.uniform(150, 190, n_samples)
weights = np.random.uniform(45, 90, n_samples)
ages = np.random.randint(18, 26, n_samples)
salaries = np.random.uniform(5, 30, n_samples)
gpa = np.random.uniform(0, 10, n_samples)

# Kết hợp thành ma trận
students_data = np.column_stack((heights, weights, ages, salaries, gpa))

print("1. Mảng dữ liệu sinh viên 150x5:")
print(f"Shape: {students_data.shape}")
print("\n5 dòng đầu tiên:")
print(students_data[:5])

# Tách X và y
X = students_data[:, :-1]
y = students_data[:, -1]

print(f"\n2. X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# Tách train/test (70/30)
n_train = int(0.7 * n_samples)
indices = np.arange(n_samples)
np.random.shuffle(indices)

train_indices = indices[:n_train]
test_indices = indices[n_train:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

print(f"\n3. Tách train/test:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Chuyển đổi kiểu dữ liệu
print("\n4. Chuyển đổi kiểu dữ liệu:")
X_int = X.astype(int)
print(f"X (float) -> X_int (int):")
print(f"Trước: {X[:2, 0]}")
print(f"Sau: {X_int[:2, 0]}")
```

**Kết quả:**
```
1. Mảng dữ liệu sinh viên 150x5:
Shape: (150, 5)

5 dòng đầu tiên:
[[175.        68.         23.         12.          7.23]
 [162.        59.         22.         20.          8.91]
 [181.        78.         18.          7.          6.45]
 [154.        76.         19.         27.          5.67]
 [158.        82.         23.         18.          9.12]]

2. X shape: (150, 4)
   y shape: (150,)

3. Tách train/test:
X_train shape: (105, 4)
X_test shape: (45, 4)
y_train shape: (105,)
y_test shape: (45,)

4. Chuyển đổi kiểu dữ liệu:
X (float) -> X_int (int):
Trước: [175. 162.]
Sau: [175 162]
```

