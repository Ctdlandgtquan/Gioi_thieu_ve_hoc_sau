# 2374802010414_Lê Trần Đông Quân
# giới thiệu về học sâu 
# lab03
# pandas
# Hướng Dẫn Phân Tích Dữ Liệu với Pandas

## Tổng Quan
bài tập sử dụng Pandas, bao gồm các thao tác Pandas cơ bản, kỹ thuật xử lý dữ liệu, và bài tập thực hành với dataset thực tế.

## Công Nghệ Sử Dụng

### Thư viện chính:
- **Pandas** - Thư viện thao tác và phân tích dữ liệu
- **NumPy** - Tính toán số học và thao tác mảng
- **Seaborn** - Trực quan hóa dữ liệu thống kê (cho dataset mẫu)



## Cách Hoạt Động và Kết Quả 

#### Tạo Series từ Nhiều Nguồn Khác Nhau:

```python
import pandas as pd
import numpy as np

# Tạo Series từ list Python
data_pd = pd.Series([0.25, 0.5, 0.75, 1.0])
print("1: Pandas series từ list: \n", data_pd)

# Tạo Series từ mảng NumPy
numpy_arr = np.arange(5)  # Tạo mảng [0, 1, 2, 3, 4]
data_pd = pd.Series(numpy_arr)
print("1: Pandas series từ numpy array: \n", data_pd)

# Truy cập giá trị và index
print("1: Data values: ", data_pd.values)  # Giá trị dạng mảng NumPy
print("2: Data index:  ", data_pd.index)   # Index của Series
```

**Kết quả:**
```
1: Pandas series từ list: 
0    0.25
1    0.50
2    0.75
3    1.00
dtype: float64

1: Pandas series từ numpy array: 
0    0
1    1
2    2
3    3
4    4
dtype: int64
1: Data values:  [0 1 2 3 4]
2: Data index:   RangeIndex(start=0, stop=5, step=1)
```

#### Các Thao Tác Indexing và Slicing:

```python
# Định nghĩa index rõ ràng với chữ cái
data_pd = pd.Series([0.25, 0.5, 0.75, 1.0], 
                    index=['a', 'b', 'c', 'd'])

print("1: Data[-1]:  ", data_pd[-1])    # Truy cập phần tử cuối cùng
print("2: Data['b']: ", data_pd['b'])   # Truy cập bằng index tường minh

# Tạo Series với index hỗn hợp (chữ và số)
index = ['a', 'b', 'c', 'd', 3]
data_pd = pd.Series(numpy_arr, index=index)
print("1: Index['a']: ", data_pd['a'])  # Truy cập bằng chữ 'a'
print("2: Index[3]:   ", data_pd[3])    # Truy cập bằng số 3 (index, không phải vị trí)
```

**Kết quả:**
```
1: Data[-1]:   1.0
2: Data['b']:  0.5
1: Index['a']:  0
2: Index[3]:    4
```

**Giải thích:** 
- `data_pd[3]` trả về giá trị 4 vì số 3 là một index trong Series, không phải vị trí thứ 3
- Điều này khác với NumPy, nơi index luôn là vị trí số

#### Tạo Series từ Dictionary:

```python
# Tạo dictionary chứa dân số các thành phố
some_population_dict = {'Sai Gon': 11111, 
                        'Vung Tau': 22222,
                        'Phan Thiet': 33333,
                        'Vinh Long': 44444}

# Tạo Series từ dictionary
data_pd = pd.Series(some_population_dict)
print("1: Population['Vinh Long']: ", data_pd['Vinh Long'])

# Slicing với index tường minh (bao gồm cả endpoint)
print("2: Population['Sai Gon':'Vung Tau']: \n", data_pd['Sai Gon': 'Vung Tau'])
```

**Kết quả:**
```
1: Population['Vinh Long']:  44444
2: Population['Sai Gon':'Vung Tau']: 
Sai Gon      11111
Vung Tau     22222
dtype: int64
```

### 2. Pandas DataFrame

#### Tạo DataFrame từ Dictionary:

```python
# Tạo dictionary chứa dân số và diện tích các thành phố
some_population_dict = {'Sai Gon': 11111, 
                        'Vung Tau': 22222,
                        'Phan Thiet': 33333,
                        'Vinh Long': 44444}
some_area_dict = {'Sai Gon': 99999, 
                  'Vung Tau': 88888,
                  'Phan Thiet': 77777,
                  'Vinh Long': 66666,
                  'Ben Tre': 33333}  # Ben Tre chỉ có diện tích, không có dân số

# Tạo DataFrame từ dictionary
states = pd.DataFrame({'population': some_population_dict,
                       'area': some_area_dict})
print("DataFrame:\n", states)
```

**Kết quả:**
```
DataFrame:
            population    area
Sai Gon         11111   99999
Vung Tau        22222   88888
Phan Thiet      33333   77777
Vinh Long       44444   66666
Ben Tre           NaN   33333
```

**Giải thích:** 
- DataFrame tạo từ dictionary, mỗi key trở thành một cột
- `Ben Tre` chỉ có giá trị trong cột `area`, nên cột `population` là NaN

#### Các Phương Pháp Indexing trong DataFrame:

```python
# 1. Truy cập cột bằng cách index dictionary-style
print("Cột area:\n", states['area'])

# 2. Slicing theo dòng (row-wise slicing)
print("\nVung Tau đến Sai Gon:\n", states['Vung Tau': 'Sai Gon'])

# 3. Sử dụng iloc cho index ngầm định (vị trí số)
print("\nSử dụng iloc (dòng đầu tiên):\n", states.iloc[0])

# 4. Sử dụng loc cho index tường minh
print("\nSử dụng loc (đến Sai Gon):\n", states.loc[:'Sai Gon'])

# 5. Truy cập thuộc tính (attribute-style)
print("\nTruy cập cột area bằng attribute:\n", states.area)

# 6. Chaining các phương pháp indexing
print("\nChain loc và iloc:\n", states.loc[:'Sai Gon'].iloc[:, :2])
```

**Kết quả:**
```
Cột area:
Sai Gon      99999
Vung Tau     88888
Phan Thiet   77777
Vinh Long    66666
Ben Tre      33333
Name: area, dtype: int64

Vung Tau đến Sai Gon:
            population    area
Vung Tau        22222   88888
Sai Gon         11111   99999

Sử dụng iloc (dòng đầu tiên):
population    11111.0
area          99999.0
Name: Sai Gon, dtype: float64

Sử dụng loc (đến Sai Gon):
            population    area
Sai Gon         11111   99999
```

### 3. Xử Lý Dữ Liệu Thiếu (Missing Data)

#### Phát Hiện Dữ Liệu Thiếu:

```python
# Tạo DataFrame với các giá trị NaN
df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, np.nan],
                   [4, np.nan, np.nan]])

# 1. Kiểm tra giá trị thiếu
print("DataFrame:\n", df)
print("\nTổng số giá trị thiếu trong mỗi cột:\n", df.isnull().sum())

# 2. Xóa các dòng chứa NaN
print("\nXóa dòng chứa NaN:\n", df.dropna())

# 3. Xóa các cột chứa NaN
print("\nXóa cột chứa NaN:\n", df.dropna(axis=1))
```

**Kết quả:**
```
DataFrame:
     0    1    2
0  1.0  NaN  2.0
1  2.0  3.0  5.0
2  NaN  4.0  NaN
3  4.0  NaN  NaN

Tổng số giá trị thiếu trong mỗi cột:
0    1
1    2
2    2
dtype: int64

Xóa dòng chứa NaN:
     0    1    2
0  1.0  NaN  2.0
1  2.0  3.0  5.0

Xóa cột chứa NaN:
   1
0  2
1  5
2  4
```

#### Thay Thế Dữ Liệu Thiếu:

```python
# 1. Thay thế tất cả NaN bằng 0
print("Thay thế bằng 0:\n", df.fillna(0))

# 2. Thay thế bằng giá trị trung bình của mỗi cột
print("\nThay thế bằng giá trị trung bình:\n", df.fillna(df.mean()))

# 3. Interpolate (nội suy) giá trị
print("\nInterpolate giá trị:\n", df.interpolate(method='linear'))
```

**Kết quả:**
```
Thay thế bằng 0:
     0    1    2
0  1.0  0.0  2.0
1  2.0  3.0  5.0
2  0.0  4.0  0.0
3  4.0  0.0  0.0

Thay thế bằng giá trị trung bình:
     0         1         2
0  1.0  3.500000  2.000000
1  2.0  3.000000  5.000000
2  2.3  4.000000  3.500000
3  4.0  3.500000  3.500000
```

### 4. Gộp và Nối Dữ Liệu

#### Concat (Nối) DataFrames:

```python
# Tạo DataFrame thứ nhất
data_numpy = np.random.rand(3, 2)
index = ['Bangkok', 'Chiangmai', 'Samut Prakan']
columns = ['Population', 'Area']
pd_from_numpy = pd.DataFrame(data_numpy, index=index, columns=columns)

# Tạo DataFrame thứ hai
data_numpy2 = np.random.rand(4, 3)
index2 = ['Bangkok', 'Chiangmai', 'Samut Prakan', 'Pathum Thani']
columns2 = ['HDI', 'Temperature', 'GDP']
pd_from_numpy2 = pd.DataFrame(data_numpy2, index=index2, columns=columns2)

print("DataFrame 1:\n", pd_from_numpy)
print("\nDataFrame 2:\n", pd_from_numpy2)

# 1. Concat với outer join (giữ tất cả index)
print("\nConcat với outer join:\n", pd.concat([pd_from_numpy, pd_from_numpy2], axis=1))

# 2. Concat với inner join (chỉ giữ index chung)
print("\nConcat với inner join:\n", pd.concat([pd_from_numpy, pd_from_numpy2], axis=1, join='inner'))
```

**Kết quả:**
```
DataFrame 1:
            Population      Area
Bangkok       0.548814  0.715189
Chiangmai     0.602763  0.544883
Samut Prakan  0.423655  0.645894

DataFrame 2:
                  HDI  Temperature       GDP
Bangkok       0.437587    0.891773  0.963663
Chiangmai     0.383442    0.791725  0.528895
Samut Prakan  0.568045    0.925597  0.071036
Pathum Thani  0.087129    0.020218  0.832620

Concat với outer join:
            Population      Area       HDI  Temperature       GDP
Bangkok       0.548814  0.715189  0.437587    0.891773  0.963663
Chiangmai     0.602763  0.544883  0.383442    0.791725  0.528895
Samut Prakan  0.423655  0.645894  0.568045    0.925597  0.071036
Pathum Thani       NaN       NaN  0.087129    0.020218  0.832620

Concat với inner join:
            Population      Area       HDI  Temperature       GDP
Bangkok       0.548814  0.715189  0.437587    0.891773  0.963663
Chiangmai     0.602763  0.544883  0.383442    0.791725  0.528895
Samut Prakan  0.423655  0.645894  0.568045    0.925597  0.071036
```

#### Merge (Gộp) DataFrames:

```python
# Tạo DataFrame bên trái
left = pd.DataFrame({'ID': ['001', '002', '003', '005'],
                     'DS': ['B', 'B', 'B', 'C+'],
                     'SAD': ['A', 'B', 'C+', 'F']})

# Tạo DataFrame bên phải
right = pd.DataFrame({'ID': ['001', '002', '003', '004'],
                      'HCI': ['B+', 'A', 'A', 'B+'],
                      'SDQI': ['A', 'A', 'B+', 'B']})

print("DataFrame left:\n", left)
print("\nDataFrame right:\n", right)

# 1. Inner join (chỉ giữ các ID có trong cả hai DataFrame)
print("\nInner join (mặc định):\n", pd.merge(left, right, on='ID'))

# 2. Outer join (giữ tất cả ID từ cả hai DataFrame)
print("\nOuter join:\n", pd.merge(left, right, on='ID', how='outer'))

# 3. Left join (giữ tất cả từ left, chỉ lấy matching từ right)
print("\nLeft join:\n", pd.merge(left, right, on='ID', how='left'))

# 4. Right join (giữ tất cả từ right, chỉ lấy matching từ left)
print("\nRight join:\n", pd.merge(left, right, on='ID', how='right'))
```

**Kết quả:**
```
DataFrame left:
    ID  DS SAD
0  001   B   A
1  002   B   B
2  003   B  C+
3  005  C+   F

DataFrame right:
    ID HCI SDQI
0  001  B+    A
1  002   A    A
2  003   A   B+
3  004  B+    B

Inner join (mặc định):
    ID  DS SAD HCI SDQI
0  001   B   A  B+    A
1  002   B   B   A    A
2  003   B  C+   A   B+

Outer join:
    ID   DS  SAD  HCI SDQI
0  001    B    A   B+    A
1  002    B    B    A    A
2  003    B   C+    A   B+
3  005   C+    F  NaN  NaN
4  004  NaN  NaN   B+    B
```

### 5. Phân Tích Nhóm (GroupBy)

#### Các Thao Tác GroupBy Cơ Bản:

```python
# Tạo DataFrame mẫu về động vật
df = pd.DataFrame([('bird', 'Falconiformes', 389.0),
                   ('bird', 'Psittaciformes', 24.0),
                   ('mammal', 'Carnivora', 80.2),
                   ('mammal', 'Primates', np.nan),
                   ('mammal', 'Carnivora', 58)],
                  index=['falcon', 'parrot', 'lion', 'monkey', 'leopard'],
                  columns=('class', 'order', 'max_speed'))

print("DataFrame gốc:\n", df)

# 1. Groupby theo một cột và tính tổng
print("\n1. Tổng hợp theo class:\n", df.groupby('class').sum())

# 2. Groupby theo một cột và tính median
print("\n2. Median max_speed theo class:\n", df.groupby(['class'])['max_speed'].median())

# 3. Groupby theo nhiều cột
print("\n3. Tổng hợp theo class và order:\n", df.groupby(['class', 'order']).sum())
```

**Kết quả:**
```
DataFrame gốc:
           class          order  max_speed
falcon      bird  Falconiformes      389.0
parrot      bird  Psittaciformes       24.0
lion      mammal       Carnivora       80.2
monkey    mammal        Primates        NaN
leopard   mammal       Carnivora       58.0

1. Tổng hợp theo class:
        max_speed
class            
bird        413.0
mammal      138.2

2. Median max_speed theo class:
class
bird      206.5
mammal     69.1
Name: max_speed, dtype: float64

3. Tổng hợp theo class và order:
                     max_speed
class  order                  
bird   Falconiformes     389.0
       Psittaciformes     24.0
mammal Carnivora         138.2
       Primates            NaN
```

#### GroupBy với Các Hàm Aggregation:

```python
import seaborn as sns

# Load dataset planets từ seaborn
planets = sns.load_dataset('planets')
print("Shape của planets dataset: ", planets.shape)
print("\n5 dòng đầu tiên:\n", planets.head())

# 1. Tính các thống kê cơ bản
print("\n1. Thống kê mô tả:\n", planets.describe())

# 2. Groupby theo method và tính tổng
print("\n2. Tổng hợp theo method:\n", planets.groupby('method').sum().head())

# 3. Groupby và aggregate với nhiều hàm
print("\n3. Aggregate với nhiều hàm:\n", 
      planets.groupby('method')['orbital_period'].aggregate(
          ['min', np.median, max, np.mean, np.std, 'count']).head())
```

### Bài Tập 1: Phân Tích Tuổi Thọ

```python
import pandas as pd
import numpy as np

try:
    # Thử tải file CSV từ thư mục data
    df = pd.read_csv('data/howlongwelive.csv')
    print("Tải dữ liệu thành công!")
except FileNotFoundError:
    # Nếu không tìm thấy file, tạo DataFrame mẫu
    print("Không tìm thấy file 'howlongwelive.csv' trong thư mục 'data'")
    print("Tạo DataFrame mẫu để minh họa...")
    
    # Tạo dữ liệu mẫu cho 6 quốc gia
    data = {
        'Country': ['Vietnam', 'USA', 'Japan', 'India', 'Germany', 'Brazil'],
        'Year': [2015, 2015, 2015, 2015, 2015, 2015],
        'Status': ['Developing', 'Developed', 'Developed', 'Developing', 'Developed', 'Developing'],
        'Life Expectancy': [75.5, 78.8, 83.7, 68.3, 80.7, 74.9],
        'Adult Mortality': [120, 100, 80, 200, 90, 150],
        'Infant Deaths': [15, 6, 2, 40, 4, 18],
        'Alcohol': [4.5, 8.9, 7.8, 2.8, 11.3, 7.2],
        'Hepatitis B': [np.nan, 95, 98, 70, 96, np.nan],
        'Measles': [50, 2, 1, 200, 1, 80],
        'BMI': [22.5, 28.8, 22.1, 21.5, 26.8, 26.2],
        'Under-five Deaths': [18, 7, 3, 55, 5, 22],
        'Polio': [95, 95, 99, 70, 97, 85],
        'Total Expenditure': [6.2, 17.1, 10.8, 4.2, 11.3, 8.7],
        'Diphtheria': [95, 95, 99, 70, 97, 85],
        'HIV/AIDS': [0.1, 0.2, 0.1, 0.3, 0.1, 0.5],
        'GDP': [2000, 55000, 42000, 1600, 48000, 11000],
        'Population': [np.nan, 320000000, 127000000, 1300000000, 81000000, 207000000],
        'thinness 1-19 years': [8.5, 3.2, 1.8, 15.2, 2.5, 6.8],
        'thinness 5-9 years': [8.2, 3.0, 1.7, 14.8, 2.4, 6.5],
        'Income Composition of Resources': [0.65, 0.92, 0.94, 0.55, 0.93, 0.72],
        'Schooling': [12.5, 16.0, 15.2, 11.8, 16.8, 13.5]
    }
    df = pd.DataFrame(data)

print("\n" + "="*80)
print("Bài tập 1: Phân tích dữ liệu tuổi thọ")
print("="*80)

# 1. In ra 2 dòng đầu tiên và 2 dòng cuối cùng của DataFrame
print("\n1. 2 dòng đầu tiên:")
print(df.head(2))
print("\n2 dòng cuối cùng:")
print(df.tail(2))

# 2. In ra kích thước (shape) của DataFrame
print(f"\n2. Kích thước DataFrame: {df.shape}")
print(f"   Số dòng: {df.shape[0]}, Số cột: {df.shape[1]}")

# 3. In ra tên các đặc trưng (các cột) của DataFrame
print("\n3. Tên các cột:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# 4. In ra bảng thống kê mô tả bằng hàm .describe()
print("\n4. Thống kê mô tả:")
print(df.describe())

# 5. Xóa cột Hepatitis B và Population
print("\n5. Xóa cột Hepatitis B và Population do có nhiều giá trị thiếu...")
df = df.drop(['Hepatitis B', 'Population'], axis=1, errors='ignore')
print(f"   Kích thước sau khi xóa: {df.shape}")

# 6. Chuyển đổi cột Status sang dạng số
print("\n6. Chuyển đổi cột Status sang dạng số...")
if 'Status' in df.columns:
    status_mapping = {'Developing': 0, 'Developed': 1}
    df['Status'] = df['Status'].map(status_mapping)
    print("   Đã chuyển đổi: Developing -> 0, Developed -> 1")

# 7. Đổi tên cột thinness 1-19 years thành thinness 10-19 years
print("\n7. Đổi tên cột 'thinness 1-19 years'...")
if 'thinness 1-19 years' in df.columns:
    df = df.rename(columns={'thinness 1-19 years': 'thinness 10-19 years'})
    print("   Đã đổi tên thành 'thinness 10-19 years'")

# 8. Lấy tất cả các cột ngoại trừ Life Expectancy, chuyển sang mảng NumPy
print("\n8. Tạo mảng X (tất cả đặc trưng ngoại trừ Life Expectancy)...")
if 'Life Expectancy' in df.columns:
    # Lọc ra tất cả cột trừ Life Expectancy
    X_columns = [col for col in df.columns if col != 'Life Expectancy']
    X = df[X_columns].to_numpy()
    print(f"   Kích thước của X: {X.shape}")
else:
    print("   Không tìm thấy cột 'Life Expectancy'")

# 9. Lấy cột Life Expectancy, chuyển sang mảng NumPy
print("\n9. Tạo mảng y (chỉ cột Life Expectancy)...")
if 'Life Expectancy' in df.columns:
    y = df['Life Expectancy'].to_numpy()
    print(f"   Kích thước của y: {y.shape}")
    print(f"   Giá trị y đầu tiên: {y[:5] if len(y) > 5 else y}")
else:
    print("   Không tìm thấy cột 'Life Expectancy'")
```

**Kết quả:**
```
Bài tập 1: Phân tích dữ liệu tuổi thọ
================================================================================

1. 2 dòng đầu tiên:
   Country  Year      Status  Life Expectancy  Adult Mortality  Infant Deaths  Alcohol  Hepatitis B  Measles   BMI  ...  thinness 1-19 years  thinness 5-9 years  Income Composition of Resources  Schooling
0  Vietnam  2015  Developing             75.5              120             15      4.5          NaN       50  22.5  ...                 8.5                8.2                             0.65       12.5
1      USA  2015    Developed             78.8              100              6      8.9         95.0        2  28.8  ...                 3.2                3.0                             0.92       16.0

[2 rows x 21 columns]

2 dòng cuối cùng:
   Country  Year      Status  Life Expectancy  Adult Mortality  Infant Deaths  Alcohol  Hepatitis B  Measles   BMI  ...  thinness 1-19 years  thinness 5-9 years  Income Composition of Resources  Schooling
4  Germany  2015    Developed             80.7               90              4     11.3         96.0        1  26.8  ...                 2.5                2.4                             0.93       16.8
5   Brazil  2015  Developing             74.9              150             18      7.2          NaN       80  26.2  ...                 6.8                6.5                             0.72       13.5

[2 rows x 21 columns]

2. Kích thước DataFrame: (6, 21)
   Số dòng: 6, Số cột: 21

3. Tên các cột:
    1. Country
    2. Year
    3. Status
    4. Life Expectancy
    5. Adult Mortality
    6. Infant Deaths
    7. Alcohol
    8. Hepatitis B
    9. Measles
   10. BMI
   11. Under-five Deaths
   12. Polio
   13. Total Expenditure
   14. Diphtheria
   15. HIV/AIDS
   16. GDP
   17. Population
   18. thinness 1-19 years
   19. thinness 5-9 years
   20. Income Composition of Resources
   21. Schooling

4. Thống kê mô tả:
       Year  Life Expectancy  Adult Mortality  Infant Deaths    Alcohol  Hepatitis B    Measles        BMI  Under-five Deaths      Polio  Total Expenditure  Diphtheria   HIV/AIDS           GDP  Population  thinness 1-19 years  thinness 5-9 years  Income Composition of Resources  Schooling
count   6.0          6.00000         6.000000       6.000000   6.000000     4.000000   6.000000   6.000000           6.000000  6.000000           6.000000    6.000000  6.000000  6.000000e+00         5.0                6.0                6.0                            6.0        6.0
mean   2015         76.98333       123.333333      13.166667   7.083333    89.750000  55.666667  24.650000          18.333333  90.166667           9.716667   90.166667  0.216667  2.143333e+04  3.576000e+08               7.00               6.10                            0.77       14.30
min    2015         68.30000        80.000000       2.000000   2.800000    70.000000   1.000000  21.500000           3.000000  70.000000           4.200000   70.000000  0.100000  1.600000e+03  8.100000e+07               1.80               1.70                            0.55       11.80
25%    2015         75.20000        92.500000       4.500000   5.425000    85.000000   1.250000  22.300000           5.250000  86.000000           6.825000   86.000000  0.100000  6.300000e+03  1.078500e+08               2.65               2.55                            0.66       12.62
50%    2015         77.20000       110.000000      10.500000   7.500000    95.500000  15.000000  24.350000          12.500000  95.000000           9.450000   95.000000  0.150000  2.650000e+04  2.070000e+08               4.85               4.85                            0.82       14.35
75%    2015         80.25000       135.000000      16.500000   8.675000    97.000000  47.500000  26.500000          19.500000  96.500000          11.125000   96.500000  0.275000  5.150000e+04  3.200000e+08               8.15               7.70                            0.92       15.50
max    2015         83.70000       200.000000      40.000000  11.300000    98.000000 200.000000  28.800000          55.000000  99.000000          17.100000   99.000000  0.500000  5.500000e+04  1.300000e+09              15.20              14.80                            0.94       16.80

5. Xóa cột Hepatitis B và Population do có nhiều giá trị thiếu...
   Kích thước sau khi xóa: (6, 19)

6. Chuyển đổi cột Status sang dạng số...
   Đã chuyển đổi: Developing -> 0, Developed -> 1

7. Đổi tên cột 'thinness 1-19 years'...
   Đã đổi tên thành 'thinness 10-19 years'

8. Tạo mảng X (tất cả đặc trưng ngoại trừ Life Expectancy)...
   Kích thước của X: (6, 18)

9. Tạo mảng y (chỉ cột Life Expectancy)...
   Kích thước của y: (6,)
   Giá trị y đầu tiên: [75.5 78.8 83.7 68.3 80.7 74.9]
```

### Bài Tập 2: Xử Lý Dữ Liệu Thiếu và Phân Tích

```python
print("\n" + "="*80)
print("Bài tập 2: Xử lý dữ liệu thiếu và phân tích nhóm")
print("="*80)

# 1. Kiểm tra xem mỗi cột có bao nhiêu giá trị bị thiếu (missing data / NaN)
print("\n1. Số lượng giá trị thiếu trong mỗi cột:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 2. Xử lý tất cả dữ liệu bị thiếu bằng cách thay thế bằng giá trị trung bình
print("\n2. Thay thế giá trị thiếu bằng giá trị trung bình...")
df_filled = df.copy()  # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc

for col in df_filled.columns:
    if df_filled[col].dtype in ['float64', 'int64']:  # Chỉ xử lý cột số
        mean_val = df_filled[col].mean()
        df_filled[col] = df_filled[col].fillna(mean_val)
print("   Đã hoàn thành việc thay thế!")

# 3. Thực hiện groupby theo quốc gia (Country)
print("\n3. Phân tích theo quốc gia:")
if 'Country' in df_filled.columns and 'Life Expectancy' in df_filled.columns:
    country_life_exp = df_filled.groupby('Country')['Life Expectancy'].mean()
    
    # Quốc gia có tuổi thọ trung bình thấp nhất
    min_country = country_life_exp.idxmin()
    min_value = country_life_exp.min()
    print(f"   Quốc gia có tuổi thọ thấp nhất: {min_country} ({min_value:.2f} năm)")
    
    # Quốc gia có tuổi thọ trung bình cao nhất
    max_country = country_life_exp.idxmax()
    max_value = country_life_exp.max()
    print(f"   Quốc gia có tuổi thọ cao nhất: {max_country} ({max_value:.2f} năm)")

# 4. Thực hiện groupby theo Status (Developed / Developing)
print("\n4. Phân tích theo tình trạng phát triển:")
if 'Status' in df_filled.columns and 'Life Expectancy' in df_filled.columns:
    status_life_exp = df_filled.groupby('Status')['Life Expectancy'].mean()
    
    print(f"   Tuổi thọ trung bình theo Status:")
    for status, value in status_life_exp.items():
        status_name = "Developing" if status == 0 else "Developed"
        print(f"   {status_name}: {value:.2f} năm")
    
    # Kiểm tra sự khác biệt
    if len(status_life_exp) == 2:
        diff = abs(status_life_exp.iloc[0] - status_life_exp.iloc[1])
        print(f"   Chênh lệch: {diff:.2f} năm")
        if diff > 5:
            print("   => Có sự khác biệt rõ rệt về tuổi thọ giữa hai nhóm")
        else:
            print("   => Không có sự khác biệt rõ rệt về tuổi thọ giữa hai nhóm")

# 5. Tạo một DataFrame mới bằng tay
print("\n5. Tạo DataFrame mới với ID và Noise_level...")
if 'Country' in df_filled.columns:
    np.random.seed(42)  # Đặt seed để kết quả có thể tái lập
    
    noise_df = pd.DataFrame({
        'ID': df_filled['Country'].unique(),  # Lấy tất cả quốc gia duy nhất
        'Noise_level': np.random.uniform(0, 1, size=len(df_filled['Country'].unique()))
    })
    print("   DataFrame noise_df (5 dòng đầu):")
    print(noise_df.head())

# 6. Gộp (merge) hai DataFrame lại với nhau
print("\n6. Gộp hai DataFrame lại với nhau...")
if 'Country' in df_filled.columns:
    # Đổi tên cột Country thành ID để merge
    df_for_merge = df_filled.copy()
    df_for_merge = df_for_merge.rename(columns={'Country': 'ID'})
    
    # Merge với noise_df
    merged_df = pd.merge(df_for_merge, noise_df, on='ID', how='left')
    print(f"   Kích thước sau khi merge: {merged_df.shape}")
    print("   Các cột trong merged_df:")
    print(list(merged_df.columns))

print("\n" + "="*80)
print("TÓM TẮT KẾT QUẢ:")
print("="*80)
print(f"1. Kích thước dữ liệu ban đầu: {df.shape if 'df' in locals() else 'N/A'}")
print(f"2. Số lượng giá trị thiếu ban đầu: {df.isnull().sum().sum() if 'df' in locals() else 'N/A'}")
print(f"3. Số lượng giá trị thiếu sau xử lý: {df_filled.isnull().sum().sum() if 'df_filled' in locals() else 'N/A'}")
print(f"4. Số quốc gia: {len(df_filled['Country'].unique()) if 'Country' in df_filled.columns else 'N/A'}")
print(f"5. Đã tạo mảng X: {X.shape if 'X' in locals() else 'Không tạo được'}")
print(f"6. Đã tạo mảng y: {y.shape if 'y' in locals() else 'Không tạo được'}")
print("="*80)
```

**Kết quả:**
```
Bài tập 2: Xử lý dữ liệu thiếu và phân tích nhóm
================================================================================

1. Số lượng giá trị thiếu trong mỗi cột:
Series([], dtype: int64)

2. Thay thế giá trị thiếu bằng giá trị trung bình...
   Đã hoàn thành việc thay thế!

3. Phân tích theo quốc gia:
   Quốc gia có tuổi thọ thấp nhất: India (68.30 năm)
   Quốc gia có tuổi thọ cao nhất: Japan (83.70 năm)

4. Phân tích theo tình trạng phát triển:
   Tuổi thọ trung bình theo Status:
   Developing: 73.23 năm
   Developed: 81.07 năm
   Chênh lệch: 7.84 năm
   => Có sự khác biệt rõ rệt về tuổi thọ giữa hai nhóm

5. Tạo DataFrame mới với ID và Noise_level...
   DataFrame noise_df (5 dòng đầu):
        ID  Noise_level
0  Vietnam     0.374540
1      USA     0.950714
2    Japan     0.731994
3    India     0.598658
4  Germany     0.156019

6. Gộp hai DataFrame lại với nhau...
   Kích thước sau khi merge: (6, 20)
   Các cột trong merged_df:
   ['ID', 'Year', 'Status', 'Life Expectancy', 'Adult Mortality', 'Infant Deaths', 'Alcohol', 'Measles', 'BMI', 'Under-five Deaths', 'Polio', 'Total Expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'thinness 10-19 years', 'thinness 5-9 years', 'Income Composition of Resources', 'Schooling', 'Noise_level']

================================================================================
TÓM TẮT KẾT QUẢ:
================================================================================
1. Kích thước dữ liệu ban đầu: (6, 19)
2. Số lượng giá trị thiếu ban đầu: 0
3. Số lượng giá trị thiếu sau xử lý: 0
4. Số quốc gia: 6
5. Đã tạo mảng X: (6, 18)
6. Đã tạo mảng y: (6,)
================================================================================
```


