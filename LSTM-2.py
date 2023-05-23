# 调用库 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers

#### 数据处理部分 ####

# 读入数据
google_stock = pd.read_excel('data2.csv', encoding='GBK')
google_stock.tail() # 查看部分数据
google_stock.head()

# 时间戳长度
time_stamp = 5 # 输入序列长度

# 划分训练集与验证集
google_stock = google_stock[['开盘价(元/点)_OpPr']]
train = google_stock[0:7000 + time_stamp]
valid = google_stock[7000 - time_stamp:8500 + time_stamp]
test = google_stock[8500 - time_stamp:]

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)
x_train, y_train = [], []


# 训练集切片
for i in range(time_stamp, len(train)-5):
    x_train.append(scaled_data[i - time_stamp:i])
    y_train.append(scaled_data[i: i+5])

x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1,5)

# 验证集切片
scaled_data = scaler.fit_transform(valid)
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid)-5):
    x_valid.append(scaled_data[i - time_stamp:i])
    y_valid.append(scaled_data[i: i+5])

x_valid, y_valid = np.array(x_valid), np.array(y_valid).reshape(-1,5)

# 测试集切片
scaled_data = scaler.fit_transform(test)
x_test, y_test = [], []
for i in range(time_stamp, len(test)-5):
    x_test.append(scaled_data[i - time_stamp:i])
    y_test.append(scaled_data[i: i+5])

x_test, y_test = np.array(x_test), np.array(y_test).reshape(-1,5)