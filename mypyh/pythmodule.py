#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
#Created on 2020年2月13日
#@author: haoch
# 文件名：pythmodule.py

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.metrics import r2_score

#并将数据加载到Pandas 的dataframe中。
df = pd.read_csv(r"C:\Users\haoch\Documents\SZ#159915.csv",encoding="gb2312",names=['date','open','high','low','close','volume','money'])
#print(df.head())

#将“日期”列转换为时间数据类型，并将“日期”列设置为索引。
df['date'] = pd.to_datetime(df['date'])

#默认的，当列变成行索引之后，原来的列就没了，但是可以通过设置drop来保留原来的列。
df = df.set_index(['date'], drop=True)

#print(df.head())
#开启一个窗口，同时设置大小，分辨率
##使用plt.figure定义一个图像窗口：大小为(10, 6).
#plt.figure(figsize=(10, 6))
#plt.plot(df['close'])
#plt.show()

#按日期“2019–01–02”将数据拆分为训练集和测试集，即在此日期之前的数据是训练数据，此之后的数据是测试数据
split_date = pd.Timestamp('2019/01/02')

df =  df['close']

train = df.loc[:split_date]

test = df.loc[split_date:]

plt.figure(figsize=(10, 6))

ax = train.plot()

test.plot(ax=ax)

plt.legend(['train', 'test']);

#plt.show()

#我们将训练和测试数据缩放为[-1，1]。

#scaler = MinMaxScaler(feature_range=(-1, 1))

#train_sc = scaler.fit_transform(train)

#test_sc = scaler.transform(test)
train_sc = train
test_sc = test
#获取训练和测试数据。

X_train = train_sc[:-1]

y_train = train_sc[1:]

X_test = test_sc[:-1]

y_test = test_sc[1:]

"""
#用于时间序列预测的简单人工神经网络
#我们创建一个序列模型。
nn_model = keras.Sequential()
#通过.add()方法添加层。
nn_model.add(keras.layers.Dense(12, input_dim=1, activation='relu'))
#将“input_dim”参数传递到第一层。
nn_model.add(keras.layers.Dense(1))
#激活函数为线性整流函数Relu（Rectified Linear Unit，也称校正线性单位）。

#通过compile方法完成学习过程的配置。

#损失函数是mean_squared_error，优化器是Adam。
nn_model.compile(loss='mean_squared_error', optimizer='adam')

#当监测到loss停止改进时，结束训练。
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=1)
#patience =2，表示经过数个周期结果依旧没有改进，此时可以结束训练。

#人工神经网络的训练时间为100个周期，每次用1个样本进行训练。
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, 
                       callbacks=[early_stop], shuffle=False)

y_pred_test_nn = nn_model.predict(X_test)

y_train_pred_nn = nn_motrain.predict(X_train)

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))
"""

#LSTM
X_train_lmse = X_train
X_test_lmse = X_test
#LSTM网络的构建和模型编译和人工神经网络相似。
lstm_model = keras.Sequential()

#LSTM有一个可见层，它有1个输入。
#隐藏层有7个LSTM神经元。
lstm_model.add(keras.layers.LSTM(7, input_shape=(1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))

#输出层进行单值预测。
lstm_model.add(keras.layers.Dense(1))

#LSTM神经元使用Relu函数进行激活。
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
#LSTM的训练时间为100个周期，每次用1个样本进行训练。
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=1)

history_lstm_model = lstm_model.fit(X_train_lmse, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])

#训练早在第10个周期就停了下来。

y_pred_test_lstm = lstm_model.predict(X_test_lmse)

y_train_pred_lstm = lstm_model.predict(X_train_lmse)

print(traine R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))


比较模型

我们比较了两种模型的测试MSE

nn_test_mse = nn_model.evaluate(X_test, y_test, batch_size=1)

lstm_test_mse = lstm_model.evaluate(X_test_lmse, y_test, batch_size=1)

print('NN: %f'%nn_test_mse)

print('LSTM: %f'%lstm_test_mse)

进行预测

nn_y_pred_test = nn_model.predict(X_test)

lstm_y_pred_test = lstm_fit2evaluateedict(X_test_lmse)

plt.figure(figsize=(10, 6))

plt.plot(fit2evaluate label='True')

plt.plot(y_pred_test_nn, label='NN')

plt.title("NN's Prediction")

plt.xlabel('Observation')

plt.ylabel('Adj Close Scaled')

plt.legend()

plt.show();

plt.figure(figsize=(10, 6))

plt.plot(y_test, label='True')

plt.plot(y_pred_test_lstm, label='LSTM')

plt.title("LSTM's Prediction")

plt.xlabel('Observation')

plt.ylabel('Adj Close scaled')

plt.legend()

plt.show();