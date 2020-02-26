#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
'''
Created on 2020年2月20日

@author: haochunyang@sina.com
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import utils
pd.set_option('display.max_columns',100)

source_fields = ['date','open','high','low','close','volume','amount']  # features
fields = ['open','high','low','close','volume','amount']
fileName = r"C:\Users\haoch\Documents\stockdata\SZ#002463.csv"
print(fileName)
data = pd.read_csv(fileName,encoding="gb2312",names=source_fields)

a = 9.39
b = 15.75
c = b/a -1
print(c)
# print(data.info())
# 
# # last = data['volume'][len(data)-2]
# # print(last)
# # 
# data['avage'] = data['amount']  / data['volume']
# data['avage3'] = (data['avage'].shift(1) + data['avage'].shift(2) + data['avage'].shift(3))/3
# data['avage5'] = (data['avage'].shift(1) + data['avage'].shift(2) + data['avage'].shift(3) + data['avage'].shift(4) + data['avage'].shift(5))/5
# data['amount3'] = (data['amount'].shift(1) + data['amount'].shift(2) + data['amount'].shift(3))/3
# data['amount5'] = (data['amount'].shift(1) + data['amount'].shift(2) + data['amount'].shift(3) + data['amount'].shift(4) + data['amount'].shift(5))/5
# data['avage10'] = (data['avage5'].shift(1) + data['avage5'].shift(2))/2
# # data['avage20'] = (data['avage10'].shift(1) + data['avage10'].shift(2))/2
# data['amount10'] =(data['amount5'].shift(1) + data['amount5'].shift(2))/2
# # data['amount20'] =(data['amount10'].shift(1) + data['amount10'].shift(2))/2
# 
# data['inout']=np.where(data['open'] > data['close'],-data['amount'],data['amount'])
# data['inout5'] = (data['inout'] + data['inout'].shift(1) + data['inout'].shift(2) + data['inout'].shift(3) + data['inout'].shift(4))/5
# 
# data.dropna(inplace=True)            
# data.reset_index(drop=True, inplace=True)
#  
# print(data.tail(20))
#  
# plt.figure(figsize=(100, 6))
#  
# plt.subplot(3,1,1)  # 将画板分为2行1两列，本幅图位于第一个位置
#     
# plt.plot(data['amount'][-200:-1], label='amount')
#     
# plt.plot(data['amount3'][-200:-1], label='amount3')
#     
# plt.plot(data['amount5'][-200:-1], label='amount5')
#   
# plt.plot(data['amount10'][-200:-1], label='amount10')
# 
# 
# # plt.plot(data['amount20'][-200:-1], label='amount20')
#  
# plt.subplot(3,1,2)  # 将画板分为2行1两列，本幅图位于第一个位置
# 
# plt.plot(data['avage'][-200:-1], label='avage')
#  
# plt.plot(data['avage3'][-200:-1], label='avage3')
#   
# plt.plot(data['avage5'][-200:-1], label='avage5')
#    
# plt.plot(data['avage10'][-200:-1], label='avage10')
#  
# plt.subplot(3,1,3)  # 将画板分为2行1两列，本幅图位于第一个位置
# plt.plot(data['inout'][-200:-1], label='inout')
# plt.plot(data['inout5'][-200:-1], label='inout5') 
# # plt.plot(data['avage20'][-200:-1], label='avage20') 
#  
# plt.title("avage")
#   
# # plt.xlabel('amount5')
# # 
# # plt.ylabel('amount10')
#   
# plt.legend()
#   
# plt.show();


# x_train = np.random.random((100, 20))
# y_train = np.random.randint(2, size=(100, 1))
# print(x_train)
# print(y_train)
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(2, size=(100, 1))
# 
# a = np.random.randint(10, size=(20, 1))
# print(a)
# y_test = utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=210)
# print(y_test)
# #a = np.array([1,2,3,4,5,6,7,8])  #一维数组
#print(a[:-1])
#shape是查看数据有多少行多少列
#print(a.shape[0])  #值为8，因为有8个数据
#print(a.shape[1])  #IndexError: tuple index out of range

# a = np.array([[1,2,3,4],[5,6,7,8]])  #二维数组
# print(a[:, :-1])
#print(a.shape[0])  #值为2，最外层矩阵有2个元素，2个元素还是矩阵。
#print(a.shape[1])  #值为4，内层矩阵有4个元素。
#print(a.shape[2])  #IndexError: tuple index out of range

#reshape()是数组array中的方法，作用是将数据重新组织