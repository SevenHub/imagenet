#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
#Created on 2020年2月20日
#@author: haochunyang@sina.com
#功能，识别相似的波形
#参考https://www.ctolib.com/topics-137982.html
# 　　1、使用L2正则化，dropout技术，扩展数据集等，有效缓解过拟合，提升了性能；
# 　　2、使用ReLU，导数为常量，可以缓解梯度下降问题，并加速训练；
# 　　3、增加Conv/Pooling与Fc层，可以改善性能。（我自己实测也是如此）

import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import random

from pathlib import Path
# from tensorflow.keras import Sequential,models,utils
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input,LSTM,Dense,Dropout,Conv1D, MaxPooling1D,Activation,Bidirectional,Embedding,GlobalAveragePooling1D
# from tensorflow.keras.backend import concatenate
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
from sklearn.preprocessing import scale
from sympy.physics import pring
from sqlalchemy.sql.expression import false
from sklearn.model_selection import train_test_split 
from sklearn.svm import LinearSVC 
from sklearn.externals import joblib


savePath = "C:\\work\\tensorflowdata\\dense\\"
saveH5FilePath = savePath+'sklearnsvc.m'

pd.set_option('display.max_columns',100)

#常量
class Config:
    source_fields = ['date','open','high','low','close','volume','amount']  # features
    fields = ['open','high','low','close','volume','amount']
#     split_date = '2018/12/03'
    feature_back_days = 20 #每个input的长度 使用过去510天的数据
    batch_size = 100 #整数，指定进行梯度下降时每个batch包含的样本数,训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步
    feature_days = 5
    
conf = Config() 
   
#读数据
class DataFilter:
    
    #r"C:\Users\haoch\Documents\SZ#159915.csv"
    def readFileData(self,fileName):
        #1.数据采集
        #并将数据加载到Pandas 的dataframe中。
        data = pd.read_csv(fileName,encoding="gb2312",names=conf.source_fields)
       
        data.dropna(inplace=True)            
        data.reset_index(drop=True, inplace=True)
#         print(data.head(50))        
        return data
    
class ModelInputData:
    train_x = []
#     train_x_aux = []
    train_y = []
    test_x = []
#     test_x_aux = []
    test_y = []
    datatime = []
    
class ModelMnger:


    allCount = 0
    errorCount = 0
    
    def getTrainingData(self,fileName,train_input,train_output):
        
        dataFilter = DataFilter()
        data = dataFilter.readFileData(fileName)  
        #记录predictions的时间，回测要用
        datatime = data['date']#[data.date>=conf.split_date]  
                
        scaledata = data[conf.fields]
        traindata = data#data[data.date<conf.split_date]
        
        data=data[data.amount>0]
        data.reset_index(drop=True, inplace=True)
#         data['return'] = data['close']

#         train_input = []
# #         train_input_aux = []
#         train_output = []

       
        find = 0
        for i in range(0, len(traindata)-conf.feature_back_days-conf.feature_days):
            
            maxClose = 0.0
            for j in range(i+conf.feature_back_days,i+conf.feature_back_days+conf.feature_days):
                if scaledata['close'][j]>maxClose:
                    maxClose = scaledata['close'][j]
            c = 0        
            if maxClose/scaledata['open'][i+conf.feature_back_days] - 1 > 0.2:
                c = 1
                find = find + 1
                train_output.append(c)
                
            if i+conf.feature_back_days >= len(scaledata):
                break
            a = scale(scaledata[i:i+conf.feature_back_days])
            train_input.append(a)                
            #如果len(traindata)-conf.feature_back_days-conf.feature_days) > 5*find 随机删除一个
                    
        if find ==0 : return None
#         print(len(traindata)-conf.feature_back_days-conf.feature_days,find)
        
#         for i in range(len(traindata), len(data)-conf.feature_back_days-conf.feature_days):
#             if i+conf.feature_back_days >= len(scaledata):
#                 break            
#             b = scale(scaledata[i:i+conf.feature_back_days])
#             test_input.append(b)
#             
#             maxClose = 0.0
#             for j in range(i+conf.feature_back_days,i+conf.feature_back_days+conf.feature_days):
#                 if scaledata['close'][j]>maxClose:
#                     maxClose = scaledata['close'][j]
#             c = 0        
#             if maxClose/scaledata['open'][i+conf.feature_back_days] - 1 > 0.2:
#                 c = 1
# #                 print(scaledata['date'][i+conf.feature_back_days])            
#             test_output.append(c)
            
        # LSTM接受数组类型的输入
        return train_input，train_output
        modelData = ModelInputData()
        modelData.datatime = datatime
        
        modelData.train_x, modelData.test_x, modelData.train_y, modelData.test_y = train_test_split(train_input, train_output, test_size=0.30, random_state=42)
        
        return modelData
    
#     def getPredictData(self,fileName,date):
#         
#         data = pd.read_csv(fileName,encoding="gb2312",names=conf.source_fields)
#         #记录predictions的时间，回测要用
# #         datatime = data['date'][data.date>=conf.split_date]  
# #         data['label'] = data['close'] / 500
# 
#         traindata = data[data.date>=date]
#         if len(traindata) <= 0:
#             return
#         
#         traindata = traindata.dropna()            
#         traindata.reset_index(drop=True, inplace=True)
#                         
#         scaledata = traindata[conf.fields]
# 
#         # 数据处理：设定每个input（30time series×6features）以及数据标准化
#         train_input = []
# #         train_input_aux = []
# #         print("#################")
# #         print(scaledata.head(conf.feature_back_days+5))
#         
#         
#         if conf.feature_back_days+conf.feature_days >= len(traindata):
#             return None
#         
#         a = scale(scaledata[0:conf.feature_back_days])
# #         print("scaledata[i+1-conf.feature_back_days:i+1]")
# #         print(scaledata[0:conf.feature_back_days]) 
# #         print("scale(scaledata[0:conf.feature_back_days])")
# #         print(a)   
#         train_input.append(a)
#             
# 
#         # LSTM接受数组类型的输入
#         modelData = ModelInputData()
#         modelData.train_x = np.array(train_input)
#         modelData.train_y = []
# #         modelData.train_x_aux = np.array(train_input_aux)
#         maxClose = 0.0
#         for j in range(conf.feature_back_days,conf.feature_back_days+conf.feature_days):
#             if traindata['close'][j]>maxClose:
#                 maxClose = traindata['close'][j]
# 
# #         open = traindata['open'][conf.feature_back_days]
# #         up = maxClose/open - 1
# #         print(traindata['date'][conf.feature_back_days],open,maxClose,up)    
#                 
#         c = 0        
#         if maxClose/traindata['open'][conf.feature_back_days] - 1 > 0.2:
#             c = 1
#         modelData.train_y.append(c) 
#         modelData.train_y.append(traindata['date'][conf.feature_back_days+1]) 
#         modelData.train_y.append(traindata['date'][conf.feature_back_days+conf.feature_days]) 
#         modelData.train_y.append(traindata['close'][conf.feature_back_days+1]) 
#         modelData.train_y.append(traindata['close'][conf.feature_back_days+conf.feature_days]) 
#         
#         return modelData
        
    def train(self,modelData,lstm_model):   
        if len(modelData.train_x) > 0:
            joblib.dump(lstm_model, saveH5FilePath)    

        
    def createModel(self):
        
        model = LinearSVC(random_state=0, tol=1e-5) 
        return model
    
    def getModel(self):
        
        h5File = Path(saveH5FilePath)
        
        if h5File.exists():
            # Recreate the exact same model, including weights and optimizer.
            lstm_model = joblib.load(saveH5FilePath)
        else:
            lstm_model = None     
               
        return lstm_model
    
    def fit2evaluate(self,lstm_model,fileName):
        
            modelData = self.getTrainingData(fileName)       
            if modelData is None:
                print("None")
                return

#             if len(modelData.test_x)>0 and len(modelData.test_y)>0 :
#                 predicted = lstm_model.predict(modelData.test_x) 
#                 accuracy_score(y_test, predicted) 
                               
            self.train(modelData,lstm_model)
            if len(modelData.test_x)>0 and len(modelData.test_y)>0 :
                predicted = lstm_model.predict(modelData.test_x) 
#                 accuracy_score(y_test, predicted)     
            
    def getMindate(self):
        path="C:\\Users\\haoch\\Documents\\stockdata"
        parents = os.listdir(path)
        minDate = datetime.datetime.strptime ("2020/02/22", '%Y/%m/%d')
        for parent in parents:
            child = os.path.join(path,parent)
            if not os.path.isdir(child):
                fileName = child
                data = pd.read_csv(fileName,encoding="gb2312",nrows=1,names=conf.source_fields)
                date = datetime.datetime.strptime (data['date'][0], '%Y/%m/%d')
                if minDate>date:
                    minDate = date
        print(minDate)#1990-12-19 00:00:00  
              
    def buySell(self,lstm_model,fileName,starDate,leng):
        #时间循环
        dateStart=datetime.datetime.strptime(starDate,'%Y/%m/%d')
#         dateEnd=datetime.datetime.now()
 
        for i in range(leng):
            day = dateStart + datetime.timedelta(days=i)
            week_day = day.strftime("%w")
            if int(week_day) > 0 and int(week_day) < 6:
                modelData = self.getPredictData(fileName,day.strftime('%Y/%m/%d'))
                if not modelData is None:
#                     predictions = lstm_model.predict(modelData.train_x, batch_size=None, verbose=0, steps=None)
#                     print(predictions[0][0])
#                     print(modelData.train_y[0])
                    self.allCount = self.allCount + 1
                    if int(predictions[0][0]) != modelData.train_y[0]:
                        self.errorCount = self.errorCount + 1
                        print("predictions[0][0]:",int(predictions[0][0]))
                        print("modelData.train_y:",modelData.train_y)

                        
    def fileTrain(self):  
                  
        lstm_model = self.getModel()
        # lstm_model.summary()
        if(lstm_model is None):
            lstm_model = self.createModel()  
            
        path="C:\\Users\\haoch\\Documents\\stockdata"
        parents = os.listdir(path)
        i=0
        for parent in parents:
            child = os.path.join(path,parent)
            if not os.path.isdir(child):
                fileName = child
        #         fileName = r"C:\Users\haoch\Documents\SH#600000.csv" #4491/4491 - 2s - loss: 0.2582 - mse: 0.2582
                print(fileName)
                i=i+1
                print("file count",len(parents),i)
                
#                 if(i<500):continue
        #         if(i!=73):
        #         if "SZ#" in fileName:
                self.fit2evaluate(lstm_model,fileName)
                
#                 modelData = modelMnger.getPredictData(fileName,'2018/12/03')#'2020/01/06' '2015/10/30'
                # 训练完成后使用

                if i >10:
                    break    
    def test(self):       
        # modelMnger.getMindate()
        lstm_model = self.getModel()
        # lstm_model.summary()
        if(lstm_model is None):
            lstm_model = self.createModel()  
            
        path="C:\\Users\\haoch\\Documents\\stockdata"
        parents = os.listdir(path)
        i=0
        for parent in parents:
            child = os.path.join(path,parent)
            if not os.path.isdir(child):
                fileName = child
        #         fileName = r"C:\Users\haoch\Documents\SH#600000.csv" #4491/4491 - 2s - loss: 0.2582 - mse: 0.2582
                print(fileName)
                i=i+1
                print("file count",len(parents),i)
                if "SZ#002017" not in fileName: continue
        #         if(i<12):continue/////
                self.errorCount = 0
                self.allCount = 0
                self.buySell(lstm_model,fileName,"2018/12/03",365)
                print(self.allCount,self.errorCount)
        
#                 if i >10:
#                     break  
                  
modelMnger = ModelMnger() 
modelMnger.fileTrain()
# modelMnger.test()