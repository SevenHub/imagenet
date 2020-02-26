#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
#Created on 2020年2月20日
#@author: haochunyang@sina.com
#功能，用过去5天的数据预测明天涨跌
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

from pathlib import Path
from tensorflow.keras import Sequential,models,utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense,Dropout,Conv1D, MaxPooling1D,Activation,Bidirectional
from tensorflow.keras.backend import concatenate
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from sklearn.preprocessing import scale
from sympy.physics import pring

savePath = "C:\\work\\tensorflowdata\\lstm\\"
saveH5FilePath = savePath+'lstm5to1updown.h5'

pd.set_option('display.max_columns',100)
#常量
class Config:
    source_fields = ['date','open','high','low','close','volume','amount']  # features
    fields = ['open','high','low','close','volume','amount']
    split_date = '2019/01/02'
    feature_back_days = 5 #每个input的长度 使用过去510天的数据
    batch_size = 100 #整数，指定进行梯度下降时每个batch包含的样本数,训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步
    feature_days = 1
    
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
    
    def getTrainingData(self,fileName):
        dataFilter = DataFilter()
        data = dataFilter.readFileData(fileName)  
        #记录predictions的时间，回测要用
        datatime = data['date'][data.date>=conf.split_date]  
                
        scaledata = data[conf.fields]
        traindata = data[data.date<conf.split_date]
        
        data=data[data.amount>0]
        data.reset_index(drop=True, inplace=True)
        data['return'] = data['close']

        train_input = []
#         train_input_aux = []
        train_output = []
        test_input = []
#         test_input_aux = []
        test_output = []
       
        for i in range(0, len(traindata)-conf.feature_back_days-conf.feature_days):
            
            a = scale(scaledata[i:i+conf.feature_back_days])
            train_input.append(a)
            c = 0    
            if scaledata['close'][i+conf.feature_back_days+conf.feature_days]>scaledata['open'][i+conf.feature_back_days+conf.feature_days]:
                c = 1
            train_output.append(c)

        for i in range(len(traindata), len(data)-conf.feature_back_days-conf.feature_days):
            b = scale(scaledata[i:i+conf.feature_back_days])
            test_input.append(b)
            
            c = 0    
            if scaledata['close'][i+conf.feature_back_days+conf.feature_days]>scaledata['open'][i+conf.feature_back_days+conf.feature_days]:
                c = 1             
            test_output.append(c)
            
        # LSTM接受数组类型的输入
        modelData = ModelInputData()
        modelData.train_x = np.array(train_input)
#         modelData.train_x_aux = np.array(train_input_aux)
        modelData.train_y = np.array(train_output)
#         print(modelData.train_y)
        modelData.test_x = np.array(test_input) 
#         modelData.test_x_aux = np.array(test_input_aux)
        modelData.test_y = np.array(test_output)
        modelData.datatime = datatime
        return modelData
    
    def getPredictData(self,fileName,date):
        
        data = pd.read_csv(fileName,encoding="gb2312",names=conf.source_fields)
        #记录predictions的时间，回测要用
#         datatime = data['date'][data.date>=conf.split_date]  
#         data['label'] = data['close'] / 500

        traindata = data[data.date>=date]
        if len(traindata) <= 0:
            return
        traindata.dropna(inplace=True)            
        traindata.reset_index(drop=True, inplace=True)
                        
        scaledata = traindata[conf.fields]

        # 数据处理：设定每个input（30time series×6features）以及数据标准化
        train_input = []
#         train_input_aux = []
#         print("#################")
#         print(scaledata.head(conf.feature_back_days+5))
        
        
        if conf.feature_back_days+conf.feature_days >= len(traindata):
            return None
        
        a = scale(scaledata[0:conf.feature_back_days])
#         print("scaledata[i+1-conf.feature_back_days:i+1]")
#         print(scaledata[0:conf.feature_back_days]) 
#         print("scale(scaledata[0:conf.feature_back_days])")
#         print(a)   
        train_input.append(a)
            

        # LSTM接受数组类型的输入
        modelData = ModelInputData()
        modelData.train_x = np.array(train_input)
        modelData.train_y = []
#         modelData.train_x_aux = np.array(train_input_aux)
        c = 0    
        if traindata['close'][conf.feature_back_days+conf.feature_days]>traindata['open'][conf.feature_back_days+conf.feature_days]:
            c = 1 
        modelData.train_y.append(c) 
        modelData.train_y.append(traindata['date'][conf.feature_back_days]) 
        modelData.train_y.append(traindata['open'][conf.feature_back_days+conf.feature_days]) 
        modelData.train_y.append(traindata['close'][conf.feature_back_days+conf.feature_days]) 
        return modelData
        
    def train(self,modelData,lstm_model):   

#         verbose = 0 #为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
        if len(modelData.train_x) > 0:
            lstm_model.fit(modelData.train_x,modelData.train_y, batch_size=conf.batch_size, epochs=10, verbose=2)    

#             lstm_model.train_on_batch(modelData.train_x,modelData.train_y)
#         保存模型
#             print(saveH5FilePath)
        lstm_model.save(saveH5FilePath)   
        
    def createModel(self):
        
        model = Sequential()
        # 输入 这是一个 Multiple Input 是指：input 为多个序列，output 为一个序列的问题
        #n_steps 为输入的 X 每次考虑几个时间步 n_features 此例中 = 2，因为输入有两个并行序列
        #model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
#         return_sequences=True,

#         model.add(LSTM(128,return_sequences=True, input_shape=(conf.feature_back_days,6)))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(16,input_shape=(conf.feature_back_days,6)))       
        #连接下面的 所以 return_sequences=True
#         model.add(Bidirectional(LSTM(64)))
        
        model.add(Dropout(0.2))
        
#         model.add(Dense(64, activation='relu'))
        
#         model.add(Dropout(0.2))
        
        model.add(Dense(16))
        
        model.add(Dense(1, activation='sigmoid'))

#         model.add(Dropout(0.5))
        
#         model.add(Activation('sigmoid'))#sigmoid 激活函数0~1 tanh:-1~1
#         model.add(Dense(2, activation='sigmoid'))
#         model.add(Dense(1))
        # 二分类问题
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

#         print("一个好的loss的定义应该使得模型在训练集上随着loss的下降accuracy渐渐提高。但也有可能只是刚开始训练的时候是这样，到达一定阶段后有可能loss下降但accuracy并不会提高")      
        return model;
    
    def getModel(self):
        
        h5File = Path(saveH5FilePath)
        
        if h5File.exists():
            # Recreate the exact same model, including weights and optimizer.
            lstm_model = models.load_model(saveH5FilePath)
        else:
            lstm_model = None     
               
        return lstm_model
    
    def fit2evaluate(self,lstm_model,fileName):
            modelData = self.getTrainingData(fileName)       
            if modelData is None:
                return
#             lstm_model.summary() 
            acc1 = 0    
            if len(modelData.test_x)>0 and len(modelData.test_y)>0 :
                # step9 测试模型
#                 score,acc = lstm_model.evaluate(modelData.test_x,modelData.test_y,batch_size=conf.batch_size)
                score,acc1 = lstm_model.test_on_batch(modelData.test_x,modelData.test_y)
#                 print("Restored model, accuracy:{:5.2f}%".format(100 * acc))
#                 print('Test score:', score)
#                 print('准确率 Test accuracy:', acc)  
                          
            self.train(modelData,lstm_model)
            
            if len(modelData.test_x)>0 and len(modelData.test_y)>0 :
                # step9 测试模型
#                 score,acc = lstm_model.evaluate(modelData.test_x,modelData.test_y,batch_size=conf.batch_size)
                
                score,acc = lstm_model.test_on_batch(modelData.test_x,modelData.test_y)
                if acc !=acc1:
                    print("Restored model, accuracy1:{:5.2f}%".format(100 * acc1))
                    print("Restored model, accuracy2:{:5.2f}%".format(100 * acc))
                    print('Test score:', score)
   
            
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
    #区间，开盘买入，最高价大于10%时，收盘价卖出
    def buySell(self,starDate):
        #时间循环
        dateStart=datetime.datetime.strptime(starDate,'%Y/%m/%d')
        dateEnd=datetime.datetime.now()
 
        for i in range((dateEnd - dateStart).days + 1):
            day = dateStart + datetime.timedelta(days=i)
            week_day = day.strftime("%w")
            if week_day > 0 and week_day < 6:
                modelData = self.getPredictData(fileName,day.strftime('%Y/%m/%d'))
                if not modelData is None:
                    predictions = lstm_model.predict(modelData.train_x, batch_size=None, verbose=0, steps=None)
                    print(predictions[0][0])
                    print(modelData.train_y[0])
                    if predictions[0][0] != modelData.train_y[0]:
                        print(predictions[0][0])
                        print(modelData.train_y)
                
modelMnger = ModelMnger()
# modelMnger.getMindate()
lstm_model = modelMnger.getModel()
# lstm_model.summary()
if(lstm_model is None):
    lstm_model = modelMnger.createModel()  
    
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
        
#         if(i<252):continue
#         if(i!=73):
#         if "SZ#" in fileName:
        modelMnger.fit2evaluate(lstm_model,fileName)
        
#         modelData = modelMnger.getPredictData(fileName,'2015/10/08')#'2020/01/06' '2015/10/30'
#         # 训练完成后使用
#         if not modelData is None:
#             predictions = lstm_model.predict(modelData.train_x, batch_size=None, verbose=0, steps=None)  
#             print("#######predictions##########")
#             print(predictions[0][0]) 
#             print("#######predictions##########")  
# #         predicted = decode_predictions(predictions, top=3)[0]
# #         print('Predicted:', predicted )
#             if predictions[0][0] > 0:
#                 print("买入")
#                 print("########################################")
#                 print(modelData.train_x)
#                 print("########################################")
#                 print(modelData.train_y)
        if i >5:
            break    
           

