#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
#Created on 2020年2月20日
#@author: haochunyang@sina.com
#功能，用过去10天的数据预测将来五天的收益大于10%
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
saveH5FilePath = savePath+'lstm2prediction2.h5'

pd.set_option('display.max_columns',100)
#常量
class Config:
    source_fields = ['date','open','high','low','close','volume','amount']  # features
    fields = ['open','high','low','close','volume','amount']
    split_date = '2015/08/24'
    feature_back_days = 10 #每个input的长度 使用过去10天的数据
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
        #print(df.head())
#         data.rename(index='id')
#         print(data.head(50))     
        #计算未来5日收益率（未来第五日的收盘价/明日的开盘价）
#        
#         data.reset_index(drop=False, inplace=True)
#       
#         data['index'] = data['index'] / conf.feature_days
#         data['index'] = data['index'].astype(int)
#         data['index'] = data['index'].astype(int)
#         data['return'] =  data.groupby(by='index').agg({'close':['max']})
#         data['return'] = data['return'] / data['open'].shift(-1) - 1 
#          
# #         print(data.head(50)) 
#         data=data[data.amount>0]
# #         data['return'] = data['return'].apply(lambda np.where(x>=0.1,1,-1)) 
#         data['return'] = data['return'].apply(lambda x:np.where(x>=0.1,1,-1))
#          #去极值
        #data['return'] = data['return'].apply(lambda x:np.where(x>=0.2,0.2,np.where(x>-0.2,x,-0.2))) 
        #去极值
#         data['return'] = data['return'].clip(-0.2, 0.2)        
#         data['return'] = data['return']*10  # 适当增大return范围，利于LSTM模型训练
        
        # 辅助输入
#         data['label'] = np.round(data['close'] / 500)
#         data['label'] = data['close'] / 500
        
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
        
#         data.reset_index(drop=False, inplace=True)
#       
#         data['index'] = data['index'] / conf.feature_days
#         data['index'] = data['index'].astype(int)
# #         data['index'] = data['index'].astype(int)
#         data['max'] =  data.groupby(by='index').agg({'close':['max']})
#         data['up'] = data['max'] / data['open'].shift(-1) - 1 
         
#         print(data.head(50)) 
        data=data[data.amount>0]
        data['return'] = data['close']
#         data['return'] = data['up'].apply(lambda x:np.where(x>=0.1,1,0))
#         print(data.head(50))    
         #去极值        
        #print(scaledata)
        # 数据处理：设定每个input（30time series×6features）以及数据标准化
        train_input = []
#         train_input_aux = []
        train_output = []
        test_input = []
#         test_input_aux = []
        test_output = []
       
        for i in range(0, len(traindata)-conf.feature_back_days-conf.feature_days):
            
            a = scale(scaledata[i:i+conf.feature_back_days])
            train_input.append(a)
            maxClose = 0.0
            for j in range(i+conf.feature_back_days,i+conf.feature_back_days+conf.feature_days):
                if data['close'][j]>maxClose:
                    maxClose = data['close'][j]
            c = 0        
            if maxClose/data['open'][i+conf.feature_back_days] - 1 > 0.1:
                c = 1
                print(data['date'][i])
                print(data['open'][i+conf.feature_back_days])
                print(maxClose)
            train_output.append(c)
            
#         train_output = utils.to_categorical(train_output,num_classes=2)    
#             ac = data['label'][i]
#             train_input_aux.append(ac)
        for i in range(len(traindata), len(data)-conf.feature_back_days-conf.feature_days):
            b = scale(scaledata[i:i+conf.feature_back_days])
            test_input.append(b)

            maxClose = 0.0
            for j in range(i+conf.feature_back_days,i+conf.feature_back_days+conf.feature_days):
                if data['close'][j]>maxClose:
                    maxClose = data['close'][j]
            
            c = 0        
            if maxClose/data['open'][i+conf.feature_back_days] - 1 > 0.1:
                c = 1
#                 print(data['date'][i])
#                 print(data['open'][i+conf.feature_back_days])
#                 print(maxClose)                
            test_output.append(c)
            
#         print("len(test_output):")     
#         print(len(test_output))    
#         if len(test_output) <1:
#             return None
        
#         test_output = utils.to_categorical(test_output,num_classes=2)        
#             ac = data['label'][i]
#             test_input_aux.append(ac)            
#         print(train_input_aux)    
#         print(train_output[0])
        # LSTM接受数组类型的输入
        modelData = ModelInputData()
        modelData.train_x = np.array(train_input)
#         modelData.train_x_aux = np.array(train_input_aux)
        modelData.train_y = np.array(train_output)
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
#         if index<0:
#             index = len(data)-1
        #print(scaledata)
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
            
#         ac = data['label'][index]
#         train_input_aux.append(ac)
#         print(train_input_aux)    
#         print(train_output[0])
        # LSTM接受数组类型的输入
        modelData = ModelInputData()
        modelData.train_x = np.array(train_input)
        modelData.train_y = []
#         modelData.train_x_aux = np.array(train_input_aux)
        maxClose = 0.0
        for j in range(conf.feature_back_days,conf.feature_back_days+conf.feature_days):
            if traindata['close'][j]>maxClose:
                maxClose = traindata['close'][j]
        
        c = 0        
        if maxClose/traindata['open'][conf.feature_back_days] - 1 > 0.1:
            c = 1
#         print(traindata['date'][0])
#         print(traindata['open'][conf.feature_back_days])
#         print(maxClose) 
        modelData.train_y.append(c) 
        modelData.train_y.append(traindata['date'][conf.feature_back_days]) 
        modelData.train_y.append(traindata['open'][conf.feature_back_days]) 
        modelData.train_y.append(maxClose) 
        modelData.train_y.append(maxClose/traindata['open'][conf.feature_back_days] - 1) 
        return modelData
        
    def train(self,modelData,lstm_model):   
        #print(train_x)
        #print(train_x.shape[1])
        #print(train_y.shape[1])
        #modelBuilder = ModelBuilder()
        #lstm_model = modelBuilder.createModel()
        #model.train(train_x, train_y, batch_size=batch_size, nb_epoch=10, verbose=2)
#         verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
        lstm_model.train(modelData.train_x,modelData.train_y, batch_size=conf.batch_size, epochs=10, verbose=2)    
            
   
    def createModel(self):
        
        model = Sequential()
        # 输入 这是一个 Multiple Input 是指：input 为多个序列，output 为一个序列的问题
        #n_steps 为输入的 X 每次考虑几个时间步 n_features 此例中 = 2，因为输入有两个并行序列
        #model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(LSTM(32, return_sequences=True, input_shape=(conf.feature_back_days,6)))  # returns a sequence of vectors of dimension 32
        #连接下面的 所以 return_sequences=True
        model.add(Bidirectional(LSTM(16)))

#         model.add(Dense(8))
        model.add(Dropout(0.5))
        
#         model.add(Activation('sigmoid'))#sigmoid 激活函数0~1 tanh:-1~1
        model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(1))
        # 二分类问题
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

#         model.add(LSTM(32, return_sequences=True,
#                        input_shape=(conf.feature_back_days,6)))  # returns a sequence of vectors of dimension 32
#         model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
#         model.add(LSTM(32))  # return a single vector of dimension 32
#       
#         model.add(Dense(2, activation='softmax'))
#         
# #         model.add(Activation('sigmoid'))
# 
#         model.compile(loss='categorical_crossentropy',
#                       optimizer='rmsprop',
#                       metrics=['accuracy'])
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
            #fileName = r"C:\Users\haoch\Documents\SZ#159915.csv"
            #print(fileName)
            #modelData = modelTraining.getTrainingData(fileName) 
            #modelTraining.train(modelData,lstm_model)#1688/1688 - 1s - loss: 0.2197 - mse: 0.2197
        
            self.train(modelData,lstm_model)# 4491/4491 - 2s - loss: 0.2584 - mse: 0.2584
                                          # 4491/4491 - 3s - loss: 0.2587 - mse: 0.2587  
            # 保存模型
#             print(saveH5FilePath)
            lstm_model.save(saveH5FilePath)
            
#             lstm_model.summary() 
            if len(modelData.test_x)>0 and len(modelData.test_y)>0 :
                # step9 测试模型
                score,acc = lstm_model.evaluate(modelData.test_x,modelData.test_y,batch_size=conf.batch_size)
                print("Restored model, accuracy:{:5.2f}%".format(100 * acc))
                print('Test score:', score)
                print('准确率 Test accuracy:', acc)            

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
        
#         if(i<73):continue
#         if(i!=73):
#         if "SZ#" in fileName:
        modelMnger.fit2evaluate(lstm_model,fileName)
        
        modelData = modelMnger.getPredictData(fileName,'2015/10/08')#'2020/01/06' '2015/10/30'
        # 训练完成后使用
        if not modelData is None:
            predictions = lstm_model.predict(modelData.train_x, batch_size=None, verbose=0, steps=None)  
            print("#######predictions##########")
            print(predictions[0][0]) 
            print("#######predictions##########")  
#         predicted = decode_predictions(predictions, top=3)[0]
#         print('Predicted:', predicted )
            if predictions[0][0] > 0:
                print("买入")
                print("########################################")
                print(modelData.train_x)
                print("########################################")
                print(modelData.train_y)
        if i >0:
            break    
           
# fileName = r"C:\Users\haoch\Documents\SH#600000.csv" #4491/4491 - 2s - loss: 0.2582 - mse: 0.2582
# modelMnger.fit2evaluate(lstm_model,fileName)
# dataFilter = DataFilter()
# data = dataFilter.readFileData(fileName)  
# #2020/02/20,11.17,11.27,11.06,11.23,36656416,408938112.00
# fileName = r"C:\Users\haoch\Documents\SH#600000.csv"

# dfScore = predictions.flatten()
# print(dfScore)
# 
#     predictions = lstm_model.predict(
#         [np.array(df['X'].values.tolist()), np.array(df['X_aux'].values)])
#     df['score'] = predictions.flatten()
# 
#     # 预测值和真实值的分布
#     T.plot(
#         df,
#         x='Y', y=['score'], chart_type='scatter',
#         title='LSTM预测结果：实际值 vs. 预测值'
#     )
