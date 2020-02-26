#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
#Created on 2020年2月20日
#@author: haochunyang@sina.com
#功能，用过去30天的数据预测将来五天的收益
#参考https://www.ctolib.com/topics-137982.html

import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import scale

#常量
class Config:
    source_fields = ['date','open','high','low','close','volume','amount']  # features
    fields = ['open','high','low','close','volume','amount']
    split_date = '2019/01/02'
    seq_len = 30 #每个input的长度
    batch = 100 #整数，指定进行梯度下降时每个batch包含的样本数,训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步
    
#读数据
class DataFilter:
    conf = Config()
    #r"C:\Users\haoch\Documents\SZ#159915.csv"
    def readFile(self,fileName):
        #1.数据采集
 readFileData并将数据加载到Pandas 的dataframe中。
        data = pd.read_csv(fileName,encoding="gb2312",names=self.conf.source_fields)
        #print(df.head())
        
        #计算未来5日收益率（未来第五日的收盘价/明日的开盘价）
        data['return'] = data['close'].shift(-5) / data['open'].shift(-1) - 1 
        
        data=data[data.amount>0]
        
        #滤除缺失数据  inplace=True：不创建新的对象，直接对原始对象进行修改；
        data.dropna(inplace=True)
         #去极值
        data['return'] = data['return'].apply(lambda x:np.where(x>=0.2,0.2,np.where(x>-0.2,x,-0.2))) 
        data['return'] = data['return']*10  # 适当增大return范围，利于LSTM模型训练
        data.reset_index(drop=True, inplace=True)
        
        return data
    
class ModelData:
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    datatime = []
    
class ModelTraining:
    conf = Config()
    
    def getTrainingData(self,fileName):
        dataFilter = DataFilter()
        data = dataFilter.readFile(fileName)  
        #记录predictions的时间，回测要用
        datatime = data['date'][readFileDatae>=self.conf.split_date]  
                
        scaledata = data[self.conf.fields]
        traindata = data[data.date<self.conf.split_date]
        #print(scaledata)
        # 数据处理：设定每个input（30time series×6features）以及数据标准化
        train_input = []
        train_output = []
        test_input = []
        test_output = []
        for i in range(self.conf.seq_len-1, len(traindata)):
            a = scale(scaledata[i+1-self.conf.seq_len:i+1])
            train_input.append(a)
            c = data['return'][i]
            train_output.append(c)
        for j in range(len(traindata), len(data)):
            b = scale(scaledata[j+1-self.conf.seq_len:j+1])
            test_input.append(b)
            c = data['return'][j]
            test_output.append(c)
            
        print(train_input[0])    
        print(train_output[0])
        # LSTM接受数组类型的输入
        modelData = ModelData()
        modelData.train_x = np.array(train_input)
        modelData.train_y = np.array(train_output)
        modelData.test_x = np.array(test_input) 
        modelData.test_y = np.array(test_output)
        modelData.datatime = datatime
        return modelData
    
    def training(self,modelData,lstm_model):   
        #print(train_x)
        #print(train_x.shape[1])
        #print(train_y.shape[1])
        #modelBuilder = ModelBuilder()
        #lstm_model = modelBuilder.createModel()
        #model.fit(train_x, train_y, batch_size=batch, nb_epoch=10, verbose=2)
        lstm_model.fit(modelData.train_x, modelData.train_y, batch_size=self.conf.batch, nb_epoch=10, verbose=2)        

class ModelBuilder:
    
    def createModel(self):
        # 构建神经网络层 1层LSTM层+3层Dense层
        # 用于1个输入情况
        lstm_model = Sequential()
        
        lstm_input = Input(shape=(30,6), name='lstm_input')
        lstm_model.add(lstm_input)
        
        #lstm_output = LSTM(128, return_sequences=True,activation=atan, dropout=0.2, recurrent_dropout=0.1)#(lstm_input)
        #lstm_model.add(lstm_output)
        #lstm_output = LSTM(128,input_shape=(30,6), activation=atan, dropout=0.2, recurrent_dropout=0.1)
        lstm_output = LSTM(128,activation='tanh', dropout=0.2, recurrent_dropout=0.1)
        lstm_model.add(lstm_output)
        
        Dense_output_1 = Dense(64, activation='linear')#(lstm_output)
        lstm_model.add(Dense_output_1)
        
        Dense_output_2 = Dense(16, activation='linear')#(Dense_output_1)
        lstm_model.add(Dense_output_2)
        
        predictions = Dense(1, activation='tanh')#(Dense_output_2)
        lstm_model.add(predictions)

        #model = Model(input=lstm_input, output=predictions)
        
        #model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return lstm_model;
           
modelBuilder = ModelBuilder()
lstm_model = modelBuilder.createModel()

modelTraining = ModelTraining()
#fileName = r"C:\Users\haoch\Documents\SZ#159915.csv"
#print(fileName)
#modelData = modelTraining.getTrainingData(fileName) 
#modelTraining.training(modelData,lstm_model)#1688/1688 - 1s - loss: 0.2197 - mse: 0.2197

fileName = r"C:\Users\haoch\Documents\SH#600000.csv" #4491/4491 - 2s - loss: 0.2582 - mse: 0.2582
print(fileName)
modelData = modelTraining.getTrainingData(fileName) 
modelTraining.training(modelData,lstm_model)# 4491/4491 - 2s - loss: 0.2584 - mse: 0.2584

'''
# 预测
predictions = lstm_model.predict(modelData.test_x)


# 预测值和真实值的关系
data1 = modelData.test_y
data2 = predictions
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data2,data1, 'o', label="data")
ax.legend(loc='best')
#plt.show()

# 如果预测值>0,取为1；如果预测值<=0,取为-1.为回测做准备
for i in range(len(predictions)):
    if predictions[i]>0:
        predictions[i]=1
    elif predictions[i]<=0:
        predictions[i]=-1

print(predictions)
# 将预测值与时间整合作为回测数据
cc = np.reshape(predictions,(len(predictions), 1),order='C')
databacktest = pd.DataFrame()
databacktest['date'] = modelData.datatime
databacktest['direction']=np.round(cc)

    
# 在沪深300上回测
def initialize(context):
    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 传入预测数据和真实数据
    context.predictions=databacktest
    
    context.hold=conf.split_date
    
# 回测引擎：每日数据处理函数，每天执行一次
def handle_data(context, data):
    current_dt = data.current_dt.strftime('%Y-%m-%d') 
    sid = context.symbol(conf.instrument)
    cur_position = context.portfolio.positions[sid].amount    # 持仓
    if cur_position==0:
        if databacktest['direction'].values[databacktest.date==current_dt]==1:
            context.order_target_percent(sid, 0.9)
            context.date=current_dt
            
    else:
        if databacktest['direction'].values[databacktest.date==current_dt]==-1:
            if context.trading_calendar.session_distance(pd.Timestamp(context.date), pd.Timestamp(current_dt))>=5:
                context.order_target(sid, 0)
                
# 调用回测引擎
m8 = M.backtest.v5(
    instruments=conf.instrument,
    start_date=conf.split_date,
    end_date=conf.end_date,
    initialize=initialize,
    handle_data=handle_data,
    order_price_field_buy='open',       # 表示 开盘 时买入
    order_price_field_sell='close',     # 表示 收盘 前卖出
    capital_base=10000, 
    benchmark='000300.SHA', 
    m_cached=False
)    
'''     
#数据去噪，小波变换
'''
x = np.array(self.stock_data.iloc[i: i + 11, j])                
(ca, cd) = pywt.dwt(x, "haar")                
cat = pywt.threshold(ca, np.std(ca), mode="soft")                
cdt = pywt.threshold(cd, np.std(cd), mode="soft")                
tx = pywt.idwt(cat, cdt, "haar")
'''
'''
#将“日期”列转换为时间数据类型，并将“日期”列设置为索引。
df['date'] = pd.to_datetime(df['date'])

#默认的，当列变成行索引之后，原来的列就没了，但是可以通过设置drop来保留原来的列。
df = df.set_index(['date'], drop=True)

#按日期“2019–01–02”将数据拆分为训练集和测试集，即在此日期之前的数据是训练数据，此之后的数据是测试数据
split_date = pd.Timestamp(split_date_s)

#df =  df['close']

train = df.loc[:split_date]

test = df.loc[split_date:]

'''
#input_shape的三个维度samples, time_steps, features
#features: 是一个原始样本的特征维数 features = 6
#time_steps: 是输入时间序列的长度，即用多少个连续样本预测一个输出。如果你希望用连续m个序列（每个序列即是一个原始样本），那么就应该设为m。
# time_steps = 1 每天对应一个输出
#samples：经过格式化后的样本数。假设原始样本(3000*6), 你选择features=6, time_steps=m,则samples=3000/m
'''一个例子
原始样本集 (3000, 6):
[[1,1,1,1,1,1] * 3000]
处理后(3000, 1, 6)
[
[[1,1,1,1,1,1]] * 3000
]
'''
'''
例如希望根据前seq_len天的收盘价预测第二天的收盘价，那么可以将data转换为(len(data)-seq_len)(seq_len+1)的数组，由于LSTM神经网络接受的input为3维数组，因此最后可将input+output转化为(len(data)-seq_len)(seq_len+1)*1的数组
前5天的收盘价预测第二天的收盘价 data转换为(3000-5)(5+1)
'''
'''
# 划分训练集和测试集的输入和输出
train_X = train #train[:, :-1] #a[:-1]表示从第0位开始直到最后一位,,不要最后一位
test_X = test #test[:, :-1]

train_Y = train['close']
test_Y = test['close']
#转化为三维数据
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

print(train_X)


train_x，test_x 的 shape 是 (None, 30, 6)，其中 None 表示样本数量
train_aux，test_aux，train_y, test_y 都是长为样本数量的向量

# 构建神经网络层 1层LSTM层+3层Dense层
lstm_input = Input(shape=(30, 6), name='lstm_input')
lstm_output = LSTM(128, activation=activation, dropout_W=0.2, dropout_U=0.1)(lstm_input)
aux_input = Input(shape=(1,), name='aux_input')
merged_data = concatenate([lstm_output, aux_input],axis=-1)
dense_output_1 = Dense(64, activation='linear')(merged_data)
dense_output_2 = Dense(16, activation='linear')(dense_output_1)
predictions = Dense(1, activation=activation)(dense_output_2)
    
model = Sequential()

#input的时间跨度为30天，每天的features为['close','open','high','low','amount','volume']共6个，因此每个input为30×6的二维向量。
model.add(LSTM(32, input_shape=(1, 6)))


model = Sequential()#定义

# 1 本层输出维度
model.add(LSTM(1,input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dropout(0.5))#可根据自身情况添加dropout,dropout比率最好的设置为0.5，因为随机生成的网络结构最多

model.add(Dense(1))#

        input_data = kl.Input(shape=(1, self.input_shape))
        lstm = kl.LSTM(5, input_shape=(1, self.input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
                       recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
        perc = kl.Dense(5, activation="sigmoid", activity_regularizer=regularizers.l2(0.005))(lstm)
        lstm2 = kl.LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
                        dropout=0.2, recurrent_dropout=0.2)(perc)
        out = kl.Dense(1, activation="sigmoid", activity_regularizer=regularizers.l2(0.001))(lstm2)


# model.add(Activation('sigmoid'))#根据情况添加激活函数

model.compile(loss=rmse_koss, optimizer='adam')#模型优化，其中loss和optimzizer等式右边的损失函数和优化函数都可以根据自身需求进行调节和定义，当然你也可以用系统默认的损失函数和优化函数，本实现定义了RMSE为损失函数，所以你需要调用K文件库进行实现

history=model.fit(train_X, train_y, epochs=1,validation_split=0.1, batch_size=180, shuffle=True)#最后一句是对模型进行传值train_X, train_y，定义epochs迭代次数等；
'''