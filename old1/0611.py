# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:20:46 2022

@author: USER
"""


import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision #
import torchvision.datasets as dset
import torchvision.transforms as tr #데이터 불러오면서 전처리를 가능한게 해주는 라이브러리 
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from haversine import haversine
import src.preprocess as pp
import src.models1 as models
import src.seq2seq as seq2seq
from sklearn.metrics import mean_squared_error as mse

from torch.autograd import Variable 


import matplotlib.pyplot as plt
import seaborn as sns

import os 

path_dir_train = 'C:/Users/USER/Desktop/YT_data/train'
path_dir_test = 'C:/Users/USER/Desktop/YT_data/test'

file_list_train = os.listdir(path_dir_train)
file_list_test = os.listdir(path_dir_test)


#트레인 데이터 전처리 
YT_dict_train ={} 
for f_name in file_list_train: 
    df = pd.read_csv(path_dir_train+'/'+f_name)
    yt_name = f_name[:-4]
    
    #datetime 변환 및 인덱스 설정
    df['evnt_dt'] = pd.to_datetime(df['evnt_dt'])
    df.set_index('evnt_dt',inplace=True)
    
    """
    #wk_id가 nan 이면 i(idle)로 변환
    df['wk_id'].fillna('i',inplace=True)
    
    #y_blk가 nan이면 '-'으로 변환
    df['y_blk'].fillna('-',inplace=True)
    df['y_bay'].fillna('-',inplace=True)
    df['y_row'].fillna('-',inplace=True)
    df['y_tier'].fillna('-',inplace=True)
    
    ########작업 분리#############
    #shift칼럼 생성해서 wk_id / y_blk / y_row / y_tier 비교해서 하나라도 달라지면 작업이 바뀐 거라 생각
    df[['shift1','shift2','shift3','shift4']] = df[['wk_id', 'y_blk', 'y_row', 'y_tier']].shift(1)
    mask = (df['shift1'] != df['wk_id']) | (df['shift2'] != df['y_blk']) | (df['shift3'] != df['y_row'])| (df['shift4'] != df['y_tier']) 
    
    #shift 칼럼 삭제
    df.drop(['shift1', 'shift2', 'shift3', 'shift4'], axis=1 , inplace=True)
    
    #각각의 작업 분리하기 위해 check 칼럼 생성 / 시작 시점:1, 나머지: 0 라벨링 
    df['check'] = 0
    df.loc[mask,'check']= 1
    """
    
    #거리 및 방향 변화
    dis_list = [np.nan]
    dir_list = [np.nan]
    for r in range(1, len(df)):
        #거리계산 
        a = (df.iloc[r-1,1], df.iloc[r-1,2])
        b = (df.iloc[r,1], df.iloc[r,2])
        dis = haversine(a,b, unit='m')
        dis_list.append(dis)
        
        #방향 변화
        dir_list.append(df.iloc[r,5] - df.iloc[r-1,5])
        
        
    df['distance'] = dis_list
    df['dir_diff'] = dir_list
    
    #distance 결측인 행 제거 
    df = df.dropna(subset=['distance'], how = 'any', axis=0)
    
    #1초 이동거리 12 이상인 애들 제거 
    df = df[df.distance<=12]
    
    #불필요 칼럼 삭제 
    df.drop(['reg_seq', 'altitude', 'position_fix', 'satelites', 'dev_id', 'cre_dt', 'cntr_dup', 
             'wk_id','y_blk', 'y_bay', 'y_row', 'y_tier', 'long_cut', 'latt_cut'], axis=1 , inplace=True)
    
    #결측시 채우기 
    if len(df) != 0:
        ts = pd.date_range(start=df.index[0],
                           end = df.index[-1],
                           freq = 'S')
        
        ts = pd.DataFrame(index = ts)
    
        df = ts.join(df)
        df = df.interpolate()
        df['equ_no'] = yt_name
        
    #인덱스 제거
    df = df.reset_index()
    
    #딱 한 번 정지 상태 외에 제거 
    temp = df.shift(1)
    df = df[~((df.velocity==0) & (temp.velocity==0))]
    
    
            
    #날짜 인덱스 제거
    #df = df.drop('index', axis=1)
    
    
    YT_dict_train[yt_name] = df
    

############################################

#트레인 데이터 전처리 
YT_dict_test={} 
for f_name in file_list_test: 
    df = pd.read_csv(path_dir_test+'/'+f_name)
    yt_name = f_name[:-4]
    
    #datetime 변환 및 인덱스 설정
    df['evnt_dt'] = pd.to_datetime(df['evnt_dt'])
    df.set_index('evnt_dt',inplace=True)
    
    
    
    
    #거리 및 방향 변화
    dis_list = [np.nan]
    dir_list = [np.nan]
    for r in range(1, len(df)):
        #거리계산 
        a = (df.iloc[r-1,1], df.iloc[r-1,2])
        b = (df.iloc[r,1], df.iloc[r,2])
        dis = haversine(a,b, unit='m')
        dis_list.append(dis)
        
        #방향 변화
        dir_list.append(df.iloc[r,5] - df.iloc[r-1,5])
        
        
    df['distance'] = dis_list
    df['dir_diff'] = dir_list
    
    #distance 결측인 행 제거 
    df = df.dropna(subset=['distance'], how = 'any', axis=0)
    
    
    
    #불필요 칼럼 삭제 
    df.drop(['reg_seq', 'altitude', 'position_fix', 'satelites', 'dev_id', 'cre_dt', 'cntr_dup', 
             'wk_id','y_blk', 'y_bay', 'y_row', 'y_tier', 'long_cut', 'latt_cut'], axis=1 , inplace=True)
    
    
    #인덱스 제거
    df = df.reset_index()
    
    
    YT_dict_test[yt_name] = df
    

############################################


#전체 데이터 합치기 
df_train = pd.DataFrame()
df_test = pd.DataFrame()

for name in YT_dict_train :
    df_train = pd.concat([df_train, YT_dict_train[name]])
    
for name in YT_dict_test :
    df_test = pd.concat([df_test, YT_dict_test[name]])
    


#정규화 (이상치 나름 제거 했으니 minmax로)
from sklearn.preprocessing import MinMaxScaler

df_train = df_train.drop(columns=['equ_no','index'], axis=1)
df_test = df_test.drop(columns='equ_no', axis=1)


scaler = [] 
for i in range(len(df_train.columns)-1):
    scaler.append(MinMaxScaler())

for i in range(len(df_train.columns)-1):
    scaler[i].fit(df_train.iloc[:,i].values.reshape(-1,1))


#스케일러 피팅 
mm  = MinMaxScaler()
mm.fit(df_train)

#시계열 생성 

w_size = 3
p_size = 3

ts_dict_train={} 
for yt in YT_dict_train:
    temp = YT_dict_train[yt].drop(['equ_no','index'], axis=1)
    
    #스케일링 
    temp = pd.DataFrame(mm.transform(temp), columns=temp.columns, index=temp.index)
    
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_train[yt] = [X_s, y_s]

ts_dict_test={} 
for yt in YT_dict_test:
    temp = YT_dict_test[yt].drop(['equ_no','evnt_dt'], axis=1)
    
    #스케일링 
    temp = pd.DataFrame(mm.transform(temp), columns=temp.columns, index=temp.index)
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_test[yt] = [X_s, y_s]    
    


#시계열 데이터 합치기 
X_train=np.zeros((1,w_size,len(df_train.columns)))
y_train=np.zeros((1,p_size,2))

X_test=np.zeros((1,w_size,len(df_train.columns)))
y_test=np.zeros((1,p_size,2))

for ts in ts_dict_train: 
    X_train = np.concatenate((X_train,ts_dict_train[ts][0]), axis=0)
    y_train = np.concatenate((y_train,ts_dict_train[ts][1]), axis=0)
        
for ts in ts_dict_test: 
    X_test = np.concatenate((X_test,ts_dict_test[ts][0]), axis=0)
    y_test = np.concatenate((y_test,ts_dict_test[ts][1]), axis=0)
    
    

#학습 데이터 생성     
X_train = X_train[1:]
y_train = y_train[1:]

X_test = X_test[1:]
y_test = y_test[1:]



train_data = pp.TensorData(X_train, y_train)
train_loader = DataLoader(train_data, batch_size = 128, shuffle=False)


test_data = pp.TensorData(X_test, y_test)
test_loader = DataLoader(test_data, batch_size = y_test.shape[0], shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available else 'cpu')



#모델 구축 

# train parameter
num_epoch = 5000
hid_dim = 256

in_dim = X_train.shape[-1]
out_dim = 2

encoder = seq2seq.Encoder(input_dim = in_dim, hidden_dim=hid_dim, num_layers=1)
decoder = seq2seq.Decoder(input_dim =2,  output_dim=out_dim, hidden_dim=hid_dim, num_layers=1)
m = seq2seq.Seq2Seq(encoder, decoder, device).to(device)


crit = torch.nn.MSELoss()
para = list(m.parameters())
optimizer = optim.Adam(para, 0.0001)
final_test_rmse = 10e+10
final_test_mape = 0


#y_test를 쭉 펼치고 
real = y_test.reshape((-1,2))

real = np.concatenate((scaler[0].inverse_transform(real[:,0].reshape(-1,1)),
                       scaler[1].inverse_transform(real[:,1].reshape(-1,1))),axis=1)

real = real.reshape(-1,p_size*2)

real = pd.DataFrame(real, columns = ['lat1','long1','lat2','long2','lat3','long3'])


import src.seq2seq as seq2seq
encoder = seq2seq.Encoder(input_dim = in_dim, hidden_dim=hid_dim, num_layers=1)
decoder = seq2seq.Decoder(input_dim =2,  output_dim=out_dim, hidden_dim=hid_dim, num_layers=1)
m = seq2seq.Seq2Seq(encoder, decoder, device).to(device)

#######################여기 밑에 다시 확인 

total_test_rmse = []
total_test_mape = []  
for e in range(num_epoch):
    #배치별 훈련
    for _, samples in enumerate(train_loader): 
        
        
        X_train, y_train = samples
        #xt = X_train.detach().cpu().numpy()
        #X_train[:,-1,0:2]
        optimizer.zero_grad()
        y_train[:,0,:]
        
        out = m(X_train.to(device), y_train.to(device), teacher_forcing_ratio = 0.5)
        
      
        
        break;
    break;
    

