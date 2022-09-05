# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:20:46 2022

@author: USER
"""
import os 

import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
# import torchvision #
# import torchvision.datasets as dset
import torchvision.transforms as tr #데이터 불러오면서 전처리를 가능한게 해주는 라이브러리 
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from haversine import haversine
import src.preprocess as pp
from src import model
from sklearn.metrics import mean_squared_error as mse

from torch.autograd import Variable 


import matplotlib.pyplot as plt
import seaborn as sns

import glob
import config 

path_dir_train = glob.glob(os.path.join(config.BASE_DIR, f"data\\train\\*.csv"))
path_dir_test = glob.glob(os.path.join(config.BASE_DIR, f"data\\test\\*.csv"))


file_list_train = os.listdir(path_dir_train)
file_list_test = os.listdir(path_dir_test)


#트레인 데이터 전처리 
YT_dict_train ={} 
for f_name in file_list_train: 
    print(path_dir_train+'/'+f_name)
    df = pd.read_csv(path_dir_train+'/'+f_name)
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
    df = df.drop('index', axis=1)
    
    
    YT_dict_train[yt_name] = df
    

############################################

#테스트 데이터 전처리 
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
            
    #날짜 인덱스 제거
    df = df.drop('evnt_dt', axis=1)
    
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

df_train = df_train.drop(columns='equ_no', axis=1)
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
    temp = YT_dict_train[yt].drop('equ_no', axis=1)
    
    #스케일링 
    temp = pd.DataFrame(mm.transform(temp), columns=temp.columns, index=temp.index)
    
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_train[yt] = [X_s, y_s]

ts_dict_test={} 
for yt in YT_dict_test:
    temp = YT_dict_test[yt].drop('equ_no', axis=1)
    
    #스케일링 
    temp = pd.DataFrame(mm.transform(temp), columns=temp.columns, index=temp.index)
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_test[yt] = [X_s, y_s]    
    
ts_dict_train

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
train_loader = DataLoader(train_data, batch_size = 64, shuffle=False)


test_data = pp.TensorData(X_test, y_test)
test_loader = DataLoader(test_data, batch_size = y_test.shape[0], shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


#모델 구축 

# train parameter
num_epoch = 1000
hid_dim = 512

in_dim = X_train.shape[-1]
out_dim = 6

m = model._LSTM2(in_dim = in_dim, hid_dim = hid_dim, out_dim = out_dim, num_layers = 1).cuda()

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

#######################여기 밑에 다시 확인 

total_test_rmse = []
total_test_mape = []  
for e in range(num_epoch):
    #배치별 훈련
    for _, samples in enumerate(train_loader): 
        X_train, y_train = samples
        optimizer.zero_grad()
        
        #6개 값 출력 
        out = m(X_train.to(device))
        
        y_train = y_train.to(device).reshape(-1,6)
    
        
        #배치별 업데이트
        loss = crit(out, y_train)
        
        loss.backward()
        optimizer.step()
        
        
    #테스트 
    with torch.no_grad():
        for idx, samples in enumerate(test_loader):
            X_test, y_test = samples
        
            out = m(X_test.to(device))
            
            #위경도 위경도 위경도 
            out = out.detach().cpu().numpy()
            
            

            #역변환 (확신이 안섬 맞는지 )
            lat = scaler[0].inverse_transform(out[:,[0,2,4]].reshape(-1,3))
            long = scaler[1].inverse_transform(out[:,[1,3,5]].reshape(-1,3)) 
            
            
            
            
            #예측 결과 (lat, long을  v 형태로 만들어야 함)
            #pred = pd.DataFrame(np.concatenate((lat,long),axis=1), columns = ['latitude','longitude'])
            
            pred = pd.DataFrame({'lat1':lat[:,0],'long1':long[:,0],
                                 'lat2':lat[:,1],'long2':long[:,1],
                                 'lat3':lat[:,2],'long3':long[:,2]})
    
            
            test_rmse = (mse(pred, real)**0.5)
            test_mape = (np.mean(abs((np.array(real)-np.array(pred))/np.array(real)))*100)
        
        
        if final_test_rmse > test_rmse:
            final_test_rmse = test_rmse
            final_test_mape = test_mape
            
        
        #평균 오차 거리 계산 
        dis_diff=[]
        
        
        for r in range(X_test.size(0)): #전체 데이터 돌면서
            
            #1초후 거리 차이 
            a1 = (pred.iloc[r,0], pred.iloc[r,1])
            b1 = (real.iloc[r,0],real.iloc[r,1])
            dis1 = haversine(a1,b1, unit='m') 
            
            #2초후 거리 차이 
            a2 = (pred.iloc[r,2], pred.iloc[r,3])
            b2 = (real.iloc[r,2],real.iloc[r,3])
            dis2 = haversine(a2,b2, unit='m') 
            
            #3초후 거리 차이 
            a3 = (pred.iloc[r,4], pred.iloc[r,5])
            b3 = (real.iloc[r,4],real.iloc[r,5])
            dis3 = haversine(a3,b3, unit='m') 
            
            dis_diff.append([dis1, dis2, dis3])
        
            
        dis_diff = pd.DataFrame(dis_diff,columns=['pred1','pred2','pred3'])
        mean_diff = np.mean(dis_diff)
        
        
        
       
        
        #테스트 시각화 
        if e%100==0:
            fig, ax = plt.subplots(1,1, figsize=(10,25))
            
            #3초 후 값만 출력
            sns.scatterplot(data=pred, x='long3', y='lat3', ax=ax, color='r')
            
            #예측값인데 오차 3m 이하인 애들
            #sns.scatterplot(data=pred[~mask], x='longitude', y='latitude', ax=ax, color='green')
            
            sns.scatterplot(data=real, x='long3', y='lat3', ax=ax, color='b')
            
            ax.set(title ='average dis diff: \n'+str(mean_diff))
            
            #ax.set(title='Average dis diff: '+str(mean_diff))
            plt.show()
            
    
        total_test_rmse.append(final_test_rmse)
        total_test_mape.append(final_test_mape)
        
        

pred[['diff1', 'diff2', 'diff3']] = dis_diff

