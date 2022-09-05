# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:09:22 2022

@author: LSJ
"""

import pandas as pd
import numpy as np

import torch 

import torch.optim as optim




from torch.utils.data import TensorDataset,DataLoader, Dataset
from haversine import haversine_vector
import src.preprocess as pp
import src.model as models
import src.seq2seq as seq2seq
from tqdm import tqdm


import matplotlib.pyplot as plt
import seaborn as sns

import os 
import time
import random
import torch.backends.cudnn as cudnn

#시드 고정
seed = 37
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, 'GPS\\data\\YT_data')

folder_list = os.listdir(data_dir)


#데이터 전처리 
YT_dict = {} 

for folder in folder_list:
    YT_dict[folder] ={}
    d = os.path.join(data_dir,folder)
    file_list = os.listdir(d)
    for file in file_list:
        df = pd.read_csv(os.path.join(d,file))
        yt_name = file[:-4]
        
        #datetime 변환 및 인덱스 설정
        df['evnt_dt'] = pd.to_datetime(df['evnt_dt'])
        #df.set_index('evnt_dt',inplace=True)
        
        
        #거리 및 방향 변화
        df_shift = df.shift(1)
        
        a = df_shift.loc[:,['latitude','longitude','direction']] 
        b = df.loc[:, ['latitude', 'longitude','direction']]
        dis = haversine_vector(a[['latitude','longitude']], b[['latitude', 'longitude']], unit='m')
        
            
        df['distance'] = dis
        df['dir_diff'] = b['direction'] - a['direction']
        
        #distance 결측인 행 제거 
        df = df.dropna(subset=['distance'], how = 'any', axis=0)
        df = df.reset_index(drop='True')
        
        #1초 이동거리 12 이상인 애들 제거 
        #df = df[df.distance<=12]
        
        #불필요 칼럼 삭제 
        df = df[['latitude','longitude','velocity','direction','equ_no','evnt_dt','distance', 'dir_diff']]
        
        
        #결측시 채우기
        ts = pd.date_range(start=df.loc[0,'evnt_dt'],
                           end = df.loc[len(df)-1,'evnt_dt'],
                           freq = 'S')
        
        ts = pd.DataFrame(index = ts)
        
        df = df.set_index('evnt_dt')
        df = ts.join(df)
        df = df.interpolate()
        
        #파일 이름과 YT넘버 동일하게 
        df['equ_no'] = yt_name
            
        #인덱스 제거
        df = df.reset_index()
        
        
        #딱 한 번 정지 상태 외에 제거 
        temp = df.shift(1)
        df = df[~((df.velocity==0) & (temp.velocity==0))]
        
        #인덱스 제거
        df = df.reset_index(drop=True)
        
        
                
        #날짜 인덱스 제거
        #df = df.drop('index', axis=1)
        
        
        YT_dict[folder][yt_name] = df
    
            

#전체 데이터 합치기 (스케일러 만들기 위해)
df_dict = {}

for folder in folder_list:
    df_dict[folder] = pd.DataFrame()
    for YT in YT_dict[folder]:
        result = pd.concat([df_dict[folder], YT_dict[folder][YT]])
        df_dict[folder] = result
    df_dict[folder] = df_dict[folder].drop(columns=['equ_no','index'], axis=1)
          
#칼럼 별 스케일러 생성 
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = {}
for c in df_dict['train'].columns:
    scaler[c] = StandardScaler()

for c in df_dict['train'].columns:
    scaler[c].fit(df_dict['train'].loc[:,c].values.reshape(-1,1))


#스케일러 피팅 
ss  = StandardScaler()
ss.fit(df_dict['train'])

#시계열 생성 

w_size = 5
p_size = 5

ts_dict = {}

for folder in folder_list:
    ts_dict[folder]={}
    for yt in YT_dict[folder]:
        temp = YT_dict[folder][yt].drop(['equ_no','index'], axis=1)
        
        #스케일링 
        temp = pd.DataFrame(ss.transform(temp), columns=temp.columns, index=temp.index)
    
    
        #시계열 생성 
        X = temp
        y = temp[['latitude','longitude']]    
        
        X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
        ts_dict[folder][yt] = [X_s, y_s]



#시계열 데이터 합치기 
data_set = {} 
for folder in folder_list:
    data_set[folder] = {}
    for i in ['X','y']:
        if i=='X':
            data_set[folder][i] = np.zeros((1,w_size,len(df_dict[folder].columns)))
        else:
            data_set[folder][i] = np.zeros((1,p_size,2))
    
for folder in folder_list:
    for ts in ts_dict[folder]:
        
        data_set[folder]['X'] = np.concatenate((data_set[folder]['X'],ts_dict[folder][ts][0]), axis=0)
        data_set[folder]['y'] = np.concatenate((data_set[folder]['y'],ts_dict[folder][ts][1]), axis=0)
    
    data_set[folder]['X'] = data_set[folder]['X'][1:]    
    data_set[folder]['y'] = data_set[folder]['y'][1:]    

#데이터 로더 생성 
data_loader = {} 
for folder in folder_list:
    temp = pp.TensorData(data_set[folder]['X'], data_set[folder]['y'])
    if folder == 'train':
        data_loader[folder] = DataLoader(temp, batch_size = 128, shuffle=True)
    elif folder == 'valid':
        data_loader[folder] = DataLoader(temp, batch_size = data_set[folder]['y'].shape[0], shuffle=True)
    elif folder == 'test':
        data_loader[folder] = DataLoader(temp, batch_size = 1, shuffle=False)
    
    
    

#모델 구축 
#hid_dim = 256

in_dim = df_dict['train'].shape[-1]
out_dim = p_size*2


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

final_test_rmse = 10e+10
final_test_mape = 0


#test 정답에 대한 실제 값 
real = data_set['test']['y'].reshape((-1,2))

real = np.concatenate((scaler['latitude'].inverse_transform(real[:,0].reshape(-1,1)),
                       scaler['longitude'].inverse_transform(real[:,1].reshape(-1,1))),axis=1)


real = real.reshape(-1,p_size*2)

columns =[]
for i in range(1,p_size+1):
    columns.append('lat'+str(i))
    columns.append('long'+str(i))
    
real = pd.DataFrame(real, columns = columns)



#모델 로드 및 테스트 
model = 'RNN(5-5)'
m = torch.load('C:/Users/USER/Dropbox/project code/GPS/'+model+'.pt', map_location=device)    

    
lat_min = df_dict['train']['latitude'].min()
lat_max = df_dict['train']['latitude'].max()

long_min = df_dict['train']['longitude'].min()
long_max = df_dict['train']['longitude'].max()


result_dir = os.path.join(parent_dir, 'GPS\\result\\'+model)

with torch.no_grad():
    present = np.array([])
    for idx, samples in enumerate(data_loader['test']):
    
        X_test, y_test = samples
        
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
    
        out = m(X_test)
    
        out = out.detach().cpu().numpy()
        
        
        #나중에 시각화 할 현재 위치 
        present_lat = scaler['latitude'].inverse_transform(X_test[0,:,[0]].detach().cpu().numpy())
        present_long = scaler['longitude'].inverse_transform(X_test[0,:,[1]].detach().cpu().numpy())
        present = np.concatenate((present_lat, present_long), axis=1)
        
        
        
        #역변환 
    
        lat = scaler['latitude'].inverse_transform(out[:,list(range(0,len(columns),2))].reshape(-1,p_size))
        long = scaler['longitude'].inverse_transform(out[:,list(range(1,len(columns)+1,2))].reshape(-1,p_size)) 
        
        
        #예측 결과 (lat, long을  v 형태로 만들어야 함)
        pred = pd.DataFrame()
        
        for i in range(p_size):
            pred['lat'+str(i+1)] = lat[:,i]
            pred['long'+str(i+1)] = long[:,i]
            
            
    
    
        #평균 오차 거리 계산 
        dis_diff=[]
        
        row_real = real.loc[idx:idx, :]
        
        #1초 뒤 예측 거리 차이 
        dis_diff = pd.DataFrame()
        for i in range(1, p_size+1):
            a = pred.loc[:, ['lat'+str(i),'long'+str(i)]]
            b = row_real.loc[:, ['lat'+str(i),'long'+str(i)]]
            dis_diff['pred'+str(i)] = haversine_vector(a,b,unit='m')
        
        mean_diff = np.mean(dis_diff)
    
   
    
        #테스트 경로 시각화     
        fig, ax = plt.subplots(1,1, figsize=(10,8))
        
        #기존 경로 시각화
        v = pd.DataFrame(present)
        #기존 경로 시각화 
        sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax, estimator = None, color='g')
        
        #예측 값 하나 샘플링 
        lat = np.array(pred.iloc[0, list(range(0,len(columns),2))]) 
        long = np.array(pred.iloc[0, list(range(1,len(columns)+1,2))]) 
        
        v = pd.DataFrame(np.stack([lat,long], 0 ).T)
        
        #예측 경로 시각화 
        sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax, estimator = None, color='r')
        
        #실제 값 하나 샘플링 
        lat = np.array(row_real.iloc[0, list(range(0,len(columns),2))]) 
        long = np.array(row_real.iloc[0, list(range(1,len(columns)+1,2))]) 
        
        v = pd.DataFrame(np.stack([lat,long], 0 ).T)
        
        #실제 경로 시각화 
        sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax, estimator = None, color='b')
        
        ax.set(title ='average dis diff: '+str(round(np.mean(mean_diff),3))+'\n'+str(round(mean_diff,3))+' - RNN',
               xlim=(lat_min,lat_max), ylim = (long_min,long_max))
        
        plt.legend(labels=['present', 'predicted', 'real'])
        
        plt.savefig(os.path.join(result_dir, str(idx)+'.png'))
        #plt.show()