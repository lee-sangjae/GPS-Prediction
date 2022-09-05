# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:23:41 2022

@author: LSJ
"""

# =============================================================================
# GPS 데이터 전처리 
# =============================================================================


import pandas as pd
import numpy as np

import torch 

import torch.optim as optim




from torch.utils.data import TensorDataset,DataLoader, Dataset
from haversine import haversine_vector, haversine
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
np.random.seed(seed)
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
    
            

#train / valid / test 분할 
df_dict = {}
for folder in folder_list:
    df_dict[folder] = pd.DataFrame()
    for YT in YT_dict[folder]:
        result = pd.concat([df_dict[folder], YT_dict[folder][YT]])
        df_dict[folder] = result
    df_dict[folder] = df_dict[folder].drop(columns=['equ_no','index'], axis=1)
    #df_dict[folder].to_csv('{}.csv'.format(folder), index = False)

# 격자 라벨링 
#주어진 데이터에서 위 경도 범위 체크 

#lat, long
#위도, 경도   
#위도: 

#경도 
max_long = 0
min_long = np.Inf

#위도
max_lat = 0
min_lat = np.Inf

#그리드 단위 
#0.000045 차이나면 5m 
#0.000009 차이나면 1m

unit = 0.000045 


for partition in df_dict:
    
    if df_dict[partition]['latitude'].min() <= min_lat:
        min_lat = df_dict[partition]['latitude'].min()
    
    if df_dict[partition]['latitude'].max() >= max_lat:
        max_lat = df_dict[partition]['latitude'].max()
        
    if df_dict[partition]['longitude'].min() <= min_long:
        min_long = df_dict[partition]['longitude'].min()
    
    if df_dict[partition]['longitude'].max() >= max_long:
        max_long = df_dict[partition]['longitude'].max()
        

long_range = max_long - min_long 
lat_range = max_lat - min_lat 

long_range // 0.000045



lat_label = np.arange(min_lat, max_lat, 0.000045)
long_label = np.arange(min_long, max_long, 0.000045)

temp = df_dict['train']
temp['lat_label'] = 0
temp['long_label'] = 0

for partition in df_dict:
    temp = df_dict[partition]
    temp['lat_label'] = 0
    temp['long_label'] = 0
    
    for idx, i in enumerate(lat_label):
        condition = temp.latitude>=i
        temp.loc[condition, 'lat_label'] = idx
    
    for idx, i in enumerate(long_label):
        condition = temp.longitude>=i
        temp.loc[condition, 'long_label'] = idx
    
    temp['grid'] = '('+temp['lat_label'].astype('str') + ', ' + temp['long_label'].astype('str') + ')'
    
    temp = temp.reset_index(drop=True)
    
    df_dict[partition] = temp
    


    

#딕셔너리 자체 저장 
import pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(df_dict, f)






# 그리드 생성 


 
    
#격자 범위 설정 
plt.figure(figsize=(18,37))
plt.xlim(0, 184)    #long 
plt.ylim(0, 368)    #lat


#격자 간격 설정
plt.xticks(range(0,184,1))
plt.yticks(range(0,368,1))

plt.grid()

plt.gca().axes.xaxis.set_ticklabels([]) #x축 눈금 없애기
plt.gca().axes.yaxis.set_ticklabels([]) #y축 눈금 없애기

#격자활용해서 경로 시각화 
#lat,lat,long,long / lat, long, long, lat
for i in range(len(df_dict['train'])):
    temp = df_dict['train'][['lat_label', 'long_label']]
    lat = temp.loc[i, 'lat_label']
    long = temp.loc[i, 'long_label']
    
    plt.fill([long, long, long+1 ,long+1], [lat, lat+1, lat+1, lat ] , color = 'gray', alpha=1)

plt.show()



a = (0,0)
b = (0.000045, 0)

haversine(a,b, unit='m')



