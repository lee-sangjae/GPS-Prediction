# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:04:44 2022

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:43:35 2022

@author: USER
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine

import os 


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

from torch.autograd import Variable 


path_dir = 'C:/Users/USER/Desktop/YT_data'
file_list = os.listdir(path_dir)


#데이터 전처리 
YT_dict ={}
for f_name in file_list: 
    df = pd.read_csv(path_dir+'/'+f_name)
    yt_name = f_name[:-4]
    
    #datetime 변환 및 인덱스 설정
    df['evnt_dt'] = pd.to_datetime(df['evnt_dt'])
    df.set_index('evnt_dt',inplace=True)
    
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
    ###########################
    #정지상태 제거 
    temp = df[ (df.velocity != 0)]
    
    
    #거리 및 방향 변화
    dis_list = [np.nan]
    dir_list = [np.nan]
    for r in range(len(df)-1):
        #거리계산 
        a = (df.iloc[r,1], df.iloc[r,2])
        b = (df.iloc[r+1,1], df.iloc[r+1,2])
        dis = haversine(a,b) * 1000
        dis_list.append(dis)
        
        #방향 변화
        dir_list.append(df.iloc[r+1,5] - df.iloc[r,5])
        
        
    df['distance'] = dis_list
    df['dir_diff'] = dir_list
    
    #distance 결측인 행 제거 
    df = df.dropna(subset=['distance'], how = 'any', axis=0)
    
    #불필요 칼럼 삭제 
    df.drop(['reg_seq', 'altitude', 'position_fix', 'satelites', 'dev_id', 'cre_dt', 'cntr_dup', 
             'wk_id','y_blk', 'y_bay', 'y_row', 'y_tier', 'long_cut', 'latt_cut', 'check'], axis=1 , inplace=True)
    
    YT_dict[yt_name] = df




#작업별 번호 매기기
for YT in YT_dict:
    
    df = YT_dict[YT] 
    n=1
    for i in range(len(df)):
        if df.iloc[i,-1] == 1:
            df.iloc[i,-1] = n
            n+=1
            

#데이터 이상한 YT 309 / 314 / 392 제거 
del YT_dict['YT309']
del YT_dict['YT314']
del YT_dict['YT392']

        
########################################################################################
#다 합친 데이터 
df = pd.DataFrame()

for name in YT_dict :
    df = pd.concat([df, YT_dict[name]])
    
df[['longitude','latitude']].plot(x='longitude', y= 'latitude', kind='scatter')

grouped = df.groupby(['longitude', 'latitude'])

d = {}
for key, group in grouped:
    d[key] = group[group.velocity !=0]
