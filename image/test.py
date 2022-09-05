# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:20:10 2022

@author: LSJ
"""

# =============================================================================
# 
# =============================================================================

import pandas as pd
import numpy as np

from tqdm import tqdm 

import matplotlib.pyplot as plt
import seaborn as sns

from haversine import haversine_vector

import os


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, 'data\\YT_data')

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
    
    
df = df_dict['train']
df = df.reset_index(drop=True)




temp = df.copy()

#rec_plot 
import pylab as pplt

def rec_plot(s, eps = None, steps = None):
    if eps == None : eps=0.01
    if steps ==None : steps = 10
    N = s.size
    S =  np.repeat(s[None,:], N, axis = 0)
    Z = np.floor(np.abs(S-S.T)/eps)
    Z[Z>steps] = steps
    
    return Z

s = np.random.random(1000)
pplt.imshow(rec_plot(s))
plt.plot(s)

s = np.array(temp[['latitude','longitude']])
pplt.imshow(rec_plot(s))
plt.plot(s)



#1. Gramian Angular Field(GAF)
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler

a = temp


