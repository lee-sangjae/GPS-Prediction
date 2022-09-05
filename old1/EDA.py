# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:40:26 2022

@author: USER
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os 

path_dir = 'C:/Users/USER/Desktop/YT_data/train'
file_list = os.listdir(path_dir)

YT_list =[] 
for f_name in file_list: 
    df = pd.read_csv(path_dir+'/'+f_name)
    yt_name = f_name[:-4]
    YT_list.append([yt_name, df])

#데이터 이상한 YT 309 / 314 제거 
YT_list.pop(3)
YT_list.pop(5)

###############YT 움직임 시각화 ##################
#YT별 이동 범위(위/경도)
max_lat=-1000
min_lat=1000
max_long=-1000
min_long=1000
for YT in YT_list:
    yt_name, df = YT
    data = df[['evnt_dt','latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
    max_latitude = data['latitude'].max()
    min_latitude = data['latitude'].min()
    
    max_longitude = data['longitude'].max()
    min_longitude = data['longitude'].min()
    
    if max_latitude>max_lat : 
        max_lat=max_latitude
       
    if min_latitude<min_lat : 
        min_lat=min_latitude
       
    if max_longitude > max_long : 
        max_long=max_longitude    
    
    if min_longitude<min_long : 
        min_long=min_longitude
 



#YT별 움직임 시각화(YT별)
fig, ax = plt.subplots(1,1, figsize=(13,10))
for YT in YT_list:
    yt_name, df = YT
    fig, ax = plt.subplots(1,1, figsize=(13,10))

    data = df[['evnt_dt','latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
    
    sns.scatterplot(data=data[['latitude','longitude']], x='latitude', y='longitude', ax=ax)
    ax.set(title=yt_name,
           xlim=(min_lat, max_lat),
           ylim=(min_long, max_long))
    plt.show()       


#YT별 움직임 시각화(전체)
fig, ax = plt.subplots(1,1, figsize=(13,10))
for YT in YT_list:
    yt_name, df = YT
    data = df[['evnt_dt','latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
    
    sns.scatterplot(data=data[['latitude','longitude']], x='latitude', y='longitude', ax=ax, color = 'r')
    ax.set(title='Whole YT',
           xlim=(min_lat, max_lat),
           ylim=(min_long, max_long))
plt.show()   

#따로 떨어져있는 YT392 제거
YT_list.pop(15)

#################################### ################## ################## ##################
########움직임 변화 없는 애들 제거################
##이전행과 속도/방향이 같은 애들 제거 ###
#vector = df[['velocity', 'direction']].values
"""
index_list=[]

for i in range (len(df)-1):
    vector = df[['velocity', 'direction']]
    
    #맨 첫번째 값도 추가해야 함
    
    #속도/방향 비교
    if not((vector.iloc[i,0]==vector.iloc[i+1,0]) & (vector.iloc[i,1]==vector.iloc[i+1,1])):
        index_list.append(i+1)
      
for i in range(len(df))
"""   

#벡터 변화 여부 필터링 
vector = df[['velocity', 'direction']]
vector_ = vector.shift(1)

filter = ((vector.velocity!=vector_.velocity) | (vector.direction!=vector_.direction))

vector = vector.loc[filter,:] #얘는 다음 시점에 벡터 변화가 있는 애들만 필터링 된 거 




#정지상태 제거된 YT 데이터(필터링) 
#fig, ax = plt.subplots(1,1, figsize=(13,10))
for YT in YT_list:
    yt_name, df = YT
    fig, ax = plt.subplots(1,1, figsize=(13,10))
    #필터링
    data = df[['evnt_dt','latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
    vector = df[['velocity', 'direction']]
    vector_ = vector.shift(1)
    
    filter = ((vector.velocity!=vector_.velocity) | (vector.direction!=vector_.direction))
    
    #얘가 필터링된 데이터 
    data = data.loc[filter,:]
    
    sns.scatterplot(data=data[['latitude','longitude']], x='latitude', y='longitude', ax=ax)
    ax.set(title='Whole YT',
           xlim=(min_lat, max_lat),
           ylim=(min_long, max_long))
    plt.show()   
    
    
        



#데이터 요약
describe = df.describe()
df.info()

#칼럼 별 고유값 개수 확인
c_list = ['position_fix','satelites', 'equ_no', 'cntr_dup', 'wk_id' ]
for c in c_list:
    print(c+'\n', df[c].value_counts())
#position_fix: 4,5,2
#satelites : 12, 11   
#equ_no: YT별로 배정된 번호인듯
#cntr_dup: 8개정도
#wk_id: 전부 U

#df.columns
data = df[['evnt_dt','latitude', 'longitude', 'altitude', 'direction', 'velocity']]
data['evnt_dt'] = pd.to_datetime(df['evnt_dt'])

#해당 차량의 데이터가 기록된 시간 
data['evnt_dt'][len(data)-1]- data['evnt_dt'][0]

sns.scatterplot(data=data[['latitude','longitude']], x='latitude', y='longitude')

