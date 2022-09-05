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

path_dir = 'C:/Users/USER/Desktop/YT_data'
file_list = os.listdir(path_dir)


#데이터 전처리 
YT_list =[] 
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
    
    YT_list.append([yt_name, df])

#작업별 번호 매기기
for YT in YT_list:
    _, df = YT 
    n=1
    for i in range(len(df)):
        if df.iloc[i,-1] == 1:
            df.iloc[i,-1] = n
            n+=1
            



df.columns
#데이터 이상한 YT 309 / 314 / 392 제거 

###############YT 움직임 시각화 ##################
#YT별 이동 범위(위/경도)
max_lat=-1000
min_lat=1000
max_long=-1000
min_long=1000
for YT in YT_list:
    yt_name, df = YT
    data = df[[#'evnt_dt',
               'latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
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
fig, ax = plt.subplots(1,1, figsize=(10,20))
for YT in YT_list:
    yt_name, df = YT
    fig, ax = plt.subplots(1,1, figsize=(13,10))

    data = df[[#'evnt_dt',
               'latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
    
    sns.scatterplot(data=data[['latitude','longitude']], x='longitude', y='latitude', ax=ax)
    ax.set(title=yt_name,
           xlim=(min_long, max_long),
           ylim=(min_lat, max_lat))
    plt.show()       


#YT별 움직임 시각화(전체)
fig, ax = plt.subplots(1,1, figsize=(10,20))
for YT in YT_list:
    yt_name, df = YT
    data = df[[#'evnt_dt',
               'latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
    
    sns.scatterplot(data=data[['latitude','longitude']], x='longitude', y='latitude', ax=ax)
    ax.set(title=yt_name,
           xlim=(min_long, max_long),
           ylim=(min_lat, max_lat))
plt.show()   


#YT 작업별로 분리 후 시각화 (작업 바뀔 때 마다 경로 표시)
#YT별 움직임 시각화(YT별)
#fig, ax = plt.subplots(1,1, figsize=(10,20))
for YT in YT_list:
    yt_name, df = YT
    fig, ax = plt.subplots(1,1, figsize=(10,25))
    
    data = df[:]
    
    #작업 시작점 따로 표시(빨간다이아)
    sns.scatterplot(data=data[(data['check']!=0) & (data['wk_id']!='i')], x='longitude', y='latitude', ax=ax, marker="D", color='r', s=100)
    
    #작업 경로 표시 (노란점)
    sns.scatterplot(data=data[(data['check']==0) & (data['wk_id']!='i')], x='longitude', y='latitude', ax=ax, marker="o", color='y', s=5)
    
    #idle 시작점 따로 표시(파란다이아)
    sns.scatterplot(data=data[(data['check']!=0) & (data['wk_id']=='i')], x='longitude', y='latitude', ax=ax, marker="D", color='b', s=100)
    
    #idle 경로 표시 (초록점)
    sns.scatterplot(data=data[(data['check']==0) & (data['wk_id']=='i')], x='longitude', y='latitude', ax=ax, marker="o", color='g', s=5)
    
    #시작점 순번 표시
    no = data[data.check!=0]
    for i in range(len(no)): 
        plt.text(no.iloc[i]['longitude'], no.iloc[i]['latitude'], s = no.iloc[i]['check'],
                 fontdict=dict(color='red',size=8),
                 bbox=dict(facecolor='yellow',alpha=0.2))
    
    
    ax.set(title=yt_name,
           xlim=(min_long, max_long),
           ylim=(min_lat, max_lat))
    plt.show()       


#경로 움직임 파악 
fig, ax = plt.subplots(1,1, figsize=(10,20))
data = df['2021-10-31-20:31':'2021-10-31-20:35'][['latitude','longitude']]

sns.scatterplot(data=data[['latitude','longitude']], x='longitude', y='latitude', ax=ax)

ax.set(title='yt',
           xlim=(min_long, max_long),
           ylim=(min_lat, max_lat))
plt.show()  
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






#정지상태 제거된 YT 데이터 
pp_YT_list = []
fig, ax = plt.subplots(1,1, figsize=(13,10))
for YT in YT_list:
    yt_name, df = YT

    #필터링
    data = df[[#'evnt_dt',
               'latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
    vector = df[['velocity', 'direction']]
    vector_ = vector.shift(1)
    
    filter = ((vector.velocity!=vector_.velocity) | (vector.direction!=vector_.direction))
    
    #얘가 필터링된 데이터 
    data = data.loc[filter,:]
    pp_YT_list.append([yt_name, data])
        



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

