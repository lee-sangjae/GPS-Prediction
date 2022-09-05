# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:49:14 2022

@author: USER
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from haversine import haversine

import os 

path_dir = 'C:/Users/USER/Dropbox/project code/GPS/data'
file_list = os.listdir(path_dir)

yt = {} 

for f_name in file_list: 
    df = pd.read_csv(path_dir+'/'+f_name)
    yt_name = f_name[:-4]
    
    #datetime 변환 및 인덱스 설정
    df['evnt_dt'] = pd.to_datetime(df['evnt_dt'])
    

    #거리 및 방향 변화
    dis_list = [np.nan]
    dir_list = [np.nan]
    for r in range(1, len(df)):
        #거리계산 
        a = (df.loc[r-1,'latitude'], df.loc[r-1,'longitude'])
        b = (df.loc[r,'latitude'], df.loc[r,'longitude'])
        dis = haversine(a,b, unit='m')
        dis_list.append(dis)
        
        #방향 변화
        dir_list.append(df.loc[r,'direction'] - df.loc[r-1,'direction'])
        
        
    df['distance'] = dis_list
    df['dir_diff'] = dir_list
    
    
    df.set_index('evnt_dt',inplace=True)
    
    #결측시 채우기 
    if len(df) != 0:
        ts = pd.date_range(start=df.index[0],
                           end = df.index[-1],
                           freq = 'S')
        
        ts = pd.DataFrame(index = ts)
    
        df = ts.join(df)
        df = df.interpolate()
        df['equ_no'] = yt_name
    
    df = df.reset_index()
    
    yt[yt_name] = df

    
    
    
    


###############YT 움직임 시각화 ##################
#YT별 이동 범위(위/경도)
max_lat=-1000
min_lat=1000
max_long=-1000
min_long=1000
for YT in yt:
    yt_name, df = YT, yt[YT]
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
for YT in yt:
    yt_name, df = YT, yt[YT]
    fig, ax = plt.subplots(1,1, figsize=(13,10))

    data = df[['evnt_dt','latitude', 'longitude', 'altitude', 'direction', 'velocity']]
    
    
    sns.scatterplot(data=data[['latitude','longitude']], x='latitude', y='longitude', ax=ax)
    ax.set(title=yt_name,
           xlim=(min_lat, max_lat),
           ylim=(min_long, max_long))
    plt.show()       
    
