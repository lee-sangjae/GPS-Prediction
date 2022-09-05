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
result_dir = os.path.join(parent_dir, 'GPS\\result')

result_list = os.listdir(result_dir)
select_list = []

size = '(5-10)'
for i in result_list:
    if size in i:
        select_list.append(i)




present = pd.read_csv('C:/Users/USER/Dropbox/project code/GPS/data/'+size+'_present.csv')
real = pd.read_csv('C:/Users/USER/Dropbox/project code/GPS/data/real'+size+'.csv')


#모델 별 결과 그리기 
model = {} 
for result in select_list:
        model[result] = pd.read_csv(os.path.join(result_dir,result+'\\'+result+'_pred.csv'))


#
del(model['LSTM(5-10)'])
del(model['Conv_LSTM(5-10)'])
del(model['RNN(5-10)'])
del(model['Conv_Seq2Seq(5-10)_new'])
del(model['Seq2Seq(5-10)'])


num = len(present)


#모델 별 시퀀스 별 오차 계산 
diff = []
model_dis_diff = {}

fig, ax = plt.subplots(1,1, figsize=(10,8))
for m in model:
    df = model[m]
    
    dis_diff = pd.DataFrame()
    for i in range(1, int(len(real.columns)/2)+1):
        a = df.loc[:, ['lat'+str(i),'long'+str(i)]]
        b = real.loc[:, ['lat'+str(i),'long'+str(i)]]
        dis_diff['pred'+str(i)] = haversine_vector(a,b,unit='m')
    
    #예측 시간 별 평균 오차 계산 
    mean = pd.DataFrame(np.mean(dis_diff), columns = [m])
    sns.lineplot(data = mean, x = mean.index, y = m)
    
    #경로 별 평균 오차 계산
    dis_diff['mean'] = dis_diff.loc[:,dis_diff.columns].mean(axis=1)
    
    model_dis_diff[m] = dis_diff
    
    #모델 별 평균 오차 계산 
    diff.append(mean.mean().values[0])
      
plt.legend(labels=['Attention', 'Conv_Seq2Seq'])     
ax.set(title = 'model sequence average error')
plt.show()

fig, ax = plt.subplots(1,1, figsize=(10,8))
sns.barplot( x = ['Attention', 'Conv_Seq2Seq'], y=diff )
ax.set(title = 'model average error')
plt.show()

##### (모델 별 오차 저장)
"""
d = []
for i in model_dis_diff:
    y = []
    df = model_dis_diff[i]
    for j in range(len(df.columns)):
        y.append(np.round(df.iloc[:,j].mean(),4))
    d.append(y)
"""

#모델 별 예측 경로 차이 많이 나는 구간 체크 
#thres_hold 이상 거리 차이나는 결과 체크 
thres_hold = 3

for m in model_dis_diff:
    df = model_dis_diff[m]
    df['error'] = 0
    df.loc[df['mean']>thres_hold,'error'] = 1
    model_dis_diff[m] = df
    
    
    
#각 모델 별 경로 예측 시각화 
max_lat= 35.1107483
min_lat= 35.0946503
max_long= 129.1020508
min_long= 129.0938873


###############################
attention = pd.read_csv('C:/Users/USER/Dropbox/project code/GPS/attention_weigth.csv')
attention.columns = ['1','2','3','4','5']

attention.loc[:, 'max'] = attention.loc[:, ['1','2','3','4','5']].max(axis=1)

condition1 = attention.loc[:, 'max'] == attention.loc[:, '1']
condition2 = attention.loc[:, 'max'] == attention.loc[:, '2']
condition3 = attention.loc[:, 'max'] == attention.loc[:, '3']
condition4 = attention.loc[:, 'max'] == attention.loc[:, '4']
condition5 = attention.loc[:, 'max'] == attention.loc[:, '5']

attention.loc[condition1, 'top1'] = '1'
attention.loc[condition2, 'top1'] = '2'
attention.loc[condition3, 'top1'] = '3'
attention.loc[condition4, 'top1'] = '4'
attention.loc[condition5, 'top1'] = '5'

attention['top1'].value_counts()
###############################





for i in range(num):    
    for m in model:
        color = ''
        
        if 'Conv_Seq2Seq' in m:
            color = 'orange'
        elif 'Conv_LSTM' in m:
            color = 'c'
        elif 'Seq2Seq' in m :
            color ='b'
        elif 'RNN' in m:
            color = 'g'
        elif 'Attention' in m :
            color = 'r'
        elif 'LSTM' in m:
            color = 'r'
        #라벨링 
        label = model_dis_diff[m].loc[i:i, 'error'].values[0]
        
        attention_label = attention.loc[i, 'top1']

        if  label== 1: #세 모델 중 하나라도 threshold 넘어가면 다 그림 
            title = ''
            p = present.loc[i:i,:]
            r = real.loc[i:i,:]
            fig, ax = plt.subplots(1,2, figsize=(20,13))
            for m in model:
                color = ''
                
                if '_new' in m:
                    color = 'r'
                    t = 'Conv_Seq2Seq_new: '
                elif 'Conv_Seq2Seq' in m:
                    color = 'orange'
                    t = 'Conv_Seq2Seq: '
                elif 'Conv_LSTM' in m:
                    color = 'c'
                    t = 'Conv_LSTM'
                elif 'Seq2Seq' in m :
                    color ='b'
                    t = 'Seq2Seq: '
                elif 'RNN' in m:
                    color = 'g'
                    t = 'RNN: '
                elif 'Attention' in m:
                    color = 'r'
                    t = 'Attention'
                elif 'LSTM' in m:
                    color = 'r'
                    t = 'LSTM: '
                
                #결과 한줄 씩 
                
                df = model[m].loc[i:i,:]
                
                #모델 별 예측 경로 
                lat = np.array(df.iloc[0, list(range(0,len(r.columns),2))]) 
                long = np.array(df.iloc[0, list(range(1,len(r.columns)+1,2))]) 
                v = pd.DataFrame(np.stack([lat,long], 0 ).T)
                
                sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax[0], estimator = None, color=color)
                sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax[1], estimator = None, color=color)
                
                
                #평균 오차 거리 
                d = np.round(model_dis_diff[m].loc[i:i, 'mean'].values[0],4)
                
                title = title + t + str(d) +'m\n'
            
            
            #현재 위치 
            lat = np.array(p.iloc[0, list(range(0,len(p.columns),2))]) 
            long = np.array(p.iloc[0, list(range(1,len(p.columns)+1,2))]) 
            v = pd.DataFrame(np.stack([lat,long], 0 ).T)
            
            sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax[0], estimator = None, color='gray')
            sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax[1], estimator = None, color='gray')
            
            #실제 정답 
            lat = np.array(r.iloc[0, list(range(0,len(r.columns),2))]) 
            long = np.array(r.iloc[0, list(range(1,len(r.columns)+1,2))]) 
            v = pd.DataFrame(np.stack([lat,long], 0 ).T)
            
            sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax[0], estimator = None, color='black')
            sns.lineplot(data=v, x=0, y=1, marker='o',ax=ax[1], estimator = None, color='black')
            
            
            
            #ax.set(xlim=(lat_min,lat_max), ylim = (long_min,long_max))
            ax[0].legend(labels=['Attention', 'Conv_Seq2Seq', 'present', 'real' ]) 
            ax[1].legend(labels=['Attention', 'Conv_Seq2Seq','present', 'real' ]) 
            
            ax[1].set_xlim([min_lat, max_lat])
            ax[1].set_ylim([min_long, max_long])
            
            ax[0].set_xlabel('latitude')
            ax[0].set_ylabel('longitude')
            
            ax[1].set_xlabel('latitude')
            ax[1].set_ylabel('longitude')
            
            plt.suptitle(title)
            
            plt.show()
        
        

        
        
        
        


      
    
   
    
        