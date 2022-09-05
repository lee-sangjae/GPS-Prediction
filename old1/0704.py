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
from haversine import haversine_vector, haversine
import src.preprocess as pp
import src.model as models
import src.seq2seq as seq2seq
from sklearn.metrics import mean_squared_error as mse

from torch.autograd import Variable 


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


path_dir_train = 'C:/Users/USER/Desktop/YT_data/train'
path_dir_valid = 'C:/Users/USER/Desktop/YT_data/valid'
path_dir_test = 'C:/Users/USER/Desktop/YT_data/test'

file_list_train = os.listdir(path_dir_train)
file_list_valid = os.listdir(path_dir_valid)
file_list_test = os.listdir(path_dir_test)


#트레인 데이터 전처리 
YT_dict_train ={} 
for f_name in file_list_train: 
    df = pd.read_csv(path_dir_train+'/'+f_name)
    yt_name = f_name[:-4]
    
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
    
    
    YT_dict_train[yt_name] = df

############################################

#검증 데이터 전처리 
YT_dict_valid={} 
for f_name in file_list_valid: 
    df = pd.read_csv(path_dir_valid+'/'+f_name)
    yt_name = f_name[:-4]
    
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
    
    
    YT_dict_valid[yt_name] = df
    

############################################

#테스트 데이터 전처리 
YT_dict_test={} 
for f_name in file_list_test: 
    df = pd.read_csv(path_dir_test+'/'+f_name)
    yt_name = f_name[:-4]
    
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
    
    
    YT_dict_test[yt_name] = df
    

############################################




#전체 데이터 합치기 (스케일러 만들기 위해)
df_train = pd.DataFrame()
df_valid = pd.DataFrame()
df_test = pd.DataFrame()

for name in YT_dict_train :
    df_train = pd.concat([df_train, YT_dict_train[name]])

for name in YT_dict_valid :
    df_valid = pd.concat([df_valid, YT_dict_valid[name]])

for name in YT_dict_test :
    df_test = pd.concat([df_test, YT_dict_test[name]])
    


#정규화 
from sklearn.preprocessing import MinMaxScaler,StandardScaler

df_train = df_train.drop(columns=['equ_no','index'], axis=1)
df_valid = df_valid.drop(columns=['equ_no','index'], axis=1)
df_test = df_test.drop(columns=['equ_no','index'], axis=1)


scaler = {}
for c in df_train.columns:
    scaler[c] = StandardScaler()

for c in df_train.columns:
    scaler[c].fit(df_train.loc[:,c].values.reshape(-1,1))



#스케일러 피팅 
ss  = StandardScaler()
ss.fit(df_train)

#시계열 생성 

w_size = 3
p_size = 5

ts_dict_train={} 
for yt in YT_dict_train:
    temp = YT_dict_train[yt].drop(['equ_no','index'], axis=1)
    
    #스케일링 
    temp = pd.DataFrame(ss.transform(temp), columns=temp.columns, index=temp.index)
    
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_train[yt] = [X_s, y_s]

ts_dict_valid={} 
for yt in YT_dict_valid:
    temp = YT_dict_valid[yt].drop(['equ_no','index'], axis=1)
    
    #스케일링 
    temp = pd.DataFrame(ss.transform(temp), columns=temp.columns, index=temp.index)
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_valid[yt] = [X_s, y_s]    

ts_dict_test={} 
for yt in YT_dict_test:
    temp = YT_dict_test[yt].drop(['equ_no','index'], axis=1)
    
    #스케일링 
    temp = pd.DataFrame(ss.transform(temp), columns=temp.columns, index=temp.index)
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_test[yt] = [X_s, y_s]    
    


#시계열 데이터 합치기 
X_train=np.zeros((1,w_size,len(df_train.columns)))
y_train=np.zeros((1,p_size,2))

X_valid=np.zeros((1,w_size,len(df_train.columns)))
y_valid=np.zeros((1,p_size,2))

X_test=np.zeros((1,w_size,len(df_train.columns)))
y_test=np.zeros((1,p_size,2))

for ts in ts_dict_train: 
    X_train = np.concatenate((X_train,ts_dict_train[ts][0]), axis=0)
    y_train = np.concatenate((y_train,ts_dict_train[ts][1]), axis=0)
    
for ts in ts_dict_valid: 
    X_valid = np.concatenate((X_valid,ts_dict_valid[ts][0]), axis=0)
    y_valid = np.concatenate((y_valid,ts_dict_valid[ts][1]), axis=0)
        
for ts in ts_dict_test: 
    X_test = np.concatenate((X_test,ts_dict_test[ts][0]), axis=0)
    y_test = np.concatenate((y_test,ts_dict_test[ts][1]), axis=0)
    
    

#학습 데이터 생성     
X_train = X_train[1:]
y_train = y_train[1:]

X_valid = X_valid[1:]
y_valid = y_valid[1:]

X_test = X_test[1:]
y_test = y_test[1:]



train_data = pp.TensorData(X_train, y_train)
train_loader = DataLoader(train_data, batch_size = 128, shuffle=True)

valid_data = pp.TensorData(X_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size = y_valid.shape[0], shuffle=True)

test_data = pp.TensorData(X_test, y_test)
test_loader = DataLoader(test_data, batch_size = y_test.shape[0], shuffle=False)







#모델 구축 
# train parameter
num_epoch = 500
hid_dim = 256

in_dim = X_train.shape[-1]
out_dim = p_size*2


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
m = models.RNN(input_size = in_dim, hidden_size = hid_dim, out_dim = out_dim, num_layers = 1, seq_length=w_size).to(device)


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

columns =[]
for i in range(1,p_size+1):
    columns.append('lat'+str(i))
    columns.append('long'+str(i))
    
real = pd.DataFrame(real, columns = columns)

#######################여기 밑에 다시 확인 
min_val_loss = np.Inf
    
#에포크 별 loss 저장 
epoch_train_loss = []
epoch_val_loss = []
   
start = time.time() 
print("Begin training...")

for e in range(1, num_epoch+1):
    running_train_loss = 0.0
    running_val_loss = 0.0
    
    train_loss_arr = []
    val_loss_arr = [] 
    #Training Loop
    m.train()
    for _, samples in enumerate(train_loader): 
        X_train, y_train = samples
        
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        
        optimizer.zero_grad()
        
        train_out = m(X_train)
        
        y_train = y_train.reshape(-1,out_dim)
        
        train_loss = crit(train_out, y_train)
        train_loss.backward()
        optimizer.step()
        
        running_train_loss += train_loss.item()
        train_loss_arr.append(train_loss.item())
        
    #해당 애포크에서의 최종 train loss
    train_loss_value = running_train_loss / len(train_loader)
    epoch_train_loss.append(train_loss_value)
    
    #검증
    m.eval()
    with torch.no_grad():
        for _, samples in enumerate(valid_loader): 
            X_val, y_val = samples
            
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            
            
            val_out = m(X_val)
            y_val = y_val.reshape(-1,out_dim)
            
            val_loss = crit(val_out, y_val)
            
            running_val_loss += val_loss.item()
            val_loss_arr.append(val_loss.item())
        #해당 애포크에서의 최종 valid loss
        val_loss_value = running_val_loss / len(valid_loader)
        epoch_val_loss.append(val_loss_value)
        
    print(f'Epoch {e} \t\t Training Loss: {train_loss_value} \t\t Validation Loss: {val_loss_value}')  
    
        

    #val loss가 감소하면 모델 저장 
    if min_val_loss > val_loss:
            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
            min_val_loss = val_loss
            
            torch.save(m.state_dict(),'best_RNN3.pth')      
            
    
    #테스트 
    with torch.no_grad():
        for idx, samples in enumerate(test_loader):
            X_test, y_test = samples
            
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            
        
            out = m(X_test)
        
            out = out.detach().cpu().numpy()
            
            
            
            
            #역변환 
        
            lat = scaler[0].inverse_transform(out[:,list(range(0,len(columns),2))].reshape(-1,p_size))
            long = scaler[1].inverse_transform(out[:,list(range(1,len(columns)+1,2))].reshape(-1,p_size)) 
            
            
            #예측 결과 (lat, long을  v 형태로 만들어야 함)
            #pred = pd.DataFrame(np.concatenate((lat,long),axis=1), columns = ['latitude','longitude'])
        
            pred = pd.DataFrame()
            
            for i in range(p_size):
                pred['lat'+str(i+1)] = lat[:,i]
                pred['long'+str(i+1)] = long[:,i]
                
                
        
        
        #평균 오차 거리 계산 
        dis_diff=[]
        
        #1초 뒤 예측 거리 차이 
        dis_diff = pd.DataFrame()
        for i in range(1, p_size+1):
            a = pred.loc[:, ['lat'+str(i),'long'+str(i)]]
            b = real.loc[:, ['lat'+str(i),'long'+str(i)]]
            dis_diff['pred'+str(i)] = haversine_vector(a,b,unit='m')
        
        mean_diff = np.mean(dis_diff)
        
        #예측 오차 3m이상 나는 애들 마스킹 
        #mask = dis_diff.dis_diff>2
        
       
        
        #테스트 시각화 
        if e%50==0:
            fig, ax = plt.subplots(1,1, figsize=(10,25))
            
            #3초 후 값만 출력
            sns.scatterplot(data=pred, x='long'+str(p_size), y='lat'+str(p_size), ax=ax, color='r')
            
            #예측값인데 오차 3m 이하인 애들
            #sns.scatterplot(data=pred[~mask], x='longitude', y='latitude', ax=ax, color='green')
            
            sns.scatterplot(data=real, x='long'+str(p_size), y='lat'+str(p_size), ax=ax, color='b')
            
            ax.set(title ='average dis diff: '+str(round(np.mean(mean_diff),3))+'\n'+str(round(mean_diff,3))+'\nepoch: '+str(e)+' - RNN')
            
            #ax.set(title='Average dis diff: '+str(mean_diff))
            plt.show()


#pred[['diff1', 'diff2', 'diff3','diff4', 'diff5']] = dis_diff
pred[['diff'+str(i) for i in range(1,p_size+1)]] = dis_diff

    
            
#전체 학습  train / val loss 시각화 
plt.plot(epoch_train_loss, 'blue', label = 'train')
plt.plot(epoch_val_loss, 'red', label = 'valid')
plt.ylim(0,0.00005)
plt.legend()
plt.title('RNN train result')
plt.show()

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간