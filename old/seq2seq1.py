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
import torchvision.transforms as tr #데이터 불러오면서 전처리를 가능한게 해주는 라이브러리 
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from haversine import haversine_vector, haversine
import src.preprocess as pp
import src.seq2seq as seq2seq
from sklearn.metrics import mean_squared_error as mse
from torch.autograd import Variable 
import random

import matplotlib.pyplot as plt
import seaborn as sns

import time
import os 
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
    df.set_index('evnt_dt',inplace=True)
    
    """
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
    """
    
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
    
    
    YT_dict_valid[yt_name] = df
    

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
    
    
    YT_dict_test[yt_name] = df
    

############################################




#전체 데이터 합치기 
df_train = pd.DataFrame()
df_valid = pd.DataFrame()
df_test = pd.DataFrame()

for name in YT_dict_train :
    df_train = pd.concat([df_train, YT_dict_train[name]])

for name in YT_dict_valid :
    df_valid = pd.concat([df_valid, YT_dict_valid[name]])

for name in YT_dict_test :
    df_test = pd.concat([df_test, YT_dict_test[name]])
    


#정규화 (이상치 나름 제거 했으니 minmax로)
from sklearn.preprocessing import MinMaxScaler

df_train = df_train.drop(columns=['equ_no','index'], axis=1)
df_valid = df_valid.drop(columns=['equ_no','evnt_dt'], axis=1)
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

w_size = 5
p_size = 5

ts_dict_train={} 
for yt in YT_dict_train:
    temp = YT_dict_train[yt].drop(['equ_no','index'], axis=1)
    
    #스케일링 
    temp = pd.DataFrame(mm.transform(temp), columns=temp.columns, index=temp.index)
    
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_train[yt] = [X_s, y_s]

ts_dict_valid={} 
for yt in YT_dict_valid:
    temp = YT_dict_valid[yt].drop(['equ_no','evnt_dt'], axis=1)
    
    #스케일링 
    temp = pd.DataFrame(mm.transform(temp), columns=temp.columns, index=temp.index)
    
    #시계열 생성 
    X = temp
    y = temp[['latitude','longitude']]    
    
    X_s, y_s = pp.make_sequene_train_dataset(X, y, w_size, p_size)
    ts_dict_valid[yt] = [X_s, y_s]    

ts_dict_test={} 
for yt in YT_dict_test:
    temp = YT_dict_test[yt].drop(['equ_no','evnt_dt'], axis=1)
    
    #스케일링 
    temp = pd.DataFrame(mm.transform(temp), columns=temp.columns, index=temp.index)
    
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
num_epoch = 3000
hid_dim = 256

in_dim = X_train.shape[-1]
out_dim = 2





#y_test를 쭉 펼치고 
real = y_test.reshape((-1,2))

real = np.concatenate((scaler[0].inverse_transform(real[:,0].reshape(-1,1)),
                       scaler[1].inverse_transform(real[:,1].reshape(-1,1))),axis=1)

real = real.reshape(-1,p_size*2)

real = pd.DataFrame(real, columns = ['lat1','long1','lat2','long2','lat3','long3','lat4','long4','lat5', 'long5'])



import src.seq2seq as seq2seq
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
encoder = seq2seq.Encoder(input_dim = in_dim, hid_dim=hid_dim, n_layers=1, dropout=0.2,)
decoder = seq2seq.Decoder(input_dim =2,  output_dim=out_dim, hid_dim=hid_dim, n_layers=1, dropout=0.2)
m = seq2seq.Seq2Seq(encoder, decoder, device).to(device)

crit = torch.nn.MSELoss()
para = list(m.parameters())
optimizer = optim.Adam(para, 0.0001)
final_test_rmse = 10e+10
final_test_mape = 0


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
        
        train_out = m(X_train, y_train, teacher_forcing_ratio = 0.5)
        
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
            
            
            val_out = m(X_val, y_val, teacher_forcing_ratio = 0.5)
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
            
            torch.save(m.state_dict(),'best_seq2seq2.pth')      
            
    
    #테스트 
    with torch.no_grad():
        for idx, samples in enumerate(test_loader):
            X_test, y_test = samples
            
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            
        
            out = m(X_test, y_test)

            out = out.reshape(-1, p_size*out_dim)   
        
            out = out.detach().cpu().numpy()
            
            
            

            #역변환 
            lat = scaler[0].inverse_transform(out[:,[0,2,4,6,8]].reshape(-1,p_size))
            long = scaler[1].inverse_transform(out[:,[1,3,5,7,9]].reshape(-1,p_size)) 
            
            
            #예측 결과 (lat, long을  v 형태로 만들어야 함)
            #pred = pd.DataFrame(np.concatenate((lat,long),axis=1), columns = ['latitude','longitude'])
            
            pred = pd.DataFrame({'lat1':lat[:,0],'long1':long[:,0],
                                 'lat2':lat[:,1],'long2':long[:,1],
                                 'lat3':lat[:,2],'long3':long[:,2],
                                 'lat4':lat[:,3],'long4':long[:,3],
                                 'lat5':lat[:,4],'long5':long[:,4]})
        
        
        #평균 오차 거리 계산 
        dis_diff=[]
        
        #1초 뒤 예측 거리 차이 
        a1 = pred.loc[:,['lat1','long1']]
        b1 = real.loc[:,['lat1','long1']]
        dis1 = haversine_vector(a1,b1,unit='m')
        
        a2 = pred.loc[:,['lat2','long2']]
        b2 = real.loc[:,['lat2','long2']]
        dis2 = haversine_vector(a2,b2,unit='m')
        
        a3 = pred.loc[:,['lat3','long3']]
        b3 = real.loc[:,['lat3','long3']]
        dis3 = haversine_vector(a3,b3,unit='m')
        
        a4 = pred.loc[:,['lat4','long4']]
        b4 = real.loc[:,['lat4','long4']]
        dis4 = haversine_vector(a4,b4,unit='m')
        
        a5 = pred.loc[:,['lat5','long5']]
        b5 = real.loc[:,['lat5','long5']]
        dis5 = haversine_vector(a5,b5,unit='m')
        
        
        
        dis_diff = pd.DataFrame(columns=['pred1','pred2','pred3','pred4', 'pred5'])
        dis_diff['pred1'] = dis1
        dis_diff['pred2'] = dis2
        dis_diff['pred3'] = dis3
        dis_diff['pred4'] = dis4
        dis_diff['pred5'] = dis5
        
        mean_diff = np.mean(dis_diff)
        
        #예측 오차 3m이상 나는 애들 마스킹 
        #mask = dis_diff.dis_diff>2
        
       
        
        #테스트 시각화 
        if e%50==0:
            fig, ax = plt.subplots(1,1, figsize=(10,25))
            
            #3초 후 값만 출력
            sns.scatterplot(data=pred, x='long5', y='lat5', ax=ax, color='r')
            
            #예측값인데 오차 3m 이하인 애들
            #sns.scatterplot(data=pred[~mask], x='longitude', y='latitude', ax=ax, color='green')
            
            sns.scatterplot(data=real, x='long5', y='lat5', ax=ax, color='b')
            
            ax.set(title ='average dis diff: '+str(round(np.mean(mean_diff),3))+'\n'+str(round(mean_diff,3))+'\nepoch: '+str(e)+' - seq2seq')
            
            #ax.set(title='Average dis diff: '+str(mean_diff))
            plt.show()

pred[['diff1', 'diff2', 'diff3','diff4', 'diff5']] = dis_diff

    
            
#전체 학습  train / val loss 시각화 
plt.plot(epoch_train_loss, 'blue', label = 'train')
plt.plot(epoch_val_loss, 'red', label = 'valid')
plt.ylim(0,0.00004)
plt.legend()
plt.title('Seq2Seq train result')
plt.show()
    
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간