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
p_size = 10

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

#real.to_csv('real(3-5).csv', index=False)
    

#모델 구축 
# train parameter
num_epoch = 3000
hid_dim = 512

in_dim = df_dict['train'].shape[-1]




#모델 학습 
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

#학습 시킬 모델 종류
import src.model as models
model = {}
out_dim = p_size*2
model['RNN'] = models.RNN(input_size = in_dim, hidden_size = hid_dim, out_dim = out_dim, num_layers = 1, seq_length=w_size).to(device)
model['LSTM'] = models.LSTM(input_size = in_dim, hidden_size = hid_dim, out_dim = out_dim, num_layers = 1, seq_length=w_size).to(device)
model['Conv_LSTM'] = models.Convolution_LSTM(input_size = in_dim, hidden_size = hid_dim, out_dim = out_dim, num_layers=1, seq_length=w_size).to(device)
    
out_dim = 2
encoder = seq2seq.Encoder(input_dim = in_dim, hid_dim=hid_dim, n_layers=1, dropout=0.2,)
decoder = seq2seq.Decoder(input_dim =2,  output_dim=out_dim, hid_dim=hid_dim, n_layers=1, dropout=0.2)
model['Seq2Seq'] = seq2seq.Seq2Seq(encoder, decoder, device).to(device)

import src.conv_seq2seq as conv_seq2seq
encoder = seq2seq.Encoder(input_dim = in_dim, hid_dim=hid_dim, n_layers=1, dropout=0.2,)
decoder = seq2seq.Decoder(input_dim =2,  output_dim=out_dim, hid_dim=hid_dim, n_layers=1, dropout=0.2)
model['Conv_Seq2Seq'] = conv_seq2seq.Seq2Seq(encoder, decoder, device).to(device)


min_val_loss = np.Inf

del(model['Conv_Seq2Seq'])
del(model['LSTM'])
del(model['RNN'])
del(model['Seq2Seq'])
    
start = time.time() 
print("Begin training...")


model
for md in model:
    
    #에포크 별 로스 저장 
    epoch_train_loss = []
    epoch_val_loss = []
    
    m = model[md]
    crit = torch.nn.MSELoss()
    para = list(m.parameters())
    optimizer = optim.Adam(para, 0.0001)
    final_test_rmse = 10e+10
    final_test_mape = 0
    
    
    for e in tqdm(range(1, num_epoch+1)):
            running_train_loss = 0.0
            running_val_loss = 0.0
            
            train_loss_arr = []
            val_loss_arr = [] 
            
            #Training Loop
            m.train()
            if (md == 'RNN') | (md=='LSTM') | (md=='Conv_LSTM'):
                out_dim = p_size*2
                for _, samples in enumerate(data_loader['train']): 
                    X_train, y_train = samples
                    
                    
                    
                    X_train = X_train.to(device)
                    y_train = y_train.to(device)
                    optimizer.zero_grad()
                    
                    X_train.size()
                    
                    train_out = m(X_train)
                    
                    y_train = y_train.reshape(-1,out_dim)
                    
                    train_loss = crit(train_out, y_train)
                    train_loss.backward()
                    optimizer.step()
                    
                    running_train_loss += train_loss.item()
                    train_loss_arr.append(train_loss.item())
                    
                #해당 애포크에서의 최종 train loss
                train_loss_value = running_train_loss / len(data_loader['train'])
                epoch_train_loss.append(train_loss_value)
                
                #검증
                m.eval()
                with torch.no_grad():
                    for _, samples in enumerate(data_loader['valid']): 
                        X_val, y_val = samples
                        
                        X_val = X_val.to(device)
                        y_val = y_val.to(device)
                        
                        
                        val_out = m(X_val)
                        y_val = y_val.reshape(-1,out_dim)
                        
                        val_loss = crit(val_out, y_val)
                        
                        running_val_loss += val_loss.item()
                        val_loss_arr.append(val_loss.item())
                    #해당 애포크에서의 최종 valid loss
                    val_loss_value = running_val_loss / len(data_loader['valid'])
                    epoch_val_loss.append(val_loss_value)
    
            
                #val loss가 감소하면 모델 저장 
                if min_val_loss > val_loss:
                        #print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
                        min_val_loss = val_loss
                        torch.save(m, f'./'+md+'({}-{})'.format(w_size,p_size)+'.pt')     
                        
            elif 'Seq2Seq' in md:
                out_dim = 2
                for _, samples in enumerate(data_loader['train']): 
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
                train_loss_value = running_train_loss / len(df_dict['train'])
                epoch_train_loss.append(train_loss_value)
                    
                
                #검증
                m.eval()
                with torch.no_grad():
                    for _, samples in enumerate(data_loader['valid']): 
                        X_val, y_val = samples
                        
                        X_val = X_val.to(device)
                        y_val = y_val.to(device)
                        
                        
                        val_out = m(X_val, y_val, teacher_forcing_ratio = 0.5)
                        val_loss = crit(val_out, y_val)
                        
                        running_val_loss += val_loss.item()
                        val_loss_arr.append(val_loss.item())
                    #해당 애포크에서의 최종 valid loss
                    val_loss_value = running_val_loss / len(data_loader['valid'])
                    epoch_val_loss.append(val_loss_value)
                
            
                #val loss가 감소하면 모델 저장 
                if min_val_loss > val_loss:
                        min_val_loss = val_loss
                        #torch.save(m, f'./'+md+'({}-{})'.format(w_size,p_size)+'.pt')     
    
    
    #전체 학습  train / val loss 시각화 
    plt.plot(epoch_train_loss, 'blue', label = 'train')
    plt.plot(epoch_val_loss, 'red', label = 'valid')
    #plt.ylim(0,0.1)
    plt.legend()
    plt.title(md+'({}-{})'.format(w_size,p_size)+' train result')
    plt.show()
    
