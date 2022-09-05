import pandas as pd
import numpy as np

from tqdm import tqdm 
import random
import argparse
import time 

import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy # Add Deepcopy for args

import src.preprocess as pp
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


import os



parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, 'GPS\\data')

data = {} 
data['train'] = pd.read_csv(os.path.join(data_dir, 'train.csv'))
data['val'] = pd.read_csv(os.path.join(data_dir, 'valid.csv'))
data['test'] = pd.read_csv(os.path.join(data_dir, 'test.csv'))

#칼럼 별 스케일러 생성 및 fitting  
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = {}
for c in data['train'].columns:
    scaler[c] = StandardScaler()

for c in data['train'].columns:
    scaler[c].fit(data['train'].loc[:,c].values.reshape(-1,1))



# ====== Random Seed Initialization ====== #
seed = 37
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "exp1_lr"
args.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# ====== Data Loading ====== #
args.batch_size = 32
args.x_frames = 10
args.y_frames = 5

# ====== Model Capacity ===== #
args.enc_in_dim = 6
args.dec_in_dim = 2

args.enc_hid_dim = 64
args.dec_hid_dim = 256

args.enc_out_dim = 1
args.dec_out_dim = 2

args.n_layers = 1

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.2
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'Adam' #'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0001
args.epoch = 2
args.act = 'tanh'

#데이터 셋으로 변환 
data_set = {} 
for partition in data:    
    df = data[partition].copy()
    #스케일링 
    for c in scaler:
        df.loc[:, c] = scaler[c].transform(df[[c]])
    
    X = df.copy() 
    y = df[['latitude', 'longitude']]
    
    data_set[partition] = pp.SeqDataset(X, y, args.x_frames, args.y_frames)

    
#데이터 로더 변환 
data_loader = {} 
for partition in data_set: 
    
    
    if partition == 'test':
        data_loader[partition] = DataLoader(data_set[partition], batch_size = len(data_set[partition]), shuffle = False)   
    
    else:
        data_loader[partition] = DataLoader(data_set[partition], batch_size = 32, shuffle = True, drop_last = True)   
    


columns =[]
for i in range(1,args.y_frames+1):
    columns.append('lat'+str(i))
    columns.append('long'+str(i))


lat_min = data['train']['latitude'].min()
lat_max = data['train']['latitude'].max()

long_min = data['train']['longitude'].min()
long_max = data['train']['longitude'].max()    


#모델 로드 및 테스트 
import pickle

with open('model(Attention)_result.pkl', 'rb') as f:
    model_result = pickle.load(f)    

model_result['Attention'] = model_result['0.0001_ELU_256']
del(model_result['0.0001_ELU_256'])

#del(model_result['Attention'])

def test(model_name, model, data_loader, args):
    model.eval()
    
    if model_name=='Attention':
        running_test_loss = 0.0
        with torch.no_grad(): 
            for _, sample in enumerate(data_loader['test']):
                X, y = sample 
                
                #데이터 GPU에 올리기 
                X = X.swapaxes(0,1).to(args.device)
                y = y.swapaxes(0,1).to(args.device)
                
                out, attention = model(X,y,False)
                print('ㅁㄴㅇㄹ')
                
                out = out.reshape(y.size(1),-1)
                y = y.swapaxes(0,1)
                y = y.reshape(y.size(0), -1)
               
                
                out = out.detach().cpu().numpy()
                
                y = y.detach().cpu().numpy()
                
                #출력값 역변환 
                lat = scaler['latitude'].inverse_transform(out[:,list(range(0,len(columns),2))].reshape(-1, args.y_frames))
                long = scaler['longitude'].inverse_transform(out[:,list(range(1,len(columns)+1,2))].reshape(-1, args.y_frames)) 
                
                #예측 결과 (lat, long을  v 형태로 만들어야 함)
                pred = pd.DataFrame()
                
                for i in range(args.y_frames):
                    pred['lat'+str(i+1)] = lat[:,i]
                    pred['long'+str(i+1)] = long[:,i]
                    
                    
                #파일 저장 
                pred.to_csv('{}_pred.csv'.format(model_name), index=False)
    
    else:
        running_test_loss = 0.0
        with torch.no_grad(): 
            for _, sample in enumerate(data_loader['test']):
                X, y = sample 
                
                #데이터 GPU에 올리기 
                X = X.to(args.device)
                y = y.to(args.device)
                
                out = model(X, y, False).reshape(y.size(0), -1)
                y = y.reshape(y.size(0), -1)
                
                out = out.detach().cpu().numpy()
                
                y = y.detach().cpu().numpy()
                
                
                #출력값 역변환 
                lat = scaler['latitude'].inverse_transform(out[:,list(range(0,len(columns),2))].reshape(-1, args.y_frames))
                long = scaler['longitude'].inverse_transform(out[:,list(range(1,len(columns)+1,2))].reshape(-1, args.y_frames)) 
                
                #예측 결과 (lat, long을  v 형태로 만들어야 함)
                pred = pd.DataFrame()
                
                for i in range(args.y_frames):
                    pred['lat'+str(i+1)] = lat[:,i]
                    pred['long'+str(i+1)] = long[:,i]
                    
                #파일 저장 
                pred.to_csv('{}_pred.csv'.format(model_name), index=False)
            
#모델별 베스트 선정 
for model in model_result:
     
    md= model_result[model]
    best_loss = np.Inf
    best_model = None
    
    #실험 파라미터 종류 
    
    val_loss = md['final_val_loss']
    
    #해당 모델 학습 그래프 
    epoch_train_loss = md['train_loss']
    epoch_val_loss = md['val_loss']
    
    #훈련 로스 시각화 
    args  = md['args']
    plt.plot(epoch_train_loss, 'blue', label='train')
    plt.plot(epoch_val_loss, 'red', label= 'valid')
    #plt.ylim(0,0.01)
    plt.legend()
    plt.title('lr: {}, act: {}, hid_dim: {}'.format( args.lr, args.act, args.enc_hid_dim))
    plt.show()
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = md['model']

    
    test(model, best_model, data_loader, args)
    
            
        
 
        
        
        
        
      
    
   
    
        