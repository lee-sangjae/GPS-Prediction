# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:45:47 2022

@author: LSJ
"""

# =============================================================================
# 
# =============================================================================


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
args.epoch = 500
args.act = 'ELU'


#real / present 생성 
df = data['test'].copy()
X = df.copy() 
y = df[['latitude', 'longitude']]
test_set = pp.SeqDataset(X, y, args.x_frames, args.y_frames)

real = np.zeros((len(test_set), args.y_frames*2))
present = np.zeros((len(test_set), args.x_frames*2))
for i, sample in enumerate(test_set):
    if i==len(test_set):
        break;
    X, y = sample 
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()    
    
    X = X[:,0:2]
    y = y[:,:]
    
    X = X.flatten()
    y = y.flatten()
    
    present[i] = X
    real[i] = y

columns =[]
for i in range(1,args.y_frames+1):
    columns.append('lat'+str(i))
    columns.append('long'+str(i))

real = pd.DataFrame(real, columns = columns)
present = pd.DataFrame(present)    

real.to_csv('real.csv', index= False)
present.to_csv('present.csv', index = False)
##==================================================##


#칼럼 별 스케일러 생성 및 fitting  
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = {}
for c in data['train'].columns:
    scaler[c] = StandardScaler()

for c in data['train'].columns:
    scaler[c].fit(data['train'].loc[:,c].values.reshape(-1,1))


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


def train(model, data_loader, optimizer, loss_fn, args):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    
    #mini-batch loss
    running_train_loss = 0.0 
    for _, sample in enumerate(data_loader['train']):
        X, y = sample 
        
        #데이터 GPU에 올리기 
        X = X.to(args.device)
        y = y.to(args.device)
        
        #initial gradient 
        model.zero_grad()
        optimizer.zero_grad()
        
        out = model(X)
        y = y.reshape(y.size(0), -1)
    
        #calculate loss 
        loss = loss_fn(out.flatten(), y.flatten())
        
        #backpropagation
        loss.backward()
        
        #parameter update
        optimizer.step() 
        
        #loss 누적 
        running_train_loss = loss.item() 
        
    running_train_loss = running_train_loss / len(data_loader['train'])
        
    return model, running_train_loss 

def validation(model, data_loader, loss_fn, args):
    model.eval()
    
    #mini-batch loss
    running_val_loss = 0.0
    
    with torch.no_grad(): 
        for _, sample in enumerate(data_loader['val']):
            X, y = sample 
            
            #데이터 GPU에 올리기 
            X = X.to(args.device)
            y = y.to(args.device)
            
            
            out = model(X)
            y = y.reshape(y.size(0), -1)
        
            
            #calculate loss 
            loss = loss_fn(out.flatten(), y.flatten())
    
            #loss 누적 
            running_val_loss = loss.item() 
            
        running_val_loss = running_val_loss / len(data_loader['val'])

    return running_val_loss 


def learning(model, partition, args):
    
    #모델 별 파라미터에 따른 결과 저장 
    result = {} 

    loss_fn = torch.nn.MSELoss()

    
    if args.optim == 'SGD':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')
    
    # epoch 별 loss
    epoch_train_loss = []
    epoch_val_loss = []
    min_val_loss = np.Inf
    
    for e in tqdm(range(1, args.epoch+1), desc='train processing...'):  # loop over the dataset multiple times
        ts = time.time()
        
        model, train_loss = train(model, data_loader, optimizer, loss_fn, args)
        val_loss = validation(model, data_loader, loss_fn, args)
        
        #val loss가 감소하면 모델 저장 
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            
            #모델 저장 
            result = {'args': args,  'final_val_loss': val_loss, 'model': model, }
            
            
        te = time.time()
        
        # ====== Add Epoch Data ====== #
        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss)
        # ============================ #
        
        print('Epoch {}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec'.format(e, train_loss, val_loss, te-ts))
    

    # ======= Add Result to Dictionary ======= #
    result['train_loss'] = epoch_train_loss
    result['val_loss'] = epoch_val_loss
    
    
    return result


# ====== Experiment Variable ====== #
name_var1 = 'lr'
name_var2 = 'x_frames'
name_var3 = 'y_frames'

list_var1 = [0.0001]
list_var2 = [10,15]
list_var3 = [ 2,5 ]

import src.model as md
model = {} 
model['RNN'] = md.RNN(6, args.enc_hid_dim , args.y_frames*2, args.n_layers, args.x_frames).to(args.device)
model['LSTM'] = md.LSTM(6, args.enc_hid_dim , args.y_frames*2, args.n_layers, args.x_frames).to(args.device)
model['GRU'] = md.GRU(6, args.enc_hid_dim , args.y_frames*2, args.n_layers, args.x_frames).to(args.device)
#model['Conv_LSTM'] = md.Convolution_LSTM(6, args.enc_hid_dim , args.y_frames*2, args.n_layers, args.x_frames).to(args.device)


model_result = {}
for m in model: 
    model_result[m] = {} 
    for var1 in list_var1:
        for var2 in list_var2:
            for var3 in list_var3:
                
                setattr(args, name_var1, var1)
                setattr(args, name_var2, var2)
                setattr(args, name_var3, var3)
                
                #실험 결과
                result = learning(model[m], partition, deepcopy(args))
                
                #훈련 로스 시각화 
                plt.plot(result['train_loss'], 'blue', label='train')
                plt.plot(result['val_loss'], 'red', label= 'valid')
                #plt.ylim(0,0.01)
                plt.legend()
                plt.title('lr: {}, x_frames: {}, y_frames {}'.format(var1, var2, var3))
                plt.show()
                
                model_result[m]['{}_{}_{}'.format(var1, var2, var3)] = result
            

#import pickle
#with open('model(RNN)_result.pkl', 'wb') as f:
#    pickle.dump(model_result, f)

##### 가장 좋은 모델 선정 ##### 
            
            
            


