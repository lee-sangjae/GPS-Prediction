# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:18:10 2022

@author: USER
"""

import torch
import torchvision.datasets as dset
from torch.utils.data import TensorDataset,DataLoader, Dataset
import numpy as np


class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data) #들어온 데이터를 텐서로 
        #self.x_data = self.x_data.permute(0,3,1,2) #이미지 개수, 채널 수, 이미지 너비, 높이
        self.y_data = torch.FloatTensor(y_data)  #들어온 데이터를 텐서로 
        self.len = self.y_data.shape[0]
        
    def __getitem__ (self, index): #
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len


def make_sequene_train_dataset(feature, label, window_size, predict_size):

    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list

    for i in range(len(feature)-window_size-predict_size+1):

        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size:i+window_size+predict_size]) #predict만큼 여러 개 
        #label_list.append(label[i+window_size+predict_size-1:i+window_size+predict_size]) #predict이후 시점 딱 한개 
    return np.array(feature_list), np.array(label_list)
    
    
    
# 시퀀스 데잍터 셋 생성 =========================================================
class SeqDataset(Dataset):
    
    def __init__(self, X, y, x_frames, y_frames):
        
        self.X = X
        self.y = y
        
        self.x_frames = x_frames
        self.y_frames = y_frames
        
        
    def __len__(self):
        return len(self.X) - (self.x_frames + self.y_frames) + 1
    
    #dataset에서 인덱스로 하나씩 꺼내오는 기능 
    def __getitem__(self, idx):
        idx += self.x_frames
        
        X = self.X.iloc[idx-self.x_frames:idx].values
        y = self.y.iloc[idx:idx+self.y_frames].values
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
    

        return X, y 