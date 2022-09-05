# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:18:10 2022

@author: USER
"""

import torch
import torchvision.datasets as dset
from torch.utils.data import TensorDataset,DataLoader, Dataset
import numpy as np
import pandas as pd
from haversine import haversine
import config

def to_datetime(data, column):
    data[column] = pd.to_datetime(data[column])
    return data


def set_index(data, column):
    data = data.set_index(column)
    return data


def insert_distance(data, column):
    dis_list = [np.nan]
    for r in range(1,len(data)):
        a = (data.iloc[r-1,1], data.iloc[r-1,2])
        b = (data.iloc[r,1], data.iloc[r,2])
        dis = haversine(a,b, unit='m')
        dis_list.append(dis)

    data[column] = dis_list
    return data


def insert_direction(data, column):
    dir_list = [np.nan]
    for r in range(1,len(data)):
        dir_list.append(data.iloc[r,5] - data.iloc[r-1,5])

    data[column] = dir_list
    return data


def drop_missing(data, subset, how, axis):
    data=data.dropna(subset=subset, how=how, axis=axis)
    return data
    

def drop_distance(data, criterion, how=["up","down"]):
    if how == "up":
        return data[data.distance <= criterion]
    elif how == "down":
        return data[data.distance >= criterion]


def drop_columns(data:pd.DataFrame, columns:list):
    return data.drop(columns)


def interpolate_vars(data, equ_no):
    #결측시 채우기 
    if len(data) != 0:
        ts = pd.date_range(start=data.index[0],
                           end = data.index[-1],
                           freq = 'S')
        
        ts = pd.DataFrame(index = ts)
    
        data = ts.join(data)
        data = data.interpolate()
        # data['equ_no'] = equ_no
    return data
    
def concat_data():
    pass


    



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
    


def make_sequence_train_dataset(feature, label, window_size, predict_size):

    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list

    for i in range(len(feature)-window_size-predict_size+1):

        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size:i+window_size+predict_size]) #predict만큼 여러 개 
        #label_list.append(label[i+window_size+predict_size-1:i+window_size+predict_size]) #predict이후 시점 딱 한개 
    return np.array(feature_list), np.array(label_list)
    
    