# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:32:50 2022

@author: USER
"""


import torch 
import torch.nn as nn
from torch.autograd import Variable
from statsmodels.tsa.seasonal import STL

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

crit = torch.nn.MSELoss()


def train(num_epochs, train_loader, val_loader, m, optimizer):
    min_val_loss = np.Inf
    
    #에포크 별 loss 저장 
    epoch_train_loss = []
    epoch_val_loss = []
    
    print("Begin training...")
    
    for epoch in range(1, num_epochs+1):
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
            
            train_out = m(X_train, y_train)
    
            train_loss = crit(train_out,y_train)
            train_loss.backward()
            optimizer.step()
            
            running_train_loss += train_loss.item()
            train_loss_arr.append(train_loss.item())
        
        #해당 애포크에서의 최종 train loss
        train_loss_value = running_train_loss / len(train_loader)
        epoch_train_loss.append(train_loss_value)
        
        #Validation Loop
        m.eval()
        with torch.no_grad():
            m.eval()
            for _, samples in enumerate(val_loader):
                X_val, y_val = samples
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                
                val_out = m(X_val, y_val)
                
                val_loss = crit(val_out, y_val)
                
                running_val_loss += val_loss.item()
                val_loss_arr.append(val_loss.item())
            
            #해당 애포크에서의 최종 valid loss
            val_loss_value = running_val_loss / len(val_loader)
            epoch_val_loss.append(val_loss_value)
                
        print(f'Epoch {epoch} \t\t Training Loss: {train_loss_value)} \t\t Validation Loss: {val_loss_value}')
        
        #해당 에포크의 배치 별 train / val loss 시각화 
        plt.plot(train_loss_arr, 'blue', label = 'train')
        plt.plot(val_loss_arr, 'red', label = 'valid')
        plt.legend()
        plt.show()
        
        
        if min_val_loss > val_loss:
            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
            min_val_loss = val_loss
            
            torch.save(m.state_dict(),'saved_model.pth')
        
    #전체 학습  train / val loss 시각화 
    plt.plot(epoch_train_loss, 'blue', label = 'train')
    plt.plot(epoch_val_loss, 'red', label = 'valid')
    plt.legend()
    plt.show()
    