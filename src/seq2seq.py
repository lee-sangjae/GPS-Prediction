# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:52:34 2022

@author: USER
"""


import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#모델 구축 
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x: (batch, sequence_length, feature_num)
        
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hid_dim)).to(device) #은닉상태 0 초기화
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hid_dim)).to(device) #셀 상태 0 초기화 
        
        outputs, (hidden, cell) = self.rnn(x, (h_0, c_0))
        
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim,  n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers


        # embedding을 입력받아 hid_dim 크기의 hidden state, cell 출력
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        
        
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        
        #batch, output_dim
        prediction = self.fc_out(hidden.squeeze(0))
        
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # encoder와 decoder의 hid_dim이 일치하지 않는 경우 에러메세지
        assert encoder.hid_dim == decoder.hid_dim, \
            'Hidden dimensions of encoder decoder must be equal'
        # encoder와 decoder의 hid_dim이 일치하지 않는 경우 에러메세지
        assert encoder.n_layers == decoder.n_layers, \
            'Encoder and decoder must have equal number of layers'

    def forward(self, x, y, is_Train = True, teacher_forcing_ratio = 0.5):
        
        
        batch_size = x.size(0)
        
        output_len = y.size(1) #예측할 시퀀스 길이(몇 시점 뒤)
       
        
        #값 하나 뽑을 거면 1될 거고, 위경도 뽑을 거면 2개 되겠지 
        output_feature_num = self.decoder.output_dim 
       

        # decoder의 output을 저장하기 위한 tensor(한 시점씩 예측)
        #outputs = torch.zeros(output_len, batch_size, output_feature_num).to(self.device)
        outputs = torch.zeros(batch_size, output_len, output_feature_num).to(self.device)
        

        # initial hidden state
        #(num_layer, batch, hidden)
        hidden, cell = self.encoder(x)
        
        
        # 첫 번째 입력값은 인코더 인풋의 마지막 시퀀스 값 
        #1:3은 디코더에 위경도가 들어가야 하니까 (요 밑에 상황에 따라 바꿔야 함 )
        input_ = x[:,-1:,1:3] #(batch, 1, feature_num(위경도))
        
        

        for t in range(0,output_len): 
            prediction, hidden, cell = self.decoder(input_, hidden, cell)
            

            # prediction 저장 *** 요기가 지금 문제  *** => 지금 prediction은 배치마다 하나씩 값 뽑아놨는데 이게 하나씩 들어가야 함 
            #prediction은 각배치에 대한 t번째 시퀀스 예측값 
            outputs[:,t,:] = prediction
            prediction = prediction.unsqueeze(1)
            
            if is_Train:
                # random.random() : [0,1] 사이 랜덤한 숫자 
                # 랜덤 숫자가 teacher_forcing_ratio보다 작으면 True니까 teacher_force=1
                teacher_force = random.random() < teacher_forcing_ratio
                
    
                #input_이 디코더의 다음 시퀀스 입력으로 들어감
                #y[t]도 각 배치에 대한 t번째 시퀀스 실제 값이 돼야 함             
                input_ = y[:,t:t+1,:] if teacher_force else prediction
            else:
                input_ = prediction
            
        return outputs