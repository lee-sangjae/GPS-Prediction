# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:07:42 2022

@author: USER
"""


import torch 
import torch.nn as nn
from torch.autograd import Variable

import pandas as pd
import numpy as np



 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Convolution_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim, num_layers, seq_length):
        super(Convolution_LSTM, self).__init__()
        #self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.conv1 = nn.Conv1d(in_channels = input_size, out_channels = 2, kernel_size = 2, stride = 1 )

        self.lstm = nn.LSTM(input_size = 2, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*4, out_dim)
        #self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ELU()
        
    def forward(self, x):
        x = x.transpose(1,2)
        
        feature_map = self.conv1(x)
        
        feature_map = feature_map.transpose(1,2)
        
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #셀 상태 0 초기화 
        
        
        output, (hn, cn) = self.lstm(feature_map, (h_0, c_0))
        
        output = output.reshape(-1, self.hidden_size*4).to(device)
        
        out = self.fc_1(output)
        
        out = self.relu(out)

        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*seq_length, out_dim)
        self.relu = nn.ELU()
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #셀 상태 0 초기화 
        
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        
        
        output = output.reshape(-1, self.hidden_size*self.seq_length).to(device)
        
        out = self.fc_1(output)
        
        out = self.relu(out)
        
        return out 
    
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim, num_layers, seq_length):
        super(GRU, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*seq_length, out_dim)
       
        self.relu = nn.ELU()
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        
        output, hn = self.gru(x, h_0)
        
        output = output.reshape(-1, self.hidden_size*self.seq_length).to(device)
        
        out = self.fc_1(output)
        
        out = self.relu(out)
    
        return out
    
class RNN(nn.Module):
    def __init__(self,  input_size, hidden_size, out_dim, num_layers, seq_length):
        super(RNN, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*seq_length, out_dim)
        
        self.relu = nn.ELU()
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        
        output, hn = self.rnn(x, h_0)
        
        output = output.reshape(-1, self.hidden_size*self.seq_length).to(device)
        
        out = self.fc_1(output)
        
        out = self.relu(out)
    
        return out
    
    
