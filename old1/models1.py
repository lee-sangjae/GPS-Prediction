# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:01:25 2021

@author: SIM
"""

import torch 
import torch.nn as nn
from torch.autograd import Variable
from statsmodels.tsa.seasonal import STL

import pandas as pd
import numpy as np



 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*num_layers, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #셀 상태 0 초기화 
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        
        
        hn = hn.view(-1, self.hidden_size*self.num_layers).to(device)
        
        
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
    
class LSTM2(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, out_dim, num_layers, seq_length):
        super(LSTM2, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*seq_length, out_dim)
        #self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ELU()
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #셀 상태 0 초기화 
        
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        
        
        output = output.reshape(-1, self.hidden_size*self.seq_length).to(device)
        
        
        #out = self.relu(output)
        out = self.fc_1(output)
        out = self.relu(out)
        #out = self.fc(out)
        #out = self.relu(out)

        return out


class _LSTM2(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(_LSTM2, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hid_dim, num_layers=self.num_layers, batch_first=True)
        
        self.linear = nn.Sequential(
            nn.Linear(self.hid_dim*self.num_layers*3, self.out_dim),
            nn.ELU()
        )
        
    def forward(self, x):
        
        hid_state = Variable(torch.zeros(self.num_layers, x.size(0), self.hid_dim)).cuda()
        cell_state = Variable(torch.zeros(self.num_layers, x.size(0), self.hid_dim)).cuda()
        
        #out은 전체 히든, hid_out은 마지막 히든 
        out, (hid_out, _) = self.lstm(x, (hid_state, cell_state))
        out = out.reshape(-1,self.hid_dim*3)
        
        out = self.linear(out)
        return out
        
class RNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size,out_dim, num_layers, seq_length):
        super(RNN, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*seq_length,out_dim)
        #self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        
        
        
        output, hn = self.rnn(x, h_0)
        
        
        output = output.reshape(-1, self.hidden_size*self.seq_length).to(device)
        
        
        ##out = self.relu(output)
        out = self.fc_1(output)
        out = self.relu(out)
        #out = self.fc(out)
        #out = self.relu(out)
        return out        