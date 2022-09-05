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


class _LSTM(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(_LSTM, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hid_dim, num_layers=self.num_layers, batch_first=True)
        
        self.linear = nn.Sequential(
            nn.Linear(self.hid_dim*self.num_layers, self.out_dim),
            nn.ELU()
        )
        
    def forward(self, x):
        
        hid_state = Variable(torch.zeros(self.num_layers, x.size(0), self.hid_dim)).cuda()
        cell_state = Variable(torch.zeros(self.num_layers, x.size(0), self.hid_dim)).cuda()
        
        _, (hid_out, _) = self.lstm(x, (hid_state, cell_state))
        hid_out = hid_out.squeeze()
        out = self.linear(hid_out)
        return out

class _LSTM2(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, device):
        super(_LSTM2, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hid_dim, num_layers=self.num_layers, batch_first=True)
        
        self.linear = nn.Sequential(
            nn.Linear(self.hid_dim*self.num_layers*3, self.out_dim),
            nn.ELU()
        )
        
    def forward(self, x):
        
        if self.device == "cpu":
            hid_state = Variable(torch.zeros(self.num_layers, x.size(0), self.hid_dim))
            cell_state = Variable(torch.zeros(self.num_layers, x.size(0), self.hid_dim))
        else:
            hid_state = Variable(torch.zeros(self.num_layers, x.size(0), self.hid_dim)).cuda()
            cell_state = Variable(torch.zeros(self.num_layers, x.size(0), self.hid_dim)).cuda()
        
        #out은 전체 히든, hid_out은 마지막 히든 
        out, (hid_out, _) = self.lstm(x, (hid_state, cell_state))
        out = out.reshape(-1,self.hid_dim*3)
        
        out = self.linear(out)
        
        
        return out