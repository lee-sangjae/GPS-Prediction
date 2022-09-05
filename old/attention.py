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
#모델 구축 
class Encoder(nn.Module):
    def __init__(self, in_dim, enc_hid_dim, dec_hid_dim, n_layers,  act, dropout=0.2):
        super().__init__()
        
        
        self.in_dim = in_dim

        self.enc_hid_dim = enc_hid_dim

        self.dec_hid_dim = dec_hid_dim

        self.n_layers = n_layers

        #활성화 함수 초기화
        if act == 'relu':
            self.act = nn.ReLU()
            
        elif act =='Sigmoid':
            self.act = nn.Sigmoid()
            
        elif act == 'Tanh':
            self.act=nn.Tanh()
        
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        
        elif act == 'PReLU':
            self.act = nn.PReLU()
            
        elif act == 'ELU':
            self.act = nn.ELU()
        
        else:
            self.act = nn.ReLU()

    
        self.rnn = nn.GRU(in_dim, enc_hid_dim, n_layers, dropout=dropout)
        
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        #x: (batch, sequence_length, feature_num)
        
        h_0 = Variable(torch.zeros(self.n_layers, x.size(1), self.enc_hid_dim)).to(device) #은닉상태 0 초기화
        #c_0 = Variable(torch.zeros(self.n_layers, x.size(1), s elf.enc_hid_dim)).to(device) #셀 상태 0 초기화 
        
        outputs, hidden = self.rnn(x, h_0)
        #outputs, (hidden, cell) = self.rnn(x, (h_0, c_0))
        
        #outputs은 attention 목적, hidden은 context vector 
        hidden = self.act(self.fc(hidden))
        
        return outputs, hidden
    
# 어텐션(Attention) 아키텍처 정의
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hidden_dim  + dec_hidden_dim), dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, enc_outputs):
        # hidden(디코더 히든): [1, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # enc_outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 인코더에서 전체 단어의 출력 정보
        batch_size = enc_outputs.shape[1]
        src_len = enc_outputs.shape[0]
        
        # 현재 디코더의 히든 상태(hidden state)를 src_len만큼 반복
        
        #여기서 히든은 디코더 히든(인코더에 들어간 시퀀스 만큼 디코더 히든도 생성  )
        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1) #(배치, 1, 디코더 히든) 그냥 배치가 맨 앞으로 오도록 조정한 듯 
        #cell = cell.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        
        enc_outputs = enc_outputs.permute(1, 0, 2) #얘도 배치가 맨앞으로 오게 변경 
        
        # hidden(디코더): [배치 크기, (인코더 인풋의) seq_length, 디코더 히든 차원]: 현재까지의 모든 단어의 정보
        # enc_outputs: [배치 크기, seq_length, 인코더 히든 차원 ]: 전체 단어의 출력 정보
        
        #디코더 히든이랑 인코더 아웃풋 이어 붙이고 
        #이어 붙이면 (배치, 인코더 시퀀스, enc_hid + dec_hid)
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        
        # energy: [배치 크기, 인코더 시퀀스, 디코더 히든]
        attention = self.v(energy).squeeze(2)
        # attention: [배치 크기, 시퀀스 길이, 1]을  squeeze(2)로 마지막 차원 줄이기 

        return F.softmax(attention, dim=1) #[배치, 시퀀스] => 각 시퀀스에 대한 어텐션 가중치 
    
class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, enc_hid_dim, dec_hid_dim, dropout_ratio, attention):
        super().__init__()

        self.output_dim = out_dim
        self.attention = attention
        
        # rnn 레이어
        self.rnn = nn.GRU((in_dim+enc_hid_dim), dec_hid_dim)

        # FC 레이어
        self.fc_out = nn.Linear((enc_hid_dim + dec_hid_dim + in_dim), out_dim)
        
        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio) 

    # 디코더는 현재까지 출력된 문장에 대한 정보를 입력으로 받아 타겟 문장을 반환     
    def forward(self, input, hidden, enc_outputs):
        
        # input: [배치 크기, 2(위 경도 )]: 인풋 시퀀스의 마지막 값 하나 
        # hidden: [1, 배치 크기, 디코더  히든 차원]
        # enc_outputs: [length, 배치 크기, 인코더 히든 차원]: 전체 단어의 출력 정보
        
        input = input.unsqueeze(0)
        # input: [시퀀스1, 배치 크기, 2(위경도 )]
    
        attention = self.attention(hidden, enc_outputs)
        # attention: [배치 크기, 시퀀스 ]: 실제 각 시퀀스에 대한 어텐선(attention) 값들
        
        attention = attention.unsqueeze(1)
        # attention: [배치 크기, 1, 시퀀스]: 실제 각 시퀀스에 대한 어텐선(attention) 값들

        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs: [배치 크기, 시퀀스, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보

        weighted = torch.bmm(attention, enc_outputs) # 행렬 곱 함수
        # weighted: [배치 크기, 1, 인코더 히든 차원 * 방향의 수]

        weighted = weighted.permute(1, 0, 2)
        # weighted: [1, 배치 크기, 인코더 히든 차원 * 방향의 수]
        
        rnn_input = torch.cat((input, weighted), dim=2)
        # rnn_input: [1, 배치 크기, 인코더 히든 차원 * 방향의 수 + 1]: 어텐션이 적용된 현재 단어 입력 정보
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [출력 개수, 배치 크기, 디코더 히든 차원 * 방향의 수]
        # hidden: [레이어 개수 * 방향의 수, 배치 크기, 디코더 히든 차원]: 현재까지의 모든 시퀀스 정보 

        # 현재 예제에서는 단어 개수, 레이어 개수, 방향의 수 모두 1의 값을 가짐
        # 따라서 output: [1, 배치 크기, 디코더 히든 차원], hidden: [1, 배치 크기, 디코더 히든 차원]
        # 다시 말해 output과 hidden의 값 또한 동일
        #assert (output == hidden).all()

        input = input.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        attention = attention.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, input), dim=1))
        # prediction = [배치 크기, 출력 차원]
        
        # (현재 출력 단어, 현재까지의 모든 단어의 정보)
        return prediction, hidden, attention   # [배치, 출력], [시퀀스(1), 배치, 디코더 히든]
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src , trg, teacher_forcing_ratio=0.5):
        
        # src: [입력 Sequence_length, 배치 크기, 입력 변수 개수]
        # trg: [출력 Sequence_length, 배치 크기, 출력 변수 개수]
        # 먼저 인코더를 거쳐 전체 출력과 문맥 벡터(context vector)를 추출
        enc_outputs, hidden = self.encoder(src)
        #[입력 시퀀스, 배치, 인코더 히든]: attention 용, [1, 배치, 디코더 히든]: 디코더에 들어갈 히든(context vector..?)

        # 디코더(decoder)의 최종 결과를 담을 텐서 객체 만들기
        trg_len = trg.shape[0] # 출력 sequence_length
        batch_size = trg.shape[1] # 배치 크기
        
        #trg_vocab_size = self.decoder.output_dim # 출력 차원
        out_dim = self.decoder.output_dim # 출력 차원
        
        #(출력 시퀀스, 배치 사이즈, 출력 변수 개수)
        outputs = torch.zeros(trg_len, batch_size, out_dim).to(self.device)
        

        # 첫 번째 입력은 항상 인풋의 마지막 시퀀스 
        #input = trg[0, :]
        #소스의 마지막 시퀀스 위경도만 
        input = src[-1, :, 1:3]

        # 타겟 단어의 개수만큼 반복하여 디코더에 포워딩(forwarding)
        for t in range(1, trg_len):
            output, hidden, attention = self.decoder(input, hidden, enc_outputs)

            outputs[t] = output # FC를 거쳐서 나온 현재의 출력 위경도 정보
            
            # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else output # 현재의 출력 결과를 다음 입력에서 넣기

        return outputs, attention #[출력 시퀀스, 배치, 출력 변수]
    

attn = Attention(128,256).to(device)

enc = Encoder(6,128,256,1,'Tanh').to(device)
dec = Decoder(2,2,128,256,0.3, attn).to(device)

model = Seq2Seq(enc, dec, device)

src = torch.zeros(5, 32, 6).to(device)
trg = torch.zeros(10, 32, 2).to(device)

pred, attention_w = model(src, trg)

enc_outputs, enc_hidden = enc(src)

trg_len = trg.shape[0]
batch_size = trg.shape[1]

out_dim = dec.output_dim

outputs = torch.zeros(trg_len, batch_size, out_dim).to(device)
input = src[-1, :, 1:3]


t=0
for t in range(0, trg_len):
    prediction, dec_hidden, attention = dec(input, enc_hidden, enc_outputs)

    outputs[t] = prediction # FC를 거쳐서 나온 현재의 출력 위경도 정보
    
    # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
    teacher_force = random.random() < 0.5
    
    #[배치, 2]가 다시 인풋으로 들어가야 함 
    input = trg[t ,:, : ] if teacher_force else prediction # 현재의 출력 결과를 다음 입력에서 넣기
    
    
    
    
    
    
    
    
            
    
    
    



