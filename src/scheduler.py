# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:39:13 2022

@author: USER
"""

import torch
import numpy as np 

#Learning Rate Schduler 
#=> 검증 데이터셋에 대해 patience 횟수만큼 오차 감소가 없다면 factor만큼 학습률 감소
class LRScheduler():
    #학습과정에서 모델 성능에 대한 개선이 없을 경우 학습률 값을 조절하여 모델의 개선 유도하는 콜백 함수
    #*콜백함수: 개발자가 명시적으로 함수를 호출하는 것이 아님.
    #개발자가 단지 함수 등록만 하고 특정 이벤트 발생에 의해 함수를 호출하고 처리하도록 하는 것이 콜백 함수 
    def __init__(self, optimizer, patience, min_lr = 1e-6, factor = 0.5 ):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor 
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',                      #언제 학습률을 조정할지에 대한 기준 
            patience = self.patience,
            factor = self.factor,            #학습률을 얼마나 감소시킬지 (factor를 곱한만큼 감소)
            min_lr = self.min_lr,
            verbose = True)                  #조기 종료의 시작과 끝을 출력하기 위해 사용 
        
    #실제 학습률 업데이트(에포크 단위로 검증 데이터 셋에 대한 오차를 받아 이전 오차와 비교하여 차이가 없다면 학습률 업데이트)
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
class EarlyStopping():
    def __init__(self, patience, path, verbose=False, delta = 0):
        self.patience = patience
        self.counter = 0
        self.best_score = None 
        self.early_stop = False 
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path         #모델이 저장될 경로 
        
    def __call__(self, val_loss, model): #에포크만큼 학습 반복하면서 best_loss 갱신, 갱신 없으면 조기 종료 후 모델 저장 
        score = -val_loss

        if self.best_score is None: #best_score에 값이 존재하지 않을 때 실행 
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta: #best_score+delta가 score보다 크면 실행 
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0    

    def save_checkpoint(self, val_loss, model): #검증 데이터 셋에 대한 오차가 감소하면 모델을 저장 
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        