# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:13:40 2022

@author: USER
"""

import numpy as np

def msee(act, pred):
 
   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
    
   return mean_diff

def rmse(act, pred):
 
   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val

def mae(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff