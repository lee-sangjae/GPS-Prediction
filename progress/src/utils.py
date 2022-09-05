import pickle 
import torch
from haversine import haversine_vector
import pandas as pd
import numpy as np
def load_pickle(path):
    with open(path, 'rb') as f:
        m = pickle.load(f)
    return m

def save_pickle(path,obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def haversine_loss(y_true:torch.Tensor,y_pred:torch.Tensor):
    loss = torch.Tensor(haversine_vector(y_true, y_pred, unit="m")).mean()
    return loss



def load_data(path):
    data=pd.read_csv(path)
    return data

