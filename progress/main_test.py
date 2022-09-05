import os, glob
import math
import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
# import torchvision #
# import torchvision.datasets as dset
import torchvision.transforms as tr #데이터 불러오면서 전처리를 가능한게 해주는 라이브러리 
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Variable 
from sklearn.metrics import mean_squared_error as mse
from haversine import haversine

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from src import preprocess as pp
from src import model
#from src import vis
from src import utils
# from src import data

import config

parser = argparse.ArgumentParser()

# preprocessing을 진행할지 말지
parser.add_argument("--preprop", '-pr', type=int, help="preprocessing을 진행할지 하지 않을지 결정")
parser.add_argument("--train", '-tr', type=int, help="data를 train 할지 말지")

args=parser.parse_args() 

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

if args.preprop == False:

    train_data=torch.load(os.path.join(config.BASE_DIR, "data\\train\\train.pkl"))
    test_data=torch.load(os.path.join(config.BASE_DIR, "data\\train\\test.pkl"))

    train_loader = DataLoader(train_data, batch_size = 64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size = test_data.shape[0], shuffle=False)

elif args.preprop == True:
    # 다른 csv 파일 있으면 안 됨
    train_data_paths=glob.glob(os.path.join(config.BASE_DIR, f"data\\train\\*.csv"))
    test_data_paths=glob.glob(os.path.join(config.BASE_DIR, f"data\\test\\*.csv"))

    train_dict, test_dict={},{}
    total_data_paths = train_data_paths.copy()
    for ts_path in test_data_paths:
        total_data_paths.append(ts_path)

    test_equ_nos=["YT341", "YT374", "YT377"]

    for data_path in total_data_paths:
        data = pd.read_csv(os.path.join(config.BASE_DIR, data_path))
        equ_no = data_path.split("\\")[-1].split(".")[0]    
        typ = "train" if equ_no not in test_equ_nos else "test"

        # all
        data = pp.to_datetime(data, column="evnt_dt")
        data = data.set_index("evnt_dt")
        data = pp.insert_distance(data, column="distance")
        # data = pp.insert_direction(data, column="direction")
        data["sin_dir"] = data["direction"].apply(math.sin)
        data["cos_dir"] = data["direction"].apply(math.cos)
        data["cos_dir_diff"] = data["cos_dir"].shift(1)-data["cos_dir"]
        data["sin_dir_diff"] = data["sin_dir"].shift(1)-data["sin_dir"]

        data=data.drop(columns=["direction"])
        b=data.shape[0]
        
        data = data.dropna(subset=["distance"], how="any", axis=0); print(f"executed type : {typ} equ_no : {equ_no} dropna(sdubset=distance) count : {b-data.shape[0]}")
        data = data.drop(columns=['reg_seq', 'altitude', 'position_fix', 'satelites', 'dev_id', 'cre_dt', 'cntr_dup', 
                                'wk_id','y_blk', 'y_bay', 'y_row', 'y_tier', 'long_cut', 'latt_cut'], axis=1)

        # train 추가 전처리
        if typ == "train":
            data = data[data.distance<=12]
            data = pp.interpolate_vars(data, equ_no)
            data = data[~((data.velocity==0) & (data.shift(1).velocity==0))] # 딱 한 번 정지 상태 외에 제거
            train_dict[equ_no] = data.reset_index(drop=True) # 날짜 인덱스 제거

        elif typ == "test":
            # test일 경우에 TimeStamp를 남겨놓는다
            test_dict[equ_no] = data.reset_index()


    #전체 데이터 합치기 
    train = pd.concat([df for _, df in train_dict.items()]).reset_index(drop=True).drop(columns="equ_no", axis=1)
    test = pd.concat([df for _, df in test_dict.items()]).reset_index(drop=True).drop(columns="equ_no", axis=1)

    #정규화 (이상치 나름 제거 했으니 minmax로)
    from sklearn.preprocessing import MinMaxScaler

    target_cols=["latitude","longitude","velocity","distance"]
    scalers = [] 
    for col_name in target_cols:
        scalers.append((col_name,MinMaxScaler()))

    for col_name, scaler in scalers: # Scaler Fitting ( train data에 대해서만 fitting 진행 )
        scaler.fit(train.loc[:,col_name].values.reshape(-1,1))
        print(f"{col_name} scaling complete")

    # train dict, test_dict 업데이트
    w_size = 3
    p_size = 3

    for typ, data_dict in [("train",train_dict),("test",test_dict)]:
        for equ_no, data in data_dict.items():
            print(f"type : {typ} equ_no : {equ_no}")
            
            for col_name, scaler in scalers: # train, test 전부 Scaling 진행
                data[col_name] = scaler.transform(data[col_name].values.reshape(-1,1))
        
            X = data
            y = data[["latitude","longitude"]]

            X,y = pp.make_sequence_train_dataset(X, y, w_size, p_size)
        
            if typ == "train": train_dict[equ_no] = [X,y] # train dict update
            if typ == "test":test_dict[equ_no] = [X,y] # test dict update

            print(f"update type : {typ} equ_no : {equ_no} dict update")


    X_train = np.concatenate([tr[0] for _, tr in train_dict.items()],axis=0)
    y_train = np.concatenate([tr[1] for _, tr in train_dict.items()],axis=0)

    X_test = np.concatenate([ts[0] for _, ts in test_dict.items()],axis=0)
    y_test = np.concatenate([ts[1] for _, ts in test_dict.items()],axis=0)

    for fn, obj in [("X_train",X_train), ("y_train",y_train), ("X_test",X_test),("y_test",y_test)]:
        utils.save_pickle(os.path.join(config.BASE_DIR, f"data\\train\\{fn}_npy.pkl"),obj)
    

    train_data = pp.TensorData(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size = 64, shuffle=False)

    # X_test에는 TimeStamp가 있으므로 자르고 float형태로 변환
    test_data = pp.TensorData(X_test[:,:,1:].astype(float), y_test)
    test_loader = DataLoader(test_data, batch_size = y_test.shape[0], shuffle=False)
   
# train model
if args.train == True:
    num_epoch = 1000
    hid_dim = 512

    # 변수 개수에 따라 변경
    in_dim = 6
    out_dim = 6

    lstm = model._LSTM2(in_dim = in_dim, hid_dim = hid_dim, out_dim = out_dim, num_layers = 1).cuda()

    crit = torch.nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), 0.0001)
    best_rmse = 10e+10
    best_mape = 0

    for i,epoch in enumerate(range(num_epoch)):
        
        for batch_idx, (X_train, y_train) in enumerate(train_loader): 
            
            optimizer.zero_grad()
            
            #6개 값 출력 
            out = lstm.forward(X_train)
            loss = crit(out.view(-1,6), y_train.view(-1,6))
            
            loss.backward()
            optimizer.step()

        i += 1

        if i%10 == 0:
            print(f"Epoch : {i}/{epoch} Loss : {loss:.10f}")

    
# 예측하고 결과만 확인
# --train = 1 로 하면 위의 결과대로 학습한 모델이 예측한다
elif args.train == False:
    model_path=os.path.join(config.BASE_DIR, "model\\best-model.pt")
    lstm = torch.load(model_path)

    
# 테스트

from src import utils
scalers=utils.load_pickle(os.path.join(config.BASE_DIR, "model\\scalers.pkl"))

cnt=0
y_preds, y_trues=[],[]
outliers=[]

p_size=3
w_size=3
lat_sclr = scalers[0]
lon_sclr = scalers[1]

with torch.no_grad():
    correct = 0
    total_loss = 0
    best_loss = 999999
    outlier_cnt=0
    for test, y_true in test_loader:
        
        y_true = y_true.reshape((-1,2))
        y_true = np.concatenate((lat_sclr[1].inverse_transform(y_true[:,0].reshape(-1,1)),
                                 lon_sclr[1].inverse_transform(y_true[:,1].reshape(-1,1))),axis=1)
        y_true = y_true.reshape(-1, p_size*2)

        y_pred=lstm(test)
        y_pred=y_pred.detach().cpu().numpy()


        y_pred[:,[0,2,4]] = lat_sclr[1].inverse_transform(y_pred[:,[0,2,4]])
        y_pred[:,[1,3,5]] = lon_sclr[1].inverse_transform(y_pred[:,[1,3,5]])

        loss = utils.haversine_loss(y_pred, y_true)
        # print(loss)
        total_loss += loss
        y_trues.append(y_true)
        y_preds.append(y_pred)
        if loss < best_loss:
            best_loss = loss
        
        cnt += 1
        

print(f"Sample Count : {cnt}")
print(f"Average Distance Diff : {total_loss/cnt}")



