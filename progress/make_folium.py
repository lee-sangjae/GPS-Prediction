import os,sys
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

import config
from src import vis

## 모델 로드
best_model = torch.load(os.path.join(config.BASE_DIR, 'model\\best-model.pt'))


test = torch.load(os.path.join(config.BASE_DIR, "data\\test\\test.pkl"))
test_loader=DataLoader(test)

# Test the model
cnt=0
y_preds, y_trues=[],[]
outliers=[]

p_size=3
w_size=3

from src import utils
scalers=utils.load_pickle(os.path.join(config.BASE_DIR, "model\\scalers.pkl"))

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
                                 lon_sclr[1].inverse_transform(y_true[:,1].reshape(-1,1))),
                                 axis=1)
        y_true = y_true.reshape(-1, p_size*2)

        y_pred=best_model(test)
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

### 시각화

path = os.path.join(config.BASE_DIR, "../../3. 데이터리소스\\csv\\1.Grid_location_mdf.csv")

sample_test_df=test_df.reset_index(drop=True).copy()
sample_test_df["longitude_y_pred"] = np.nan
sample_test_df["latitude_y_pred"] = np.nan

# lon과 lat에 대한 pred값이 들어있는 test_df를 생성

lon_y_pred, lat_y_pred=[],[]
for idx, y_pred in enumerate(y_preds):
    
    if (idx+1)%6==0:
        sample_test_df.loc[idx,"longitude_y_pred"]=y_pred.squeeze().numpy()[0]
        sample_test_df.loc[idx,"latitude_y_pred"]=y_pred.squeeze().numpy()[1]

# tr_number가 겹치지 않게 sampling

sample_idx=[i for i in range(0,test_df.tr_number.unique().shape[0]) if i%6 == 0]
sample_ts_numbers=test_df.groupby(by="tr_number").evnt_dt.max().iloc[sample_idx].index

# sample_idx = np.random.randint(0,sample_ts_numbers.shape[0], 100) # Random Sample 
# sample_idx = test_df.groupby(by=["tr_number"]).evnt_dt.max().index[:50] # 시간 순 Cut
sample_test_df=sample_test_df[sample_test_df.tr_number.isin(sample_ts_numbers)]

m = vis.MakeMap(vis.grid_locations, sample_test_df, 
            lat_colname="latitude",lon_colname="longitude", 
            y_pred_lat_colname="latitude_y_pred",y_pred_lon_colname="longitude_y_pred", 
            date_colname="evnt_dt")
m.initialize_map()
m.get_styles()
# m.draw_grid()
# m.draw_point(0.5)
m.draw_point(0.5, isin_predict=True, add_last_point=False)
m.save("../Result/y_true_y_pred_vis.html")
m.return_map()




