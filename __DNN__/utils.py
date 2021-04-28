import sys
import os
import json
from math import *

import numpy as np
import pandas as pd

sys.path.append("../")
from config import *

def load_data(appliance, file, data_proportion = [0.9, 0.95]):
    # train : validate : testing -> [0.9, 0.05, 0.05]
    df = pd.read_csv(file).sample(frac=1)
    list_agg_power = [ json.loads(l) for l in df[df['appliance'] == appliance]['agg_power'].tolist()]
    list_app_power = [ json.loads(l) for l in df[df['appliance'] == appliance]['app_power'].tolist()]
    
    length = len(list_app_power)
    
    l = []
    for i in range(3):
        if i == 0:
            lb = 0
            ub = floor(length*data_proportion[0])
        elif i == 1:
            lb = floor(length*data_proportion[0])
            ub = floor(length*data_proportion[1])
        else:
            lb = floor(length*data_proportion[1])
            ub = length
        
        x = np.array(list_agg_power[lb:ub], dtype=np.float64)
        y = np.array(list_app_power[lb:ub], dtype=np.float64)
        l.append(x)
        l.append(y)
        
    x_train, y_train, x_test, y_test, x_val, y_val = l
        
    return x_train, y_train, x_test, y_test, x_val, y_val

def save_history(appliance, model_name, time_cost, history, mae, mape, nrmse, total_mape):
    result_path = './result/history.csv'
    if not os.path.isfile(result_path):
        pd.DataFrame({
            'appliance': [],
            'model': [],
            'time_cost':[],
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': [],
            'test_mae': [],
            'test_mape': [],
            'test_nrmse':[],
            'test_total_mape': []
            }, dtype=np.int32).to_csv(result_path, mode='w', index=False)
        
    
    pd.DataFrame({
    'appliance': [appliance],
    'model': [model_name],
    'time_cost':[time_cost],
    'loss': [history.history['loss']],
    'val_loss': [history.history['val_loss']],
    'mae': [history.history['mae']],
    'val_mae': [history.history['mae']],
    'test_mae': [mae],
    'test_mape': [mape],
    'test_nrmse':[nrmse],
    'test_total_mape': [total_mape]
    }, dtype=np.int32).to_csv(result_path, header=None, mode='a', index=False)