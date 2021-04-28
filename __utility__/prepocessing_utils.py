import sys
sys.path.append("../")
import os
import random

import pandas as pd
import numpy as np

from config import *


def get_channel(houseId, appliance):
    if appliance == 'aggregate':
        return 1
    with open('../{}house_{}/labels.dat'.format(DATA_ORIGIN_DIR, houseId), 'r') as f:
        for line in f.readlines():
            id, name = line.strip().split()
            if name == appliance:
                return eval(id)
    return None

def get_dataframe(house_id, appliance):
    # first col -> time; snd col -> power
    data_folder = DATA_ORIGIN_DIR
    channel_id = get_channel(house_id, appliance)
    if channel_id:
        return pd.read_csv('../{}house_{}/channel_{}.dat'.format(
            data_folder, house_id, channel_id),
                           sep=' ', header=None, dtype=np.int64)
    else:
        return pd.DataFrame(data={
            0:[],
            1:[]
        })

def get_activations(house_id, appliance):
    df = get_dataframe(house_id, appliance)
    on_power_threshold = APP_CONFIG[appliance]['on_power_threshold']
    min_on_duration = APP_CONFIG[appliance]['min_on_duration']
    max_off_duration = APP_CONFIG[appliance]['max_off_duration']
    
    df = df[df[1]>=on_power_threshold]
    
    list_st = []
    list_et = []
    if df.empty:
        pass
    else:
        start_time = None
        end_time = None
        off_during = None
        for index, (timestamp, power) in df.iterrows():
            if start_time is None:
                start_time = timestamp
                # off_during = 0
                end_time = timestamp
            else:
                if timestamp - end_time < max_off_duration:
                    end_time = timestamp
                else:
                    if end_time-start_time > min_on_duration:
                        list_st.append(start_time)
                        list_et.append(end_time)
                    start_time = timestamp
                    end_time = timestamp
    return pd.DataFrame({
        'start_time': list_st,
        'end_time': list_et
    })

def load_activations(house_id, appliance):
    df = pd.read_csv('../' + DATA_PREPROCESS_DIR + 'activation.csv')
    return df[(df['appliance'] == appliance) & (df['house_id'] == house_id)]

def load_all_activations():
    return pd.read_csv('../' + DATA_PREPROCESS_DIR + 'activation.csv')

def load_origin_df(house_id, appliance):
    channel_id = get_channel(house_id, appliance)
    if channel_id == None:
        return pd.DataFrame({
            0: [],
            1: []
        })
    else:
        return pd.read_csv('../{}house_{}/channel_{}.dat'.format(DATA_ORIGIN_DIR, house_id, channel_id),
                           sep=' ', header=None, dtype=np.int64)

def get_data(df, start_time, end_time, step_size=DEFAULT_STEP_SIZE, filling_type=0):
    data = [0 for i in range((end_time-start_time)//step_size)]
    # data = np.zeros((end_time-start_time)//step_size + 1, dtype=np.int64)
    
    df = df[(df[0]>=start_time-FORWARD_FILLING_WINDOW) & (df[0]<=end_time)]
    
    df_iterator = df.iterrows()
    i = 0
    cur_time = start_time
    try:
        ub_time, ub_power = next(df_iterator)[1]
        lb_time, lb_power = next(df_iterator)[1]
    except:
        return data
        
    while cur_time < end_time and i < (end_time-start_time)//step_size:
        while cur_time < ub_time or cur_time >= lb_time:
            try: 
                ub_time, ub_power = lb_time, lb_power
                lb_time, lb_power = next(df_iterator)[1]
            except:
                return data
        
        if ub_time + FORWARD_FILLING_WINDOW > cur_time:
            data[i] = ub_power
        
        i += 1
        cur_time = start_time + step_size*i
    
    return data

def get_samples(appliance, app_df, agg_df, start_time, end_time, number_samples, 
                window_size_config=JACK_KELLY_INPUT_CONFIG):
    window_size = window_size_config[appliance]
    list_app, list_aggre, list_st, list_et = [], [], [], []
    app_df = app_df[(app_df[0]>=end_time-window_size-FORWARD_FILLING_WINDOW) & (app_df[0]<=start_time+window_size)]
    agg_df = agg_df[(agg_df[0]>=end_time-window_size-FORWARD_FILLING_WINDOW) & (agg_df[0]<=start_time+window_size)]
    
    for i in range(number_samples):
        if window_size >= end_time - start_time:
            s_t = random.randint(end_time-window_size, start_time)
            app_data = get_data(app_df, s_t, s_t+window_size)
            agg_data = get_data(agg_df, s_t, s_t+window_size)


            list_app.append(app_data)
            list_aggre.append(agg_data)
            list_st.append(s_t)
            list_et.append(s_t+window_size)
    
    return list_app, list_aggre, list_st, list_et

def random_samples(house_id, appliance, number_samples):
    df_app = load_origin_df(house_id, appliance)
    df_agg = load_origin_df(house_id, 'aggregate')
    
    sample_length = JACK_KELLY_INPUT_CONFIG[appliance]
    t_min, t_max = df_app[0].min(), df_app[0].max()-sample_length
    
    l_app, l_agg, l_st, l_et = [], [], [], []
    for i in range(number_samples):
        print(i, end='\r')
        rand_st = random.randint(t_min, t_max)
        l_app.append(get_data(df_app, rand_st, rand_st + sample_length))
        l_agg.append(get_data(df_agg, rand_st, rand_st + sample_length))
        l_st.append(rand_st)
        l_et.append(rand_st + sample_length)
    
    return l_app, l_agg, l_st, l_et

def get_gaussian_noise(length, size):
    # create gaussian noise based on aggreagate data distribution
    # must > 0
    
    list_series = [load_origin_df(i, 'aggregate')[1] for i in range(1, 6)]
    power = pd.concat(list_series)
    m = power.mean()
    std = power.std()
    return np.array([[ int(i) if i > 0 else 0 for i in np.random.normal(m, std, length)] for i in range(size)])