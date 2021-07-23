import time
import json
import pandas as pd
import numpy as np
from tensorflow import keras

#### constant ####
RANDOM_SEED = 2021

# X=agg_power |  Y = [std_3, activate, app_power]
def load_data(appliance, lst_x_names, lst_y_names, 
              real_data_only=True, off_on_ratio=1, random_seed=RANDOM_SEED):
    print('Loading Data for {}'.format(appliance))
    st = time.time()
    df = pd.read_csv('..//data/dataset/{}_beta.csv'.format(appliance), 
                    dtype={'s2q_agg_power': str})

    # get real data only
    if real_data_only:
        df = df[df['house_id'] != -1]
        
    # get on & off data
    df_on = df[df['contain_activation'] == 1][lst_x_names + lst_y_names]
    df_off = df[df['contain_activation'] == 0].sample(
        frac=off_on_ratio, 
        random_state=random_seed, 
        replace= True if off_on_ratio > 1 else False
    )[lst_x_names + lst_y_names]
    
    # concat on and off, then shuffle
    df = pd.concat([df_on, df_off]).sample(frac=1, random_state=random_seed)
    
    X = [ np.array([ json.loads(l) for l in df[x_name]]) for x_name in lst_x_names]
    Y = [ np.array([ json.loads(l) for l in df[y_name]]) for y_name in lst_y_names]
    
    print('Time elapse: {:.2f}s'.format(time.time()-st))
    return X, Y

'''
def load_data(appliance, x_name, y_name, 
              real_data_only=True, off_on_ratio=1, random_seed=RANDOM_SEED):
    print('Loading Data for {}'.format(appliance))
    st = time.time()
    df = pd.read_csv('..//data/dataset/{}_beta.csv'.format(appliance), 
                    dtype={'s2q_agg_power': str})

    # get real data only
    if real_data_only:
        df = df[df['house_id'] != -1]
        
    # get on & off data
    df_on = df[df['contain_activation'] == 1][[x_name, y_name]]
    df_off = df[df['contain_activation'] == 0].sample(
        frac=off_on_ratio, 
        random_state=random_seed, 
        replace= True if off_on_ratio > 1 else False
    )[[x_name, y_name]]
    
    # concat on and off, then shuffle
    df = pd.concat([df_on, df_off]).sample(frac=1, random_state=random_seed)

    X = np.array([ json.loads(l) for l in df[x_name]])
    Y = np.array([ json.loads(l) for l in df[y_name]])
    print('Time elapse: {:.2f}s'.format(time.time()-st))
    return X, Y
'''



def split_data(data, ratios):
    left_index = 0
    right_index = 0
    result = []
    for r in ratios:
        left_index = right_index
        right_index += int(data.shape[0]*r)
        split_data = data[left_index:right_index]
        result.append(split_data)
    return result

class STDScaler:
    def __init__(self, l):
        self.fit(l)
    
    def fit(self, l):
        self.std = np.array(l).std()
        self.mean = np.array(l).mean()
        
    def transfer(self, l):
        return (l - self.mean) / self.std
    
    def transfer_back(self, l):
        return l * self.std + self.mean
    
class MinMaxScaler:
    def __init__(self, l):
        self.fit(l)
    
    def fit(self, l):
        self.max = np.array(l).max()
        self.min = np.array(l).min()
        
    def transfer(self, l):
        return (l - self.min) / (self.max - self.min)
    
    def transfer_back(self, l):
        return l * (self.max - self.min) + self.min
    
class S2SDataGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32, shuffle=True):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(self.X.shape[0])
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.X.shape[0] // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        selected = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return (
            self.X[selected],
            self.Y[selected]
        )

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

class S2SDataGenerator_beta(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32, shuffle=True):
        self.X, self.Y= X, Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(self.X[0].shape[0])
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.X[0].shape[0] // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        selected = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return (
            [ var[selected] for var in self.X],
            [ var[selected] for var in self.Y],
        )

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        
class S2S_with_STD_DataGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, stds, batch_size=32, shuffle=True):
        self.X, self.Y, self.stds = X, Y, stds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(self.X.shape[0])
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.X.shape[0] // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        selected = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return (
            self.X[selected],
            [self.Y[selected], self.stds[selected]]
        )

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indexes) 


class S2PDataGenerator(keras.utils.Sequence):
    def __init__(self, appliance, 
                 generator_type='train', ratios={'train': 0.9,'val': 0.05,'test': 0.05},
                 sequence_length=599, batch_size=32,shuffle=True):
        self.X, self.Y = get_dataset(appliance, x_name='s2q_agg_power', y_name='app_power')
        if generator_type == 'train':
            split_index = int(self.X.shape[0]*ratios['train'])
            self.X = self.X[:split_index]
            self.Y = self.Y[:split_index]
        elif generator_type == 'val':
            split_index1 = int(self.X.shape[0]*ratios['train'])
            split_index2 = int(self.X.shape[0]*(ratios['train'] + ratios['val']))
            self.X = self.X[split_index1:split_index2]
            self.Y = self.Y[split_index1:split_index2]
        elif generator_type == 'test':
            split_index = int(self.X.shape[0]*(ratios['train'] + ratios['val']))
            self.X = self.X[split_index:]
            self.Y = self.Y[split_index:]
        else:
            print('error!')
        
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return (self.X.shape[1]-self.sequence_length + 1) * self.X.shape[0] // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        return (self.get_batch_data(self.X, index, self.batch_size, self.sequence_length),
                self.get_batch_data(self.Y, index, self.batch_size, 1))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            p = np.random.permutation(self.X.shape[0])
            self.X = self.X[p]
            self.Y = self.Y[p]

    def get_batch_data(self, data, index, batch_size, sequence_length):
        num_chuncks = data.shape[1] - sequence_length + 1
        num_each_chunck = data.shape[0]

        chunck_index = index*batch_size // num_each_chunck
        chunck_offset = index*batch_size % num_each_chunck

        if chunck_offset + batch_size <= num_each_chunck:
            return data[:,chunck_index:chunck_index+sequence_length][chunck_offset:chunck_offset + batch_size]
        elif chunck_index < num_chuncks-1:
            part_1 = data[:,chunck_index:chunck_index+sequence_length][chunck_offset:num_each_chunck]
            part_2 = data[:,chunck_index+1:chunck_index+1+sequence_length][0:batch_size - (num_each_chunck-chunck_offset)]
            return np.concatenate((part_1,part_2), axis=0)
        else:
            return data[:,chunck_index:chunck_index+sequence_length][chunck_offset:num_each_chunck]
