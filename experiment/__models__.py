import os
import time
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Reshape, Conv1D, Subtract, Activation, Flatten, Lambda, Add, Multiply, Bidirectional, Dense, BatchNormalization, SpatialDropout1D, LSTM
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model


WEIGHT_PATH = './model_weights/'

# call back funct: at each end of epoch, record number of epoch done. 
class TrainingCheckPointCallback(keras.callbacks.Callback):
    def __init__(self, weight_folder):
        super().__init__()
        self.path = weight_folder + 'train.log'
        self.start_time = -1
    
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if self.start_time == -1:
            time_cost = -1
        else:
            time_cost = time.time()-self.start_time
        
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                num_epoch_trained = eval(f.readlines()[-1])[0]
        else:
            num_epoch_trained = -1
            
        with open(self.path, 'a') as f:
            f.write('{},{}\n'.format(num_epoch_trained + 1, time_cost))

# Model prototype
class NetworkPrototype:
    def __init__(self, input_length, output_length, appliance, model_type, epochs, model_name, auto_build=True, save_path=WEIGHT_PATH):
        # model_type: 'reg or clf'
        self.name = '{}_{}_{}to{}_{}'.format(appliance, model_name, input_length, output_length, model_type)
        self.model_type = model_type
        self.input_length = input_length
        self.output_length =output_length
        self.epochs = epochs
        self.save_path = save_path
        if auto_build:
            self.build_model()
        
    def get_layers(self, inputs):
        return None
    
    def build_model(self):
        inputs = Input(shape=(self.input_length,))
        x = Reshape((self.input_length, 1))(inputs)
        
        _ = self.get_layers(x)
        outputs = Flatten()(_)
        model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        if self.model_type == 'reg' or self.model_type == 'reg_std_loss':
            loss = 'mse'
            metrics=['mae']
        elif self.model_type == 'clf':
            loss = 'binary_crossentropy'
            metrics=['acc']
        else:
            print('----- model type error! -----')
            
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )    
            
        self.model = model
    
    def recompile_model(self, optimizer, loss, metrics):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
                 
    def train(self, train_generator, val_generator, continue_train=True):
        weight_folder = '{}{}_{}/'.format(self.save_path, self.name, self.epochs)
        if not os.path.exists(weight_folder):
            os.makedirs(weight_folder)
        
        num_epoch_trained = self.__get_num_epoch_trained__()
        epochs_left = self.epochs - num_epoch_trained
        
        if num_epoch_trained == self.epochs:
            return
        
        if num_epoch_trained > 0:
            self.load_weights()
        
        
        if self.model_type == 'reg' or self.model_type == 'reg_std_loss':
            monitor = 'val_mae'
        elif self.model_type == 'clf':
            monitor = 'val_acc'
        elif self.model_type == 'reg_std':
            monitor = 'disagg_output_mae'
        else:
            print('----- model type error! -----')
        
        self.model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            epochs=epochs_left,
            verbose=1,
            callbacks = [
                EarlyStopping(monitor=monitor, patience=3),
                TrainingCheckPointCallback(weight_folder),
                ModelCheckpoint(
                    filepath=weight_folder + 'weights.hdf5',
                    monitor=monitor,
                    save_weights_only=True,
                    save_best_only=True,
                    mode='auto'
                )
            ]
        )
    
    def load_weights(self):
        weigth_path = '{}{}_{}/weights.hdf5'.format(self.save_path, self.name, self.epochs)
        if os.path.exists(weigth_path):
            self.model.load_weights(weigth_path)
    
    def __get_num_epoch_trained__(self):
        train_path = '{}{}_{}/train.log'.format(self.save_path, self.name, self.epochs)
        try:
            with open(train_path, 'r') as f:
                return eval(f.readlines()[-1])[0]
        except:
            return 0
    
# 1. Existing Models
## 1.1. LSTM & DAE models from Jack Kelly's Paper "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation" (2015) 
## https://arxiv.org/abs/1507.06594
# dAE
class JKDAE(NetworkPrototype):
    def __init__(self, input_length, output_length, appliance, model_type, epochs, model_name='jkdae', auto_build=True, save_path=WEIGHT_PATH):
        super().__init__(input_length, output_length, appliance, model_type, epochs, model_name=model_name, auto_build=auto_build, save_path=save_path)

    def get_layers(self, inputs):
        x = Conv1D(8, 4, activation="linear", padding="same", strides=1)(inputs)
        x = Dense((self.input_length -3 ) * 8, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense((self.input_length - 3) * 8, activation='relu')(x)
        outputs = Conv1D(1, 4, activation="linear", padding="same", strides=1)(x)
        
        return outputs
# LSTM
class JKLSTM(NetworkPrototype):
    def __init__(self, input_length, output_length, appliance, model_type, epochs, model_name='jklstm', auto_build=True, save_path=WEIGHT_PATH):
        super().__init__(input_length, output_length, appliance, model_type, epochs, model_name=model_name, auto_build=auto_build, save_path=save_path)

    def get_layers(self, inputs):
        x = Conv1D(16, 4, activation="linear", padding="same", strides=1)(inputs)
        x = Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat')(x)
        x = Bidirectional(LSTM(256, return_sequences=True, stateful=False), merge_mode='concat')(x)
        x = Dense(128, activation='tanh')(x)
        outputs = Dense(1)(x)

        return outputs

## 1.2. sequence-to-point & sequence-to-sequence models from Chaoyun Zhang's 
## "Sequence-to-point learning with neural networks for non-intrusive loadmonitoring" (2017)
## https://arxiv.org/pdf/1612.09106.pdf
##https://github.com/MingjunZhong/NeuralNetNilm
# sequence-to-sequence model
class S2S(NetworkPrototype):
    def __init__(self, input_length, output_length, appliance, model_type, epochs, model_name='s2s', auto_build=True, save_path=WEIGHT_PATH):
        super().__init__(input_length, output_length, appliance, model_type, epochs, model_name=model_name, auto_build=auto_build, save_path=save_path)

    def get_layers(self, inputs):
        x = Conv1D(30, 4, activation="relu", padding="same", strides=1)(inputs)
        x = Conv1D(30, 8, activation="relu", padding="same", strides=1)(x)
        x = Conv1D(40, 6, activation="relu", padding="same", strides=1)(x)
        x = Conv1D(50, 5, activation="relu", padding="same", strides=1)(x)
        x = Conv1D(50, 5, activation="relu", padding="same", strides=1)(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(self.output_length)(x)
        outputs = Reshape((self.output_length, 1))(x)

        return outputs
    
# sequence-to-point model
class S2P(S2S):
    def __init__(self, input_length, output_length, appliance, model_type, epochs, model_name='s2p', auto_build=True, save_path=WEIGHT_PATH):
        super().__init__(input_length, output_length, appliance, model_type, epochs, model_name=model_name, auto_build=auto_build, save_path=save_path)

## 1.3. BitcnNILM Model from 
## "Sequence to Point Learning Based on Bidirectional Dilated Residual Network for Non-Intrusive Load Monitoring" (2020)
## https://arxiv.org/ftp/arxiv/papers/2006/2006.00250.pdf
## https://github.com/linfengYang/BitcnNILM
# BiTCNResidual -> seq to point        
class BiTCNResidual(NetworkPrototype):
    def __init__(self, input_length, output_length, appliance, model_type, epochs, model_name='bi_tcn_residual', auto_build=True, save_path=WEIGHT_PATH):
        super().__init__(input_length, output_length, appliance, model_type, epochs, model_name=model_name, auto_build=auto_build, save_path=save_path)

    def get_layers(self, inputs, 
                   nb_filters=128, dilations=[1,2,4,8,16,32,64,128], 
                   filter_length=3,dropout = 0.3):
        x = Conv1D(filters=nb_filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(inputs)

        skip_connections = []
        for d in dilations:
            x, skip_out = self.residual_block(x, dilation_rate=d,nb_filters=nb_filters,kernel_size=filter_length, 
                                         padding = 'same', activation = 'relu', dropout_rate=dropout)
            skip_connections.append(skip_out)
        x = Add()(skip_connections)
        x = Lambda(lambda t: t[:, -1, :])(x)
        x = Dense(self.output_length, activation='linear')(x)
        outputs = Reshape((self.output_length, 1))(x)
        # outputs = Flatten()(x) 

        return outputs
    
    def residual_block(self, x, dilation_rate, nb_filters, kernel_size, 
                       padding='same', activation='relu', dropout_rate=0, kernel_initializer='he_normal'):
        # input layer
        prev_x = x
        
        # 1) two dialted cnn layer
        for k in range(2):
            x = Conv1D(filters=nb_filters,
                       kernel_size=kernel_size,
                       dilation_rate=dilation_rate,
                       kernel_initializer=kernel_initializer,
                       padding='same')(x)
            x = BatchNormalization()(x)  
            x = Activation('relu')(x)
            x = SpatialDropout1D(rate=dropout_rate)(x)

        # 2) one traditional cnn layer 
        prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
        
        # linear combine 1) and 2)
        res_x = Add()([prev_x, x])
        res_x = Activation(activation)(res_x)
        
        # return output & input layer
        return res_x, x
    
## 1.4. Fully Convolutional Network Model from 
## "NON-INTRUSIVE LOAD MONITORING WITH FULLY CONVOLUTIONAL NETWORKS" (2018)
## https://arxiv.org/pdf/1812.03915.pdf
## https://github.com/cbrewitt/nilm_fcn
# BiTCNResidual -> seq to point     
class FullyConvolutionalNetwork(NetworkPrototype):
    def __init__(self, input_length, output_length, appliance, model_type, epochs, model_name='fcn', auto_build=True, save_path=WEIGHT_PATH):
        super().__init__(input_length, output_length, appliance, model_type, epochs, model_name=model_name, auto_build=auto_build, save_path=save_path)

    def get_layers(self, inputs):
        offset = (self.input_length - self.output_length)//2
        
        # CNN
        x = Conv1D(128, 9, padding='same', activation='relu', dilation_rate=1, name='cnn_pre')(inputs)
        
        # dilated CNN
        num_dilated_layers = 8
        for rate in [ 2**i for i in range(1,num_dilated_layers+1)]:
            x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=rate, name='dil_{}'.format(rate))(x)
        
        # CNN
        x = Conv1D(256, 1, padding='same', activation='relu', name="cnn_post_1")(x)
        x = Conv1D(1, 1, padding='same', activation=None, name="cnn_post_2")(x)

        if self.input_length > self.output_length:
            x = Lambda(lambda x: x[:, offset:-offset])(x)
            
        return x

    
    
# 2. Model with STD block    
def root_N_activation(N):
    def __(x):
        return K.sqrt(x/N)
    return __

def square_activation(x):
    return K.square(x)

def keep_same_init(index):
    def __(shape,  dtype=None):
        weights = np.zeros(shape=shape)
        weights[index, 0, 0] = 1
        return K.variable(weights, dtype=dtype)
    return __

class STD(NetworkPrototype):
    def __init__(self, input_length, output_length, appliance, model_type, 
                 epochs, prototype_class, filter_size=3, save_path=WEIGHT_PATH):
        self.prototype = prototype_class(input_length, output_length, appliance, model_type, epochs, auto_build=False)
        self.model_type = model_type
        self.input_length = input_length
        self.output_length =output_length
        self.filter_size = filter_size
        self.epochs = epochs
        self.save_path = save_path
        self.name = self.prototype.name
        self.build_model()
    
    def get_layers(self, inputs, is_trainable=True):
        # support functions
        sequence_length = inputs.shape[1]
        x = inputs
        # 1. get mean of each windows
        _ = Conv1D(
            filters = 1,
            kernel_size = self.filter_size,
            strides=1,
            padding="same",
            kernel_initializer = tf.keras.initializers.Constant(1/self.filter_size),
            name = 'mean_layer'
        )
        _.trainable = is_trainable
        mean_layer = _(x)

        # 2. get each items of each windows
        list_square_layers = []
        for index in range(self.filter_size):
            _ = Conv1D(
                filters = 1,
                kernel_size = self.filter_size,
                strides=1,
                padding="same",
                kernel_initializer = keep_same_init(index),
                name = 'single_value_at_{}'.format(index)
            )
            _.trainable = is_trainable
            item_layer = _(x)

            subtract_layer = Subtract(name='mean_sub_at_{}'.format(index))([mean_layer, item_layer])
            square_layer = Activation(square_activation, name='square_mean_sub_at_{}'.format(index))(subtract_layer)
            list_square_layers.append(square_layer)
            # list_square_layers.append(subtract_layer)

        sum_layer = Add(name="sum_square_mean_sub")(list_square_layers)
        outputs = Activation(root_N_activation(self.filter_size), name='root_mean')(sum_layer)

        return outputs
    
    def build_model(self, std_weight=0.2):
        inputs = Input(shape=(self.input_length,))
        x = Reshape((self.input_length, 1))(inputs)
        
        # disaggregation output
        _ = self.prototype.get_layers(x)
        disagg_output = Flatten(name='disagg_output')(_)
        
        # std block
        __ = self.get_layers(_)
        std_out = Flatten(name='std_output')(__)
        
        model = Model(
             inputs,
             outputs=[disagg_output, std_out], 
             name=self.name
        )
        
        
        loss_weights={"disagg_output": 1-std_weight, "std_output": std_weight}
        if self.model_type == 'reg_std':
            loss={"disagg_output": MeanSquaredError(), "std_output": MeanSquaredError()}
            metrics = ['mae']
        elif self.model_type == 'clf_std':
            loss = {"disagg_output": BinaryCrossentropy(), "std_output": BinaryCrossentropy()}
            metrics=['acc']
        else:
            print('----- model type error! -----')
            
        model.compile(
            optimizer='adam',
            loss=loss,
            loss_weights = loss_weights,
            metrics=metrics
        )    
            
        self.model = model
    
    def recompile_model(self, optimizer, loss, metrics, loss_weights):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights = loss_weights,
            metrics=metrics
        )
