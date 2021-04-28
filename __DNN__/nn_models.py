from keras.models import Sequential
from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, Reshape
from keras.callbacks import ModelCheckpoint


# LSTM
def RNN_model(sequence_length):
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))

    #Bi-directional LSTMs
    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, stateful=False), merge_mode='concat'))

    # Fully Connected Layers
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam',metrics=['mae'])
    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model

# DAE
def DAE_model(sequence_length):
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dropout(0.2))
    model.add(Dense((sequence_length-0)*8, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense((sequence_length-0)*8, activation='relu'))

    model.add(Dropout(0.2))

    # 1D Conv
    model.add(Reshape(((sequence_length-0), 8)))
    model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model

# CNN
def S2S_model(sequence_length):
    model = Sequential()

    # layer 1
    model.add(Conv1D(10, 30, activation="relu", input_shape=(sequence_length,1), padding="same", strides=1))

    # layer 2
    model.add(Conv1D(8, 30, activation="relu", padding="same", strides=1))

    # layer 3
    model.add(Conv1D(5, 40, activation="relu", padding="same", strides=1))

    # layer 4
    model.add(Conv1D(5, 50, activation="relu", padding="same", strides=1))

    # layer 5
    model.add(Conv1D(5, 50, activation="relu", padding="same", strides=1))

    # layer 6
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    
    # output layer
    model.add(Dense(sequence_length))
    model.add(Reshape((sequence_length, 1)))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

# MLP
def MLP_model(sequence_length):
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dropout(0.2))
    model.add(Dense((sequence_length-0)*2, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense((sequence_length-0)*4, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense((sequence_length-0)*8, activation='relu'))

    model.add(Dropout(0.2))

    # 1D Conv
    model.add(Reshape(((sequence_length-0), 8)))
    model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model