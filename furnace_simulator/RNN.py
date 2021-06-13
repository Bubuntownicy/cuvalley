import pickle

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from data.data_getter import data_getter4
'''
X = pickle.load(open("X_19042021.pickle", "rb"))
y = pickle.load(open("Y_19042021.pickle", "rb"))
X = list(map(float, X))
y = list(map(float, y))
X = np.asarray(X)
y = np.asarray(y)
'''

X, y = data_getter4()
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

def train_nn(time_steps=200, observations=2, test_size=200, epochs=510, batch_size=512, save_name="furnace_model"):
    X_train = X[test_size:]
    y_train = y[test_size:]
    X_test = X[:test_size]
    y_test = y[:test_size]

    print(X_train.shape, y_train.shape)

    X_train = X_train.reshape(-1, time_steps, observations)
    y_train = y_train.reshape(-1, time_steps, 1)
    X_test = X_test.reshape(-1, time_steps, observations)
    y_test = y_test.reshape(-1, time_steps, 1)
    print(X_train.shape, y_train.shape)

    ckpt_callback = ModelCheckpoint(filepath=save_name, verbose=1, save_freq='epoch', period=10)

    model = Sequential()
    model.add(LSTM(64, input_shape=(time_steps, observations)))
    model.add(Dense(32))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    # fit the model

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test),
              callbacks=[ckpt_callback])


if __name__ == '__main__':
    train_nn()
