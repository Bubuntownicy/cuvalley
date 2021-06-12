import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, SimpleRNN, Input, Embedding, GRU, Flatten, LSTM

X = pickle.load(open("X_19042021.pickle", "rb"))
y = pickle.load(open("Y_19042021.pickle", "rb"))
X = list(map(float, X))
y = list(map(float, y))
X = np.asarray(X)
y = np.asarray(y)

#print(len(y))
#print(len(X))

#print(X.shape)
#X = X/255.0

X_train = X[10000:]
y_train = y[10000:]
X_test = X[:10000]
y_test = y[:10000]

X_train = X_train.reshape(2351, 1, 400)
y_train = y_train.reshape(2351, 1, 400)
X_test  = X_test.reshape(25, 1, 400)
y_test = y_test.reshape(25, 1, 400)

model = Sequential()
model.add(LSTM(512, activation='relu', kernel_initializer='he_normal', input_shape=(1, 400)))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
# fit the model
model.fit(X_train, y_train, epochs=512, batch_size=256, verbose=2, validation_data=(X_test, y_test))