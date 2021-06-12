import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, SimpleRNN, InputLayer, Embedding, GRU

'''
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X = np.asarray(X)
y = np.asarray(y)

X = X/255.0

X_train = X[128:]
y_train = y[128:]
X_test = X[:128]
y_test = y[:128]
'''

model = Sequential()
model.add(Embedding(input_dim=600, output_dim=600))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(SimpleRNN(128))

model.add(Dense(1))

model.summary()


model.compile(optimizer = "adam", 
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ["accuracy"])

'''
model.fit(X_train, y_train, 
          validation_data=(X_test, y_test),
          batch_size=32, epochs=8, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest accuracy: ", test_acc)

model.save('saved_model')
'''