from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


def predict_one(input_data, weights="checkpoint"):
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, 400)))
    model.add(Dense(32))
    model.add(Dense(1))
    model.load_weights(weights)

    pred = model.predict(input_data)

    return pred

if __name__ == '__main__':
    predict_one()