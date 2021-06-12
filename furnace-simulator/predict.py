import numpy as np
from tensorflow.keras.models import load_model


def predict_one(input_data, model="furnace_model"):
    model = load_model(model)
    pred = model.predict(input_data)

    return pred


def predict_series(input_data, model="furnace_model"):
    model = load_model(model)
    pred = [model.predict(input_data[:i + 1, :])[0][0] for i in range(input_data.shape[1])]

    return pred


if __name__ == '__main__':
    print(predict_series(np.array([[[1]] * 10])))
