# For correcting the gait
import h5py
import keras
import numpy as np
from keras.models import load_model


def Correct(data, modelname):
    model = load_model(modelname)
    x_action = data['x_action']
    y_action = model.predict(x_action)
    y_res = []
    for i in range(y_action.shape[0]):
        if y_action[i] == [1, 0, 0, 0]:
            y_res.append(1)
        elif y_action[i] == [0, 1, 0, 0]:
            y_res.append(2)
        elif y_action[i] == [0, 0, 1, 0]:
            y_res.append(3)
        elif y_action[i] == [0, 0, 0, 1]:
            y_res.append(4)

    return y_res
