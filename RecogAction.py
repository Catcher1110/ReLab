# For correcting the gait
import h5py
import keras
import numpy as np
from keras.models import load_model


def Recog_Action(data, modelname):
    #  data: a dictionary made by RecogData.Recog_data()
    model = load_model(modelname)
    x_action = data['x_action']
    y_action = model.predict(x_action)
    y_res = []

    #  For the 25 in the beginning
    for i in range(25):
        y_res.append(0)

    for i in range(y_action.shape[0]):
        index = 0
        for j in range(4):
            if y_action[i, j] == max(y_action[i]) and y_action[i, j] >= 0.7:
                y_res.append(j+1)
                index = 1
                break
        if index == 0:
            y_res.append(0)

    return y_res
