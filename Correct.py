# For correcting the gait
import h5py
import keras
import numpy as np
from keras.models import load_model


def Correct(data, modelname):
    #  data: a dictionary made by CorrectData.Correct_Data()
    model = load_model(modelname)
    x_correct, y_wrong = data['x_correct'], data['y_wrong']
    y_correct = model.predict(x_correct)
    correct_data = data['raw_Angle']
    correct_data[x_correct.shape[1]:, 3] = y_correct[:, 0]
    correct_data[x_correct.shape[1]:, 10] = y_correct[:, 1]

    return correct_data
