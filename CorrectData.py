import readfile
import numpy as np
import Calangle


def Train_Correct_Data(raw_data):
    Angle = Calangle.Cal_angle(1, raw_data)
    # Length of the sequence
    seq_len = 25
    sequence_length = seq_len + 1
    data_temp = []
    for index in range(len(Angle) - sequence_length + 1):
        data_temp.append(Angle[index:index + sequence_length, :])

    data_temp = np.array(data_temp)
    np.random.shuffle(data_temp)
    data = {}
    data['x_train'] = np.delete(data_temp, [3, 10], 2)[:, :-1, :]
    data['y_train'] = np.delete(data_temp, [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13], 2)[:, -1, :]

    return data


def Correct_Data(raw_data):
    Angle = Calangle.Cal_angle(1, raw_data)
    # Length of the sequence
    seq_len = 25
    sequence_length = seq_len + 1
    data_temp = []
    for index in range(len(Angle) - sequence_length + 1):
        data_temp.append(Angle[index:index + sequence_length, :])

    data_temp = np.array(data_temp)
    data = {}
    data['x_correct'] = np.delete(data_temp, [3, 10], 2)[:, :-1, :]
    data['y_wrong'] = np.delete(data_temp, [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13], 2)[:, -1, :]
    data['raw_Angle'] = np.array(Angle)

    return data
