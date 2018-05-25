import readfile
import numpy as np
import Calangle
import Calfoot


def Process_data(filename):
    raw_data = readfile.loadfile(filename)

    data = Calfoot.Cal_foot(raw_data)
    p1, p2 = [], []
    for i in range(len(data['Position_r'])):
        p1.append(np.sqrt(np.sum(np.square(data['Position_r'][i, :]))))
        p2.append(np.sqrt(np.sum(np.square(data['Position_l'][i, :]))))

    i = 0
    while p1[i] == 0 and p2[i] == 0:
        i = i + 1

    j = len(p1) - 1
    last_p1, last_p2 = p1[-1], p2[-1]
    while abs(p1[j] - last_p1) < 0.01 and abs(p2[j] - last_p2) < 0.01:
        j = j - 1

    Angle = Calangle.Cal_angle(1, raw_data)[i:j, :]

    temp = np.zeros([Angle.shape[0], 1])
    if filename[0:3] == 'zou':
        temp = np.append(temp+1, np.zeros([Angle.shape[0], 3]), 1)
    elif filename[0:3] == 'lou':
        temp = np.append(temp, temp + 1, 1)
        temp = np.append(temp, np.zeros([Angle.shape[0], 2]), 1)
    elif filename[0:3] == 'zuo':
        temp = np.append(np.zeros([Angle.shape[0], 2]), temp + 1, 1)
        temp = np.append(temp, np.zeros([Angle.shape[0], 1]), 1)
    elif filename[0:3] == 'you':
        temp = np.append(np.zeros([Angle.shape[0], 3]), temp + 1, 1)
    Angle = np.append(Angle, temp, 1)
    # Length of the sequence
    seq_len = 25
    data_temp = []
    for index in range(len(Angle) - seq_len + 1):
        data_temp.append(Angle[index:index + seq_len, :])

    data = np.array(data_temp)

    return data


def Train_Recog_Data(filename, *args):
    '''
    raw_data = readfile.loadfile(filename)
    Angle = Calangle.Cal_angle(1, raw_data)
    # Length of the sequence
    seq_len = 25
    # sequence_length = seq_len + 1
    data_temp = []
    for index in range(len(Angle) - seq_len + 1):
        data_temp.append(Angle[index:index + seq_len, :])

    data_temp = np.array(data_temp)'''

    data_temp = Process_data(filename)
    for i in range(len(args)):
        data_temp = np.append(data_temp, Process_data(args[i]), 0)

    np.random.shuffle(data_temp)
    data = {}
    data['x_train'] = np.delete(data_temp, [14, 15, 16, 17], 2)[:, :, :]
    data['y_train'] = np.delete(data_temp, range(14), 2)[:, -1, :]

    return data


def Recog_Data(raw_data):
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
