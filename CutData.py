import numpy as np
import matplotlib.pyplot as plt
import readfile
import Calfoot

def Cut_data(raw_data):
    # Delete the stationary data
    data = Calfoot.Cal_foot(raw_data)
    print len(data['Position_r'])

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

    res = {'Stationary_r': data['Stationary_r'][i:j], 'Position_r': data['Position_r'][i:j],
           'Stationary_l': data['Stationary_l'][i:j], 'Position_l': data['Position_l'][i:j]}

    return res
