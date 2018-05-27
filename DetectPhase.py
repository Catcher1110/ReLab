import Calangle
import Calfoot
import readfile
import numpy as np


def Detect_phase(filename):
    # filename: string
    raw_data = readfile.loadfile(filename)
    Angle = Calangle.Cal_angle(1, raw_data)
    Foot = Calfoot.Cal_foot(raw_data)

    #  Determine when the person is walking
    p1, p2 = [], []
    for i in range(len(Foot['Position_r'])):
        p1.append(np.sqrt(np.sum(np.square(Foot['Position_r'][i, :]))))
        p2.append(np.sqrt(np.sum(np.square(Foot['Position_l'][i, :]))))

    i = 0
    while p1[i] == 0 and p2[i] == 0:
        i = i + 1

    j = len(p1) - 1
    last_p1, last_p2 = p1[-1], p2[-1]
    while abs(p1[j] - last_p1) < 0.01 and abs(p2[j] - last_p2) < 0.01:
        j = j - 1

    data = np.zeros([i]).tolist()
    temp = np.zeros([len(Foot['Position_r']) - j]).tolist()
    detect_data = phase_detect(Angle[i:j , :])

    data = data + detect_data + temp

    return data


def phase_detect(Angle):
    # Angle shape: N * 14 list
    return
