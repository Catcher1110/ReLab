import numpy as np
import matplotlib.pyplot as plt
import Quaternions as Qu
import OtherFunction as otf


def Cal_foot(data):
    SensorNumber = 7
    K = np.array(data)
    # print("K.shape: " + str(K.shape))

    m = len(K)
    # print("m: " + str(m))

    Length = m / SensorNumber - 1
    # print("Length: " + str(Length))

    # Time = np.delete(K, range(8), 1)
    # time_step = sum(Time) / Length / 1000.0
    # print("time_step: " + str(time_step))

    Time_index = [0.0]
    for i in range(Length):
        #  Time_index.append(Time_index[-1] + Time[7 * i + 6] / 1000.0)
        Time_index.append(Time_index[-1] + 1 / 30.0)
    Time_index = np.array(Time_index)
    # print("Time_index: " + str(Time_index.shape))

    # Update the acc to the absolute coordinate axis
    for i in range(len(K)):
        q = np.array([K[i, 4], K[i, 5], K[i, 6], K[i, 7]])
        RotateMat = Qu.quat2dcm(q)
        temp_acc = np.dot([K[i, 1], K[i, 2], K[i, 3]], RotateMat)
        for j in range(3):
            K[i, j + 1] = temp_acc[j]

    # Divide the data by seven sensors
    K = K.tolist()
    rawData = [[], [], [], [], [], [], []]
    # print("K[1][0]: " + str(K[1][0]))
    for i in range(m):
        rawData[int(K[i][0])].append(K[i])
    K = np.array(K)

    filterData = np.array(rawData)
    # print("filterData: " + str(filterData.shape))

    start_thread = 0.04
    ini_time = np.zeros([1, SensorNumber])
    '''
    for i in range(SensorNumber):
        time = 0
        while time < m:
            if filterData[i, time, 3] - filterData[i, time+1, 3] > start_thread:
                ini_time[0, i] = time
            time = time + 1
    '''
    staticBias = np.zeros([SensorNumber, 3])

    for i in range(SensorNumber):
        ini_time[0, i] = 47
        for j in range(3):
            for k in range(int(ini_time[0, i])):
                staticBias[i, j] = filterData[i, k, j + 1] + staticBias[i, j]
            staticBias[i, j] = staticBias[i, j] / ini_time[0, i]

    for i in range(SensorNumber):
        for j in range(3):
            for k in range(Length):
                filterData[i, k, j + 1] = filterData[i, k, j + 1] - staticBias[i, j]

    sizeofdata = filterData.shape
    # print("sizeofdata: " + str(sizeofdata))

    add_acc = np.zeros([SensorNumber, sizeofdata[1]])
    # print("add_acc: " + str(add_acc.shape))

    for j in range(SensorNumber):
        for i in range(sizeofdata[1]):
            add_acc[j, i] = np.sqrt(np.square(filterData[j, i, 1])
                                    + np.square(filterData[j, i, 2])
                                    + np.square(filterData[j, i, 3]))

    add_accFilt_r = add_acc[0, :]
    add_accFilt_l = add_acc[6, :]
    # print("add_accFilt: " + str(add_accFilt.shape))

    station_mark_r = 0.142
    station_mark_l = 0.142

    stationary_r = []
    for i in add_accFilt_r:
        if i < station_mark_r:
            stationary_r.append(1)
        else:
            stationary_r.append(0)
    stationary_r = np.array(stationary_r)
    # print("stationary.shape: " + str(stationary.shape))

    stationary_l = []
    for i in add_accFilt_l:
        if i < station_mark_l:
            stationary_l.append(1)
        else:
            stationary_l.append(0)
    stationary_l = np.array(stationary_l)

    vel_reight = np.zeros(filterData[0, :, 1:4].shape)
    # print("vel_reight.shape: " + str(vel_reight.shape))

    for i in range(1, len(vel_reight)):
        vel_reight[i, :] = vel_reight[i - 1, :] + filterData[0, i, 1:4] * 9.8 * (Time_index[i] - Time_index[i - 1])
        if stationary_r[i] == 1:
            vel_reight[i, :] = [0.0, 0.0, 0.0]

    vel_left = np.zeros(filterData[6, :, 1:4].shape)
    # print("vel_left.shape: " + str(vel_left.shape))
    for i in range(1, len(vel_left)):
        vel_left[i, :] = vel_left[i - 1, :] + filterData[6, i, 1:4] * 9.8 * (Time_index[i] - Time_index[i - 1])
        if stationary_l[i] == 1:
            vel_left[i, :] = [0.0, 0.0, 0.0]

    velDrift_r = np.zeros(vel_reight.shape)
    stationaryStart_r = otf.findval(stationary_r, -1)
    # print("stationaryStart: " + str(stationaryStart[0:6]))
    stationaryEnd_r = otf.findval(stationary_r, 1)
    # print("stationaryEnd: " + str(stationaryEnd[0:6]))

    velDrift_l = np.zeros(vel_left.shape)
    stationaryStart_l = otf.findval(stationary_l, -1)
    stationaryEnd_l = otf.findval(stationary_l, 1)


    delStart_r = []
    delEnd_r = []

    for i in range(len(stationaryEnd_r)):
        if stationaryStart_r[i + 1] - stationaryEnd_r[i] <= 0:
            delStart_r.append(i + 1)
            delEnd_r.append(i)
    stationaryStart_r = np.delete(stationaryStart_r, delStart_r, axis=0)
    stationaryEnd_r = np.delete(stationaryEnd_r, delEnd_r, axis=0)

    delStart_l = []
    delEnd_l = []
    for i in range(len(stationaryEnd_l)):
        if stationaryStart_l[i + 1] - stationaryEnd_l[i] <= 0:
            delStart_l.append(i + 1)
            delEnd_l.append(i)
    stationaryStart_l = np.delete(stationaryStart_l, delStart_l, axis=0)
    stationaryEnd_l = np.delete(stationaryEnd_l, delEnd_l, axis=0)

    station_r = stationaryStart_r.tolist() + stationaryEnd_r.tolist()
    station_r.sort()
    station_r.append(None)
    index, val, j = 0, 1, 0
    stationary_sort_r = []
    for i in range(int(len(stationary_r))):
        if i == station_r[index]:
            if j % 2 == 0:
                val = val - 1
            else:
                val = val + 1
            j = j + 1
            index = index + 1
        stationary_sort_r.append(val)

    for i in range(len(stationaryEnd_r)):
        driftRate = vel_reight[stationaryEnd_r[i] - 1, :] / (stationaryEnd_r[i] - stationaryStart_r[i])
        enum = np.array(range(1, 1 + int(stationaryEnd_r[i] - stationaryStart_r[i])))
        enum = enum.reshape([enum.shape[0], 1])
        drift = np.array([np.dot(enum, driftRate[0]), np.dot(enum, driftRate[1]), np.dot(enum, driftRate[2])])
        drift = drift.reshape([3, drift.shape[1]]).T
        velDrift_r[stationaryStart_r[i]:stationaryEnd_r[i], :] = drift

    vel_reight = vel_reight - velDrift_r

    pos_right = np.zeros(vel_reight.shape)
    for i in range(1, len(vel_reight)):
        pos_right[i, :] = pos_right[i - 1, :] + vel_reight[i, :] * (Time_index[i] - Time_index[i - 1])

    # Left foot
    station_l = stationaryStart_l.tolist() + stationaryEnd_l.tolist()
    station_l.sort()
    station_l.append(None)
    index, val, j = 0, 1, 0
    stationary_sort_l = []
    for i in range(int(len(stationary_l))):
        if i == station_l[index]:
            if j % 2 == 0:
                val = val - 1
            else:
                val = val + 1
            j = j + 1
            index = index + 1
        stationary_sort_l.append(val)

    for i in range(len(stationaryEnd_l)):
        driftRate = vel_left[stationaryEnd_l[i] - 1, :] / (stationaryEnd_l[i] - stationaryStart_l[i])
        enum = np.array(range(1, 1 + int(stationaryEnd_l[i] - stationaryStart_l[i])))
        enum = enum.reshape([enum.shape[0], 1])
        drift = np.array([np.dot(enum, driftRate[0]), np.dot(enum, driftRate[1]), np.dot(enum, driftRate[2])])
        drift = drift.reshape([3, drift.shape[1]]).T
        velDrift_l[stationaryStart_l[i]:stationaryEnd_l[i], :] = drift

    vel_left = vel_left - velDrift_l

    pos_left = np.zeros(vel_left.shape)
    for i in range(1, len(vel_left)):
        pos_left[i, :] = pos_left[i - 1, :] + vel_left[i, :] * (Time_index[i] - Time_index[i - 1])


    '''posi_r = []
    for i in range(len(pos_right)):
        posi_r.append(np.sqrt(np.square(pos_right[i, 0]) + np.square(pos_right[i, 1]) + np.square(pos_right[i, 2])))

    station_r.pop()
    support = []
    swing = []
    for i in range(len(stationaryEnd_r) - 1):
        support.append(Time_index[stationaryEnd_r[i]] - Time_index[stationaryStart_r[i]])
        swing.append(Time_index[stationaryStart_r[i + 1]] - Time_index[stationaryEnd_r[i]])

    displacement = []
    for i in range(len(stationaryEnd_r)):
        pos_start = pos_right[stationaryEnd_r[i]]
        pos_end = pos_right[stationaryStart_r[i]]
        displacement.append(otf.displace(pos_start, pos_end))'''

    res = {'Stationary_r': stationary_sort_r, 'Position_r': pos_right,
           'Stationary_l': stationary_sort_l, 'Position_l': pos_left}

    '''frep = open('report.txt', 'w')
    frep.write(filename + '\r\n')
    frep.write('Support Phase of Right Leg    Displacement of Right Leg' + '\r\n')
    for i in range(len(support)):
        temp = 100 * support[i] / (support[i] + swing[i])
        frep.write('%.2f' % temp + '%' + '  %.3f' % (displacement[i]) + ' m' + '\r\n')

    frep.close()'''

    return res
