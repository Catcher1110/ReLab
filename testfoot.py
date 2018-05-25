import numpy as np
import matplotlib.pyplot as plt
import readfile
import Calfoot
import CutData


filename = 'louti1.txt'
data = readfile.loadfile(filename)
# res = Calfoot.Cal_foot(data)
print len(data)

res = CutData.Cut_data(data)
print len(res['Position_r'])
f1 = plt.figure(1)
p1, p2 = [], []
for i in range(len(res['Position_r'])):
    p1.append(np.sqrt(np.sum(np.square(res['Position_r'][i, :]))))
    p2.append(np.sqrt(np.sum(np.square(res['Position_l'][i, :]))))

# SensorNumber = 7
# K = np.array(res['Stationary_r'])
# m = len(K)
Length = len(res['Stationary_r']) - 1  # m / SensorNumber - 1
Time_index = [0.0]
for i in range(Length):
    Time_index.append(Time_index[-1] + 1 / 30.0)
Time_index = np.array(Time_index)

plt.plot(Time_index, p1)
plt.plot(Time_index, p2)

f2 = plt.figure(2)
plt.plot(Time_index, res['Position_r'][:, 0])
plt.plot(Time_index, res['Position_r'][:, 1])
plt.plot(Time_index, res['Position_r'][:, 2])

f3 = plt.figure(3)
plt.plot(Time_index, res['Stationary_r'])

f3 = plt.figure(4)
plt.plot(Time_index, res['Stationary_l'])

plt.show()
