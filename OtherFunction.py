import numpy as np


def finddiff(Array, val):
    temp = [0]
    for i in range(len(Array) - 1):
        temp.append(Array[i+1] - Array[i])
    res = []
    for i in range(len(temp)):
        if temp[i] == val:
            res.append(i)
    return res


def findval(TarArray, val):
    Tarlist = TarArray.tolist()
    temp = []
    for i in range(1, len(Tarlist)):
        temp.append(Tarlist[i] - Tarlist[i-1])
    temp.insert(0, 0)
    res = []
    for i in range(len(temp)):
        if temp[i] == val:
            res.append(i)
    res = np.array(res)
    return res


def displace(arr1, arr2):
    res = np.sqrt(np.square(arr1[0] - arr2[0])
                  + np.square(arr1[1] - arr2[1])
                  + np.square(arr1[2] - arr2[2]))
    return res