import numpy as np
import quaternion


def quaternProd(a, b):
    ab = np.zeros(a.shape)
    if len(a.shape) == 1:
        ab[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        ab[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        ab[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        ab[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    else:
        for i in range(a.shape[0]):
            ab[i, 0] = a[i, 0] * b[i, 0] - a[i, 1] * b[i, 1] - a[i, 2] * b[i, 2] - a[i, 3] * b[i, 3]
            ab[i, 1] = a[i, 0] * b[i, 1] + a[i, 1] * b[i, 0] + a[i, 2] * b[i, 3] - a[i, 3] * b[i, 2]
            ab[i, 2] = a[i, 0] * b[i, 2] - a[i, 1] * b[i, 3] + a[i, 2] * b[i, 0] + a[i, 3] * b[i, 1]
            ab[i, 3] = a[i, 0] * b[i, 3] + a[i, 1] * b[i, 2] - a[i, 2] * b[i, 1] + a[i, 3] * b[i, 0]
    return ab


def quaternConj(q):
    qConj = np.zeros(q.shape)
    if len(q.shape) == 1:
        qConj[0] = q[0]
        qConj[1] = -q[1]
        qConj[2] = -q[2]
        qConj[3] = -q[3]
    elif len(q.shape) == 2 and q.shape[0] != 1:
        qConj[:, 0] = q[:, 0]
        qConj[:, 1] = -q[:, 1]
        qConj[:, 2] = -q[:, 2]
        qConj[:, 3] = -q[:, 3]
    else:
        q = q[0, :]
        qConj[0] = q[0]
        qConj[1] = -q[1]
        qConj[2] = -q[2]
        qConj[3] = -q[3]
    return qConj


def quaternRotate(v, q):
    temp = np.zeros([v.shape[0], v.shape[1] + 1])
    temp[:, 1:] = v
    v0XYZ = quaternProd(quaternProd(q, temp), quaternConj(q))
    res = v0XYZ[:, 1:4]
    return res


def quatWAvgMarkley(Q, weights):
    M = np.zeros([4, 4])
    n = Q.shape[0]
    wSum = 0

    for i in range(n):
        q = Q[i, :].reshape([4, 1])
        w_i = weights[i]
        M = M + w_i*(q*q.T)
        wSum = wSum + w_i

    M = (1.0/wSum)*M
    temp, Qavg = np.linalg.eig(M)
    index = temp.argsort()
    return Qavg[:, index[-1]].reshape(4, 1)

def quatmultiply(q, r):
    qq = quaternion.as_quat_array(q)
    rr = quaternion.as_quat_array(r)
    qr = np.multiply(qq, rr)
    qMul = quaternion.as_float_array(qr)
    return qMul


def quatinv(q):
    if len(q.shape) == 1:
        qinv = quaternConj(q)/(quatnorm(q) * np.ones([1, 4]))
    elif len(q.shape) == 2 and q.shape[0] != 1:
        qinv = np.zeros(q.shape)
        for i in range(q.shape[0]):
            qinv[i] = quaternConj(q[i])/(quatnorm(q[i])*np.ones([1, 4]))
    else:
        q = q[0, :]
        qinv = quaternConj(q) / (quatnorm(q) * np.ones([1, 4]))
    return qinv


def quatnorm(q):
    if len(q.shape) == 1:
        qout = np.sum(np.square(q))
    else:
        qnorm = np.zeros([q.shape[0], 1])
        for i in range(q.shape[0]):
            qnorm[i] = np.sqrt(np.sum(np.square(q[i])))
        qout = np.square(qnorm)
    return qout


def quat2dcm(q):
    qin = q/np.sqrt(sum(np.square(q)))
    dcm = np.zeros([3, 3])
    dcm[0, 0] = np.square(qin[0]) + np.square(qin[1]) - np.square(qin[2]) - np.square(qin[3])
    dcm[0, 1] = 2.0 * (qin[1] * qin[2] + qin[3] * qin[0])
    dcm[0, 2] = 2.0 * (qin[1] * qin[3] - qin[0] * qin[2])
    dcm[1, 0] = 2.0 * (qin[1] * qin[2] - qin[3] * qin[0])
    dcm[1, 1] = np.square(qin[0]) - np.square(qin[1]) + np.square(qin[2]) - np.square(qin[3])
    dcm[1, 2] = 2.0 * (qin[2] * qin[3] + qin[0] * qin[1])
    dcm[2, 0] = 2.0 * (qin[1] * qin[3] + qin[0] * qin[2])
    dcm[2, 1] = 2.0 * (qin[2] * qin[3] - qin[1] * qin[0])
    dcm[2, 2] = np.square(qin[0]) - np.square(qin[1]) - np.square(qin[2]) + np.square(qin[3])
    return dcm
