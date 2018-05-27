import numpy as np
import Quaternions as Qu
import readfile


def Cal_angle(scale, data):
    # Scale is a dictionary, containing the length information
    # data shape: (7 * N) * 9 list
    height = 180.0 # scale['height']
    Upper_leg_length = 0.232 * height  # scale['Upper_leg_length']
    Lower_leg_length = 0.247 * height  # scale['Lower_leg_length']
    foot_size = 0.05 * height  # scale['foot_size']
    width = 0.203 * height  # scale['width']

    # Read the data
    data = np.array(data)
    m = data.shape[0]
    l = m / 7
    K = np.delete(data, [1, 2, 3, 8], 1)

    # Initial pose
    O1 = np.array([0, 0, 0])
    O1_q0 = np.array([np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2])

    O2 = np.array([0, 0.5 * width, -0.25 * width])
    O2_q0 = np.array([np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2])
    O1O2_ini = np.array([0, 0, -0.5 * width, -0.25 * width])

    O3 = np.array([0, 0.5 * width, -Upper_leg_length])
    O3_q0 = np.array([np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2])
    O2O3_ini = np.array([0, 0, 0, -Upper_leg_length])

    O4 = np.array([0, 0.5 * width, -(Upper_leg_length + Lower_leg_length)])
    O4_q0 = np.array([np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2])
    O3O4_ini = np.array([0, 0, 0, -Lower_leg_length])

    O5 = np.array([foot_size, 0.5 * width, -(Upper_leg_length + Lower_leg_length)])
    O4O5_ini = np.array([0, foot_size, 0, 0])

    O6 = np.array([0, -0.5 * width, -0.25 * width])
    O6_q0 = np.array([np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2])
    O1O6_ini = np.array([0, 0, 0.5 * width, -0.25 * width])

    O7 = np.array([0, -0.5 * width, -Upper_leg_length])
    O7_q0 = np.array([np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2])
    O6O7_ini = np.array([0, 0, 0, -Upper_leg_length])

    O8 = np.array([0, -0.5 * width, -(Upper_leg_length + Lower_leg_length)])
    O8_q0 = np.array([np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2])
    O7O8_ini = np.array([0, 0, 0, -Lower_leg_length])

    O9 = np.array([foot_size, -0.5 * width, -(Upper_leg_length + Lower_leg_length)])
    O8O9_ini = np.array([0, foot_size, 0, 0])

    # Calibration
    L = K[0:35, 1:5]
    H = np.zeros([5, 4])
    J = np.zeros([7, 5, 4])
    weights = np.ones([5, 1]) / 5

    for i in range(7):
        for j in range(5):
            H[j, :] = L[i + j * 7, :]
            J[i, j, :] = L[i + j * 7, :]
    J = list(J)

    Q1_mast0_ave = Qu.quatWAvgMarkley(J[3], weights)
    Q1_bias = Qu.quatmultiply(Qu.quatinv(Q1_mast0_ave.T), O1_q0)
    Q2_mast0_ave = Qu.quatWAvgMarkley(J[4], weights)
    Q2_bias = Qu.quatmultiply(Qu.quatinv(Q2_mast0_ave.T), O2_q0)
    Q3_mast0_ave = Qu.quatWAvgMarkley(J[5], weights)
    Q3_bias = Qu.quatmultiply(Qu.quatinv(Q3_mast0_ave.T), O3_q0)
    Q4_mast0_ave = Qu.quatWAvgMarkley(J[6], weights)
    Q4_bias = Qu.quatmultiply(Qu.quatinv(Q4_mast0_ave.T), O4_q0)
    Q6_mast0_ave = Qu.quatWAvgMarkley(J[2], weights)
    Q6_bias = Qu.quatmultiply(Qu.quatinv(Q6_mast0_ave.T), O6_q0)
    Q7_mast0_ave = Qu.quatWAvgMarkley(J[1], weights)
    Q7_bias = Qu.quatmultiply(Qu.quatinv(Q7_mast0_ave.T), O7_q0)
    Q8_mast0_ave = Qu.quatWAvgMarkley(J[0], weights)
    Q8_bias = Qu.quatmultiply(Qu.quatinv(Q8_mast0_ave.T), O8_q0)

    # Calculation
    Q1_mast = []
    Q2_mast = []
    Q3_mast = []
    Q4_mast = []
    Q6_mast = []
    Q7_mast = []
    Q8_mast = []

    for i in range(m):
        if K[i, 0] == 3:
            Q1_mast.append(K[i, 1:5])
        elif K[i, 0] == 4:
            Q2_mast.append(K[i, 1:5])
        elif K[i, 0] == 5:
            Q3_mast.append(K[i, 1:5])
        elif K[i, 0] == 6:
            Q4_mast.append(K[i, 1:5])
        elif K[i, 0] == 2:
            Q6_mast.append(K[i, 1:5])
        elif K[i, 0] == 1:
            Q7_mast.append(K[i, 1:5])
        elif K[i, 0] == 0:
            Q8_mast.append(K[i, 1:5])

    Q1_mast = np.array(Q1_mast)
    Q2_mast = np.array(Q2_mast)
    Q3_mast = np.array(Q3_mast)
    Q4_mast = np.array(Q4_mast)
    Q6_mast = np.array(Q6_mast)
    Q7_mast = np.array(Q7_mast)
    Q8_mast = np.array(Q8_mast)

    PPP = np.zeros([l, 8, 6])
    Angle = np.zeros([l, 14])

    for i in range(l):
        O1_qt = Qu.quatmultiply(Q1_mast[i, :], Q1_bias)[0, :]

        O2_qt = Qu.quatmultiply(Q2_mast[i, :], Q2_bias)[0, :]

        O1O2 = Qu.quatmultiply(Qu.quatmultiply(O1_qt, O1O2_ini), Qu.quatinv(O1_qt))[0, :]
        P_O2 = O1O2[1:4]
        O3_qt = Qu.quatmultiply(Q3_mast[i, :], Q3_bias)[0, :]

        O3_t = O1O2 + Qu.quatmultiply(Qu.quatmultiply(O2_qt, O2O3_ini), Qu.quatinv(O2_qt))[0, :]
        O2O3 = Qu.quatmultiply(Qu.quatmultiply(O2_qt, O2O3_ini), Qu.quatinv(O2_qt))[0, :]
        P_O3 = O3_t[1:4]

        O4_qt = Qu.quatmultiply(Q4_mast[i, :], Q4_bias)[0, :]

        O4_t = O3_t + Qu.quatmultiply(Qu.quatmultiply(O3_qt, O3O4_ini), Qu.quatinv(O3_qt))[0, :]
        O3O4 = Qu.quatmultiply(Qu.quatmultiply(O3_qt, O3O4_ini), Qu.quatinv(O3_qt))[0, :]
        P_O4 = O4_t[1:4]

        # angle_knee1 = acos(dot(O2O3(2:4), O3O4(2:4)) / (norm(O2O3(2:4))*norm(O3O4(2:4))))*(180 / pi);
        q_knee1 = Qu.quatmultiply(Qu.quatinv(O3_qt), O2_qt)[0, :]
        angle_knee1_new = abs(np.arcsin(2 * (q_knee1[0] * q_knee1[2] - q_knee1[1] * q_knee1[3])))

        O5_t = O4_t + Qu.quatmultiply(Qu.quatmultiply(O4_qt, O4O5_ini), Qu.quatinv(O4_qt))[0, :]
        O4O5 = Qu.quatmultiply(Qu.quatmultiply(O4_qt, O4O5_ini), Qu.quatinv(O4_qt))[0, :]
        P_O5 = O5_t[1:4]

        # angle_foot1 = acos(dot(O3O4(2:4), O4O5(2:4)) / (norm(O3O4(2:4))*norm(O4O5(2:4))))*(180 / pi);

        O6_qt = Qu.quatmultiply(Q6_mast[i, :], Q6_bias)[0, :]

        O1O6 = Qu.quatmultiply(Qu.quatmultiply(O1_qt, O1O6_ini), Qu.quatinv(O1_qt))[0, :]
        P_O6 = O1O6[1:4]
        O7_qt = Qu.quatmultiply(Q7_mast[i, :], Q7_bias)[0, :]

        O7_t = O1O6 + Qu.quatmultiply(Qu.quatmultiply(O6_qt, O6O7_ini), Qu.quatinv(O6_qt))[0, :]
        O6O7 = Qu.quatmultiply(Qu.quatmultiply(O6_qt, O6O7_ini), Qu.quatinv(O6_qt))[0, :]
        P_O7 = O7_t[1:4]

        O8_qt = Qu.quatmultiply(Q8_mast[i, :], Q8_bias)[0, :]
        O8_t = O7_t + Qu.quatmultiply(Qu.quatmultiply(O7_qt, O7O8_ini), Qu.quatinv(O7_qt))[0, :]
        O7O8 = Qu.quatmultiply(Qu.quatmultiply(O7_qt, O7O8_ini), Qu.quatinv(O7_qt))[0, :]
        P_O8 = O8_t[1:4]

        q_knee2 = Qu.quatmultiply(Qu.quatinv(O7_qt), O6_qt)[0, :]
        angle_knee2_new = np.arcsin(2 * (q_knee2[0] * q_knee2[2] - q_knee2[1] * q_knee2[3]))

        O9_t = O8_t + Qu.quatmultiply(Qu.quatmultiply(O8_qt, O8O9_ini), Qu.quatinv(O8_qt))[0, :]
        P_O9 = O9_t[1:4]

        q_up1 = Qu.quatmultiply(Qu.quatinv(O2_qt), O1_qt)[0, :]
        angle_up1_a = np.arctan2(2 * (q_up1[0] * q_up1[1] + q_up1[2] * q_up1[3]),
                                 1 - 2 * (np.square(q_up1[1]) + np.square(q_up1[2])))
        angle_up1_b = np.arcsin(2 * (q_up1[0] * q_up1[2] - q_up1[1] * q_up1[3]))
        angle_up1_c = np.arctan2(2 * (q_up1[0] * q_up1[3] + q_up1[1] * q_up1[2]),
                                 1 - 2 * (np.square(q_up1[2]) + np.square(q_up1[3])))

        q_knee1 = Qu.quatmultiply(Qu.quatinv(O3_qt), O2_qt)[0, :]
        angle_knee1_new = abs(np.arcsin(2 * (q_knee1[0] * q_knee1[2] - q_knee1[1] * q_knee1[3])))

        q_ft1 = Qu.quatmultiply(Qu.quatinv(O4_qt), O3_qt)[0, :]
        angle_ft1_a = np.arctan2(2 * (q_ft1[0] * q_ft1[1] + q_ft1[2] * q_ft1[3]),
                                 1 - 2 * (np.square(q_ft1[1]) + np.square(q_ft1[2])))
        angle_ft1_b = np.arcsin(2 * (q_ft1[0] * q_ft1[2] - q_ft1[1] * q_ft1[3]))
        angle_ft1_c = np.arctan2(2 * (q_ft1[0] * q_ft1[3] + q_ft1[1] * q_ft1[2]),
                                 1 - 2 * (np.square(q_ft1[2]) + np.square(q_ft1[3])))

        q_up2 = Qu.quatmultiply(Qu.quatinv(O6_qt), O1_qt)[0, :]
        angle_up2_a = np.arctan2(2 * (q_up2[0] * q_up2[1] + q_up2[2] * q_up2[3]),
                                 1 - 2 * (np.square(q_up2[1]) + np.square(q_up2[2])))
        angle_up2_b = np.arcsin(2 * (q_up2[0] * q_up2[2] - q_up2[1] * q_up2[3]))
        angle_up2_c = np.arctan2(2 * (q_up2[0] * q_up2[3] + q_up2[1] * q_up2[2]),
                                 1 - 2 * (np.square(q_up2[2]) + np.square(q_up2[3])))

        q_knee2 = Qu.quatmultiply(Qu.quatinv(O7_qt), O6_qt)[0, :]
        angle_knee2_new = np.arcsin(2 * (q_knee2[0] * q_knee2[2] - q_knee2[1] * q_knee2[3]))

        q_ft2 = Qu.quatmultiply(Qu.quatinv(O8_qt), O7_qt)[0, :]
        angle_ft2_a = np.arctan2(2 * (q_ft2[0] * q_ft2[1] + q_ft2[2] * q_ft2[3]),
                                 1 - 2 * (np.square(q_ft2[1]) + np.square(q_ft2[2])))
        angle_ft2_b = np.arcsin(2 * (q_ft2[0] * q_ft2[2] - q_ft2[1] * q_ft2[3]))
        angle_ft2_c = np.arctan2(2 * (q_ft2[0] * q_ft2[3] + q_ft2[1] * q_ft2[2]),
                                 1 - 2 * (np.square(q_ft2[2]) + np.square(q_ft2[3])))

        PPP[i, 0, 0] = P_O2[0]
        PPP[i, 0, 1] = P_O2[1]
        PPP[i, 0, 2] = P_O2[2]

        PPP[i, 1, 0] = P_O3[0]
        PPP[i, 1, 1] = P_O3[1]
        PPP[i, 1, 2] = P_O3[2]
        PPP[i, 1, 3] = Angle[i, 0] = angle_up1_a
        PPP[i, 1, 4] = Angle[i, 1] = angle_up1_b
        PPP[i, 1, 5] = Angle[i, 2] = angle_up1_c

        PPP[i, 2, 0] = P_O4[0]
        PPP[i, 2, 1] = P_O4[1]
        PPP[i, 2, 2] = P_O4[2]
        PPP[i, 2, 3] = Angle[i, 3] = angle_knee1_new

        PPP[i, 3, 0] = P_O5[0]
        PPP[i, 3, 1] = P_O5[1]
        PPP[i, 3, 2] = P_O5[2]
        PPP[i, 3, 3] = Angle[i, 4] = angle_ft1_a
        PPP[i, 3, 4] = Angle[i, 5] = angle_ft1_b - np.pi * 90/180
        PPP[i, 3, 5] = Angle[i, 6] = angle_ft1_c

        PPP[i, 4, 0] = P_O6[0]
        PPP[i, 4, 1] = P_O6[1]
        PPP[i, 4, 2] = P_O6[2]

        PPP[i, 5, 0] = P_O7[0]
        PPP[i, 5, 1] = P_O7[1]
        PPP[i, 5, 2] = P_O7[2]
        PPP[i, 5, 3] = Angle[i, 7] = angle_up2_a
        PPP[i, 5, 4] = Angle[i, 8] = angle_up2_b
        PPP[i, 5, 5] = Angle[i, 9] = angle_up2_c

        PPP[i, 6, 0] = P_O8[0]
        PPP[i, 6, 1] = P_O8[1]
        PPP[i, 6, 2] = P_O8[2]
        PPP[i, 6, 3] = Angle[i, 10] = angle_knee2_new

        PPP[i, 7, 0] = P_O9[0]
        PPP[i, 7, 1] = P_O9[1]
        PPP[i, 7, 2] = P_O9[2]
        PPP[i, 7, 3] = Angle[i, 11] = angle_ft2_a
        PPP[i, 7, 4] = Angle[i, 12] = angle_ft2_b - np.pi * 90/180
        PPP[i, 7, 5] = Angle[i, 13] = angle_ft2_c

    return Angle
