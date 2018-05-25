import time
import h5py
import keras
import InitModel
import RecogData
import readfile
import CutData
import numpy as np


data = RecogData.Train_Recog_Data('zou41.txt', 'zou51.txt', 'zou52.txt',
                                  'zou53.txt', 'zuo.txt', 'zuo2.txt',
                                  'you.txt', 'louti1.txt', 'louti2.txt',
                                  'louti3.txt', 'louti4.txt',)
RecogModel = InitModel.Train_model(data, [14, 256, 32, 4], 'Recogmodel.h5')

