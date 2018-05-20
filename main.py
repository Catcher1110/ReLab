import Calangle  # Calculate the angle of each joint
import Calfoot   # Calculate the state and the position of each foot
import CorrectData  # Pre-process the data for correcting
import Correct  # Correct the Knee angle
import TrainCorrect  # Train the model
import readfile  # Read the raw file and return a list(N * 9)
import h5py

filename = 'zou4.TXT'
raw_data = readfile.loadfile(filename)
# raw_data is a list(N * 9)
Angle = Calangle.Cal_angle(1, raw_data)
# Angle is a list((N/7) * 14)
Foot = Calfoot.Cal_foot(raw_data)
# Foot is a dictionary, the keys are 'Stationary_r', 'Position_l', 'Position_r', 'Stationary_l'
# which mean whether the foot is on the field and what the position of each foot is

modelname = ''
Data_for_Cor = CorrectData.Correct_Data(raw_data)
# Data for Correcting the Knee joint
Data_for_Tra = CorrectData.Train_Correct_Data(raw_data)
# Data for Training the model
TrainCorrect.Train_model(Data_for_Tra, modelname)
# Train or update the model and save it
# The previous one is saved 'modelname_old'

Corrected_Data = Correct.Correct(Data_for_Cor, modelname)
# Return the corrected angle data, which has the same size with Angle
