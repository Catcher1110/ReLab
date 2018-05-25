import requests
import json
import numpy as np
import PostData
import GetData
import readfile


url = "http://118.25.109.179:8001/"
header = {'Authorization': 'Token 97e1c787ea683c018f08dbc7572468e415039907'}

# Scale of the body
height = 180.0  # scale['height']
Upper_leg_length = 0.232 * height  # scale['Upper_leg_length']
Lower_leg_length = 0.247 * height  # scale['Lower_leg_length']
foot_size = 0.05 * height  # scale['foot_size']
width = 0.203 * height  # scale['width']
thigh_l_len = thigh_r_len = Upper_leg_length
crus_l_len = crus_r_len = Lower_leg_length
foot_l_len = foot_r_len = foot_size
hipjoint_len = width

bodydata = {str(thigh_l_len):thigh_l_len, str(thigh_r_len):thigh_r_len,
            str(crus_l_len):crus_l_len, str(crus_r_len):crus_r_len,
            str(foot_l_len):foot_l_len, str(foot_r_len):foot_r_len,
            str(hipjoint_len):hipjoint_len}

data = readfile.loadfile('zou41.txt')
data = np.array(data)
acc = np.delete(data, [4, 5, 6, 7], 1)
acc = acc.tolist()
qu = np.delete(data, [1, 2, 3], 1)
qu = qu.tolist()
postdata = {'name': 'test', 'gather_time': '2018-5-22T21:00', 'device_id': '1',
            'exercise_id': '7', 'acceleration': str(acc), 'quaternians': str(qu),
            'myoelectricity': ''}
re = PostData.postfile(url, header, 'firsthand', postdata)

