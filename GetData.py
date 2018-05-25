import requests
import json
import ast


def getfile(url, header, datatype, id):
    # datatype is firsthand, excercise, processed, etc. String
    url = url + datatype + '/' + str(id)
    raw_data = requests.get(url, headers=header)
    raw_data = raw_data.json()
    if datatype == 'firsthand':
        data = {'acceleration': ast.literal_eval(raw_data['acceleration']),
                'quaternion': ast.literal_eval(raw_data['quaternians'])}
    elif datatype == 'excercise':
        data = {}
    elif datatype == 'processed':
        data = {}
    elif datatype == 'reports':
        data = {}
    elif datatype == 'results':
        data = {}
    elif datatype == 'suits':
        data = {}

    return data
