import requests
import json


def postfile(url, header, datatype, data):
    url = url + datatype + '/'
    res = requests.post(url, headers=header, json=data)
    return res