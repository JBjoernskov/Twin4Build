import requests
import json
import sys
import os
import numpy as np
import datetime
import matplotlib.dates as mdates
import time
import requests
import json
import matplotlib.pyplot as plt 
import unittest

if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
    print(file_path)
from twin4build.utils.uppath import uppath

@unittest.skipIf(True, 'Currently not used')
def test_kafka():
    query_filename = os.path.join(uppath(os.path.abspath(__file__), 1), "test_query.json")
    with open(query_filename, 'rb') as f:
        json_file = json.load(f)
    url = "http://tek-cfei-dockerintrumentation0a.tek.sdu.dk:8080/query"
    payload = json.dumps(json_file)
    headers = {
    'Content-Type': 'application/json'
    }
    start_time = time.time()
    print("Requesting data...")
    response = requests.request("POST", url, headers=headers, data=payload)
    print(f"Time used: {int((time.time()-start_time)/60)} minutes")
    content = response.content
    print(response.status_code)
    response_filename = os.path.join(uppath(os.path.abspath(__file__), 1), "PIR.txt") #1766
    with open(response_filename, 'wb') as f:
        f.write(content)

        
    data = [json.loads(line) for line in open(response_filename, 'rb')]
    data = data[1:] #remove header information
    print(data)

    # ex_list = []
    # for row in data:
    #     if row[1]["sensor/type"] not in ex_list:
    #         ex_list.append(row[1]["sensor/type"])
    # print(ex_list)

    # ex_list = []
    # for row in data:
    #     if row[-3] not in ex_list:
    #         print(row[-3])
    #         ex_list.append(row[-3])

    # print("------")
    # ex_list = []
    # for row in data:
    #     if row[5] not in ex_list:
    #         print(row[5])
    #         ex_list.append(row[5])



    # time_series = np.array([row for row in data])
    time_series = np.array([row[0][0] for row in data])
    # ['Aktual Temperatur', 
    # 'Værdi Status Lysrække 2', 
    # 'Spjæld Status', 
    # 'Aktuel Luxværdi Pir Række 1', 
    # 'Værdi Status Lysrække 1', 
    # 'Radiator Ventil Status', 
    # 'Spjæld', 
    # 'Aktual CO2', 
    # 'Værdi Status Lysrække 3', 
    # 'Tilstedeværelse log', 
    # 'Klimatisering', 
    # 'Auto/man Pir', 
    # 'Værdi Lyszone 3', 
    # 'Manuel Dæmp Input / Increase or decrease', 
    # 'Manuel Dæmp Input / Step Code', 
    # 'Tænd/sluk  Lys Tavle', 
    # 'Køl Styring Signal', 
    # 'TRIN/STOP Manuel', 
    # 'Manuel Dæmp Input', 
    # 'Værdi Lyszone 2', 
    # 'Værdi Lyszone 1', 
    # 'CO2 Styring Signal', 
    # 'OP/NED Manuel', 
    # 'Lås Gardiner op', 
    # 'Booket Signal', 
    # 'Master tigger', 
    # 'Rum Setpunkt visning', 
    # 'Aktuel Luxværdi Pir række 2', 
    # 'Aktuel Luxværdi Pir Række 3', 
    # 'Lux Setpunkt Instil Pir Række 2', 
    # 'Lux Setpunkt Instil Pir Række 3', 
    # 'Lux Setpunkt Instil Pir Række 1', 
    # 'Setpunkt Rum']
    # key = "Rum Setpunkt visning"
    # time_series = np.array([row[0][0] for row in data if row[1]["sensor/type"]==key])
    # time_series = np.array([row[0][0] for row in data if row[-3]==key])
    # print(time_series)

    time_series = time_series[time_series[:, 0].argsort()]
    data_dict = dict()
    # print(time_series)
    data_dict["time"] = np.vectorize(lambda data:datetime.datetime.fromtimestamp(data)) (time_series[:,0])
    data_dict["value"] = time_series[:,1]
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.plot(data_dict["time"], data_dict["value"])
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    plt.show()