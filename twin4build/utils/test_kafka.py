import requests
import json
from uppath import uppath
import os
import numpy as np
import datetime
import matplotlib.dates as mdates
import time
with open(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\BuildingEnergyModel\twin4build\utils\test_query.json", 'rb') as f:
    json_file = json.load(f)
import requests
import json

# url = "http://tek-cfei-dockerintrumentation0a.tek.sdu.dk:8080/query"

# payload = json.dumps(json_file)
# headers = {
#   'Content-Type': 'application/json'
# }


# start_time = time.time()
# print("Requesting data...")
# response = requests.request("POST", url, headers=headers, data=payload)
# print(f"Time used: {int((time.time()-start_time)/60)} minutes")
# content = response.content

response_filename = os.path.join(uppath(__file__, 1), "test_response.txt")
# print(response.status_code)





# #########
# with open(response_filename, 'wb') as f:
#     f.write(content)

    
data = [json.loads(line) for line in open(response_filename, 'rb')]
data = data[1:] #remove header information
time_series = np.array([row[0][0] for row in data])

time_series = time_series[time_series[:, 0].argsort()]

data_dict = dict()

# print(time_series)
data_dict["time"] = np.vectorize(lambda data:datetime.datetime.fromtimestamp(data)) (time_series[:,0])
data_dict["value"] = time_series[:,1]

print(time_series.shape)

import matplotlib.pyplot as plt 


fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
ax.plot(data_dict["time"], data_dict["value"])
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
plt.show()