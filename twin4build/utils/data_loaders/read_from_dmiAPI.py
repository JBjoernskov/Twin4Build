import requests
import json
import sys
import os
import numpy as np
import datetime
import requests
import json
import zipfile
import pandas as pd

if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
    print(file_path)
from twin4build.utils.uppath import uppath



# text = archive.open('2022-01-01.txt').split(b"\n")


# for line in text:
#     print(line)
#     json.loads(line)

stepSize = 24*60*60 #Days
startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0) #datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0) 
endPeriod = datetime.datetime(year=2023, month=2, day=15, hour=0, minute=0, second=0) #datetime.datetime(year=2023, month=2, day=15, hour=0, minute=0, second=0)
constructed_time_list = np.array([startPeriod + datetime.timedelta(seconds=dt) for dt in range(0, int((endPeriod-startPeriod).total_seconds()),stepSize)])

dates_temperature = []
dates_irradiation = []
values_temperature = []
values_irradiation = []
year = ""
for date in constructed_time_list:
    if date.year!=year:
        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", f"weather_DMI_{date.year}.zip")
        archive = zipfile.ZipFile(filename, 'r')
    year = date.year
    filename = f"{date.year}-{date:%m}-{date:%d}.txt"
    print(f"Extracting file: \"{filename}\"")
    data = [json.loads(line) for line in archive.open(filename, 'r')]
    data_by_stationId = [line for line in data if line["properties"]["stationId"]=="06126"]
    temperature = [line for line in data_by_stationId if line["properties"]["parameterId"]=="temp_dry"]
    irradiation = [line for line in data_by_stationId if line["properties"]["parameterId"]=="radia_glob"]

    print("--------")
    print(date)
    for line in temperature:
        print(line["properties"]["observed"], line["properties"]["value"])
        
    

    for line in temperature:
        dates_temperature.append(line["properties"]["observed"])
        values_temperature.append(line["properties"]["value"])

        
    for line in irradiation:
        dates_irradiation.append(line["properties"]["observed"])
        values_irradiation.append(line["properties"]["value"])


format = "%Y-%m-%dT%H:%M:%SZ"
time_temperature = np.vectorize(lambda data:datetime.datetime.strptime(data, format)) (np.array(dates_temperature))
values_temperature = np.array(values_temperature)


time_irradiation = np.vectorize(lambda data:datetime.datetime.strptime(data, format)) (np.array(dates_irradiation))
values_irradiation = np.array(values_irradiation)


df_temperature = pd.DataFrame()
df_temperature.insert(0, "Time stamp", time_temperature)
df_temperature.insert(0, "temperature", values_temperature)
df_temperature.sort_values(by='Time stamp', ascending = True, inplace = True)

df_irradiation = pd.DataFrame()
df_irradiation.insert(0, "Time stamp", time_irradiation)
df_irradiation.insert(0, "global irradiation", values_irradiation)
df_irradiation.sort_values(by='Time stamp', ascending = True, inplace = True) 

df = pd.DataFrame()
df.insert(0, "Time stamp", df_irradiation["Time stamp"])
df.insert(1, "outdoorTemperature", df_temperature["temperature"])
df.insert(2, "globalIrradiation", df_irradiation["global irradiation"])

df.set_index("Time stamp", inplace=True)
filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "weather_DMI.csv")
df.to_csv(filename, date_format=format, index=True)



import matplotlib.pyplot as plt
print(df)
df.plot(subplots=True)
plt.show()


