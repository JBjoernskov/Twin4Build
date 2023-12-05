import os
import sys
import datetime
from dateutil.tz import tzutc
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    print(file_path)
    sys.path.append(file_path)

from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_sampler import data_sampler
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery_system import AirToAirHeatRecoverySystem
from twin4build.saref.measurement.measurement import Measurement
import pwlf
def test():
    stepSize = 600
    startTime = datetime.datetime(year=2021, month=12, day=6, hour=0, minute=0, second=0) 
    endTime = datetime.datetime(year=2023, month=2, day=1, hour=0, minute=0, second=0)
    format = "%m/%d/%Y %I:%M:%S %p"

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_BMS.csv")
    weather = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=60)
    weather["outdoorTemperature"] = (weather["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius

    plt.figure()
    plt.plot(weather["Time stamp"], weather["outdoorTemperature"])

    indices = np.where(weather['outdoorTemperature'].isna())[0]
    for idx in indices:
        weather.iloc[idx:idx+20,1] = np.nan

    

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_DMI.csv")
    weather_dmi = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=999999)
    

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VA01.csv")
    VA01 = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=999999)
    VA01["FTF1_SV"] = (VA01["FTF1_SV"]-32)*5/9 #convert from fahrenheit to celcius
    VA01["FTT1"] = (VA01["FTT1"]-32)*5/9 #convert from fahrenheit to celcius
    VA01["FTF1"] = (VA01["FTF1"]-32)*5/9 #convert from fahrenheit to celcius




    input = pd.DataFrame()
    
    input.insert(0, "outdoorTemperature", weather["outdoorTemperature"])
    input.insert(0, "FTF1_SV", VA01["FTF1_SV"])
    input.insert(0, "FTT1", VA01["FTT1"])
    input.insert(0, "FTF1", VA01["FTF1"])
    input.insert(0, "temperature", weather_dmi["outdoorTemperature"])
    input.insert(0, "time", weather["Time stamp"])

    # input[(input["time"].dt.hour < 10) & (input["time"].dt.hour > 3)] = np.nan
    # input[(input["time"].dt.hour < 5) | (input["time"].dt.hour > 7)] = np.nan
    input = input.replace([np.inf, -np.inf], np.nan).dropna()#.reset_index()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.plot(input["time"], input["temperature"], color="red")
    ax.plot(input["time"], input["outdoorTemperature"], color="blue")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.plot(input["time"], input["FTF1_SV"], color="red")
    # ax.plot(input["Time stamp"], input["FTT1"], color="blue")
    ax.plot(input["time"], input["FTF1"], color="black")

    model = pwlf.PiecewiseLinFit(input['outdoorTemperature'], input['FTF1_SV'])
    res = model.fit(3)
    slopes = model.calc_slopes()
    print(slopes)
    print(res)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.scatter(input["outdoorTemperature"], input["FTF1_SV"])
    ax.set_xlabel("outdoorTemperature")
    ax.set_ylabel("waterTemperatureSetpoint")
    # ax.scatter(test["outdoorTemperature"], model.predict(test["outdoorTemperature"]), color="blue")
    # ax.set_xlabel("time")
    # ax.set_ylabel("outdoor temperature")


    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(input["outdoorTemperature"], input["FTF1_SV"], input["time"].dt.hour)
    ax.set_xlabel("outdoorTemperature")
    ax.set_ylabel("waterTemperatureSetpoint")
    plt.show()





if __name__ == '__main__':
    test()
