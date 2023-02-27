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

from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery_model import AirToAirHeatRecoveryModel
from twin4build.saref.measurement.measurement import Measurement
import pwlf
def test():
    air_to_air_heat_recovery = AirToAirHeatRecoveryModel(
                specificHeatCapacityAir = Measurement(hasValue=1000),
                eps_75_h = 0.8,
                eps_75_c = 0.8,
                eps_100_h = 0.8,
                eps_100_c = 0.8,
                primaryAirFlowRateMax = Measurement(hasValue=25000/3600*1.225),
                secondaryAirFlowRateMax = Measurement(hasValue=25000/3600*1.225),
                subSystemOf = [],
                input = {},
                output = {},
                savedInput = {},
                savedOutput = {},
                createReport = True,
                connectedThrough = [],
                connectsAt = [],
                id = "AirToAirHeatRecovery")


    stepSize = 600
    startPeriod = datetime.datetime(year=2021, month=12, day=6, hour=0, minute=0, second=0) 
    endPeriod = datetime.datetime(year=2022, month=12, day=15, hour=0, minute=0, second=0)
    format = "%m/%d/%Y %I:%M:%S %p"

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_BMS.csv")
    weather = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=60)
    weather["outdoorTemperature"] = (weather["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius

    plt.figure()
    plt.plot(weather["Time stamp"], weather["outdoorTemperature"])

    indices = np.where(weather['outdoorTemperature'].isna())[0]
    for idx in indices:
        weather.iloc[idx:idx+20,1] = np.nan

    

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_DMI.csv")
    weather_dmi = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
    

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VA01.csv")
    VA01 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
    VA01["FTF1_SV"] = (VA01["FTF1_SV"]-32)*5/9 #convert from fahrenheit to celcius
    VA01["FTT1"] = (VA01["FTT1"]-32)*5/9 #convert from fahrenheit to celcius
    VA01["FTF1"] = (VA01["FTF1"]-32)*5/9 #convert from fahrenheit to celcius




    test = pd.DataFrame()
    
    test.insert(0, "outdoorTemperature", weather["outdoorTemperature"])
    test.insert(0, "FTF1_SV", VA01["FTF1_SV"])
    test.insert(0, "FTT1", VA01["FTT1"])
    test.insert(0, "FTF1", VA01["FTF1"])
    test.insert(0, "temperature", weather_dmi["outdoorTemperature"])
    test.insert(0, "Time stamp", weather["Time stamp"])


    test = test.replace([np.inf, -np.inf], np.nan).dropna()#.reset_index()

    # test = test.set_index("Time stamp")




    # test = test[(test["Time stamp"].dt.hour >= 10)]# | (test["Time stamp"].dt.hour <= 5)]





    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(test.index, test["outdoorTemperature"], test["FTF1_SV"])
    # ax.set_xlabel("time")
    # ax.set_ylabel("outdoorTemperature")
    # ax.set_zlabel("FTF1_SV")
    # plt.show()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.plot(test["Time stamp"], test["temperature"], color="red")
    ax.plot(test["Time stamp"], test["outdoorTemperature"], color="blue")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.plot(test["Time stamp"], test["FTF1_SV"], color="red")
    ax.plot(test["Time stamp"], test["FTT1"], color="blue")
    ax.plot(test["Time stamp"], test["FTF1"], color="black")

    model = pwlf.PiecewiseLinFit(test['outdoorTemperature'], test['FTF1_SV'])
    res = model.fit(3)
    slopes = model.calc_slopes()
    print(slopes)
    print(res)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.scatter(test["outdoorTemperature"], test["FTF1_SV"])
    ax.set_xlabel("outdoorTemperature")
    ax.set_ylabel("waterTemperatureSetpoint")
    # ax.scatter(test["outdoorTemperature"], model.predict(test["outdoorTemperature"]), color="blue")
    # ax.set_xlabel("time")
    # ax.set_ylabel("outdoor temperature")
    plt.show()





if __name__ == '__main__':
    test()
