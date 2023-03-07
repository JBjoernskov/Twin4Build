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
<<<<<<< Updated upstream
    file_path = uppath(os.path.abspath(__file__), 11)
=======
    #change the number here according to your requirement
    #desired path looks like this "D:\Projects\Twin4Build
    file_path = uppath(os.path.abspath(__file__), 11)
    #file_path = uppath(os.path.abspath(__file__), 9)
>>>>>>> Stashed changes
    print(file_path)
    sys.path.append(file_path)
    
    

from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery_model import AirToAirHeatRecoveryModel
from twin4build.saref.measurement.measurement import Measurement
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
                saveSimulationResult = True,
                connectedThrough = [],
                connectsAt = [],
                id = "AirToAirHeatRecovery")

    input = pd.DataFrame()

    stepSize = 600
    startPeriod = datetime.datetime(year=2021, month=10, day=1, hour=0, minute=0, second=0, tzinfo=tzutc()) 
    endPeriod = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    format = "%m/%d/%Y %I:%M:%S %p"

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "weather_BMS.csv")
    weather = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    temp = weather.copy()
    # weather["outdoorTemperature"] = (weather["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_efficiency.csv")
    VE02_efficiency = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_airflowrate_supply_kg_s.csv")
    VE02_primaryAirFlowRate = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_primaryAirFlowRate["primaryAirFlowRate"] = VE02_primaryAirFlowRate["primaryAirFlowRate"]*0.0283168466/60*1.225 #convert from cubic feet per minute to kg/s

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_airflowrate_exhaust_kg_s.csv")
    VE02_secondaryAirFlowRate = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_secondaryAirFlowRate["secondaryAirFlowRate"] = VE02_secondaryAirFlowRate["secondaryAirFlowRate"]*0.0283168466/60*1.225 #convert from cubic feet per minute to kg/s

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_FTU1.csv")
    VE02_FTU1 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_FTU1["FTU1"] = (VE02_FTU1["FTU1"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_FTG_MIDDEL.csv")
    VE02_FTG_MIDDEL = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_FTG_MIDDEL["FTG_MIDDEL"] = (VE02_FTG_MIDDEL["FTG_MIDDEL"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_FTI_KALK_SV.csv")
    VE02_FTI_KALK_SV = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_FTI_KALK_SV["FTI_KALK_SV"] = (VE02_FTI_KALK_SV["FTI_KALK_SV"]-32)*5/9 #convert from fahrenheit to celcius

    
    
    # primaryTemperatureIn can be calculated based on logged efficiency and temperature measurements.
    # However, primaryTemperatureIn should also be equal to outdoor temperature, which is available.
    # Outdoor temperature is therefore currently used. 
    primaryTemperatureIn = (VE02_efficiency["efficiency"]/100*VE02_FTU1["FTU1"]-VE02_FTG_MIDDEL["FTG_MIDDEL"])/(VE02_efficiency["efficiency"]/100-1)




    input.insert(0, "time", VE02_FTI_KALK_SV["Time stamp"])
    input.insert(0, "primaryAirFlowRate", VE02_primaryAirFlowRate["primaryAirFlowRate"])
    input.insert(0, "secondaryAirFlowRate", VE02_secondaryAirFlowRate["secondaryAirFlowRate"])
    # input.insert(0, "primaryTemperatureIn", primaryTemperatureIn)
    input.insert(0, "primaryTemperatureIn", weather["outdoorTemperature"])
    input.insert(0, "secondaryTemperatureIn", VE02_FTU1["FTU1"])
    input.insert(0, "primaryTemperatureOutSetpoint", VE02_FTI_KALK_SV["FTI_KALK_SV"])
    input.insert(0, "primaryTemperatureOut", VE02_FTG_MIDDEL["FTG_MIDDEL"])


    tol = 1e-5
    input_plot = input.iloc[20000:21000,:].reset_index(drop=True)
    output_plot = input_plot["primaryTemperatureOut"].to_numpy()



    input.replace([np.inf, -np.inf], np.nan, inplace=True)
    input = (input.loc[(input["primaryAirFlowRate"]>tol) | (input["secondaryAirFlowRate"]>tol)]).dropna().reset_index(drop=True) # Filter data to remove 0 airflow data
    output = input["primaryTemperatureOut"].to_numpy()
    input.drop(columns=["primaryTemperatureOut"])



    start_pred = air_to_air_heat_recovery.do_period(input_plot) ####
    fig, ax = plt.subplots(2)
    ax[0].plot(start_pred, color="black", linestyle="dashed", label="predicted")
    ax[0].plot(output_plot, color="blue", label="Measured")
    ax[0].set_title('Before calibration')
    fig.legend()
    input.set_index("time")
    input.plot(subplots=True)
    air_to_air_heat_recovery.calibrate(input=input, output=output)
    end_pred = air_to_air_heat_recovery.do_period(input_plot)
    ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
    ax[1].plot(output_plot, color="blue", label="Measured")
    ax[1].set_title('After calibration')


    for a in ax:
        a.set_ylim([18,22])
    plt.show()


if __name__ == '__main__':
    test()
