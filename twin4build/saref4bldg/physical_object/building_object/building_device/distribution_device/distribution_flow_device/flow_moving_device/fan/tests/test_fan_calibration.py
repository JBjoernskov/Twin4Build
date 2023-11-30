import os
import sys
import datetime
from dateutil.tz import tzutc
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 9)
    print(file_path)
    sys.path.append(file_path)
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_sampler import data_sampler
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan_system import FanSystem
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.constants import Constants
from twin4build.utils.preprocessing.get_measuring_device_from_df import get_measuring_device_from_df
from twin4build.utils.preprocessing.get_measuring_device_error import get_measuring_device_error
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.utils.time_series_input import TimeSeriesInput
import twin4build.utils.plot.plot as plot

def test():

    input = pd.DataFrame()
    stepSize = 60
    # startTime = datetime.datetime(year=2022, month=2, day=1, hour=10, minute=0, second=0)
    # endTime = datetime.datetime(year=2022, month=2, day=1, hour=16, minute=0, second=0)
    # startTime = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
    # endTime = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0)
    startTime = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0)
    endTime = datetime.datetime(year=2022, month=2, day=1, hour=21, minute=0, second=0)

    format = "%m/%d/%Y %I:%M:%S %p"

    fan = FanSystem(nominalAirFlowRate=Measurement(hasValue=11),
                   nominalPowerRate=Measurement(hasValue=8000),
                   c1=0.027828,
                   c2=0.026583, 
                   c3=-0.087069, 
                   c4=1.030920,
                   saveSimulationResult=True,
                   id="fan")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_FTG_MIDDEL.csv")
    FTG_MIDDEL = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_airflowrate_supply_kg_s.csv")
    airFlowRate = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_power_VI.csv")
    VE02_power_VI = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=9999)
        

    colors = sns.color_palette("deep")
    blue = colors[0]
    orange = colors[1]
    green = colors[2]
    red = colors[3]
    purple = colors[4]
    brown = colors[5]
    pink = colors[6]
    grey = colors[7]
    beis = colors[8]
    sky_blue = colors[9]
    load_params()


    input.insert(0, "time", airFlowRate["Time stamp"])
    input.insert(0, "inletAirTemperature", FTG_MIDDEL["FTG_MIDDEL"])
    input.insert(0, "airFlowRate", airFlowRate["primaryAirFlowRate"])
    input.insert(0, "Power", VE02_power_VI["VE02_power_VI"])

    tol = 10
    input = input[input["Power"]>tol]
    # input.replace([np.inf, -np.inf], np.nan, inplace=True)
    # input = input.iloc[:-shift,:]
    # input.dropna(inplace=True)
    output = input["Power"].to_numpy()

    input.drop(columns=["Power"], inplace=True)


    start_pred = fan.do_period(input=input, stepSize=stepSize, vectorize=True)

    colors = sns.color_palette("deep")
    fig, ax = plt.subplots()
    ax.plot(start_pred, color="black", linestyle="dashed", label="predicted")
    ax.plot(output, color=colors[0], label="Measured")
    ax.set_title('Using mapped nominal conditions')
    ax.set_xlabel("Timestep (10 min)")
    ax.set_ylabel("Heat [W]")
    ax.legend(loc="upper left")

    fig, ax = plt.subplots(2)
    ax[0].plot(start_pred, color="black", linestyle="dashed", label="predicted")
    ax[0].plot(output, color="blue", label="Measured")
    ax[0].set_title('Using mapped nominal conditions')
    fig.legend()
    # input = input.set_index("time")
    # input.plot(subplots=True)
    fan.calibrate(input=input, output=output, stepSize=stepSize, vectorize=True)
    end_pred = fan.do_period(input, stepSize=stepSize, vectorize=True)
    ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
    ax[1].plot(output, color="blue", label="Measured")
    ax[1].set_title('After calibration')



    

    fig, ax = plt.subplots(2)
    ax[0].scatter(input["airFlowRate"], output, color="blue", label="Measured")
    ax[0].scatter(input["airFlowRate"], start_pred, color="black", linestyle="dashed", label="predicted")
    
    ax[0].set_title('Using mapped nominal conditions')
    fig.legend()
    # input = input.set_index("time")
    # input.plot(subplots=True)
    fan.calibrate(input=input, output=output, stepSize=stepSize, vectorize=True)
    end_pred = fan.do_period(input, stepSize=stepSize, vectorize=True)
    ax[1].scatter(input["airFlowRate"], output, color="blue", label="Measured")
    ax[1].plot(input["airFlowRate"], end_pred, color="black", linestyle="dashed", label="predicted")

    
    
    ax[1].set_title('After calibration')

    plt.show()
if __name__ == '__main__':
    test()
