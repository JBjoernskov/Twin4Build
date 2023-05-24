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

from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_FMUmodel import SpaceHeaterModel
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.constants import Constants


def valve_model(u, waterFlowRateMax):
    valve_authority = 1
    u_norm = u/(u**2*(1-valve_authority)+valve_authority)**(0.5)
    m_w = u_norm*waterFlowRateMax
    return m_w


def test():
    stepSize = 60
    space_heater = SpaceHeaterModel(
                    outputCapacity = Measurement(hasValue=2671),
                    # outputCapacity = Measurement(hasValue=1432*5),
                    temperatureClassification = "45/30-21",
                    # temperatureClassification = "70/40-20",
                    thermalMassHeatCapacity = Measurement(hasValue=500000),
                    stepSize = stepSize,
                    saveSimulationResult = True,
                    id = "space_heater")

    # waterFlowRateMax = abs(space_heater.outputCapacity.hasValue/Constants.specificHeatCapacity["water"]/(space_heater.nominalSupplyTemperature-space_heater.nominalReturnTemperature))
    waterFlowRateMax = 0.0222222
    input = pd.DataFrame()

    # startPeriod = datetime.datetime(year=2021, month=12, day=20, hour=0, minute=0, second=0) 
    # endPeriod = datetime.datetime(year=2021, month=12, day=28, hour=0, minute=0, second=0)

    # startPeriod = datetime.datetime(year=2022, month=12, day=1, hour=0, minute=0, second=0) 
    # endPeriod = datetime.datetime(year=2022, month=12, day=31, hour=0, minute=0, second=0)
    startPeriod = datetime.datetime(year=2021, month=12, day=20, hour=0, minute=0, second=0) 
    endPeriod = datetime.datetime(year=2021, month=12, day=28, hour=0, minute=0, second=0)
    format = "%m/%d/%Y %I:%M:%S %p"

    # "%d-%M-%yyyy %HH:%mm"


    response_filename = os.path.join(uppath(os.path.abspath(__file__), 10), "test", "data", "time_series_data", "OE20-601b-2_kafka_temperature.txt")
    data = [json.loads(line) for line in open(response_filename, 'rb')]
    data = data[1:] #remove header information
    data = np.array([row[0][0] for row in data])
    data = data[data[:, 0].argsort()]
    constructed_time_list,constructed_value_list,got_data = sample_data(data=data, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, dt_limit=1200)
 

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "OE20-601b-2.csv")
    space = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "OE20-601b-2_heat_consumption_Dec_2021.csv")
    heat = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VA01_FTF1_SV.csv")
    VA01_FTF1_SV = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    shift = int(1*3600/stepSize)
    input.insert(0, "time", space["Time stamp"])
    input.insert(0, "indoorTemperature", space["Indoor air temperature (Celcius)"])
    # input.insert(0, "waterFlowRate", space["Space heater valve position (0-100%)"]*waterFlowRateMax/100)
    input.insert(0, "waterFlowRate", valve_model(space["Space heater valve position (0-100%)"]/100, waterFlowRateMax))
    input.insert(0, "supplyWaterTemperature", VA01_FTF1_SV["FTF1_SV"])
    input.insert(0, "Power", heat["Effekt [kWh]"])

    tol = 1e-5
    x = input["Power"]
    x[(x<tol) & (input["waterFlowRate"]>tol)] = np.nan
    input["Power"] = x.interpolate()

    input.replace([np.inf, -np.inf], np.nan, inplace=True)

    # input = input.iloc[:-shift,:]
    input["Power"] = input["Power"].shift(-shift)
    input.dropna(inplace=True)
    output = input["Power"].to_numpy()*1000
    # output = np.cumsum(output*stepSize/3600/1000)
    
    fig, ax = plt.subplots(4, sharex=True)
    input.set_index("time").plot(subplots=True, ax=ax)
    for i in range(0,3,1):
        ax[i].set_xlabel("")
        for tick in ax[i].xaxis.get_minor_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax[i].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
    for a in ax:
        a.legend(prop={'size': 14})
    
    plt.show()
    
    input.drop(columns=["Power"], inplace=True)
    # input = input.iloc[0:-shift]
    # output = output[shift:]
    input = input.set_index("time")

    

    # output_csv = pd.DataFrame()
    # output_csv.insert(0,"Power", output)

    space_heater.output["outletTemperature"] = input["indoorTemperature"].iloc[0]
    space_heater.initialize()
    start_pred = space_heater.do_period(input, stepSize=stepSize) ####


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
    space_heater.calibrate(input=input, output=output, stepSize=stepSize)

    end_pred = space_heater.do_period(input, stepSize=stepSize)
    ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
    ax[1].plot(output, color="blue", label="Measured")
    ax[1].set_title('After calibration')


    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width, box.height-0.05])

    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width, box.height-0.05])

    fig.text(0.5, 0.01, "Timestep (10 min)", va='center', ha='center', rotation='horizontal', fontsize=25, color="black")
    fig.text(0.05, 0.5, "Power [kW]", va='center', ha='center', rotation='vertical', fontsize=25, color="black")

    cumsum_output_meas = np.cumsum(output*stepSize/3600/1000)
    cumsum_output_pred = np.cumsum(start_pred*stepSize/3600/1000)
    fig, ax = plt.subplots(1)
    ax.plot(cumsum_output_meas, color="blue")
    ax.plot(cumsum_output_pred, color="black")

    plt.show()

    

if __name__ == '__main__':
    test()
