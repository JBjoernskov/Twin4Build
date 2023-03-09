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
    #change the number here according to your requirement
    #desired path looks like this "D:\Projects\Twin4Build
    file_path = uppath(os.path.abspath(__file__), 11)
    #file_path = uppath(os.path.abspath(__file__), 9)
    #print(file_path)
    sys.path.append(file_path)

from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_model import SpaceHeaterModel
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.constants import Constants
def test():
    stepSize = 600
    space_heater = SpaceHeaterModel(
                    outputCapacity = Measurement(hasValue=2689),
                    temperatureClassification = "45/30-21",
                    thermalMassHeatCapacity = Measurement(hasValue=50000),
                    stepSize = stepSize,
                    saveSimulationResult = True,
                    id = "space_heater")

    waterFlowRateMax = abs(space_heater.outputCapacity.hasValue/Constants.specificHeatCapacity["water"]/(space_heater.nominalSupplyTemperature-space_heater.nominalReturnTemperature))
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "radiator_input.csv")
    filehandler = open(filename, 'rb')
    input = pd.read_csv(filehandler, low_memory=False)
    input["waterFlowRate"] = input["waterFlowRate"]*waterFlowRateMax
    input["supplyWaterTemperature"] = 40
    input = input.set_index("time")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "radiator_output.csv")
    filehandler = open(filename, 'rb')
    output = pd.read_csv(filehandler, low_memory=False)
    output = output["Power"].to_numpy()*1000
    output = np.cumsum(output*stepSize/3600/1000)

    space_heater.initialize()


    start_pred = space_heater.do_period(input, stepSize=stepSize) ####
    fig, ax = plt.subplots(2)
    ax[0].plot(start_pred, color="black", linestyle="dashed", label="predicted")
    ax[0].plot(output, color="blue", label="Measured")
    ax[0].set_title('Before calibration')
    fig.legend()
    # input = input.set_index("time")
    input.plot(subplots=True)
    space_heater.calibrate(input=input, output=output, stepSize=stepSize)
    end_pred = space_heater.do_period(input, stepSize=stepSize)
    ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
    ax[1].plot(output, color="blue", label="Measured")
    ax[1].set_title('After calibration')

    fig, ax = plt.subplots()
    arr = np.array(space_heater.savedOutput["outletWaterTemperature"])
    print(arr.shape)
    for i in range(arr.shape[1]):
        ax.plot(arr[:,i])
    plt.show()


if __name__ == '__main__':
    test()
