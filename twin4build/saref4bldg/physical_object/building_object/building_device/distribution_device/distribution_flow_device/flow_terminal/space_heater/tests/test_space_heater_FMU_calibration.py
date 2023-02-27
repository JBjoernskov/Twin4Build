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
    file_path = uppath(os.path.abspath(__file__), 9)
    print(file_path)
    sys.path.append(file_path)

from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_FMUmodel import SpaceHeaterModel
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.constants import Constants
def test():
    stepSize = 600
    space_heater = SpaceHeaterModel(
                    specificHeatCapacityWater = Measurement(hasValue=4180),
                    outputCapacity = Measurement(hasValue=2689),
                    temperatureClassification = "45/30-21",
                    thermalMassHeatCapacity = Measurement(hasValue=50000),
                    stepSize = stepSize, 
                    subSystemOf = [],
                    input = {"supplyWaterTemperature": 75},
                    output = {"outletWaterTemperature": 22,
                                "Energy": 0},
                    savedInput = {},
                    savedOutput = {},
                    createReport = True,
                    connectedThrough = [],
                    connectsAt = [],
                    id = "space_heater")


    space_heater.initialize()
    parameters = {"Q_flow_nominal": space_heater.outputCapacity.hasValue,
                    "T_a_nominal": space_heater.nominalSupplyTemperature,
                    "T_b_nominal": space_heater.nominalReturnTemperature}

    space_heater.set_parameters(parameters)


    waterFlowRateMax = abs(space_heater.outputCapacity.hasValue/Constants.specificHeatCapacity["Water"]/(space_heater.nominalSupplyTemperature-space_heater.nominalReturnTemperature))
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




    space_heater.heatTransferCoefficient = 5.54273276
    space_heater.thermalMassHeatCapacity.hasValue = 20.57764311
    start_pred = space_heater.do_period(input) ####
    

    


    fig, ax = plt.subplots(2)
    ax[0].plot(start_pred, color="black", linestyle="dashed", label="predicted")
    ax[0].plot(output, color="blue", label="Measured")
    ax[0].set_title('Before calibration')
    fig.legend()
    # input = input.set_index("time")
    input.plot(subplots=True)
    # plt.show()
    space_heater.calibrate(input=input, output=output)
    space_heater.reset()
    parameters = {"Radiator.UAEle": 0.70788274}
    space_heater.set_parameters(parameters)
    end_pred = space_heater.do_period(input)
    ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
    ax[1].plot(output, color="blue", label="Measured")
    ax[1].set_title('After calibration')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.set_size_inches(15,8)

    # for a in ax:
        # a.set_ylim([18,22])
    # plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


if __name__ == '__main__':
    test()
