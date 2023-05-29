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
    file_path = uppath(os.path.abspath(__file__), 4)
    print(file_path)
    sys.path.append(file_path)

from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_FMUmodel import CoilModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan_FMUmodel import FanModel
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.constants import Constants
from twin4build.utils.preprocessing.get_measuring_device_from_df import get_measuring_device_from_df
from twin4build.utils.preprocessing.get_measuring_device_error import get_measuring_device_error
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.utils.time_series_input import TimeSeriesInput
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve_model import ValveModel
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator

import twin4build.utils.plot.plot as plot

def extend_model(self):
    coil = CoilModel(
                    airFlowRateMax=None,
                    airFlowRateMin=None,
                    nominalLatentCapacity=None,
                    nominalSensibleCapacity=Measurement(hasValue=96000),
                    nominalUa=None,
                    operationTemperatureMax=None,
                    operationTemperatureMin=None,
                    placementType=None,
                    operationMode=None,
                    saveSimulationResult = True,
                    id="coil")

    fan = FanModel(capacityControlType = None,
                    motorDriveType = None,
                    nominalAirFlowRate = Measurement(hasValue=5),
                    nominalPowerRate = None,
                    nominalRotationSpeed = None,
                    nominalStaticPressure = None,
                    nominalTotalPressure = Measurement(hasValue=4000),
                    operationTemperatureMax = None,
                    operationTemperatureMin = None,
                    operationalRiterial = None,
                    operationMode = None,
                    saveSimulationResult = True,
                    id="fan")
    
    valve = ValveModel(closeOffRating=None,
                    flowCoefficient=None,
                    size=None,
                    testPressure=None,
                    valveMechanism=None,
                    valveOperation=None,
                    valvePattern=None,
                    workingPressure=None,
                    waterFlowRateMax=0.888888,
                    valveAuthority=1.,
                    saveSimulationResult = True,
                    id="valve")
    

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02_FTG_MIDDEL.csv")
    FTG_MIDDEL = TimeSeriesInput(filename=filename, id="FTG_MIDDEL")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02_airflowrate_supply_kg_s.csv")
    airFlowRate = TimeSeriesInput(filename=filename, id="airFlowRate")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02_MVV1.csv")
    valvePosition = TimeSeriesInput(filename=filename, id="valvePosition")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02_FTF1.csv")
    inletWaterTemperature = TimeSeriesInput(filename=filename, id="inletWaterTemperature")


    self.add_component(coil)
    self.add_component(fan)
    self.add_component(FTG_MIDDEL)
    self.add_component(airFlowRate)
    self.add_component(valvePosition)
    self.add_component(valve)
    self.add_component(inletWaterTemperature)

    self.add_connection(airFlowRate, fan, "airFlowRate", "airFlowRate")
    self.add_connection(FTG_MIDDEL, fan, "FTG_MIDDEL", "inletAirTemperature")
    

    self.add_connection(valvePosition, valve, "valvePosition", "valvePosition")
    self.add_connection(inletWaterTemperature, coil, "inletWaterTemperature", "inletWaterTemperature")
    self.add_connection(valve, coil, "waterFlowRate", "waterFlowRate")
    self.add_connection(airFlowRate, coil, "airFlowRate", "airFlowRate")
    self.add_connection(fan, coil, "outletAirTemperature", "inletAirTemperature")
    

def test():
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


    stepSize = 60
    startPeriod = datetime.datetime(year=2022, month=1, day=1, hour=8, minute=0, second=0) 
    endPeriod = datetime.datetime(year=2022, month=1, day=1, hour=20, minute=0, second=0)

    Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False)

    simulator = Simulator(do_plot=True)
    simulator.simulate(model,
                        stepSize=stepSize,
                        startPeriod = startPeriod,
                        endPeriod = endPeriod)
    
    plt.figure()
    plt.plot(model.component_dict["fan"].savedOutput["Power"])
    plot.plot_fan(model, simulator, "fan")
    plt.show()


if __name__ == '__main__':
    test()
