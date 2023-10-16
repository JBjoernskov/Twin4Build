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
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_DryCoilDiscretizedEthyleneGlycolWater30Percent_wbypass_FMUmodel import CoilModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan_FMUmodel import FanModel
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.constants import Constants
from twin4build.utils.preprocessing.get_measuring_device_from_df import get_measuring_device_from_df
from twin4build.utils.preprocessing.get_measuring_device_error import get_measuring_device_error
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.utils.time_series_input import TimeSeriesInput
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve_wbypass_full_FMUmodel import ValveModel
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.monitor.monitor import Monitor
from twin4build.saref.device.meter.meter_model import MeterModel
from twin4build.saref.property_.power.power import Power
from twin4build.saref.property_.flow.flow import Flow
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.device.sensor.sensor_model import SensorModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_FMUmodel import ControllerModel

from twin4build.utils.uppath import uppath

import twin4build.utils.plot.plot as plot

 
def extend_model(self):
    doUncertaintyAnalysis = False
    air_flow_property = Flow()
    air_flow_meter = MeterModel(
                    measuresProperty=air_flow_property,
                    saveSimulationResult = True,
                    id="fan flow meter")

    fan_power_property = Power()
    fan_power_meter = MeterModel(
                    measuresProperty=fan_power_property,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="fan power meter")
    

    
    fan_inlet_air_temperature_property = Temperature()
    fan_inlet_air_temperature_sensor = SensorModel(
                    measuresProperty=fan_inlet_air_temperature_property,
                    saveSimulationResult = True,
                    id="fan inlet air temperature sensor")
    
    coil_outlet_air_temperature_property = Temperature()
    coil_outlet_air_temperature_sensor = SensorModel(
                    measuresProperty=coil_outlet_air_temperature_property,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="coil outlet air temperature sensor")
    
    
    coil_outlet_water_temperature_property = Temperature()
    coil_outlet_water_temperature_sensor = SensorModel(
                    measuresProperty=coil_outlet_water_temperature_property,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="coil outlet water temperature sensor")

    coil_inlet_water_temperature_property = Temperature()
    coil_inlet_water_temperature_sensor = SensorModel(
                    measuresProperty=coil_inlet_water_temperature_property,
                    saveSimulationResult = True,
                    id="coil inlet water temperature sensor")
    
    supply_water_temperature_property = Temperature()
    supply_water_temperature_sensor = SensorModel(
                    measuresProperty=supply_water_temperature_property,
                    saveSimulationResult = True,
                    id="supply water temperature sensor")
    
    valve_position_property = Temperature()
    valve_position_sensor = SensorModel(
                    measuresProperty=valve_position_property,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="valve position sensor")
    


    coil = CoilModel(
                    airFlowRateMax=None,
                    airFlowRateMin=None,
                    nominalLatentCapacity=None,
                    nominalSensibleCapacity=Measurement(hasValue=96000),
                    nominalUa=Measurement(hasValue=1000),
                    operationTemperatureMax=None,
                    operationTemperatureMin=None,
                    placementType=None,
                    operationMode=None,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="coil")

    fan = FanModel(capacityControlType = None,
                    motorDriveType = None,
                    nominalAirFlowRate = Measurement(hasValue=11.55583), #11.55583
                    nominalPowerRate = Measurement(hasValue=8000), #8000
                    nominalRotationSpeed = None,
                    nominalStaticPressure = None,
                    nominalTotalPressure = Measurement(hasValue=557),
                    operationTemperatureMax = None,
                    operationTemperatureMin = None,
                    operationalRiterial = None,
                    operationMode = None,
                    hasProperty = [fan_power_property],
                    saveSimulationResult=True,
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="fan")
    
    valve = ValveModel(closeOffRating=None,
                    flowCoefficient=Measurement(hasValue=8.7),
                    size=None,
                    testPressure=None,
                    valveMechanism=None,
                    valveOperation=None,
                    valvePattern=None,
                    workingPressure=None,
                    saveSimulationResult=True,
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="valve")
    
    controller = ControllerModel(subSystemOf = None,
                                isContainedIn = None,
                                controlsProperty = coil_outlet_air_temperature_property,
                                saveSimulationResult=True,
                                doUncertaintyAnalysis=doUncertaintyAnalysis,
                                id="controller")


    coil_outlet_air_temperature_property.isPropertyOf = coil
    valve_position_property.isPropertyOf = coil

    self.add_supply_air_temperature_setpoint_schedule_from_csv()

    supply_air_temperature_setpoint_schedule = self.component_dict["Supply air temperature setpoint"]
    
    
    

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02_FTG_MIDDEL.csv")
    FTG_MIDDEL = TimeSeriesInput(filename=filename, id="FTG_MIDDEL")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02_airflowrate_supply_kg_s.csv")
    airFlowRate = TimeSeriesInput(filename=filename, id="air flow rate")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02_MVV1.csv")
    valvePosition = TimeSeriesInput(filename=filename, id="valve position")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VA01_FTF1.csv")
    supplyWaterTemperature = TimeSeriesInput(filename=filename, id="Supply water temperature")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02_FTU1.csv")
    return_flow_temperature_schedule = TimeSeriesInput(id="Exhaust flow temperature data", filename=filename, saveSimulationResult = self.saveSimulationResult)
    
    self.add_component(coil_outlet_water_temperature_sensor)
    self.add_component(fan_inlet_air_temperature_sensor)
    self.add_component(air_flow_meter)
    self.add_component(coil_outlet_air_temperature_sensor)
    self.add_component(supply_water_temperature_sensor)
    self.add_component(coil)
    self.add_component(fan)
    self.add_component(FTG_MIDDEL)
    self.add_component(airFlowRate)
    # self.add_component(valvePosition)
    # self.add_component(valve)
    self.add_component(supplyWaterTemperature)
    self.add_component(fan_power_meter)
    self.add_component(controller)
    self.add_component(return_flow_temperature_schedule)
    self.add_component(valve_position_sensor)

    self.add_connection(supply_air_temperature_setpoint_schedule, controller, "scheduleValue", "setpointValue")
    self.add_connection(coil_outlet_air_temperature_sensor, controller, "outletAirTemperature", "actualValue")
    self.add_connection(return_flow_temperature_schedule, supply_air_temperature_setpoint_schedule, "returnAirTemperature", "returnAirTemperature")
    self.add_connection(controller, coil, "inputSignal", "valvePosition")
    self.add_connection(coil, valve_position_sensor, "valvePosition", "valvePosition")

    self.add_connection(coil, coil_outlet_water_temperature_sensor, "outletWaterTemperature", "outletWaterTemperature")
    self.add_connection(FTG_MIDDEL, fan_inlet_air_temperature_sensor, "FTG_MIDDEL", "inletAirTemperature")
    self.add_connection(fan_inlet_air_temperature_sensor, fan, "inletAirTemperature", "inletAirTemperature")
    self.add_connection(air_flow_meter, fan, "airFlowRate", "airFlowRate")
    self.add_connection(coil, coil_outlet_air_temperature_sensor, "outletAirTemperature", "outletAirTemperature")
    self.add_connection(supply_water_temperature_sensor, coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(airFlowRate, air_flow_meter, "airFlowRate", "airFlowRate")
    # self.add_connection(valvePosition, valve, "valvePosition", "valvePosition")
    self.add_connection(supplyWaterTemperature, supply_water_temperature_sensor, "supplyWaterTemperature", "supplyWaterTemperature")
    # self.add_connection(valve, coil, "waterFlowRate", "waterFlowRate")
    self.add_connection(air_flow_meter, coil, "airFlowRate", "airFlowRate")
    self.add_connection(fan, coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(fan, fan_power_meter, "Power", "Power")
    

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
    startPeriod = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0) 
    endPeriod = datetime.datetime(year=2022, month=2, day=1, hour=21, minute=0, second=0)

    Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False)

    simulator = Simulator(model=model,
                            do_plot=False)
    

    ################################ SET PARAMETERS #################################
    coil = model.component_dict["coil"]
    # valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]

    # fan.nominalPowerRate.hasValue = 7500
    # fan.nominalAirFlowRate.hasValue = 10
    targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dpCoil_nominal", "dpPump", "dpValve_nominal", "dpSystem"],
                                    fan: ["c1", "c2", "c3", "c4", "f_total"],
                                    controller: ["kp", "Ti", "Td"]}
    x0 = {coil: [1.5, 10, 15, 15, 15, 8000, 1.5, 1.5, 10000, 10000, 10000, 10000, 10000],
                fan: [0.08, -0.05, 1.31, -0.55, 0.89],
                controller: [50, 50, 50]}
    theta = np.array([val for lst in x0.values() for val in lst])
    flat_component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
    flat_attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]
    model.set_parameters_from_array(theta, flat_component_list, flat_attr_list)
    #################################################################
    # simulator.simulate(model=model,
    #                 startPeriod=startPeriod,
    #                 endPeriod=endPeriod,
    #                 stepSize=stepSize)
    

    
    
    

    monitor = Monitor(model)
    monitor.monitor(startPeriod=startPeriod,
                    endPeriod=endPeriod,
                    stepSize=stepSize,
                    do_plot=True)
    
    print(monitor.get_MSE())
    print(monitor.get_RMSE())

    



    id_list = ["fan power meter", "fan power meter", "coil outlet air temperature sensor", "coil outlet water temperature sensor"]
    # id_list = ["Space temperature sensor", "VE02 Primary Airflow Temperature AHR sensor", "VE02 Primary Airflow Temperature AHC sensor"]

    # "VE02 Primary Airflow Temperature AHR sensor": "VE02_FTG_MIDDEL",
    #                      "VE02 Primary Airflow Temperature AHC sensor": "VE02_FTI1",
    fig,ax = plt.subplots()
    ax.plot(model.component_dict["fan"].savedInput["inletAirTemperature"], label="inletAirTemperature")
    ax.plot(model.component_dict["fan"].savedOutput["outletAirTemperature"], label="outletAirTemperature")
    ax.legend()


    # facecolor = tuple(list(beis)+[0.5])
    # edgecolor = tuple(list((0,0,0))+[0.5])
    # for id_ in id_list:
    #     fig,axes = monitor.plot_dict[id_]
    #     key = list(model.component_dict[id_].inputUncertainty.keys())[0]
    #     output = np.array(model.component_dict[id_].savedOutput[key])
    #     outputUncertainty = np.array(model.component_dict[id_].savedOutputUncertainty[key])
    #     axes[0].fill_between(monitor.simulator.dateTimeSteps, y1=output-outputUncertainty, y2=output+outputUncertainty, facecolor=facecolor, edgecolor=edgecolor, label="Prediction uncertainty")
    #     for ax in axes:
    #         myFmt = mdates.DateFormatter('%H')
    #         ax.xaxis.set_major_formatter(myFmt)
    #         h, l = ax.get_legend_handles_labels()
    #         n = len(l)
    #         box = ax.get_position()
    #         ax.set_position([0.12, box.y0, box.width, box.height])
    #         ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.15), prop={'size': 8}, ncol=n)
    #         ax.yaxis.label.set_size(15)
        # for ax in axes:
        #     h, l = ax.get_legend_handles_labels()
        #     n = len(l)
        #     box = ax.get_position()
        #     ax.set_position([0.12, box.y0, box.width, box.height])
        #     ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.15), prop={'size': 8}, ncol=n)
        #     ax.yaxis.label.set_size(15)
        #     # ax.axvline(line_date, color=monitor.colors[3])
        #     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    

    monitor.save_plots()

    plot.plot_fan(model, monitor.simulator, "fan")

    plt.show()


if __name__ == '__main__':
    test()
