import os
import sys
import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import unittest
from dateutil import tz
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    print(file_path)
    sys.path.append(file_path)

import twin4build as tb
from twin4build.saref.measurement.measurement import Measurement
from twin4build.saref.property_value.property_value import PropertyValue
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.monitor.monitor import Monitor
from twin4build.utils.piecewise_linear_schedule import PiecewiseLinearScheduleSystem
from twin4build.saref.device.meter.meter_system import MeterSystem
from twin4build.saref.property_.power.power import Power
from twin4build.saref.property_.flow.flow import Flow
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.device.sensor.sensor_system import SensorSystem
from twin4build.utils.on_off_system import OnOffSystem
import twin4build.components as components
from twin4build.utils.uppath import uppath
import twin4build.utils.plot.plot as plot
import pandas as pd

 
def fcn(self):
    # filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "fan_airflow.csv")
    filename = "fan_airflow.csv"
    fan_airflow_property = Flow()
    fan_airflow_meter = components.MeterSystem(
                    observes=fan_airflow_property,
                    filename=filename,
                    saveSimulationResult = True,
                    id="fan airflow meter")

    filename = "supply_fan_power.csv"
    fan_power_property = Power()
    fan_power_meter = components.MeterSystem(
                    observes=fan_power_property,
                    filename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="fan power meter")
    
    filename = "fan_inlet_air_temperature.csv"
    fan_inlet_air_temperature_property = Temperature()
    fan_inlet_air_temperature_sensor = components.SensorSystem(
                    observes=fan_inlet_air_temperature_property,
                    filename=filename,
                    saveSimulationResult = True,
                    id="fan inlet air temperature sensor")
    
    filename = "coil_outlet_air_temperature.csv"
    coil_outlet_air_temperature_property = Temperature()
    coil_outlet_air_temperature_sensor = components.SensorSystem(
                    observes=coil_outlet_air_temperature_property,
                    filename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="coil outlet air temperature sensor")
    
    filename = "coil_outlet_water_temperature.csv"
    coil_outlet_water_temperature_property = Temperature()
    coil_outlet_water_temperature_sensor = components.SensorSystem(
                    observes=coil_outlet_water_temperature_property,
                    filename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="coil outlet water temperature sensor")
                    
    filename = "coil_inlet_water_temperature.csv"
    coil_inlet_water_temperature_property = Temperature()
    coil_inlet_water_temperature_sensor = components.SensorSystem(
                    observes=coil_inlet_water_temperature_property,
                    filename=filename,
                    saveSimulationResult = True,
                    id="coil inlet water temperature sensor")

    filename = "coil_valve_position.csv"
    coil_valve_position_property = OpeningPosition()
    coil_valve_position_sensor = components.SensorSystem(
                    observes=coil_valve_position_property,
                    filename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="valve position sensor")

    filename = "supply_water_temperature_setpoint.csv"
    # filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "coil_supply_water_temperature_energykey.csv")
    supply_water_temperature_property = Temperature()
    supply_water_temperature_sensor = components.SensorSystem(
                    observes=supply_water_temperature_property,
                    filename=filename,
                    saveSimulationResult = True,
                    id="supply water temperature sensor")
    
    filename = "supply_air_temperature_setpoint.csv"
    # filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "coil_supply_water_temperature_energykey.csv")
    supply_air_temperature_property = Temperature()
    supply_air_temperature_setpoint = components.SensorSystem(
                    observes=supply_air_temperature_property,
                    filename=filename,
                    saveSimulationResult = True,
                    id="supply air temperature setpoint")
    
    # supply_water_temperature_schedule = supply_air_temperature_setpoint_schedule = tb.ScheduleSystem(
    #         weekDayRulesetDict = {
    #             "ruleset_default_value": 45,
    #             "ruleset_start_minute": [],
    #             "ruleset_end_minute": [],
    #             "ruleset_start_hour": [],
    #             "ruleset_end_hour": [],
    #             "ruleset_value": []},
    #         saveSimulationResult = True,
    #         id = "supply water temperature schedule")
    

    # filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "return_airflow_temperature.csv")
    # return_airflow_temperature_property = Temperature()
    # return_airflow_temperature_sensor = SensorSystem(
    #                                 observes=return_airflow_temperature_property,
    #                                 filename=filename,
    #                                 saveSimulationResult = True,
    #                                 doUncertaintyAnalysis=False,
    #                                 id="return airflow temperature sensor")
    
    coil = components.CoilPumpValveFMUSystem(
                    airFlowRateMax=None,
                    airFlowRateMin=None,
                    nominalLatentCapacity=None,
                    nominalSensibleCapacity=PropertyValue(hasValue=96000),
                    nominalUa=PropertyValue(hasValue=1000),
                    flowCoefficient=PropertyValue(hasValue=8.7),
                    dp1_nominal=1500,
                    dpSystem=0,
                    tau_w_inlet=1,
                    tau_w_outlet=1,
                    tau_air_outlet=1,
                    operationTemperatureMax=None,
                    operationTemperatureMin=None,
                    placementType=None,
                    operationMode=None,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="coil_pump_valve")

    fan = components.FanFMUSystem(capacityControlType = None,
                    motorDriveType = None,
                    nominalAirFlowRate = PropertyValue(hasValue=11.55583), #11.55583
                    nominalPowerRate = PropertyValue(hasValue=8000), #8000
                    nominalRotationSpeed = None,
                    nominalStaticPressure = None,
                    nominalTotalPressure = PropertyValue(hasValue=557),
                    operationTemperatureMax = None,
                    operationTemperatureMin = None,
                    operationalRiterial = None,
                    operationMode = None,
                    hasProperty = [fan_power_property, fan_airflow_property],
                    saveSimulationResult=True,
                    doUncertaintyAnalysis=False,
                    id="fan")
    
    controller = components.FMUPIDControllerSystem(subSystemOf = None,
                                isContainedIn = None,
                                observes = coil_outlet_air_temperature_property,
                                saveSimulationResult=True,
                                doUncertaintyAnalysis=False,
                                id="controller")
    
    # supply_air_temperature_setpoint_schedule = PiecewiseLinearScheduleSystem(
    #         weekDayRulesetDict = {
    #             "ruleset_default_value": {"X": [20, 22.5],
    #                                       "Y": [23, 20.5]},
    #             "ruleset_start_minute": [],
    #             "ruleset_end_minute": [],
    #             "ruleset_start_hour": [],
    #             "ruleset_end_hour": [],
    #             "ruleset_value": []},
    #         saveSimulationResult = True,
    #         id = "Supply air temperature setpoint")
    
    # supply_air_temperature_setpoint_schedule = tb.ScheduleSystem(
    #         weekDayRulesetDict = {
    #             "ruleset_default_value": 21,
    #             "ruleset_start_minute": [],
    #             "ruleset_end_minute": [],
    #             "ruleset_start_hour": [],
    #             "ruleset_end_hour": [],
    #             "ruleset_value": []},
    #         saveSimulationResult = True,
    #         id = "Supply air temperature setpoint")
    
    # on_off = OnOffSystem(threshold=0.01,
    #                      is_off_value=0,
    #                      saveSimulationResult = True,
    #                     id = "On-off switch")


    coil_outlet_air_temperature_property.isPropertyOf = coil
    coil_valve_position_property.isPropertyOf = coil



    
    self.add_connection(coil_outlet_air_temperature_sensor, controller, "outletAirTemperature", "actualValue")
    # self.add_connection(return_airflow_temperature_sensor, supply_air_temperature_setpoint_schedule, "returnAirTemperature", "returnAirTemperature")
    self.add_connection(controller, coil, "inputSignal", "valvePosition")
    # self.add_connection(fan_airflow_meter, on_off, "airFlowRate", "criteriaValue")
    # self.add_connection(supply_air_temperature_setpoint_schedule, on_off, "scheduleValue", "value")
    # self.add_connection(supply_air_temperature_setpoint_schedule, on_off, "scheduleValue", "value")
    self.add_connection(supply_air_temperature_setpoint, controller, "supplyAirTemperatureSetpoint", "setpointValue")
    # self.add_connection(coil, coil_valve_position_sensor, "valvePosition", "valvePosition")
    self.add_connection(controller, coil_valve_position_sensor, "inputSignal", "valvePosition")
    self.add_connection(coil, coil_outlet_water_temperature_sensor, "outletWaterTemperature", "outletWaterTemperature")
    self.add_connection(fan_inlet_air_temperature_sensor, fan, "inletAirTemperature", "inletAirTemperature")
    self.add_connection(fan_airflow_meter, fan, "airFlowRate", "airFlowRate")
    self.add_connection(coil, coil_outlet_air_temperature_sensor, "outletAirTemperature", "outletAirTemperature")
    self.add_connection(coil, coil_inlet_water_temperature_sensor, "inletWaterTemperature", "inletWaterTemperature")
    self.add_connection(supply_water_temperature_sensor, coil, "supplyWaterTemperature", "supplyWaterTemperature")
    # self.add_connection(supply_water_temperature_schedule, coil, "scheduleValue", "supplyWaterTemperature")
    self.add_connection(fan_airflow_meter, coil, "airFlowRate", "airFlowRate")
    self.add_connection(fan, coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(fan, fan_power_meter, "Power", "Power")

def export_csv(simulator):
    model = simulator.model
    df_input = pd.DataFrame()
    df_output = pd.DataFrame()
    df_input.insert(0, "time", simulator.dateTimeSteps)
    df_output.insert(0, "time", simulator.dateTimeSteps)

    for component in model.get_component_by_class(model.component_dict, components.SensorSystem):
        if component.isPhysicalSystem:
            df = component.physicalSystem.df
            print(component.savedInput.keys())
            print(list(component.savedInput.keys())[0])
            df.iloc[:,0] = component.savedInput[list(component.savedInput.keys())[0]]
            name = component.filename.replace(".csv", "_test_synthetic.csv")
            df.set_index("time").to_csv(name)
    
    for component in model.get_component_by_class(model.component_dict, components.MeterSystem):
        if component.isPhysicalSystem:
            df = component.physicalSystem.df
            df.iloc[:,0] = simulator.dateTimeSteps
            df.iloc[:,1] = component.savedInput[list(component.savedInput.keys())[0]]
            df.set_index("time").to_csv("test_"+component.filename)

@unittest.skipIf(False, 'Currently not used')
def test_LBNL_bypass_coil_model():
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
    startTime = datetime.datetime(year=2022, month=2, day=1, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen")) 
    endTime = datetime.datetime(year=2022, month=2, day=2, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    model = Model(id="test_LBNL_bypass_coil_model", saveSimulationResult=True)
    model.load_model(infer_connections=False, fcn=fcn)


    

    ################################ SET PARAMETERS #################################
    coil = model.component_dict["coil_pump_valve"]
    # valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]

    x0 = {coil: [1.5, 10, 15, 15, 15, 2000, 1, 1, 5000, 2000, 25000, 25000, 1, 1, 1],
            fan: [0.08, -0.05, 1.31, -0.55, 0.89],
            controller: [1, 0.1, 0.1]}
    targetParameters = {
                    coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dp1_nominal", "dpPump", "dpSystem", "tau_w_inlet", "tau_w_outlet", "tau_air_outlet"],
                    fan: ["c1", "c2", "c3", "c4", "f_total"],
                    controller: ["kp", "Ti", "Td"]}
    

    theta = np.array([val for lst in x0.values() for val in lst])
    flat_component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
    flat_attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]
    model.set_parameters_from_array(theta, flat_component_list, flat_attr_list)
    #################################################################
    simulator = Simulator(model=model)
    simulator.simulate(model=model,
                    startTime=startTime,
                    endTime=endTime,
                    stepSize=stepSize)
    
    # export_csv(simulator)
    # aa
    

    monitor = Monitor(model=model)
    monitor.monitor(startTime=startTime,
                    endTime=endTime,
                    stepSize=stepSize,
                    show=True)
    
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
    plot.plot_fan(model, monitor.simulator, "fan", show=False)


if __name__=="__main__":
    test_LBNL_bypass_coil_model()