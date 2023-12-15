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
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_DryCoilDiscretizedEthyleneGlycolWater30Percent_wbypass_FMUmodel import CoilSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan_system_fmu import FanSystem
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve_wbypass_full_FMUmodel import ValveSystem
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
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_system_fmu import ControllerSystem
from twin4build.utils.uppath import uppath
import twin4build.utils.plot.plot as plot

 
def fcn(self):
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "fan_airflow.csv")
    fan_airflow_property = Flow()
    fan_airflow_meter = MeterSystem(
                    measuresProperty=fan_airflow_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    id="fan airflow meter")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "supply_fan_power.csv")
    fan_power_property = Power()
    fan_power_meter = MeterSystem(
                    measuresProperty=fan_power_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="fan power meter")
    
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "fan_inlet_air_temperature.csv")
    fan_inlet_air_temperature_property = Temperature()
    fan_inlet_air_temperature_sensor = SensorSystem(
                    measuresProperty=fan_inlet_air_temperature_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    id="fan inlet air temperature sensor")
    
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "coil_outlet_air_temperature.csv")
    coil_outlet_air_temperature_property = Temperature()
    coil_outlet_air_temperature_sensor = SensorSystem(
                    measuresProperty=coil_outlet_air_temperature_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="coil outlet air temperature sensor")
    
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "coil_outlet_water_temperature.csv")
    coil_outlet_water_temperature_property = Temperature()
    coil_outlet_water_temperature_sensor = SensorSystem(
                    measuresProperty=coil_outlet_water_temperature_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="coil outlet water temperature sensor")
                    
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "coil_inlet_water_temperature.csv")
    coil_inlet_water_temperature_property = Temperature()
    coil_inlet_water_temperature_sensor = SensorSystem(
                    measuresProperty=coil_inlet_water_temperature_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    id="coil inlet water temperature sensor")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "coil_valve_position.csv")
    coil_valve_position_property = OpeningPosition()
    coil_valve_position_sensor = SensorSystem(
                    measuresProperty=coil_valve_position_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="valve position sensor")

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "supply_water_temperature_setpoint.csv")
    supply_water_temperature_property = Temperature()
    supply_water_temperature_sensor = SensorSystem(
                    measuresProperty=supply_water_temperature_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    id="supply water temperature sensor")
    

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "return_airflow_temperature.csv")
    return_airflow_temperature_property = Temperature()
    return_airflow_temperature_sensor = SensorSystem(
                                    measuresProperty=return_airflow_temperature_property,
                                    physicalSystemFilename=filename,
                                    saveSimulationResult = True,
                                    doUncertaintyAnalysis=False,
                                    id="return airflow temperature sensor")
    


    coil = CoilSystem(
                    airFlowRateMax=None,
                    airFlowRateMin=None,
                    nominalLatentCapacity=None,
                    nominalSensibleCapacity=Measurement(hasValue=96000),
                    nominalUa=Measurement(hasValue=1000),
                    flowCoefficient=8.7,
                    dp1_nominal=1500,
                    operationTemperatureMax=None,
                    operationTemperatureMin=None,
                    placementType=None,
                    operationMode=None,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=False,
                    id="coil")

    fan = FanSystem(capacityControlType = None,
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
                    doUncertaintyAnalysis=False,
                    id="fan")
    
    valve = ValveSystem(closeOffRating=None,
                    flowCoefficient=Measurement(hasValue=8.7),
                    size=None,
                    testPressure=None,
                    valveMechanism=None,
                    valveOperation=None,
                    valvePattern=None,
                    workingPressure=None,
                    saveSimulationResult=True,
                    doUncertaintyAnalysis=False,
                    id="valve")
    
    controller = ControllerSystem(subSystemOf = None,
                                isContainedIn = None,
                                controlsProperty = coil_outlet_air_temperature_property,
                                saveSimulationResult=True,
                                doUncertaintyAnalysis=False,
                                id="controller")
    
    supply_air_temperature_setpoint_schedule = PiecewiseLinearScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": {"X": [20, 22.5],
                                          "Y": [23, 20.5]},
                "ruleset_start_minute": [],
                "ruleset_end_minute": [],
                "ruleset_start_hour": [],
                "ruleset_end_hour": [],
                "ruleset_value": []},
            saveSimulationResult = True,
            id = "Supply air temperature setpoint")
    
    on_off = OnOffSystem(threshold=0.01,
                         is_off_value=0,
                         saveSimulationResult = True,
                        id = "On-off switch")


    coil_outlet_air_temperature_property.isPropertyOf = coil
    coil_valve_position_property.isPropertyOf = coil  


    fan_airflow_meter
    fan_power_meter
    fan_inlet_air_temperature_sensor
    coil_outlet_air_temperature_sensor
    coil_outlet_water_temperature_sensor
    coil_inlet_water_temperature_sensor
    coil_valve_position_sensor
    return_airflow_temperature_sensor

    
    self.add_connection(coil_outlet_air_temperature_sensor, controller, "outletAirTemperature", "actualValue")
    self.add_connection(return_airflow_temperature_sensor, supply_air_temperature_setpoint_schedule, "returnAirTemperature", "returnAirTemperature")
    self.add_connection(controller, coil, "inputSignal", "valvePosition")
    self.add_connection(fan_airflow_meter, on_off, "airFlowRate", "criteriaValue")
    self.add_connection(supply_air_temperature_setpoint_schedule, on_off, "scheduleValue", "value")
    self.add_connection(on_off, controller, "value", "setpointValue")
    self.add_connection(coil, coil_valve_position_sensor, "valvePosition", "valvePosition")
    self.add_connection(coil, coil_outlet_water_temperature_sensor, "outletWaterTemperature", "outletWaterTemperature")
    self.add_connection(fan_inlet_air_temperature_sensor, fan, "inletAirTemperature", "inletAirTemperature")
    self.add_connection(fan_airflow_meter, fan, "airFlowRate", "airFlowRate")
    self.add_connection(coil, coil_outlet_air_temperature_sensor, "outletAirTemperature", "outletAirTemperature")
    self.add_connection(coil, coil_inlet_water_temperature_sensor, "inletWaterTemperature", "inletWaterTemperature")
    self.add_connection(supply_water_temperature_sensor, coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(fan_airflow_meter, coil, "airFlowRate", "airFlowRate")
    self.add_connection(fan, coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(fan, fan_power_meter, "Power", "Power")
    
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
    startTime = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen")) 
    endTime = datetime.datetime(year=2022, month=2, day=1, hour=21, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False, fcn=fcn)


    

    ################################ SET PARAMETERS #################################
    coil = model.component_dict["coil"]
    # valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]

    # fan.nominalPowerRate.hasValue = 7500
    # fan.nominalAirFlowRate.hasValue = 10
    targetParameters = {
                        # coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dpCoil_nominal", "dpPump", "dpValve_nominal", "dpSystem"],
                        coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dp1_nominal"],
                        fan: ["c1", "c2", "c3", "c4", "f_total"],
                        controller: ["kp", "Ti", "Td"]}
    x0 = {coil: [0.5, 3.33, 23.95, 25.71, 18.17, 2705.49, 0.5, 2.05, 262677.29, 1942.25],
                fan: [0.07, -0.02, 1, 0.13, 0.71],
                controller: [0.001, 0.2, 0]}
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