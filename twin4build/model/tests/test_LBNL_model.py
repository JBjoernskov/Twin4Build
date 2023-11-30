import os
import sys
import datetime
import numpy as np
import seaborn as sns
import unittest
from dateutil.tz import gettz
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    print(file_path)
    sys.path.append(file_path)
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_DryCoilDiscretizedEthyleneGlycolWater30Percent_FMUmodel import CoilSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan_system_fmu import FanSystem
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.plot.plot import load_params
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve_wbypass_full_FMUmodel import ValveSystem
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.monitor.monitor import Monitor
from twin4build.saref.device.meter.meter_system import MeterSystem
from twin4build.saref.property_.power.power import Power
from twin4build.saref.property_.flow.flow import Flow
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.device.sensor.sensor_system import SensorSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_system_fmu import ControllerSystem
from twin4build.utils.uppath import uppath
from twin4build.utils.piecewise_linear_schedule import PiecewiseLinearSchedule
import twin4build.utils.plot.plot as plot


def extend_model(self):
    doUncertaintyAnalysis = False

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
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
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
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="coil outlet air temperature sensor")
    
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "coil_outlet_water_temperature.csv")
    coil_outlet_water_temperature_property = Temperature()
    coil_outlet_water_temperature_sensor = SensorSystem(
                    measuresProperty=coil_outlet_water_temperature_property,
                    physicalSystemFilename=filename,
                    saveSimulationResult = True,
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
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
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="valve position sensor")
    

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "return_airflow_temperature.csv")
    return_airflow_temperature_property = Temperature()
    return_airflow_temperature_sensor = SensorSystem(
                                    measuresProperty=return_airflow_temperature_property,
                                    physicalSystemFilename=filename,
                                    saveSimulationResult = True,
                                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                                    id="return airflow temperature sensor")
    

    coil = CoilSystem(
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
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
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
                    doUncertaintyAnalysis=doUncertaintyAnalysis,
                    id="valve")
    
    controller = ControllerSystem(subSystemOf = None,
                                isContainedIn = None,
                                controlsProperty = coil_outlet_air_temperature_property,
                                saveSimulationResult=True,
                                doUncertaintyAnalysis=doUncertaintyAnalysis,
                                id="controller")
    
    supply_air_temperature_setpoint_schedule = PiecewiseLinearSchedule(
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


    coil_outlet_air_temperature_property.isPropertyOf = coil
    coil_valve_position_property.isPropertyOf = valve

    self.add_connection(supply_air_temperature_setpoint_schedule, controller, "scheduleValue", "setpointValue")
    self.add_connection(coil_outlet_air_temperature_sensor, controller, "outletAirTemperature", "actualValue")
    self.add_connection(return_airflow_temperature_sensor, supply_air_temperature_setpoint_schedule, "returnAirTemperature", "returnAirTemperature")
    self.add_connection(controller, valve, "inputSignal", "valvePosition")
    self.add_connection(valve, coil_valve_position_sensor, "valvePosition", "valvePosition")
    self.add_connection(coil, coil_outlet_water_temperature_sensor, "outletWaterTemperature", "outletWaterTemperature")
    self.add_connection(fan_inlet_air_temperature_sensor, fan, "inletAirTemperature", "inletAirTemperature")
    self.add_connection(fan_airflow_meter, fan, "airFlowRate", "airFlowRate")
    self.add_connection(coil, coil_outlet_air_temperature_sensor, "outletAirTemperature", "outletAirTemperature")
    self.add_connection(coil_inlet_water_temperature_sensor, coil, "inletWaterTemperature", "inletWaterTemperature")
    self.add_connection(valve, coil, "waterFlowRate", "waterFlowRate")
    self.add_connection(fan_airflow_meter, coil, "airFlowRate", "airFlowRate")
    self.add_connection(fan, coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(fan, fan_power_meter, "Power", "Power")
    
@unittest.skipIf(False, 'Currently not used')
def test_LBNL_model():
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
    startPeriod = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")) 
    endPeriod = datetime.datetime(year=2022, month=2, day=1, hour=21, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))

    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False, extend_model=extend_model)
    simulator = Simulator(model=model)
    

    ################################ SET PARAMETERS #################################
    coil = model.component_dict["coil"]
    valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]

    # fan.nominalPowerRate.hasValue = 7500
    # fan.nominalAirFlowRate.hasValue = 10
    targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
                                    valve: ["mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dpCoil_nominal", "riseTime"],
                                    fan: ["c1", "c2", "c3", "c4", "f_total"],
                                    controller: ["kp", "Ti", "Td"]}
    x0 = {coil: [1.5, 10, 15, 15, 15, 8000],
                valve: [1.5, 1.5, 10000, 10000, 1],
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
    monitor.save_plots()
    plot.plot_fan(model, monitor.simulator, "fan", show=False)

if __name__=="__main__":
    test_LBNL_model()