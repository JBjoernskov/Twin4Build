import os
import datetime
import pandas as pd
import unittest
import dateutil
from dateutil import tz
import pytz
import sys
# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
import twin4build as tb
import twin4build.utils.plot.plot as plot
from twin4build.utils.uppath import uppath
def fcn(self):
    '''
        The fcn() function adds connections between components in a system model, 
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system 
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''
    occupancy_schedule = tb.ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [3,5,20,25,27,7,3]}, #35
            add_noise = True,
            saveSimulationResult = True,
            id = "Occupancy schedule")
    
    indoor_temperature_setpoint_schedule = tb.ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [7],
                "ruleset_end_hour": [17],
                "ruleset_value": [25]},
            weekendRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [7],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            mondayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [7],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            saveSimulationResult = True,
            id = "Temperature setpoint schedule")

    supply_water_temperature_setpoint_schedule = tb.PiecewiseLinearScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": {"X": [-5, 5, 7],
                                          "Y": [58, 65, 60.5]},
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [5],
                "ruleset_end_hour": [7],
                "ruleset_value": [{"X": [-7, 5, 9],
                                    "Y": [72, 55, 50]}]},
            saveSimulationResult = True,
            id = "Supply water temperature")
    
    supply_air_temperature_schedule = tb.ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 21,
                "ruleset_start_minute": [],
                "ruleset_end_minute": [],
                "ruleset_start_hour": [],
                "ruleset_end_hour": [],
                "ruleset_value": []},
            saveSimulationResult = True,
            id = "Supply air temperature")
    

    space = tb.BuildingSpaceSystem(id="OE20-601b-2", airVolume=466.54)
    temperature_controller = tb.ControllerSystem(id="Temperature controller", K_p=2.50773924e-01, K_i=4.38174242e-01, K_d=0)
    co2_controller = tb.RulebasedControllerSystem(id="CO2 controller")
    supply_damper = tb.DamperSystem(id="Supply damper", airFlowRateMax=tb.PropertyValue(hasValue=0.544444444))
    exhaust_damper = tb.DamperSystem(id="Exhaust damper", airFlowRateMax=tb.PropertyValue(hasValue=0.544444444))
    space_heater = tb.SpaceHeaterSystem(id="Space heater", 
                                        heatTransferCoefficient=8.31495759e+01,
                                        thermalMassHeatCapacity=tb.PropertyValue(hasValue=2.72765272e+06),
                                        temperatureClassification=tb.PropertyValue("45/30-21"))
    valve = tb.ValveSystem(id="Valve", waterFlowRateMax=0.0202, valveAuthority=1)
    heating_meter = tb.MeterSystem(id="Heating meter")
    temperature_sensor = tb.SensorSystem(id="Temperature sensor")
    co2_sensor = tb.SensorSystem(id="CO2 sensor")
    outdoor_environment = self.component_dict["outdoor_environment"]


    # self.add_connection(co2_controller, supply_damper, "inputSignal", "damperPosition")
    # self.add_connection(co2_controller, exhaust_damper, "inputSignal", "damperPosition")
    # self.add_connection(co2_sensor, co2_controller, "measuredValue", "actualValue")
    # self.add_connection(supply_damper, space, "airFlowRate", "supplyAirFlowRate")
    # self.add_connection(exhaust_damper, space, "airFlowRate", "returnAirFlowRate")
    # self.add_connection(supply_damper, space, "damperPosition", "supplyDamperPosition")
    # self.add_connection(exhaust_damper, space, "damperPosition", "returnDamperPosition")
    self.add_connection(outdoor_environment, space, "outdoorTemperature", "outdoorTemperature")
    self.add_connection(outdoor_environment, space, "globalIrradiation", "globalIrradiation")
    self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
    self.add_connection(supply_water_temperature_setpoint_schedule, space_heater, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(supply_water_temperature_setpoint_schedule, space, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(supply_air_temperature_schedule, space, "scheduleValue", "supplyAirTemperature")
    self.add_connection(space, temperature_sensor, "indoorTemperature", "measuredValue")
    self.add_connection(space, co2_sensor, "indoorCo2Concentration", "measuredValue")
    self.add_connection(indoor_temperature_setpoint_schedule, temperature_controller, "scheduleValue", "setpointValue")
    self.add_connection(temperature_sensor, temperature_controller, "measuredValue", "actualValue")
    self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")
    self.add_connection(valve, space, "valvePosition", "valvePosition")
    self.add_connection(temperature_controller, valve, "inputSignal", "valvePosition")
    self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")
    self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature")


    t = tb.Temperature()
    c = tb.Co2()
    temperature_controller.observes = t
    co2_controller.observes = c
    t.isPropertyOf = space
    c.isPropertyOf = space



    initial_temperature = 21
    custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
    self.set_custom_initial_dict(custom_initial_dict)


class TestOU44RoomCase(unittest.TestCase):
    @unittest.skipIf(True, 'Currently not used')
    def test_OU44_room_case(self, show=False):
        stepSize = 600 #Seconds
        startTime = datetime.datetime(year=2022, month=1, day=3, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime = datetime.datetime(year=2022, month=1, day=8, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

        model = tb.Model(id="default", saveSimulationResult=True)

        filename = os.path.join(uppath(os.path.abspath(__file__), 1), "weather_DMI.csv")
        model.add_outdoor_environment_system(filename=filename)
        model.load_model(fcn=fcn, infer_connections=False)
        

        simulator = tb.Simulator()
        simulator.simulate(model,
                            stepSize=stepSize,
                            startTime=startTime,
                            endTime=endTime)
        # export_csv(simulator)

        space_name = "OE20-601b-2"
        space_heater_name = "Space heater"
        temperature_controller_name = "Temperature controller"
        CO2_controller_name = "CO2 controller"
        damper_name = "Supply damper"

        plot.plot_space_temperature(model, simulator, space_name)
        plot.plot_space_CO2(model, simulator, space_name)
        plot.plot_outdoor_environment(model, simulator)
        plot.plot_space_heater(model, simulator, space_heater_name)
        plot.plot_space_heater_energy(model, simulator, space_heater_name)
        plot.plot_temperature_controller(model, simulator, temperature_controller_name)
        plot.plot_CO2_controller_rulebased(model, simulator, CO2_controller_name)
        plot.plot_space_wDELTA(model, simulator, space_name)
        plot.plot_space_energy(model, simulator, space_name)
        plot.plot_damper(model, simulator, damper_name, show)

        
if __name__=="__main__":
    d = TestOU44RoomCase()
    d.test_OU44_room_case(show=True)