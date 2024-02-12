import os
import sys
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# Only for testing before distributing package.
# If the package is installed, this is not needed.
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
    
from twin4build.evaluator.evaluator import Evaluator
from twin4build.model.model import Model
from twin4build.utils.schedule import ScheduleSystem
from twin4build.utils.flow_junction_system import FlowJunctionSystem
from twin4build.utils.piecewise_linear_schedule import PiecewiseLinearScheduleSystem

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

def fcn1(self):
    '''
        outdoor_environment: A component representing the outdoor environment of the building.
        supply_air_temperature_setpoint_schedule: A component representing the setpoint temperature for supply air to the building.
        supply_water_temperature_setpoint_schedule: A component representing the setpoint temperature for supply water to the building.
        space: A component representing the indoor space being conditioned by the HVAC system.
        heating_coil: A component representing the heating coil in the HVAC system.
        indoor_temperature_setpoint_schedule: A ScheduleSystem object representing the desired indoor temperature setpoints over time.
    '''
    logger.info("[Extend Model1 - Test Evaluator]")

    occupancy_schedule = ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [3,5,20,25,27,7,3]}, #35
            add_noise = True,
            saveSimulationResult = True,
            id = "OE20-601b-2| Occupancy schedule")
    
    indoor_temperature_setpoint_schedule = ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [4],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            weekendRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [4],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            saveSimulationResult = True,
            id = "OE20-601b-2| Temperature setpoint schedule")

    supply_water_temperature_setpoint_schedule = PiecewiseLinearScheduleSystem(
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
            id = "Heating system| Supply water temperature schedule")

    self._add_component(occupancy_schedule)
    self._add_component(indoor_temperature_setpoint_schedule)
    self._add_component(supply_water_temperature_setpoint_schedule)
    initial_temperature = 21
    custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
    self.set_custom_initial_dict(custom_initial_dict)

def fcn2(self):
    '''
        supply_air_temperature_setpoint_schedule: a component representing the supply air temperature setpoint schedule for the building's HVAC system.
        supply_water_temperature_setpoint_schedule: a component representing the supply water temperature setpoint schedule for the building's HVAC system.
        space: a component representing a space within the building.
        heating_coil: a component representing the heating coil of the building's HVAC system.
    '''
    logger.info("[Extend Model2 - Test Evaluator]")

    indoor_temperature_setpoint_schedule = ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [5],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            weekendRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [5],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            saveSimulationResult = True,
            id = "OE20-601b-2| Temperature setpoint schedule")
    
    occupancy_schedule = ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [3,5,20,25,27,7,3]}, #35
            add_noise = True,
            saveSimulationResult = True,
            id = "OE20-601b-2| Occupancy schedule")

    supply_water_temperature_setpoint_schedule = PiecewiseLinearScheduleSystem(
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
            id = "Heating system| Supply water temperature schedule")

    self._add_component(occupancy_schedule)
    self._add_component(indoor_temperature_setpoint_schedule)
    self._add_component(supply_water_temperature_setpoint_schedule)
    initial_temperature = 21
    custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
    self.set_custom_initial_dict(custom_initial_dict)

def fcn3(self):
    '''
        supply_air_temperature_setpoint_schedule: a component representing the supply air temperature setpoint schedule for the building's HVAC system.
        supply_water_temperature_setpoint_schedule: a component representing the supply water temperature setpoint schedule for the building's HVAC system.
        space: a component representing a space within the building.
        heating_coil: a component representing the heating coil of the building's HVAC system.
    '''
    logger.info("[Extend Model2 - Test Evaluator]")
    occupancy_schedule = ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [3,5,20,25,27,7,3]}, #35
            add_noise = True,
            saveSimulationResult = True,
            id = "OE20-601b-2| Occupancy schedule")
    
    indoor_temperature_setpoint_schedule = ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [6],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            weekendRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [6],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            saveSimulationResult = True,
            id = "OE20-601b-2| Temperature setpoint schedule")

    supply_water_temperature_setpoint_schedule = PiecewiseLinearScheduleSystem(
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
            id = "Heating system| Supply water temperature schedule")

    self._add_component(occupancy_schedule)
    self._add_component(indoor_temperature_setpoint_schedule)
    self._add_component(supply_water_temperature_setpoint_schedule)
    initial_temperature = 21
    custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
    self.set_custom_initial_dict(custom_initial_dict)


def fcn4(self):

    '''
        supply_air_temperature_setpoint_schedule: a component representing the supply air temperature setpoint schedule for the building's HVAC system.
        supply_water_temperature_setpoint_schedule: a component representing the supply water temperature setpoint schedule for the building's HVAC system.
        space: a component representing a space within the building.
        heating_coil: a component representing the heating coil of the building's HVAC system.
    '''

    logger.info("[Extend Model2 - Test Evaluator]")
    occupancy_schedule = ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [3,5,20,25,27,7,3]}, #35
            add_noise = True,
            saveSimulationResult = True,
            id = "OE20-601b-2| Occupancy schedule")
    
    indoor_temperature_setpoint_schedule = ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [7],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            weekendRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [7],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            saveSimulationResult = True,
            id = "OE20-601b-2| Temperature setpoint schedule")

    supply_water_temperature_setpoint_schedule = PiecewiseLinearScheduleSystem(
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
            id = "Heating system| Supply water temperature schedule")

    self._add_component(occupancy_schedule)
    self._add_component(indoor_temperature_setpoint_schedule)
    self._add_component(supply_water_temperature_setpoint_schedule)
    initial_temperature = 21
    custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
    self.set_custom_initial_dict(custom_initial_dict)


def test_evaluator():
    '''
        The evaluation uses an evaluator object that calculates and plots the energy and discomfort 
        levels for two measuring devices - a space temperature sensor and a heating meter. 
        It then plots the measured and simulated space temperature for both models. 
        The first model is a baseline and the second model includes a night setback feature. 
        The code saves all the plots as images.
    '''
    logger.info("[Test Evaluator] : Entered Test Function")
    weather_data_filename = os.path.join(uppath(os.path.abspath(__file__), 3), "model", "tests", "test_data.csv")
    filename = "configuration_template_OU44_room_case.xlsx"
    model1 = Model(id="Baseline", saveSimulationResult=True)
    model1.add_outdoor_environment(filename=weather_data_filename)
    model1.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn1)

    model2 = Model(id="Night setback 5 AM", saveSimulationResult=True)
    model2.add_outdoor_environment(filename=weather_data_filename)
    model2.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn2)

    model3 = Model(id="Night setback 6 AM", saveSimulationResult=True)
    model3.add_outdoor_environment(filename=weather_data_filename)
    model3.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn3)

    model4 = Model(id="Night setback 7 AM", saveSimulationResult=True)
    model4.add_outdoor_environment(filename=weather_data_filename)
    model4.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn4)
    
    evaluator = Evaluator()
    stepSize = 600 #Seconds
    startTime = datetime.datetime(year=2022, month=1, day=3, hour=0, minute=0, second=0) #piecewise 20.5-23
    endTime = datetime.datetime(year=2022, month=1, day=8, hour=0, minute=0, second=0) #piecewise 20.5-23

    models = [model1, model2, model3, model4]
    measuring_devices = ["OE20-601b-2| temperature sensor", "OE20-601b-2| Heating meter"]
    evaluation_metrics = ["T", "T"]
    evaluator.evaluate(startTime=startTime,
                    endTime=endTime,
                    stepSize=stepSize,
                    models=models,
                    measuring_devices=measuring_devices,
                    evaluation_metrics=evaluation_metrics)
    
    
    # The rest is plotting. The evaluate method plots 
    fig, ax = evaluator.bar_plot_dict[measuring_devices[0]]
    fig.text(0.015, 0.5, r"Discomfort [Kh]", va='center', ha='center', rotation='vertical', fontsize=13, color="black")
    fig.set_size_inches(7, 5/2)
    fig.suptitle("Discomfort", fontsize=18)
    ax.set_xlabel(None)
    ax.set_xticks([])
    # fig.savefig(f"{measuring_devices[0]}_bar_scenario.png", dpi=300)

    fig, ax = evaluator.bar_plot_dict[measuring_devices[1]]
    fig.text(0.015, 0.5, r"Energy [kWh]", va='center', ha='center', rotation='vertical', fontsize=13, color="black")
    fig.set_size_inches(7, 5/2)
    ax.set_xlabel(None)
    ax.set_xticks([])
    # fig.savefig(f"{measuring_devices[1]}_bar_scenario.png", dpi=300)
    plt.show()

    logger.info("[Test Evaluator] : Exited from Test Function")
    


if __name__ == '__main__':
    test_evaluator()


