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
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

def extend_model1(self):

    '''
        outdoor_environment: A component representing the outdoor environment of the building.
        supply_air_temperature_setpoint_schedule: A component representing the setpoint temperature for supply air to the building.
        supply_water_temperature_setpoint_schedule: A component representing the setpoint temperature for supply water to the building.
        space: A component representing the indoor space being conditioned by the HVAC system.
        heating_coil: A component representing the heating coil in the HVAC system.
        indoor_temperature_setpoint_schedule: A Schedule object representing the desired indoor temperature setpoints over time.
    '''

    logger.info("[Extend Model1 - Test Evaluator]")
    
    # node_E = [v for v in self.system_dict["ventilation"]["V1"].hasSubSystem if isinstance(v, Node) and v.operationMode == "exhaust"][0]
    outdoor_environment = self.component_dict["Outdoor environment"]
    supply_air_temperature_setpoint_schedule = self.component_dict["Supply air temperature setpoint"]
    supply_water_temperature_setpoint_schedule = self.component_dict["Supply water temperature setpoint"]
    space = self.component_dict["Space"]
    heating_coil = self.component_dict["Heating coil"]
    # self.add_connection(node_E, supply_air_temperature_setpoint_schedule, "flowTemperatureOut", "exhaustAirTemperature")
    # self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
    # self.add_connection(supply_air_temperature_setpoint_schedule, space, "supplyAirTemperatureSetpoint", "supplyAirTemperature") #############
    # self.add_connection(supply_water_temperature_setpoint_schedule, space, "supplyWaterTemperatureSetpoint", "supplyWaterTemperature") ########
    # self.add_connection(heating_coil, space, "airTemperatureOut", "supplyAirTemperature") #############

    indoor_temperature_setpoint_schedule = Schedule(
            weekDayRulesetDict = {
                "ruleset_default_value": 21,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [4],
                "ruleset_end_hour": [20],
                "ruleset_value": [21]},
            weekendRulesetDict = {
                "ruleset_default_value": 21,
                "ruleset_start_minute": [],
                "ruleset_end_minute": [],
                "ruleset_start_hour": [],
                "ruleset_end_hour": [],
                "ruleset_value": []},
            saveSimulationResult = True,
            id = "Temperature setpoint schedule")
    self.component_dict["Temperature setpoint schedule"] = indoor_temperature_setpoint_schedule

def extend_model2(self):

    '''
        supply_air_temperature_setpoint_schedule: a component representing the supply air temperature setpoint schedule for the building's HVAC system.
        supply_water_temperature_setpoint_schedule: a component representing the supply water temperature setpoint schedule for the building's HVAC system.
        space: a component representing a space within the building.
        heating_coil: a component representing the heating coil of the building's HVAC system.
    '''

    logger.info("[Extend Model2 - Test Evaluator]")
    
    # node_E = [v for v in self.system_dict["ventilation"]["V1"].hasSubSystem if isinstance(v, Node) and v.operationMode == "exhaust"][0]
    outdoor_environment = self.component_dict["Outdoor environment"]
    supply_air_temperature_setpoint_schedule = self.component_dict["Supply air temperature setpoint"]
    supply_water_temperature_setpoint_schedule = self.component_dict["Supply water temperature setpoint"]
    space = self.component_dict["Space"]
    heating_coil = self.component_dict["Heating coil"]
    # self.add_connection(node_E, supply_air_temperature_setpoint_schedule, "flowTemperatureOut", "exhaustAirTemperature")
    # self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
    # self.add_connection(supply_air_temperature_setpoint_schedule, space, "supplyAirTemperatureSetpoint", "supplyAirTemperature") #############
    # self.add_connection(supply_water_temperature_setpoint_schedule, space, "supplyWaterTemperatureSetpoint", "supplyWaterTemperature") ########
    # self.add_connection(heating_coil, space, "airTemperatureOut", "supplyAirTemperature") #############

    indoor_temperature_setpoint_schedule = Schedule(
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
            mondayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [5],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            saveSimulationResult = True,
            id = "Temperature setpoint schedule")
    self.component_dict["Temperature setpoint schedule"] = indoor_temperature_setpoint_schedule



def extend_model3(self):

    '''
        supply_air_temperature_setpoint_schedule: a component representing the supply air temperature setpoint schedule for the building's HVAC system.
        supply_water_temperature_setpoint_schedule: a component representing the supply water temperature setpoint schedule for the building's HVAC system.
        space: a component representing a space within the building.
        heating_coil: a component representing the heating coil of the building's HVAC system.
    '''

    logger.info("[Extend Model2 - Test Evaluator]")
    
    # node_E = [v for v in self.system_dict["ventilation"]["V1"].hasSubSystem if isinstance(v, Node) and v.operationMode == "exhaust"][0]
    outdoor_environment = self.component_dict["Outdoor environment"]
    supply_air_temperature_setpoint_schedule = self.component_dict["Supply air temperature setpoint"]
    supply_water_temperature_setpoint_schedule = self.component_dict["Supply water temperature setpoint"]
    space = self.component_dict["Space"]
    heating_coil = self.component_dict["Heating coil"]
    # self.add_connection(node_E, supply_air_temperature_setpoint_schedule, "flowTemperatureOut", "exhaustAirTemperature")
    # self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
    # self.add_connection(supply_air_temperature_setpoint_schedule, space, "supplyAirTemperatureSetpoint", "supplyAirTemperature") #############
    # self.add_connection(supply_water_temperature_setpoint_schedule, space, "supplyWaterTemperatureSetpoint", "supplyWaterTemperature") ########
    # self.add_connection(heating_coil, space, "airTemperatureOut", "supplyAirTemperature") #############

    indoor_temperature_setpoint_schedule = Schedule(
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
            mondayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [6],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            saveSimulationResult = True,
            id = "Temperature setpoint schedule")
    self.component_dict["Temperature setpoint schedule"] = indoor_temperature_setpoint_schedule


def extend_model4(self):

    '''
        supply_air_temperature_setpoint_schedule: a component representing the supply air temperature setpoint schedule for the building's HVAC system.
        supply_water_temperature_setpoint_schedule: a component representing the supply water temperature setpoint schedule for the building's HVAC system.
        space: a component representing a space within the building.
        heating_coil: a component representing the heating coil of the building's HVAC system.
    '''

    logger.info("[Extend Model2 - Test Evaluator]")
    
    # node_E = [v for v in self.system_dict["ventilation"]["V1"].hasSubSystem if isinstance(v, Node) and v.operationMode == "exhaust"][0]
    outdoor_environment = self.component_dict["Outdoor environment"]
    supply_air_temperature_setpoint_schedule = self.component_dict["Supply air temperature setpoint"]
    supply_water_temperature_setpoint_schedule = self.component_dict["Supply water temperature setpoint"]
    space = self.component_dict["Space"]
    heating_coil = self.component_dict["Heating coil"]
    # self.add_connection(node_E, supply_air_temperature_setpoint_schedule, "flowTemperatureOut", "exhaustAirTemperature")
    # self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
    # self.add_connection(supply_air_temperature_setpoint_schedule, space, "supplyAirTemperatureSetpoint", "supplyAirTemperature") #############
    # self.add_connection(supply_water_temperature_setpoint_schedule, space, "supplyWaterTemperatureSetpoint", "supplyWaterTemperature") ########
    # self.add_connection(heating_coil, space, "airTemperatureOut", "supplyAirTemperature") #############

    indoor_temperature_setpoint_schedule = Schedule(
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
            mondayRulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [7],
                "ruleset_end_hour": [17],
                "ruleset_value": [21]},
            saveSimulationResult = True,
            id = "Temperature setpoint schedule")
    self.component_dict["Temperature setpoint schedule"] = indoor_temperature_setpoint_schedule


def test():
    
    logger.info("[Test Evaluator] : Entered Test Function")
    
    '''
        The evaluation uses an evaluator object that calculates and plots the energy and discomfort 
        levels for two measuring devices - a space temperature sensor and a heating meter. 
        It then plots the measured and simulated space temperature for both models. 
        The first model is a baseline and the second model includes a night setback feature. 
        The code saves all the plots as images.
    '''

    Model.extend_model = extend_model1
    # filename = "configuration_template_1space_1v_1h_0c_test_new_layout_simple_naming.xlsx"
    # filename = "configuration_template_1space_BS2023.xlsx"
    filename = "configuration_template_1space_BS2023_no_sensor.xlsx"
    model1 = Model(id="Baseline", saveSimulationResult=True)
    model1.load_BS2023_model(filename)
    
    Model.extend_model = extend_model2
    model2 = Model(id="Night setback 5 AM", saveSimulationResult=True)
    model2.load_BS2023_model(filename)

    Model.extend_model = extend_model3
    model3 = Model(id="Night setback 6 AM", saveSimulationResult=True)
    model3.load_BS2023_model(filename)

    Model.extend_model = extend_model4
    model4 = Model(id="Night setback 7 AM", saveSimulationResult=True)
    model4.load_BS2023_model(filename)
    

    evaluator = Evaluator()
    stepSize = 600 #Seconds
    startPeriod = datetime.datetime(year=2022, month=1, day=3, hour=0, minute=0, second=0) #piecewise 20.5-23
    endPeriod = datetime.datetime(year=2022, month=1, day=8, hour=0, minute=0, second=0) #piecewise 20.5-23
    # startPeriod = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) #piecewise 20.5-23
    # endPeriod = datetime.datetime(year=2022, month=2, day=1, hour=0, minute=0, second=0) #piecewise 20.5-23

    models = [model1, model2, model3, model4]
    measuring_devices = ["Space temperature sensor", "Heating meter"]
    evaluation_metrics = ["T", "T"]
    evaluator.evaluate(startPeriod=startPeriod,
                    endPeriod=endPeriod,
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
    fig.savefig(f"{measuring_devices[0]}_bar_scenario.png", dpi=300)

    fig, ax = evaluator.bar_plot_dict[measuring_devices[1]]
    fig.text(0.015, 0.5, r"Energy [kWh]", va='center', ha='center', rotation='vertical', fontsize=13, color="black")
    fig.set_size_inches(7, 5/2)
    ax.set_xlabel(None)
    ax.set_xticks([])
    fig.savefig(f"{measuring_devices[1]}_bar_scenario.png", dpi=300)

    # plt.show()
    
    measuring_device = "Space temperature sensor"
    df_actual_readings = evaluator.simulator.get_actual_readings(startPeriod, endPeriod, stepSize)      
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

    compared_model = models[2]
    property_1 = models[0].component_dict[measuring_device].measuresProperty
    schedule_readings1 = property_1.isControlledByDevice.savedInput["setpointValue"]
    property_2 = compared_model.component_dict[measuring_device].measuresProperty
    schedule_readings2 = property_2.isControlledByDevice.savedInput["setpointValue"]
    fig,axes = plt.subplots(2, sharex=True)
    axes[0].plot(df_actual_readings["time"], df_actual_readings[measuring_device], color=blue, label=f"Physical")
    axes[0].plot(df_actual_readings["time"], evaluator.simulation_readings_dict[measuring_device]["Baseline"], color="black", label=f"Virtual", linestyle="dashed")
    axes[0].fill_between(df_actual_readings["time"], min(schedule_readings2), schedule_readings1, facecolor="black", edgecolor=red ,alpha=0.3, label=r"Constant setpoint", linewidth=3)
    axes[0].legend(prop={'size': 8}, loc="best")
    fig.set_size_inches(7, 5)
    fig.suptitle(measuring_device, fontsize=18)
    axes[1].plot(df_actual_readings["time"], evaluator.simulation_readings_dict[measuring_device][compared_model.id], color="black", label=f"Virtual", linestyle="dashed")
    axes[1].fill_between(df_actual_readings["time"], min(schedule_readings2), schedule_readings2, facecolor="black", edgecolor=red, alpha=0.3, label=r"Variable setpoint", linewidth=3)
    axes[1].legend(prop={'size': 8}, loc="best")
    axes[1].set_ylim(axes[0].get_ylim())
    fig.text(0.015, 0.74, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=13, color="black")
    fig.text(0.015, 0.3, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=13, color="black")
    for ax in axes:
        myFmt = mdates.DateFormatter('%a')
        ax.xaxis.set_major_formatter(myFmt)
        ax.xaxis.set_tick_params(rotation=45)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.label.set_size(15)
        h, l = ax.get_legend_handles_labels()
        n = len(l)
        box = ax.get_position()
        ax.set_position([0.12, box.y0, box.width, box.height])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.15), prop={'size': 8}, ncol=n)
    fig.savefig(f"{measuring_device}_scenario.png", dpi=300)
    plt.show()

    
    logger.info("[Test Evaluator] : EXited from Test Function")
    


if __name__ == '__main__':
    test()


