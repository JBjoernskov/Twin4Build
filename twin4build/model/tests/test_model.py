import os
import sys
import datetime
from dateutil.tz import tzutc
import pandas as pd
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
import twin4build.utils.plot.plot as plot
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")
logger.disabled = True


def extend_model(self):

    '''
        The extend_model() function adds connections between components in a system model, 
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system 
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''

    logger.info("[Test Model] : Entered in Extend Model Function")

    # node_E = [v for v in self.system_dict["ventilation"]["V1"].hasSubSystem if isinstance(v, Node) and v.operationMode == "return"][0]
    # outdoor_environment = self.component_dict["Outdoor environment"]
    # supply_air_temperature_setpoint_schedule = self.component_dict["Supply air temperature setpoint"]
    # supply_water_temperature_setpoint_schedule = self.component_dict["Supply water temperature setpoint"]
    # space = self.component_dict["Space"]
    # heating_coil = self.component_dict["Heating coil"]
    # self.add_connection(node_E, supply_air_temperature_setpoint_schedule, "flowTemperatureOut", "returnAirTemperature")
    # self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
    # self.add_connection(supply_air_temperature_setpoint_schedule, space, "supplyAirTemperatureSetpoint", "supplyAirTemperature") #############
    # self.add_connection(supply_water_temperature_setpoint_schedule, space, "supplyWaterTemperatureSetpoint", "supplyWaterTemperature") ########
    # self.add_connection(heating_coil, space, "airTemperatureOut", "supplyAirTemperature") #############


    
    # indoor_temperature_setpoint_schedule = Schedule(
    #         weekDayRulesetDict = {
    #             "ruleset_default_value": 20,
    #             "ruleset_start_minute": [0],
    #             "ruleset_end_minute": [0],
    #             "ruleset_start_hour": [7],
    #             "ruleset_end_hour": [17],
    #             "ruleset_value": [21]},
    #         weekendRulesetDict = {
    #             "ruleset_default_value": 20,
    #             "ruleset_start_minute": [0],
    #             "ruleset_end_minute": [0],
    #             "ruleset_start_hour": [7],
    #             "ruleset_end_hour": [17],
    #             "ruleset_value": [21]},
    #         mondayRulesetDict = {
    #             "ruleset_default_value": 20,
    #             "ruleset_start_minute": [0],
    #             "ruleset_end_minute": [0],
    #             "ruleset_start_hour": [7],
    #             "ruleset_end_hour": [17],
    #             "ruleset_value": [21]},
    #         saveSimulationResult = True,
    #         id = "Temperature setpoint schedule")

    # self.component_dict["Temperature setpoint schedule"] = indoor_temperature_setpoint_schedule

    
    logger.info("[Test Model] : Exited from Extend Model Function")



def test():
    
    logger.info("[Test Model] : Entered in Test Function")

    stepSize = 600 #Seconds
    # startPeriod = datetime.datetime(year=2022, month=10, day=23, hour=0, minute=0, second=0)
    # endPeriod = datetime.datetime(year=2022, month=11, day=6, hour=0, minute=0, second=0)
    # startPeriod = datetime.datetime(year=2022, month=1, day=3, hour=0, minute=0, second=0) #piecewise 20.5-23
    # endPeriod = datetime.datetime(year=2022, month=1, day=8, hour=0, minute=0, second=0) #piecewise 20.5-23
    startPeriod = datetime.datetime(year=2022, month=9, day=10, hour=0, minute=0, second=0)
    endPeriod = datetime.datetime(year=2022, month=10, day=20, hour=0, minute=0, second=0)
    # startPeriod = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) #piecewise 20.5-23
    # endPeriod = datetime.datetime(year=2022, month=2, day=1, hour=0, minute=0, second=0) #piecewise 20.5-23
    Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    # filename = "configuration_template_1space_1v_1h_0c_test_new_layout_simple_naming.xlsx"
    filename = "configuration_template_1space_BS2023_no_sensor.xlsx"
    model.load_BS2023_model(filename)


    simulator = Simulator()
    simulator.simulate(model,
                        stepSize=stepSize,
                        startPeriod = startPeriod,
                        endPeriod = endPeriod)

    space_name = "Space"
    space_heater_name = "Space heater"
    temperature_controller_name = "Temperature controller"
    CO2_controller_name = "CO2 controller"
    air_to_air_heat_recovery_name = "Air to air heat recovery"
    heating_coil_name = "Heating coil"
    supply_fan_name = "Supply fan"
    damper_name = "Supply damper"
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.plot(simulator.dateTimeSteps,
             model.component_dict["Heating coil temperature sensor"].savedOutput["airTemperatureOut"])

    # plot.plot_space_temperature(model, simulator, space_name)
    plot.plot_space_CO2(model, simulator, space_name)
    plot.plot_weather_station(model, simulator)
    plot.plot_space_heater(model, simulator, space_heater_name)
    plot.plot_space_heater_energy(model, simulator, space_heater_name)
    plot.plot_temperature_controller(model, simulator, temperature_controller_name)
    # plot.plot_CO2_controller(model, simulator, CO2_controller_name)
    plot.plot_heat_recovery_unit(model, simulator, air_to_air_heat_recovery_name)
    plot.plot_heating_coil(model, simulator, heating_coil_name)
    # plot.plot_supply_fan(model, simulator, supply_fan_name)
    # plot.plot_supply_fan_energy(model, simulator, supply_fan_name)
    # plot.plot_supply_fan_energy(model, simulator, "Exhaust fan")
    plot.plot_space_wDELTA(model, simulator, space_name)
    plot.plot_space_energy(model, simulator, space_name)
    plot.plot_supply_damper(model, simulator, damper_name)
    import matplotlib.pyplot as plt
    plt.show()

    logger.info("[Test Model] : Exited from Test Function")




if __name__ == '__main__':
    # import cProfile
    # import pstats
    # import io
    # from line_profiler import LineProfiler

    # pr = cProfile.Profile()
    # pr.enable()


    # lp = LineProfiler()
    # lp.add_function(BuildingSpaceModel.get_temperature)
    # lp.add_function(BuildingSpaceModel.do_step)   # add additional function to profile
    # lp_wrapper = lp(test)
    # lp_wrapper()
    # lp.print_stats()

    test()

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()
    # with open('profile.txt', 'w+') as f:
    #     f.wrrite(s.getvalue())