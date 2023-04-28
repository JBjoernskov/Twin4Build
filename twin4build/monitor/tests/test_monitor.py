import os
import sys
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.ticker as ticker
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.monitor.monitor import Monitor
from twin4build.model.model import Model
from twin4build.utils.plot.plot import bar_plot_line_format
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node

def extend_model(self):
    node_E = [v for v in self.system_dict["ventilation"]["V1"].hasSubSystem if isinstance(v, Node) and v.operationMode == "exhaust"][0]
    outdoor_environment = self.component_dict["Outdoor environment"]
    supply_air_temperature_setpoint_schedule = self.component_dict["V1 Supply air temperature setpoint"]
    supply_water_temperature_setpoint_schedule = self.component_dict["H1 Supply water temperature setpoint"]
    space = self.component_dict["Space"]
    heating_coil = self.component_dict["Heating coil"]
    self.add_connection(node_E, supply_air_temperature_setpoint_schedule, "flowTemperatureOut", "exhaustAirTemperature")
    self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
    # self.add_connection(supply_air_temperature_setpoint_schedule, space, "supplyAirTemperatureSetpoint", "supplyAirTemperature") #############
    self.add_connection(supply_water_temperature_setpoint_schedule, space, "supplyWaterTemperatureSetpoint", "supplyWaterTemperature") ########
    self.add_connection(heating_coil, space, "airTemperatureOut", "supplyAirTemperature") #############

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

def test():

    # Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    # filename = "configuration_template_1space_1v_1h_0c_test_new_layout_simple_naming.xlsx"
    filename = "configuration_template_1space_BS2023.xlsx"
    model.load_BS2023_model(filename)

    monitor = Monitor(model)
    stepSize = 600 #Seconds 
    startPeriod = datetime.datetime(year=2022, month=10, day=23, hour=0, minute=0, second=0)
    endPeriod = datetime.datetime(year=2022, month=11, day=6, hour=0, minute=0, second=0)
    # startPeriod = datetime.datetime(year=2022, month=1, day=3, hour=0, minute=0, second=0) #piecewise 20.5-23
    # endPeriod = datetime.datetime(year=2022, month=1, day=17, hour=0, minute=0, second=0) #piecewise 20.5-23
    # startPeriod = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) #piecewise 20.5-23
    # endPeriod = datetime.datetime(year=2022, month=2, day=1, hour=0, minute=0, second=0) #piecewise 20.5-23
    monitor.monitor(startPeriod=startPeriod,
                    endPeriod=endPeriod,
                    stepSize=stepSize)
 
    # The rest is just formatting the resulting plot
    line_date = datetime.datetime(year=2022, month=10, day=27, hour=8, minute=23, second=0) ## At this time, the supply temperature setpoint is changed to constant 19 Deg 
    id_list = ["Space temperature sensor", "Heat recovery temperature sensor", "Heating coil temperature sensor"]
    # id_list = ["Space temperature sensor", "VE02 Primary Airflow Temperature AHR sensor", "VE02 Primary Airflow Temperature AHC sensor"]

    # "VE02 Primary Airflow Temperature AHR sensor": "VE02_FTG_MIDDEL",
    #                      "VE02 Primary Airflow Temperature AHC sensor": "VE02_FTI1",
    for id_ in id_list:
        fig,axes = monitor.plot_dict[id_]
        
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            n = len(l)
            box = ax.get_position()
            ax.set_position([0.12, box.y0, box.width, box.height])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.15), prop={'size': 8}, ncol=n)
            ax.yaxis.label.set_size(15)
            # ax.axvline(line_date, color=monitor.colors[3])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

            # df = pd.DataFrame()
            # df.insert(0, "time", monitor.simulator.dateTimeSteps)
            # df = df.set_index("time")
            # ax.set_xticklabels(map(bar_plot_line_format, df.index, [evaluation_metric]*len(df.index)))

    fig,axes = monitor.plot_dict["monitor"]
    # for ax in axes:
    #     ax.axvline(line_date, color=monitor.colors[3])
    monitor.save_plots()
    plt.show()

if __name__ == '__main__':
    test()


