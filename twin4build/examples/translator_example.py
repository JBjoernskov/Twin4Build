import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")

import twin4build as tb
import datetime
from dateutil import tz
import numpy as np
import matplotlib.pyplot as plt
import twin4build.examples.utils as utils
import torch
# torch.autograd.set_detect_anomaly(True)
def fcn(self):
    supply_water_schedule = tb.ScheduleSystem(
    weekDayRulesetDict = {
        "ruleset_default_value": 60,
        "ruleset_start_minute": [],
        "ruleset_end_minute": [],
        "ruleset_start_hour": [],
        "ruleset_end_hour": [],
        "ruleset_value": []
    },
    id="supply_water_schedule"
    )
    boundary_temp_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 21,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        id="boundary_temp_schedule"
    )

    self.add_connection(boundary_temp_schedule, self.components["020B"], "scheduleValue", "boundaryTemperature")
    self.add_connection(supply_water_schedule, self.components["020B_space_heater"], "scheduleValue", "supplyWaterTemperature") # Add missing input

    self.components["020B_temperature_sensor"].useSpreadsheet = True
    self.components["020B_temperature_sensor"].filename = utils.get_path(["parameter_estimation_example", "temperature_sensor.csv"])

    self.components["020B_co2_sensor"].useSpreadsheet = True
    self.components["020B_co2_sensor"].filename = utils.get_path(["parameter_estimation_example", "co2_sensor.csv"])


    self.components["020B_valve_position_sensor"].useSpreadsheet = True
    self.components["020B_valve_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "valve_position_sensor.csv"])

    self.components["020B_damper_position_sensor"].useSpreadsheet = True
    self.components["020B_damper_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "damper_position_sensor.csv"])

    self.components["BTA004"].useSpreadsheet = True
    self.components["BTA004"].filename = utils.get_path(["parameter_estimation_example", "supply_air_temperature.csv"])

    self.components["020B_co2_setpoint"].weekDayRulesetDict = {"ruleset_default_value": 900,
                                                                    "ruleset_start_minute": [],
                                                                    "ruleset_end_minute": [],
                                                                    "ruleset_end_hour": [],
                                                                    "ruleset_start_hour": [],
                                                                    "ruleset_value": []}
    self.components["020B_occupancy_profile"].weekDayRulesetDict = {"ruleset_default_value": 0,
                                                                    "ruleset_start_minute": [],
                                                                    "ruleset_end_minute": [],
                                                                    "ruleset_start_hour": [],
                                                                    "ruleset_end_hour": [],
                                                                    "ruleset_value": []}
    self.components["020B_temperature_heating_setpoint"].useSpreadsheet = True
    self.components["020B_temperature_heating_setpoint"].filename = utils.get_path(["parameter_estimation_example", "temperature_heating_setpoint.csv"])
    
    self.components["outdoor_environment"].useSpreadsheet = True
    self.components["outdoor_environment"].filename_outdoorTemperature = utils.get_path(["parameter_estimation_example", "outdoor_environment.csv"])
    self.components["outdoor_environment"].datecolumn_outdoorTemperature = 0
    self.components["outdoor_environment"].valuecolumn_outdoorTemperature = 1
    
    self.components["outdoor_environment"].filename_globalIrradiation = utils.get_path(["parameter_estimation_example", "outdoor_environment.csv"])
    self.components["outdoor_environment"].datecolumn_globalIrradiation = 0
    self.components["outdoor_environment"].valuecolumn_globalIrradiation = 2
    
    self.components["outdoor_environment"].filename_outdoorCo2Concentration = utils.get_path(["parameter_estimation_example", "outdoor_environment.csv"])
    self.components["outdoor_environment"].datecolumn_outdoorCo2Concentration = 0
    self.components["outdoor_environment"].valuecolumn_outdoorCo2Concentration = 3


def main():
    # Create a new model
    model = tb.Model(id="translator_example")
    
    # Load the model from semantic file
    filename = utils.get_path(["parameter_estimation_example", "one_room_example_model.xlsm"])
    model.load(semantic_model_filename=filename, fcn=fcn, verbose=False)
    model.serialize()


    # Set up simulation parameters
    simulator = tb.Simulator(model)
    stepSize = 1200  # 40 minutes in seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                    tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=1, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen"))
    heating_controller = model.components["020B_temperature_heating_controller"]

    
    
    
    # # Run initial simulation for comparison
    simulator.simulate(
        stepSize=stepSize,
        startTime=startTime,
        endTime=endTime
    )

    
    # Plot initial results
    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B", "indoorTemperature", "output"),
            ("outdoor_environment", "outdoorTemperature", "output"),
            (heating_controller.id, "setpointValue", "input"),
            # (estimator.actual_readings[model.components["020B_temperature_sensor"].id], "Actual temperature"),
        ],
        components_2axis=[
            ("020B_space_heater", "Power", "output"),
            ("020B", "heatGain", "input"),

        ],
        components_3axis=[
            # ("020B_space_heater", "waterFlowRate", "input"),
            # (heating_controller.id, "setpointValue", "input"),
            (heating_controller.id, "inputSignal", "output"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Water flow rate [m³/s]",
        title="Before calibration",
        show=True,
        nticks=11
    )



    # Lets serialize the model for later use
    model.serialize()

if __name__ == "__main__":
    main()