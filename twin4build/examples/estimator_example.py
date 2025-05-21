import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")

import twin4build as tb
import datetime
from dateutil import tz
import numpy as np
import matplotlib.pyplot as plt
import twin4build.examples.utils as utils

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
    self.add_connection(supply_water_schedule, self.components["[020B][020B_space_heater]"], "scheduleValue", "supplyWaterTemperature") # Add missing input
    self.components["020B_temperature_sensor"].filename = utils.get_path(["parameter_estimation_example", "temperature_sensor.csv"])
    self.components["020B_co2_sensor"].filename = utils.get_path(["parameter_estimation_example", "co2_sensor.csv"])
    self.components["020B_valve_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "valve_position_sensor.csv"])
    self.components["020B_damper_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "damper_position_sensor.csv"])
    self.components["BTA004"].filename = utils.get_path(["parameter_estimation_example", "supply_air_temperature.csv"])
    self.components["020B_co2_setpoint"].weekDayRulesetDict = {"ruleset_default_value": 900,
                                                                    "ruleset_start_minute": [],
                                                                    "ruleset_end_minute": [],
                                                                    "ruleset_end_hour": [],
                                                                    "ruleset_start_hour": [],
                                                                    "ruleset_value": []}
    self.components["020B_temperature_heating_setpoint"].useFile = True
    self.components["020B_temperature_heating_setpoint"].filename = utils.get_path(["parameter_estimation_example", "temperature_heating_setpoint.csv"])
    self.components["outdoor_environment"].filename = utils.get_path(["parameter_estimation_example", "outdoor_environment.csv"])

def main():
    # Create a new model
    model = tb.Model(id="building_space_with_space_heater_model")
    
    # Load the model from semantic file
    filename = utils.get_path(["parameter_estimation_example", "one_room_example_model.xlsm"])
    model.load(semantic_model_filename=filename, fcn=fcn, verbose=False)

    # Set up simulation parameters
    simulator = tb.Simulator(model)
    stepSize = 1200  # 10 minutes in seconds
    startTime = datetime.datetime(
        year=2024, month=1, day=4, hour=0, minute=0, second=0,
        tzinfo=tz.gettz("Europe/Copenhagen")
    )
    endTime = datetime.datetime(
        year=2024, month=1, day=10, hour=0, minute=0, second=0,
        tzinfo=tz.gettz("Europe/Copenhagen")
    )

    # Run initial simulation to get "measured" data
    simulator.simulate(
        stepSize=stepSize,
        startTime=startTime,
        endTime=endTime
    )

    # Define parameters to estimate
    targetParameters = {
        "private": {
            "BuildingSpace": {
                "components": model.components["[020B][020B_space]"],
                "x0": [2000000.0, 10000000.0, 500000.0, 800000.0, 0.005, 0.005],  # Initial values
                "lb": [1000000.0, 5000000.0, 250000.0, 400000.0, 0.001, 0.001],   # Lower bounds
                "ub": [4000000.0, 20000000.0, 1000000.0, 1600000.0, 0.01, 0.01]    # Upper bounds
            }
        }
    }

    # Define target measuring devices
    targetMeasuringDevices = {
        "020B_temperature_sensor": {
            "standardDeviation": 0.1  # Temperature measurement uncertainty in °C
        }
    }

    # Create estimator
    estimator = tb.Estimator(model)

    # Run parameter estimation
    result = estimator.estimate(
        targetParameters=targetParameters,
        targetMeasuringDevices=targetMeasuringDevices,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize,
        method="TORCH",  # Use PyTorch-based optimization
        options={
            "lr": 0.01,
            "iterations": 1000,
            "scheduler_type": "reduce_on_plateau",
            "scheduler_params": {
                "mode": "min",
                "factor": 0.95,
                "patience": 10,
                "threshold": 1e-3
            }
        }
    )

    # Print results
    print("\nParameter Estimation Results:")
    print("----------------------------")
    for i, (component_id, component_attr) in enumerate(zip(result["component_id"], result["component_attr"])):
        print(f"{component_id}.{component_attr}: {result['result_x'][i]:.2f}")

    # Plot results
    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("[020B][020B_space]", "indoorTemperature", "output"),
            ("outdoor_environment", "outdoorTemperature", "output"),
        ],
        components_2axis=[
            ("[020B][020B_space_heater]", "Power", "output"),
        ],
        components_3axis=[
            ("[020B][020B_space_heater]", "waterFlowRate", "input"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Water flow rate [m³/s]",
        show=True,
        nticks=11
    )

if __name__ == "__main__":
    main() 