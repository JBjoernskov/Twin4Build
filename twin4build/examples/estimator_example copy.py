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
            "ruleset_default_value": 21.00780578056241,
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
    self.components["020B_occupancy_profile"].weekDayRulesetDict = {"ruleset_default_value": 0,
                                                                    "ruleset_start_minute": [],
                                                                    "ruleset_end_minute": [],
                                                                    "ruleset_start_hour": [],
                                                                    "ruleset_end_hour": [],
                                                                    "ruleset_value": []}
    self.components["020B_temperature_heating_setpoint"].useSpreadsheet = True
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
    stepSize = 2400  # 10 minutes in seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                    tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=1, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen"))

    space = model.components["020B"]
    space_heater = model.components["020B_space_heater"]
    heating_controller = model.components["020B_temperature_heating_controller"]
    co2_controller = model.components["020B_co2_controller"]
    space_heater_valve = model.components["020B_space_heater_valve"]
    supply_damper = model.components["020B_room_supply_damper"]
    exhaust_damper = model.components["020B_room_exhaust_damper"]

    targetParameters = {"private": {
                                # Thermal parameters
                                "thermal.C_air": {"components": [space], "x0": 2e+6, "lb": 1e+6, "ub": 1e+7},                # Thermal capacitance of indoor air [J/K]
                                "thermal.C_wall": {"components": [space], "x0": 2e+6, "lb": 1e+6, "ub": 1e+7},               # Thermal capacitance of exterior wall [J/K]
                                "thermal.C_boundary": {"components": [space], "x0": 5e+5, "lb": 1e+4, "ub": 1e+6},               # Thermal capacitance of exterior wall [J/K]
                                "thermal.R_out": {"components": [space], "x0": 0.05, "lb": 0.01, "ub": 1},                # Thermal resistance between wall and outdoor [K/W]
                                "thermal.R_in": {"components": [space], "x0": 0.05, "lb": 0.01, "ub": 1},                 # Thermal resistance between wall and indoor [K/W]
                                "thermal.R_boundary": {"components": [space], "x0": 0.01, "lb": 0.0001, "ub": 1},                 # Thermal resistance between wall and indoor [K/W]
                                "thermal.f_wall": {"components": [space], "x0": 0.3, "lb": 0, "ub": 1},         # Radiation factor for exterior wall
                                "thermal.f_air": {"components": [space], "x0": 0.1, "lb": 0, "ub": 1},          # Radiation factor for air
                                "thermal.Q_occ_gain": {"components": [space], "x0": 100.0, "lb": 10, "ub": 200},   # Heat gain per occupant [W]

                                # Mass parameters
                                "mass.V": {"components": [space], "x0": 100, "lb": 10, "ub": 1000}, # Volume of the space [m³]
                                "mass.G_occ": {"components": [space], "x0": 8.18e-6, "lb": 1e-8, "ub": 1e-4}, # CO2 generation rate per occupant [ppm·kg/s]
                                "mass.m_inf": {"components": [space], "x0": 0.001, "lb": 1e-6, "ub": 0.3}, # Infiltration rate [kg/s]

                                # Space heater parameters
                                "thermalMassHeatCapacity": {"components": [space_heater], "x0": 10000, "lb": 1000, "ub": 50000}, # Thermal mass heat capacity [J/K]
                                "UA": {"components": [space_heater], "x0": 30, "lb": 1, "ub": 100}, # Thermal conductance [W/K]

                                # Heating controller parameters
                                "kp": {"components": [heating_controller, co2_controller], "x0": [0.001, -0.001], "lb": [1e-5, -1], "ub": [1, -1e-5]}, # Proportional gain
                                "Ti": {"components": [heating_controller, co2_controller], "x0": [10, 10], "lb": [1, 1], "ub": [100, 100]}, # Integral gain
                                "Td": {"components": [heating_controller, co2_controller], "x0": [0, 0], "lb": [0, 0], "ub": [1, 1]}, # Derivative gain

                                # Space heater valve parameters
                                "waterFlowRateMax": {"components": [space_heater_valve], "x0": 0.01, "lb": 1e-6, "ub": 0.1}, # Maximum water flow rate [m³/s]
                                "valveAuthority": {"components": [space_heater_valve], "x0": 0.8, "lb": 0.4, "ub": 1}, # Valve authority

                                # Damper parameters
                                "a": {"components": [supply_damper, exhaust_damper], "x0": 1, "lb": 1, "ub": 10}, # Shape parameter
                                "nominalAirFlowRate": {"components": [supply_damper, exhaust_damper], "x0": 0.001, "lb": 1e-5, "ub": 1}, # Maximum water flow rate [m³/s]
                                }}

    targetMeasuringDevices = [model.components["020B_valve_position_sensor"],
                                model.components["020B_temperature_sensor"],
                                model.components["020B_co2_sensor"],
                                model.components["020B_damper_position_sensor"]]
    
    
    
    # # Run initial simulation for comparison
    # simulator.simulate(
    #     stepSize=stepSize,
    #     startTime=startTime,
    #     endTime=endTime
    # )

    
    
    # # Plot results
    # fig, axes = tb.plot.plot_component(
    #     simulator,
    #     components_1axis=[
    #         ("020B", "indoorTemperature", "output"),
    #         ("outdoor_environment", "outdoorTemperature", "output"),
    #         (heating_controller.id, "setpointValue", "input"),
    #         # (estimator.actual_readings[model.components["020B_temperature_sensor"].id], "Actual temperature"),
    #     ],
    #     components_2axis=[
    #         ("020B_space_heater", "Power", "output"),
    #         ("020B", "heatGain", "input"),

    #     ],
    #     components_3axis=[
    #         # ("020B_space_heater", "waterFlowRate", "input"),
    #         # (heating_controller.id, "setpointValue", "input"),
    #         (heating_controller.id, "inputSignal", "output"),
    #     ],
    #     ylabel_1axis="Temperature [°C]",
    #     ylabel_2axis="Power [W]",
    #     ylabel_3axis="Water flow rate [m³/s]",
    #     show=False,
    #     nticks=11
    # )

    # # Run simulation
    # simulator.simulate(
    #     stepSize=stepSize,
    #     startTime=startTime,
    #     endTime=endTime
    # )
    
    
    # # Plot results
    # fig, axes = tb.plot.plot_component(
    #     simulator,
    #     components_1axis=[
    #         ("020B", "indoorTemperature", "output"),
    #         ("outdoor_environment", "outdoorTemperature", "output"),
    #         (heating_controller.id, "setpointValue", "input"),
    #         # (estimator.actual_readings[model.components["020B_temperature_sensor"].id], "Actual temperature"),
    #     ],
    #     components_2axis=[
    #         ("020B_space_heater", "Power", "output"),
    #         ("020B", "heatGain", "input"),

    #     ],
    #     components_3axis=[
    #         # ("020B_space_heater", "waterFlowRate", "input"),
    #         (heating_controller.id, "inputSignal", "output"),
    #     ],
    #     ylabel_1axis="Temperature [°C]",
    #     ylabel_2axis="Power [W]",
    #     ylabel_3axis="Water flow rate [m³/s]",
    #     show=True,
    #     nticks=11
    # )

    # Create estimator
    estimator = tb.Estimator(simulator)



    options = {"max_nfev": 100}
    
    # Time and run LS_AD method
    result_ad = estimator.estimate(
        targetParameters=targetParameters,
        targetMeasuringDevices=targetMeasuringDevices,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize,
        n_initialization_steps=20,
        method="LS_AD",  # Use PyTorch-based optimization
        options=options,
    )
    acc_energy_before = simulator.model.components["020B_space_heater"].output["Power"].history.sum()

    # Plot results
    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B", "indoorTemperature", "output"),
            # ("outdoor_environment", "outdoorTemperature", "output"),
            (heating_controller.id, "setpointValue", "input"),
            (estimator.actual_readings[model.components["020B_temperature_sensor"].id], "Actual temperature"),
        ],
        components_2axis=[
            ("020B_space_heater", "Power", "output"),
            # ("020B", "heatGain", "input"),

        ],
        components_3axis=[
            ("020B_space_heater_valve", "valvePosition", "output"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Valve position [0-1]",
        show=False,
        nticks=11
    )

    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B_valve_position_sensor", "measuredValue", "input"),
            (estimator.actual_readings[model.components["020B_valve_position_sensor"].id], "Actual valve position"),
        ],
        ylabel_1axis="Valve position [0-1]",
        show=False,
        nticks=11
    )

    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B_temperature_sensor", "measuredValue", "input"),
            (estimator.actual_readings[model.components["020B_temperature_sensor"].id], "Actual temperature"),
        ],
        ylabel_1axis="Temperature [°C]",
        show=False,
        nticks=11
    )

    #########################################################
    # Now, we remove the controller to test how much energy we can save by optimizing valve position directly
    # model.remove_component(heating_controller)

    # # Create water flow schedule
    # valve_position_schedule = tb.ScheduleSystem(
    #     weekDayRulesetDict = {
    #         "ruleset_default_value": 1,
    #         "ruleset_start_minute": [0, 0],
    #         "ruleset_end_minute": [0, 0],
    #         "ruleset_start_hour": [0, 6],
    #         "ruleset_end_hour": [6, 24],
    #         "ruleset_value": [0, 1]
    #     },
    #     id="valve_position_schedule"
    # )
    # model.add_connection(valve_position_schedule, space_heater_valve, "scheduleValue", "valvePosition")
    # model.load()
    # Define optimization targets
    # decisionVariables = [
    #     (valve_position_schedule, "scheduleValue", 0, 1)  # Optimize water flow rate
    # ]
    #####################################################


    #####################################################
    schedule = tb.ScheduleSystem(
        weekDayRulesetDict = {
            "ruleset_default_value": 15,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute": [0, 0],
            "ruleset_start_hour": [0, 6],
            "ruleset_end_hour": [6, 24],
            "ruleset_value": [15, 25]
        },
        id="temperature_heating_setpoint_schedule"
    )
    schedule.useSpreadsheet = True
    schedule.filename = utils.get_path(["parameter_estimation_example", "temperature_heating_setpoint.csv"])
    #####################################################


    

    

    decisionVariables = [
        (model.components["020B_temperature_heating_setpoint"], "scheduleValue", 15, 25)  # Optimize water flow rate
    ]
    
    minimize = [
        (space_heater, "Power")  # Minimize power consumption
    ]

    inequalityConstraints = [
        (space, "indoorTemperature", "lower", schedule)   # Temperature should not fall below heating setpoint
    ]



    stepSize = 2400  # 10 minutes in seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                    tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=11, day=29, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen"))

    # Create optimizer
    optimizer = tb.Optimizer(simulator)

    # Run optimization
    optimizer.optimize(
        decisionVariables=decisionVariables,
        minimize=minimize,
        equalityConstraints=None,
        inequalityConstraints=inequalityConstraints,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize,
        lr=0.1,  # Start with a higher learning rate
        iterations=5000,
        scheduler_type="reduce_on_plateau",
        scheduler_params={
            "mode": "min",       # Reduce LR when loss stops decreasing
            "factor": 0.95,      # Multiply LR by this factor when plateau is detected
            "patience": 10,      # Number of epochs with no improvement after which LR will be reduced
            "threshold": 1e-3    # Threshold for measuring the new optimum
        }
    )

    acc_energy_after = simulator.model.components["020B_space_heater"].output["Power"].history.sum()


    print("Energy before optimization: ", acc_energy_before)
    print("Energy after optimization: ", acc_energy_after)
    print(f"Energy saved: {acc_energy_before - acc_energy_after} [J]")


    # Plot optimization results
    # fig, axes = tb.plot.plot_component(
    #     simulator,
    #     components_1axis=[
    #         ("020B", "indoorTemperature", "output"),
    #         ("020B_temperature_heating_setpoint", "scheduleValue", "output"),
    #     ],
    #     components_2axis=[
    #         ("020B_space_heater", "Power", "output"),
    #     ],
    #     components_3axis=[
    #         ("valve_position_schedule", "scheduleValue", "output"),
    #     ],
    #     ylabel_1axis="Temperature [°C]",
    #     ylabel_2axis="Power [W]",
    #     ylabel_3axis="Valve position [0-1]",
    #     show=True,
    #     nticks=11
    # )

    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B", "indoorTemperature", "output"),
            ("020B_temperature_heating_setpoint", "scheduleValue", "output"),
            (schedule, "scheduleValue", "output"),
        ],
        components_2axis=[
            ("020B_space_heater", "Power", "output")
            ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        show=True,
        nticks=11
    )

if __name__ == "__main__":
    main()