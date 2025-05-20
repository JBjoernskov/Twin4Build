import sys
sys.path.append("c:/Users/jabj/Documents/python/Twin4Build/")
import twin4build as tb
import datetime
from dateutil import tz
import copy
import numpy as np
import torch
import cProfile
import pstats
import matplotlib.pyplot as plt
def main():
    # Create a new model
    model = tb.Model(id="building_space_with_space_heater_model")

    # Create building space
    building_space = tb.BuildingSpaceThermalTorchSystem(
        C_air=2000000.0,
        C_wall=10000000.0,
        C_int=500000.0,
        C_boundary=800000.0,
        R_out=0.005,
        R_in=0.005,
        R_int=100000,
        R_boundary=10000,
        f_wall=0,
        f_air=0,
        Q_occ_gain=100.0,
        CO2_occ_gain=0.004,
        CO2_start=400.0,
        infiltrationRate=0.0,
        airVolume=100.0,
        id="BuildingSpace"
    )

    # Create space heater
    space_heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=2000.0,
        T_a_nominal_sh=60.0,
        T_b_nominal_sh=30.0,
        TAir_nominal_sh=21.0,
        thermalMassHeatCapacity=500000.0,
        nelements=3,
        id="SpaceHeater"
    )

    # Create schedules
    occupancy_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [8, 9, 10, 12, 14, 16, 18],
            "ruleset_end_hour": [9, 10, 12, 14, 16, 18, 20],
            "ruleset_value": [0, 0, 0, 0, 0, 0, 0]
        },
        id="OccupancySchedule"
    )
    outdoor_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 10.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 12, 18, 21, 23, 24],
            "ruleset_end_hour": [6, 12, 18, 21, 23, 24, 24],
            "ruleset_value": [5.0, 8.0, 15.0, 12.0, 8.0, 5.0, 5.0]
        },
        id="OutdoorTemperature"
    )
    # outdoor_temp = tb.ScheduleSystem(
    #     weekDayRulesetDict={
    #         "ruleset_default_value": 10.0,
    #         "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
    #         "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
    #         "ruleset_start_hour": [0, 6, 12, 18, 21, 23, 24],
    #         "ruleset_end_hour": [6, 12, 18, 21, 23, 24, 24],
    #         "ruleset_value": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
    #     },
    #     id="OutdoorTemperature"
    # )
    solar_radiation = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 9, 12, 15, 18, 24],
            "ruleset_end_hour": [6, 9, 12, 15, 18, 24, 24],
            "ruleset_value": [0.0, 100, 300, 300, 100, 0.0, 0.0]
        },
        id="SolarRadiation"
    )
    supply_air_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 20.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [0, 0, 0, 0.0, 0.0, 0.0, 0.0]
        },
        id="SupplyAirTemperature"
    )

    boundary_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 20.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [20, 20, 20, 20, 20, 20, 20]
        },
        id="BoundaryTemperature"
    )
    supply_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 60.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [60, 60, 60, 60, 60, 60, 60]
        },
        id="SupplyTempSchedule"
    )
    mf = space_heater.Q_flow_nominal_sh/4180/(space_heater.T_a_nominal_sh-space_heater.T_b_nominal_sh)
    waterflow_schedule = tb.ScheduleSystem(
        weekDayRulesetDict = {
            "ruleset_default_value": 0,    # Default flow rate when no rule applies [m³/s]
            "ruleset_start_minute": [0,0], # Start minutes for each period
            "ruleset_end_minute": [0,0],   # End minutes for each period
            "ruleset_start_hour": [8, 19], # Start hours (8:00 and 19:00)
            "ruleset_end_hour": [16, 20],  # End hours (16:00 and 20:00)
            # Flow rate calculation: Q/(cp*ΔT) where:
            # Q = 2000W (heating power)
            # cp = 4180 J/kg·K (water specific heat)
            # ΔT = 60-30 = 30K (temperature difference)
            "ruleset_value": [mf, mf]  # [m³/s]
        },
        id="Waterflow schedule"
    )

    # Remove air_flow and add supplyAirFlowRate and exhaustAirFlowRate schedules
    supply_air_flow = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 0.0},
        id="SupplyAirFlow"
    )
    exhaust_air_flow = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 0.0},
        id="ExhaustAirFlow"
    )

    # Connect schedules to building space
    model.add_connection(occupancy_schedule, building_space, "scheduleValue", "numberOfPeople")
    model.add_connection(outdoor_temp, building_space, "scheduleValue", "outdoorTemperature")
    model.add_connection(solar_radiation, building_space, "scheduleValue", "globalIrradiation")
    model.add_connection(supply_air_flow, building_space, "scheduleValue", "supplyAirFlowRate")
    model.add_connection(exhaust_air_flow, building_space, "scheduleValue", "exhaustAirFlowRate")
    model.add_connection(supply_air_temp, building_space, "scheduleValue", "supplyAirTemperature")
    model.add_connection(boundary_temp, building_space, "scheduleValue", "T_boundary")

    # Connect schedules to space heater
    model.add_connection(supply_temp, space_heater, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(waterflow_schedule, space_heater, "scheduleValue", "waterFlowRate")

    # Connect building space indoorTemperature to space heater input
    model.add_connection(building_space, space_heater, "indoorTemperature", "indoorTemperature")

    # Connect space heater output to building space input
    model.add_connection(space_heater, building_space, "Power", "Q_sh")

    # Load the model
    model.load()

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

    # Run simulation
    simulator.simulate(
        model,
        stepSize=stepSize,
        startTime=startTime,
        endTime=endTime
    )

    # Plot before optimization results
    tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("BuildingSpace", "indoorTemperature", "output"),
            ("BuildingSpace", "wallTemperature", "output"),
            ("BuildingSpace", "outdoorTemperature", "input"),
            # ("SpaceHeater", "outletWaterTemperature", "output"),
        ],
        components_2axis=[
            ("SpaceHeater", "Power", "output"),
        ],
        components_3axis=[
            ("SpaceHeater", "waterFlowRate", "input"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Water flow rate [m³/s]",
        show=False,
        nticks=11
    )


    cooling_setpoint = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 26.0,  # Default cooling setpoint
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 17, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 17, 24, 0, 0, 0, 0],
            "ruleset_value": [26.0, 24.0, 30.0, 26.0, 26.0, 26.0, 26.0]  # Unoccupied: 26°C, Occupied: 24°C
        },
        id="CoolingSetpoint"
    )

    heating_setpoint = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 18.0,  # Default heating setpoint
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 17, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [18.0, 21.0, 18.0, 18.0, 18.0, 18.0, 18.0]  # Unoccupied: 18°C, Occupied: 21°C
        },
        weekendRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_hour": [0, 0, 0, 0, 0, 0, 0],
        },
        id="HeatingSetpoint"
    )
    
    # Define optimization targets
    decisionVariables = [
        (waterflow_schedule, "scheduleValue", 0, mf)  # Optimize the water flow rate with lower bound 0 and upper bound=inf
    ]
    
    minimize = [
        (space_heater, "Power")  # Target the indoor temperature
    ]

    inequalityConstraints = [
        (building_space, "indoorTemperature", "upper", cooling_setpoint),  # Temperature should not exceed cooling setpoint
        (building_space, "indoorTemperature", "lower", heating_setpoint)   # Temperature should not fall below heating setpoint
    ]




    optimizer = tb.Optimizer(simulator)



    optimizer.optimize(
        decisionVariables=decisionVariables,
        minimize=minimize,
        equalityConstraints=None,
        inequalityConstraints=inequalityConstraints,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize,
        lr=10,  # Start with a higher learning rate since we'll be decreasing it
        iterations=1000,
        scheduler_type="reduce_on_plateau",
        scheduler_params={
            "mode": "min",       # Reduce LR when loss stops decreasing
            "factor": 0.95,      # Multiply LR by this factor when plateau is detected (must be < 1.0)
            "patience": 10,      # Number of epochs with no improvement after which LR will be reduced
            "threshold": 1e-3    # Threshold for measuring the new optimum
        }
    )

    model.add_component(cooling_setpoint)
    model.add_component(heating_setpoint)

    # Plot before optimization results
    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("BuildingSpace", "indoorTemperature", "output"),
            # ("BuildingSpace", "wallTemperature", "output"),
            ("BuildingSpace", "outdoorTemperature", "input"),
            ("HeatingSetpoint", "scheduleValue", "output"),
            ("CoolingSetpoint", "scheduleValue", "output"),
            # ("SpaceHeater", "outletWaterTemperature", "output"),
        ],
        components_2axis=[
            ("SpaceHeater", "Power", "output"),
        ],
        components_3axis=[
            ("SpaceHeater", "waterFlowRate", "input"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Water flow rate [m³/s]",
        show=True,
        nticks=11
    )

    # for i, timestep in enumerate(simulator.dateTimeSteps):
    #     print("----")
    #     print("index: ", i)
    #     print("timestep: ", timestep)
    #     print("waterflow_schedule.output['scheduleValue']: ", waterflow_schedule.output["scheduleValue"].history[i])
    #     print("building_space.output['indoorTemperature']: ", building_space.output["indoorTemperature"].history[i])
    #     print("heating_setpoint.output['scheduleValue']: ", heating_setpoint.output["scheduleValue"].history[i])
    #     print("difference: ", building_space.output["indoorTemperature"].history[i] - heating_setpoint.output["scheduleValue"].history[i])

    # # Store original values for comparison
    # print("\n### BEFORE MANUAL CHANGE ###")
    # orig_indoor_temp = building_space.output["indoorTemperature"].history.clone().detach()
    # orig_waterflow = waterflow_schedule.output["scheduleValue"].history.clone().detach()
    # orig_setpoint = heating_setpoint.output["scheduleValue"].history.clone().detach()
    
    # # Calculate and print original constraint violations
    # orig_violations = torch.relu(orig_setpoint - orig_indoor_temp)  # Lower constraint violations (temp < heating setpoint)
    # orig_violation_sum = torch.sum(orig_violations).item()
    # orig_violation_count = torch.sum(orig_violations > 0).item()
    # print(f"Original violations - Count: {orig_violation_count}, Sum: {orig_violation_sum}")
    
    # # Calculate minimization term
    # orig_power = space_heater.output["Power"].history.clone().detach()
    # orig_power_sum = torch.sum(orig_power).item()
    # print(f"Original power sum: {orig_power_sum}")
    
    # # Save the original max values from the optimizer for comparison
    # print("\n### ORIGINAL OPTIMIZER STATE ###")
    # if hasattr(optimizer, 'max_values'):
    #     orig_max_values = optimizer.max_values.copy()
    #     print(f"Original max_values: {orig_max_values}")
    # else:
    #     orig_max_values = {}
    #     print("No max_values attribute found in optimizer")
    
    # waterflow_schedule.output["scheduleValue"]._normalized_history.requires_grad = False
    # waterflow_schedule.output["scheduleValue"]._normalized_history[20:23] = waterflow_schedule.output["scheduleValue"]._normalized_history[24]
    # waterflow_schedule.output["scheduleValue"]._normalized_history.requires_grad = True
    
    # # Run closure to get updated simulation
    # print("\n### AFTER MANUAL CHANGE ###")
    # closure = optimizer.closure()
    # print("loss: ", closure)
    
    # # Compare new values
    # new_indoor_temp = building_space.output["indoorTemperature"].history.clone().detach()
    # new_waterflow = waterflow_schedule.output["scheduleValue"].history.clone().detach()
    # new_setpoint = heating_setpoint.output["scheduleValue"].history.clone().detach()
    
    # # Calculate and print new constraint violations
    # new_violations = torch.relu(new_setpoint - new_indoor_temp)  # Lower constraint violations (temp < heating setpoint)
    # new_violation_sum = torch.sum(new_violations).item()
    # new_violation_count = torch.sum(new_violations > 0).item()
    # print(f"New violations - Count: {new_violation_count}, Sum: {new_violation_sum}")
    
    # # Calculate new minimization term
    # new_power = space_heater.output["Power"].history.clone().detach()
    # new_power_sum = torch.sum(new_power).item()
    # print(f"New power sum: {new_power_sum}")
    
    # # Plot before optimization results
    # fig, axes = tb.plot.plot_component(
    #     simulator,
    #     components_1axis=[
    #         ("BuildingSpace", "indoorTemperature", "output"),
    #         # ("BuildingSpace", "wallTemperature", "output"),
    #         ("BuildingSpace", "outdoorTemperature", "input"),
    #         ("HeatingSetpoint", "scheduleValue", "output"),
    #         ("CoolingSetpoint", "scheduleValue", "output"),
    #         # ("SpaceHeater", "outletWaterTemperature", "output"),
    #     ],
    #     components_2axis=[
    #         ("SpaceHeater", "Power", "output"),
    #     ],
    #     components_3axis=[
    #         ("SpaceHeater", "waterFlowRate", "input"),
    #     ],
    #     ylabel_1axis="Temperature [°C]",
    #     ylabel_2axis="Power [W]",
    #     ylabel_3axis="Water flow rate [m³/s]",
    #     show=True,
    #     nticks=11
    # )

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')#.reverse_order()
    # stats.print_stats(0.05)
