import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
import datetime
import torch
from twin4build.optimizer.optimizer import Optimizer

def main():
    # Create a new model
    model = tb.Model(id="building_space_test_model")

    # Create building space
    building_space = tb.BuildingSpaceStateSpace(
        C_air=5000000.0,
        C_wall=10000000.0,
        C_int=500000.0,
        C_boundary=800000.0,
        R_out=0.01,
        R_in=0.01,
        R_int=100000,
        R_boundary=10000,
        f_wall=0,
        f_air=0,
        Q_occ_gain=100.0,
        CO2_occ_gain=0.004,
        CO2_start=400.0,
        infiltration=0.0,
        airVolume=100.0,
        id="BuildingSpace"
    )

    # Create minimal required schedules
    occupancy_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [0]
        },
        id="OccupancySchedule"
    )

    outdoor_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 10.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [10.0]
        },
        id="OutdoorTemperature"
    )

    solar_radiation = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [0.0]
        },
        id="SolarRadiation"
    )

    air_flow = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [0.0]
        },
        id="AirFlow"
    )

    supply_air_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 20.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [20.0]
        },
        id="SupplyAirTemperature"
    )

    outdoor_co2 = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 400.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [400.0]
        },
        id="OutdoorCO2"
    )

    boundary_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 20.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [20.0]
        },
        id="BoundaryTemperature"
    )

    # Create a simple heating schedule
    heating_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 1000.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [1000.0]
        },
        id="HeatingSchedule"
    )

    boundary_temp_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 20.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
        },
    )

    # Connect schedules to building space
    model.add_connection(occupancy_schedule, building_space, "scheduleValue", "numberOfPeople")
    model.add_connection(outdoor_temp, building_space, "scheduleValue", "outdoorTemperature")
    model.add_connection(solar_radiation, building_space, "scheduleValue", "globalIrradiation")
    model.add_connection(air_flow, building_space, "scheduleValue", "airFlowRate")
    model.add_connection(supply_air_temp, building_space, "scheduleValue", "supplyAirTemperature")
    model.add_connection(outdoor_co2, building_space, "scheduleValue", "outdoorCo2Concentration")
    model.add_connection(boundary_temp, building_space, "scheduleValue", "boundaryTemperature")
    model.add_connection(heating_schedule, building_space, "scheduleValue", "heatGain")
    model.add_connection(boundary_temp_schedule, building_space, "scheduleValue", "boundaryTemperature")

    # Load the model
    model.load()

    # Set up simulation parameters
    simulator = tb.Simulator(model)
    stepSize = 600  # 10 minutes in seconds
    startTime = datetime.datetime(
        year=2024, month=1, day=1, hour=0, minute=0, second=0,
        tzinfo=tz.gettz("Europe/Copenhagen")
    )
    endTime = datetime.datetime(
        year=2024, month=1, day=2, hour=0, minute=0, second=0,
        tzinfo=tz.gettz("Europe/Copenhagen")
    )

    # Define optimization targets
    targetInputs = {
        heating_schedule: "scheduleValue"  # Optimize the heating schedule
    }
    
    targetOutputs = {
        building_space: "indoorTemperature"  # Target the indoor temperature
    }

    # Run optimization
    optimizer = Optimizer(simulator)
    optimizer.optimize(
        targetInputs=targetInputs,
        targetOutputs=targetOutputs,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize
    )

    # Plot results
    tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("BuildingSpace", "indoorTemperature", "output"),
            ("BuildingSpace", "wallTemperature", "output"),
            ("BuildingSpace", "outdoorTemperature", "input"),
        ],
        components_2axis=[
            ("BuildingSpace", "Q_sh", "input"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        show=True,
        nticks=11
    )

if __name__ == "__main__":
    main()