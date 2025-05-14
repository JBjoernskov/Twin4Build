import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
import datetime
from dateutil import tz
import copy
import numpy as np
import torch
from twin4build.optimizer.optimizer import Optimizer
def main():
    # Create a new model
    model = tb.Model(id="building_space_with_space_heater_model")

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

    # Create space heater
    space_heater = tb.SpaceHeaterStateSpace(
        Q_flow_nominal_sh=1000.0,
        T_a_nominal_sh=60.0,
        T_b_nominal_sh=30.0,
        TAir_nominal_sh=21.0,
        thermalMassHeatCapacity=10000.0,
        nelements=10,
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
    air_flow = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [0, 0, 0, 0.0, 0.0, 0.0, 0.0]
        },
        id="AirFlow"
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
    outdoor_co2 = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 400.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [0, 0, 0, 0.0, 0.0, 0.0, 0.0]
        },
        id="OutdoorCO2"
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

    # Connect schedules to building space
    model.add_connection(occupancy_schedule, building_space, "scheduleValue", "numberOfPeople")
    model.add_connection(outdoor_temp, building_space, "scheduleValue", "outdoorTemperature")
    model.add_connection(solar_radiation, building_space, "scheduleValue", "globalIrradiation")
    model.add_connection(air_flow, building_space, "scheduleValue", "airFlowRate")
    model.add_connection(supply_air_temp, building_space, "scheduleValue", "supplyAirTemperature")
    model.add_connection(outdoor_co2, building_space, "scheduleValue", "outdoorCo2Concentration")
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
    stepSize = 600  # 10 minutes in seconds
    startTime = datetime.datetime(
        year=2024, month=1, day=1, hour=0, minute=0, second=0,
        tzinfo=tz.gettz("Europe/Copenhagen")
    )
    endTime = datetime.datetime(
        year=2024, month=1, day=5, hour=0, minute=0, second=0,
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
            ("BuildingSpace", "Q_sh", "input"),
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

    # Define optimization targets
    targetInputs = {
        waterflow_schedule: "scheduleValue"  # Optimize the water flow rate
    }
    
    targetOutputs = {
        building_space: "indoorTemperature"  # Target the indoor temperature
    }

    optimizer = Optimizer(simulator)
    optimizer.optimize(
        targetInputs=targetInputs,
        targetOutputs=targetOutputs,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize
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
            ("BuildingSpace", "Q_sh", "input"),
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

if __name__ == "__main__":
    main() 