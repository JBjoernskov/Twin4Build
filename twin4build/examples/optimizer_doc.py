# %pip install git+https://github.com/JBjoernskov/Twin4Build.git # Uncomment in google colab
# Standard library imports
import datetime
import sys

# Third party imports
from dateutil import tz

# Local application imports
import twin4build as tb


def main():
    # Create a new model
    model = tb.Model(id="optimizer_doc")

    # Create a building space with thermal parameters
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
        id="BuildingSpace",
    )

    # Create space heater
    space_heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=2000.0,
        T_a_nominal_sh=60.0,
        T_b_nominal_sh=30.0,
        TAir_nominal_sh=21.0,
        thermalMassHeatCapacity=500000.0,
        nelements=3,
        id="SpaceHeater",
    )

    print("Building space component created:")
    print(building_space)
    print("\nSpace heater component created:")
    print(space_heater)

    # Create a schedule for occupancy
    occupancy_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [8, 9, 10, 12, 14, 16, 18],
            "ruleset_end_hour": [9, 10, 12, 14, 16, 18, 20],
            "ruleset_value": [0, 0, 0, 0, 0, 0, 0],
        },
        id="OccupancySchedule",
    )

    # Create an outdoor temperature profile
    outdoor_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 10.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 12, 18, 21, 23, 24],
            "ruleset_end_hour": [6, 12, 18, 21, 23, 24, 24],
            "ruleset_value": [5.0, 8.0, 15.0, 12.0, 8.0, 5.0, 5.0],
        },
        id="OutdoorTemperature",
    )

    # Create a solar radiation profile
    solar_radiation = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 9, 12, 15, 18, 24],
            "ruleset_end_hour": [6, 9, 12, 15, 18, 24, 24],
            "ruleset_value": [0.0, 100, 300, 300, 100, 0.0, 0.0],
        },
        id="SolarRadiation",
    )

    # Create supply and exhaust air flow schedules
    supply_air_flow = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 0.0}, id="SupplyAirFlow"
    )
    exhaust_air_flow = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 0.0}, id="ExhaustAirFlow"
    )

    # Create a supply air temperature schedule
    supply_air_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 20.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [0, 0, 0, 0.0, 0.0, 0.0, 0.0],
        },
        id="SupplyAirTemperature",
    )

    # Calculate nominal water flow rate
    mf = (
        space_heater.Q_flow_nominal_sh
        / 4180
        / (space_heater.T_a_nominal_sh - space_heater.T_b_nominal_sh)
    )

    # Create water flow schedule
    waterflow_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute": [0, 0],
            "ruleset_start_hour": [8, 19],
            "ruleset_end_hour": [16, 20],
            "ruleset_value": [mf, mf],
        },
        id="WaterflowSchedule",
    )

    # Create supply water temperature schedule
    supply_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 60.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [60, 60, 60, 60, 60, 60, 60],
        },
        id="SupplyTempSchedule",
    )

    # Example continues with other components...

    # Connect schedules to building space
    model.add_connection(
        occupancy_schedule, building_space, "scheduleValue", "numberOfPeople"
    )
    model.add_connection(
        outdoor_temp, building_space, "scheduleValue", "outdoorTemperature"
    )
    model.add_connection(
        solar_radiation, building_space, "scheduleValue", "globalIrradiation"
    )
    model.add_connection(
        supply_air_flow, building_space, "scheduleValue", "supplyAirFlowRate"
    )
    model.add_connection(
        exhaust_air_flow, building_space, "scheduleValue", "exhaustAirFlowRate"
    )
    model.add_connection(
        supply_air_temp, building_space, "scheduleValue", "supplyAirTemperature"
    )

    # Connect schedules to space heater
    model.add_connection(
        supply_temp, space_heater, "scheduleValue", "supplyWaterTemperature"
    )
    model.add_connection(
        waterflow_schedule, space_heater, "scheduleValue", "waterFlowRate"
    )

    # Connect building space indoorTemperature to space heater input
    model.add_connection(
        building_space, space_heater, "indoorTemperature", "indoorTemperature"
    )

    # Connect space heater output to building space input
    model.add_connection(space_heater, building_space, "Power", "heatGain")

    # Load the model
    model.load()

    model._simulation_model.visualize(literals=False)


if __name__ == "__main__":
    main()
