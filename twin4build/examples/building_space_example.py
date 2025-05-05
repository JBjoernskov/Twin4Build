import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
import datetime
from dateutil import tz

def main():
    # Create a new model
    model = tb.Model(id="building_space_model")
    
    # Create a building space with thermal parameters
    building_space = tb.BuildingSpaceStateSpace(
        # Thermal parameters
        C_air=1000000.0,        # Thermal capacitance of indoor air [J/K]
        C_wall=3000000.0,      # Thermal capacitance of exterior wall [J/K]
        C_int=500000.0,        # Thermal capacitance of interior wall [J/K]
        C_boundary=800000.0,   # Thermal capacitance of boundary wall [J/K]
        R_out=0.03,            # Thermal resistance between wall and outdoor [K/W]
        R_in=0.01,             # Thermal resistance between wall and indoor [K/W]
        R_int=100000,            # Thermal resistance between interior wall and indoor [K/W]
        R_boundary=10000,      # Thermal resistance of boundary [K/W]
        
        # Heat gain parameters
        f_wall=0.3,            # Radiation factor for exterior wall
        f_air=0.1,             # Radiation factor for air
        Q_occ_gain=100.0,      # Heat gain per occupant [W]
        CO2_occ_gain=0.004,    # CO2 generation per person [kg/s]
        CO2_start=400.0,       # Initial CO2 concentration [ppm]
        
        # Optional parameters
        infiltration=0.0,      # Air infiltration rate [m³/s]
        airVolume=100.0,       # Air volume [m³]
        id="BuildingSpace"
    )
    
    # Create a schedule for occupancy
    occupancy_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [8, 9, 10, 12, 14, 16, 18],
            "ruleset_end_hour": [9, 10, 12, 14, 16, 18, 20],
            "ruleset_value": [0, 0, 0, 0, 0, 0, 0]  # Number of occupants
        },
        id="OccupancySchedule"
    )
    
    # Create an outdoor temperature profile
    outdoor_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 10.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 12, 18, 21, 23, 24],
            "ruleset_end_hour": [6, 12, 18, 21, 23, 24, 24],
            "ruleset_value": [5.0, 8.0, 15.0, 12.0, 8.0, 5.0, 5.0]  # Temperature in °C
        },
        id="OutdoorTemperature"
    )
    
    # Create a solar radiation profile
    solar_radiation = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 9, 12, 15, 18, 24],
            "ruleset_end_hour": [6, 9, 12, 15, 18, 24, 24],
            "ruleset_value": [0.0, 0, 0, 0, 0, 0.0, 0.0]  # Solar radiation in W/m²
        },
        id="SolarRadiation"
    )

    # Create an air flow rate schedule
    air_flow = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [0, 0, 0, 0.0, 0.0, 0.0, 0.0]  # Air flow rate in m³/s
        },
        id="AirFlow"
    )

    # Create a supply air temperature schedule
    supply_air_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 20.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [0, 0, 0, 0.0, 0.0, 0.0, 0.0]  # Temperature in °C
        },
        id="SupplyAirTemperature"
    )

    # Create an outdoor CO2 concentration schedule
    outdoor_co2 = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 400.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [0, 0, 0, 0.0, 0.0, 0.0, 0.0]  # CO2 concentration in ppm
        },
        id="OutdoorCO2"
    )

    # Create a space heater schedule
    space_heater = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 8, 16, 18, 0, 0],
            "ruleset_end_hour": [6, 8, 16, 18, 24, 0, 0],
            "ruleset_value": [0, 300, 500, 0, 0.0, 0.0, 0.0]  # Heat input in W
        },
        id="SpaceHeater"
    )

    # Create a boundary temperature schedule
    boundary_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 20.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
            "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
            "ruleset_value": [20, 20, 20, 20, 20, 20, 20]  # Temperature in °C
        },
        id="BoundaryTemperature"
    )

    # Connect components
    model.add_connection(occupancy_schedule, building_space, "scheduleValue", "numberOfPeople")
    model.add_connection(outdoor_temp, building_space, "scheduleValue", "outdoorTemperature")
    model.add_connection(solar_radiation, building_space, "scheduleValue", "globalIrradiation")
    model.add_connection(air_flow, building_space, "scheduleValue", "airFlowRate")
    model.add_connection(supply_air_temp, building_space, "scheduleValue", "supplyAirTemperature")
    model.add_connection(outdoor_co2, building_space, "scheduleValue", "outdoorCo2Concentration")
    model.add_connection(space_heater, building_space, "scheduleValue", "Q_sh")
    model.add_connection(boundary_temp, building_space, "scheduleValue", "T_boundary")
    
    # Load the model
    model.load()

    
    # Set up simulation parameters
    simulator = tb.Simulator()
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
    
    # Plot results
    tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("BuildingSpace", "indoorTemperature", "output"),
            ("BuildingSpace", "wallTemperature", "output"),
        ],
        components_2axis=[
            # ("BuildingSpace", "globalIrradiation", "input"),
            ("BuildingSpace", "Q_sh", "input")
        ],
        components_3axis=[
            ("BuildingSpace", "outdoorTemperature", "input"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Radiation/Heat",
        ylabel_3axis="Temperature [°C]",
        show=True,
        nticks=11
    )

if __name__ == "__main__":
    main() 