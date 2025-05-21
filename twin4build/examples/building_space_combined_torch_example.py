import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
import datetime
from dateutil import tz
import numpy as np
import torch
from twin4build.systems.building_space.building_space_torch_system import BuildingSpaceTorchSystem

def main():
    # Define parameters for the thermal and mass models
    thermal_kwargs = dict(
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
        infiltration=0.0,
        airVolume=100.0,
        id="BuildingSpaceThermal"
    )
    mass_kwargs = dict(
        V=120.0,  # Example: 100 m^3 * 1.2 kg/m^3
        G_occ=1e-6,
        m_inf=120/3600,
        id="BuildingSpaceMass"
    )
    # Create the combined building space model
    building_space = BuildingSpaceTorchSystem(thermal_kwargs=thermal_kwargs, mass_kwargs=mass_kwargs, id="BuildingSpaceCombined")

    # Create schedules for all required inputs
    schedules = {
        "outdoorTemperature": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 10.0}, id="OutdoorTemperature"),
        "globalIrradiation": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 0.0}, id="SolarRadiation"),
        "supplyAirFlowRate": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 0.1}, id="SupplyAirFlow"),
        "exhaustAirFlowRate": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 0.1}, id="ExhaustAirFlow"),
        "supplyAirTemperature": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 20.0}, id="SupplyAirTemperature"),
        "numberOfPeople": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 2}, id="OccupancySchedule"),
        "heatGain": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 0.0}, id="SpaceHeaterQ"),
        "boundaryTemperature": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 20.0}, id="BoundaryTemperature"),
        "outdoorCO2": tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 400.0}, id="OutdoorCO2"),
    }

    # Create a model and add the building space and schedules
    model = tb.Model(id="building_space_combined_torch_model")
    model.add_component(building_space)
    for name, sched in schedules.items():
        model.add_component(sched)
        model.add_connection(sched, building_space, "scheduleValue", name)

    # Load the model
    model.load()

    # Set up simulation parameters
    stepSize = 600  # 10 minutes in seconds
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=2, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    simulator = tb.Simulator(model)

    # Run simulation
    simulator.simulate(
        stepSize=stepSize,
        startTime=startTime,
        endTime=endTime
    )

    # Plot results: temperature and CO2
    tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("BuildingSpaceCombined", "indoorTemperature", "output"),
            ("BuildingSpaceCombined", "wallTemperature", "output"),
        ],
        components_2axis=[
            ("BuildingSpaceCombined", "indoorCO2", "output"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="CO2 [ppm]",
        show=True
    )

if __name__ == "__main__":
    main() 