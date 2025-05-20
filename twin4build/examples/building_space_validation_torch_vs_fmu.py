import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
import datetime
from dateutil import tz

def main():
    # Model parameters (should match for both models)
    thermal_kwargs = dict(
        C_air=1000000.0,
        C_wall=3000000.0,
        C_int=500000.0,
        C_boundary=1,
        R_out=0.03,
        R_in=0.01,
        R_int=10000,
        R_boundary=10000,
        f_wall=1,
        f_air=1,
        Q_occ_gain=100.0,
        id="BuildingSpaceThermal"
    )
    mass_kwargs = dict(
        infiltrationRate=0,#120.0*1.225/3600, # 120 m3/h
        airVolume=120.0,
        CO2_occ_gain=1e-6,#1e-6,
        id="BuildingSpaceMass"
    )
    # Instantiate both models
    torch_model = tb.BuildingSpaceTorchSystem(thermal_kwargs=thermal_kwargs, mass_kwargs=mass_kwargs, id="TorchModel")
    fmu_thermal_kwargs = {k: v for k, v in thermal_kwargs.items() if k != "id"}
    fmu_model = tb.BuildingSpace0AdjBoundaryOutdoorFMUSystem(
        **fmu_thermal_kwargs,
        airVolume=mass_kwargs["airVolume"],
        CO2_occ_gain=mass_kwargs["CO2_occ_gain"],
        CO2_start=400,
        C_supply=400,
        fraRad_sh=0.0,
        Q_flow_nominal_sh=1000.0,
        T_a_nominal_sh=60.0,
        T_b_nominal_sh=30.0,
        TAir_nominal_sh=21.0,
        n_sh=1.24,
        infiltration=mass_kwargs["infiltrationRate"],
        id="FmuModel"
    )

    # Define common schedules
    outdoor_temp_schedule = {
        "ruleset_default_value": 10.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 6, 12, 18, 21, 23, 24],
        "ruleset_end_hour": [6, 12, 18, 21, 23, 24, 24],
        "ruleset_value": [5.0, 8.0, 15.0, 12.0, 8.0, 5.0, 5.0]  # Temperature in °C
    }
    
    solar_radiation_schedule = {
        "ruleset_default_value": 0.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 6, 9, 12, 15, 18, 24],
        "ruleset_end_hour": [6, 9, 12, 15, 18, 24, 24],
        "ruleset_value": [0.0, 100, 500, 300, 100, 0.0, 0.0]  # Solar radiation in W/m²
    }
    airflow = 0.2# + 120.0*1.225/3600
    air_flow_schedule = {
        "ruleset_default_value": 0.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
        "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
        "ruleset_value": [0.0, airflow, 0.0, 0.0, 0.0, 0.0, 0.0]  # Air flow rate in m³/s
    }
    
    supply_air_temp_schedule = {
        "ruleset_default_value": 20.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
        "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
        "ruleset_value": [20.0, 22.0, 20.0, 20.0, 20.0, 20.0, 20.0]  # Temperature in °C
    }
    
    occupancy_schedule = {
        "ruleset_default_value": 0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [8, 9, 12, 14, 16, 18, 20],
        "ruleset_end_hour": [9, 12, 14, 16, 18, 20, 24],
        "ruleset_value": [5, 10, 8, 10, 5, 2, 0]  # Number of occupants
    }
    
    heating_schedule = {
        "ruleset_default_value": 0.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 6, 8, 16, 18, 0, 0],
        "ruleset_end_hour": [6, 8, 16, 18, 24, 0, 0],
        "ruleset_value": [0, 0, 0, 0, 0.0, 0.0, 0.0]  # Heat input in W
    }
    
    boundary_temp_schedule = {
        "ruleset_default_value": 20.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
        "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
        "ruleset_value": [20, 22, 20, 20, 20, 20, 20]  # Temperature in °C
    }
    
    co2_schedule = {
        "ruleset_default_value": 400.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
        "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
        "ruleset_value": [400, 400, 400, 400, 400, 400, 400]  # CO2 concentration in ppm
    }
    
    water_flow_schedule = {
        "ruleset_default_value": 0.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 6, 8, 16, 18, 0, 0],
        "ruleset_end_hour": [6, 8, 16, 18, 24, 0, 0],
        "ruleset_value": [0, 0, 0, 0, 0.0, 0.0, 0.0]  # Water flow rate in m³/s
    }
    
    supply_water_temp_schedule = {
        "ruleset_default_value": 60.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 8, 16, 0, 0, 0, 0],
        "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
        "ruleset_value": [60, 60, 60, 60, 60, 60, 60]  # Temperature in °C
    }

    # Torch model schedules (private)
    outdoor_temp_torch = tb.ScheduleSystem(weekDayRulesetDict=outdoor_temp_schedule, id="OutdoorTemperature_Torch")
    global_irradiation_torch = tb.ScheduleSystem(weekDayRulesetDict=solar_radiation_schedule, id="SolarRadiation_Torch")
    supply_air_flow_torch = tb.ScheduleSystem(weekDayRulesetDict=air_flow_schedule, id="SupplyAirFlow_Torch")
    exhaust_air_flow_torch = tb.ScheduleSystem(weekDayRulesetDict=air_flow_schedule, id="ExhaustAirFlow_Torch")
    supply_air_temp_torch = tb.ScheduleSystem(weekDayRulesetDict=supply_air_temp_schedule, id="SupplyAirTemperature_Torch")
    number_of_people_torch = tb.ScheduleSystem(weekDayRulesetDict=occupancy_schedule, id="OccupancySchedule_Torch")
    Q_sh_torch = tb.ScheduleSystem(weekDayRulesetDict=heating_schedule, id="SpaceHeaterQ_Torch")
    T_boundary_torch = tb.ScheduleSystem(weekDayRulesetDict=boundary_temp_schedule, id="BoundaryTemperature_Torch")
    supply_air_co2_torch = tb.ScheduleSystem(weekDayRulesetDict=co2_schedule, id="SupplyAirCO2_Torch")

    # FMU model schedules (private)
    outdoor_temp_fmu = tb.ScheduleSystem(weekDayRulesetDict=outdoor_temp_schedule, id="OutdoorTemperature_FMU")
    global_irradiation_fmu = tb.ScheduleSystem(weekDayRulesetDict=solar_radiation_schedule, id="SolarRadiation_FMU")
    supply_air_flow_fmu = tb.ScheduleSystem(weekDayRulesetDict=air_flow_schedule, id="SupplyAirFlow_FMU")
    supply_air_temp_fmu = tb.ScheduleSystem(weekDayRulesetDict=supply_air_temp_schedule, id="SupplyAirTemperature_FMU")
    number_of_people_fmu = tb.ScheduleSystem(weekDayRulesetDict=occupancy_schedule, id="OccupancySchedule_FMU")
    T_boundary_fmu = tb.ScheduleSystem(weekDayRulesetDict=boundary_temp_schedule, id="BoundaryTemperature_FMU")
    water_flow_fmu = tb.ScheduleSystem(weekDayRulesetDict=water_flow_schedule, id="WaterFlowRate_FMU")
    supply_water_temp_fmu = tb.ScheduleSystem(weekDayRulesetDict=supply_water_temp_schedule, id="SupplyWaterTemperature_FMU")
    outdoor_co2_fmu = tb.ScheduleSystem(weekDayRulesetDict=co2_schedule, id="OutdoorCO2_FMU")

    # Torch model
    torch_model_instance = tb.Model(id="torch_validation_model")
    torch_model_instance.add_component(torch_model)
    torch_model_instance.add_connection(supply_air_flow_torch, torch_model, "scheduleValue", "supplyAirFlowRate")
    torch_model_instance.add_connection(exhaust_air_flow_torch, torch_model, "scheduleValue", "exhaustAirFlowRate")
    torch_model_instance.add_connection(Q_sh_torch, torch_model, "scheduleValue", "heatGain")
    torch_model_instance.add_connection(supply_air_co2_torch, torch_model, "scheduleValue", "supplyAirCo2Concentration")
    torch_model_instance.add_connection(outdoor_temp_torch, torch_model, "scheduleValue", "outdoorTemperature")
    torch_model_instance.add_connection(global_irradiation_torch, torch_model, "scheduleValue", "globalIrradiation")
    torch_model_instance.add_connection(supply_air_temp_torch, torch_model, "scheduleValue", "supplyAirTemperature")
    torch_model_instance.add_connection(number_of_people_torch, torch_model, "scheduleValue", "numberOfPeople")
    torch_model_instance.add_connection(T_boundary_torch, torch_model, "scheduleValue", "boundaryTemperature")
    torch_model_instance.load()

    # FMU model
    fmu_model_instance = tb.Model(id="fmu_validation_model")
    fmu_model_instance.add_component(fmu_model)
    fmu_model_instance.add_connection(supply_air_flow_fmu, fmu_model, "scheduleValue", "airFlowRate")
    fmu_model_instance.add_connection(water_flow_fmu, fmu_model, "scheduleValue", "waterFlowRate")
    fmu_model_instance.add_connection(supply_air_temp_fmu, fmu_model, "scheduleValue", "supplyAirTemperature")
    fmu_model_instance.add_connection(outdoor_temp_fmu, fmu_model, "scheduleValue", "outdoorTemperature")
    fmu_model_instance.add_connection(global_irradiation_fmu, fmu_model, "scheduleValue", "globalIrradiation")
    fmu_model_instance.add_connection(number_of_people_fmu, fmu_model, "scheduleValue", "numberOfPeople")
    fmu_model_instance.add_connection(T_boundary_fmu, fmu_model, "scheduleValue", "boundaryTemperature")
    fmu_model_instance.add_connection(supply_water_temp_fmu, fmu_model, "scheduleValue", "supplyWaterTemperature")
    fmu_model_instance.add_connection(outdoor_co2_fmu, fmu_model, "scheduleValue", "outdoorCo2Concentration")
    fmu_model_instance.load()

    # Simulation parameters
    stepSize = 600
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=2, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    # Simulate both models
    simulator_torch = tb.Simulator(torch_model_instance)
    simulator_fmu = tb.Simulator(fmu_model_instance)
    simulator_torch.simulate(torch_model_instance, stepSize=stepSize, startTime=startTime, endTime=endTime)
    simulator_fmu.simulate(fmu_model_instance, stepSize=stepSize, startTime=startTime, endTime=endTime)

    # Plot comparison
    tb.plot.plot_component(
        simulator_torch,
        components_1axis=[
            ("TorchModel", "indoorTemperature", "output"),
            ("TorchModel", "outdoorTemperature", "input"),
        ],
        components_2axis=[
            ("TorchModel", "indoorCo2Concentration", "output"),
        ],
        components_3axis=[
            ("TorchModel", "supplyAirFlowRate", "input"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="CO2 concentration [ppm]",
        ylabel_3axis="Air flow rate [m³/s]",
        show=False
    )

    # Plot comparison
    tb.plot.plot_component(
        simulator_fmu,
        components_1axis=[
            ("FmuModel", "indoorTemperature", "output"),
            ("FmuModel", "outdoorTemperature", "input"),
        ],
        components_2axis=[
            ("FmuModel", "indoorCo2Concentration", "output"),
        ],
        components_3axis=[
            ("FmuModel", "airFlowRate", "input"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="CO2 concentration [ppm]",
        ylabel_3axis="Air flow rate [m³/s]",
        show=True
    )

if __name__ == "__main__":
    main() 