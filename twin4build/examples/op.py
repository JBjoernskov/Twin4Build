# %pip install twin4build # Uncomment in google colab
import twin4build as tb
import datetime
from dateutil import tz
# Create a new model
model = tb.Model(id="optimizer_example")
print(model)
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
        "ruleset_value": [0, 0, 0, 0, 0, 0, 0]
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
        "ruleset_value": [5.0, 8.0, 15.0, 12.0, 8.0, 5.0, 5.0]
    },
    noise_day_range=6,
    noise_hour_range=0.5,
    add_noise=True,
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
        "ruleset_value": [0.0, 100, 300, 300, 100, 0.0, 0.0]
    },
    id="SolarRadiation"
)

# Create supply and exhaust air flow schedules
supply_air_flow = tb.ScheduleSystem(
    weekDayRulesetDict={"ruleset_default_value": 0.0},
    id="SupplyAirFlow"
)
exhaust_air_flow = tb.ScheduleSystem(
    weekDayRulesetDict={"ruleset_default_value": 0.0},
    id="ExhaustAirFlow"
)

# Create a supply air temperature schedule
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

# Calculate nominal water flow rate
mf = space_heater.Q_flow_nominal_sh/4180/(space_heater.T_a_nominal_sh-space_heater.T_b_nominal_sh)

# Create water flow schedule
waterflow_schedule = tb.ScheduleSystem(
    weekDayRulesetDict = {
        "ruleset_default_value": 0,
        "ruleset_start_minute": [0,0],
        "ruleset_end_minute": [0,0],
        "ruleset_start_hour": [8, 19],
        "ruleset_end_hour": [16, 20],
        "ruleset_value": [mf, mf]
    },
    id="WaterflowSchedule"
)

# Create supply water temperature schedule
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

# Create heating and cooling setpoints
heating_setpoint = tb.ScheduleSystem(
    weekDayRulesetDict={
        "ruleset_default_value": 18.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 8, 17, 0, 0, 0, 0],
        "ruleset_end_hour": [8, 16, 24, 0, 0, 0, 0],
        "ruleset_value": [18.0, 21.0, 18.0, 18.0, 18.0, 18.0, 18.0]
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

cooling_setpoint = tb.ScheduleSystem(
    weekDayRulesetDict={
        "ruleset_default_value": 26.0,
        "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
        "ruleset_start_hour": [0, 8, 17, 0, 0, 0, 0],
        "ruleset_end_hour": [8, 17, 24, 0, 0, 0, 0],
        "ruleset_value": [26.0, 24.0, 30.0, 26.0, 26.0, 26.0, 26.0]
    },
    id="CoolingSetpoint"
)

# Connect schedules to building space
model.add_connection(occupancy_schedule, building_space, "scheduleValue", "numberOfPeople")
model.add_connection(outdoor_temp, building_space, "scheduleValue", "outdoorTemperature")
model.add_connection(solar_radiation, building_space, "scheduleValue", "globalIrradiation")
model.add_connection(supply_air_flow, building_space, "scheduleValue", "supplyAirFlowRate")
model.add_connection(exhaust_air_flow, building_space, "scheduleValue", "exhaustAirFlowRate")
model.add_connection(supply_air_temp, building_space, "scheduleValue", "supplyAirTemperature")

# Connect schedules to space heater
model.add_connection(supply_temp, space_heater, "scheduleValue", "supplyWaterTemperature")
model.add_connection(waterflow_schedule, space_heater, "scheduleValue", "waterFlowRate")

# Connect building space indoorTemperature to space heater input
model.add_connection(building_space, space_heater, "indoorTemperature", "indoorTemperature")

# Connect space heater output to building space input
model.add_connection(space_heater, building_space, "Power", "heatGain")

# Load the model
model.load()
# Set up simulation parameters
simulator = tb.Simulator(model)
step_size = 2400  # 40 minutes in seconds
# start_time = datetime.datetime(
#     year=2024, month=1, day=4, hour=0, minute=0, second=0,
#     tzinfo=tz.gettz("Europe/Copenhagen")
# )
# end_time = datetime.datetime(
#     year=2024, month=1, day=10, hour=0, minute=0, second=0,
#     tzinfo=tz.gettz("Europe/Copenhagen")
# )

start_time = [datetime.datetime(year=2024, month=1, day=4, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen")),
                datetime.datetime(year=2024, month=1, day=12, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen"))]           
end_time = [datetime.datetime(year=2024, month=1, day=10, hour=0, minute=0, second=0,
                            tzinfo=tz.gettz("Europe/Copenhagen")),
                 datetime.datetime(year=2024, month=1, day=14, hour=0, minute=0, second=0,
                            tzinfo=tz.gettz("Europe/Copenhagen"))]

# Run simulation
simulator.simulate(
    step_size=step_size,
    start_time=start_time,
    end_time=end_time
)

# Plot initial results
fig, axes = tb.plot.plot_component(
    simulator,
    components_1axis=[
        ("BuildingSpace", "indoorTemperature", "output"),
        ("BuildingSpace", "wallTemperature", "output"),
        ("BuildingSpace", "outdoorTemperature", "input"),
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



# Define optimization targets
variables = [
    (waterflow_schedule, "scheduleValue", 0, mf)  # Change water flow rate
]

objectives = [
    (space_heater, "Power", "min")  # Minimize power consumption
]

ineq_cons = [
    (building_space, "indoorTemperature", "upper", cooling_setpoint),  # Temperature should not exceed cooling setpoint
    (building_space, "indoorTemperature", "lower", heating_setpoint)   # Temperature should not fall below heating setpoint
]

# Create optimizer
optimizer = tb.Optimizer(simulator)

# Run optimization with Scipy solver
# On a normal laptop cpu, this takes around 2 minutes
options = {
    "maxiter": 150,
    "ftol": 1e-12,
    "disp": True
}
optimizer.optimize(
    start_time=start_time,
    end_time=end_time,
    step_size=step_size,
    variables=variables,
    objectives=objectives,
    eq_cons=None,
    ineq_cons=ineq_cons,
    method="scipy",
    options=options
)

# Add setpoints to model for plotting
model.add_component(cooling_setpoint)
model.add_component(heating_setpoint)

# Plot optimization results
fig, axes = tb.plot.plot_component(
    simulator,
    components_1axis=[
        ("BuildingSpace", "indoorTemperature", "output"),
        ("BuildingSpace", "outdoorTemperature", "input"),
        ("HeatingSetpoint", "scheduleValue", "output"),
        ("CoolingSetpoint", "scheduleValue", "output"),
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