"""
Using the estimator example model, we will use the fact that the two damper systems are similar and 
are in the same group on the execution order. A successful compilation will yield a model system with
only one lumped damper system.
"""


import twin4build as tb
import datetime
from dateutil import tz
import twin4build.examples.utils as utils

model = tb.Model(id="model_compilator_example")

# Load the model from semantic file
filename_simulation = utils.get_path(["estimator_example", "instance_graph.ttl"])
model.load(simulation_model_filename=filename_simulation, verbose=0)
print(model)

### Set up simulation parameters

model.components["020B_temperature_sensor"].filename = utils.get_path(["estimator_example", "temperature_sensor.csv"])
model.components["020B_co2_sensor"].filename = utils.get_path(["estimator_example", "co2_sensor.csv"])
model.components["020B_valve_position_sensor"].filename = utils.get_path(["estimator_example", "valve_position_sensor.csv"])
model.components["020B_damper_position_sensor"].filename = utils.get_path(["estimator_example", "damper_position_sensor.csv"])
model.components["BTA004"].filename = utils.get_path(["estimator_example", "supply_air_temperature.csv"])
model.components["020B_temperature_heating_setpoint"].filename = utils.get_path(["estimator_example", "temperature_heating_setpoint.csv"])
model.components["outdoor_environment"].filename_outdoorTemperature = utils.get_path(["estimator_example", "outdoor_environment.csv"])
model.components["outdoor_environment"].filename_globalIrradiation = utils.get_path(["estimator_example", "outdoor_environment.csv"])
model.components["outdoor_environment"].filename_outdoorCo2Concentration = utils.get_path(["estimator_example", "outdoor_environment.csv"])


# Model compilation
compiled_model = model.build_compiled_model()
print(compiled_model)
compiled_model.simulation_model.load(verbose=0, validate_model=True)
compiled_model.simulation_model.visualize()

simulator = tb.Simulator(model)
step_size = 1200  # 20 minutes in seconds
start_time = [datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen")),
                datetime.datetime(year=2023, month=12, day=2, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen"))]           
end_time = [datetime.datetime(year=2023, month=12, day=1, hour=0, minute=0, second=0,
                            tzinfo=tz.gettz("Europe/Copenhagen")),
                 datetime.datetime(year=2023, month=12, day=5, hour=0, minute=0, second=0,
                            tzinfo=tz.gettz("Europe/Copenhagen"))]
# start_time = [datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
#                                 tzinfo=tz.gettz("Europe/Copenhagen"))]           
# end_time = [datetime.datetime(year=2023, month=11, day=27, hour=12, minute=0, second=0,
#                             tzinfo=tz.gettz("Europe/Copenhagen"))]

print(f"Simulation period: {start_time} to {end_time}")
print(f"Step size: {step_size} seconds ({step_size/60:.1f} minutes)")


# Run initial simulation for comparison
simulator.simulate(
    step_size=step_size,
    start_time=start_time,
    end_time=end_time
)

# Plot initial results
fig, axes = tb.plot.plot_component(
    simulator,
    components_1axis=[
        ("020B", "indoorTemperature", "output"),
        ("outdoor_environment", "outdoorTemperature", "output"),
        ("020B_temperature_heating_controller", "setpointValue", "input"),
    ],
    components_2axis=[
        ("020B_space_heater", "Power", "output"),
        ("020B", "heatGain", "input"),
    ],
    components_3axis=[
        ("020B_temperature_heating_controller", "inputSignal", "output"),
    ],
    ylabel_1axis="Temperature [°C]",
    ylabel_2axis="Power [W]",
    ylabel_3axis="Water flow rate [m³/s]",
    title="Before calibration",
    show=True,
    nticks=11
)
