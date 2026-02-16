"""
Controller Identification for Mortar bldg34 Room RM1100

This script identifies the control logic for the VAV box serving this room by:
- Using real building data from the Mortar bldg34 database
- Testing PID controllers (which cover the full control range including on-off behavior)
- Learning which sensors and setpoints drive the damper and reheat valve
- Identifying PID parameters (Kp, Ti, Td)

Control Logic in VAV Systems:
- Zone temperature too high → increase damper (more cooling air)
- Zone temperature too low → increase reheat valve (more heating)
- Damper position often has minimum flow requirement
- Reheat valve typically uses PI control for zone temperature

Room RM1100:
- Zone temperature sensor
- Zone temperature setpoint
- Damper position command
- Reheat valve command
- Supply air flow measurement
"""

# Standard library imports
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Local application imports
import twin4build as tb
from twin4build.utils.data_loaders.load import load_from_database
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)

# ==========================================================================
# CONFIGURATION
# ==========================================================================

# Database configuration for Mortar bldg34
db_config = {
    "table_name": "mortar_bldg34",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

# Time range for identification
# The mortar dataset has data from 2011 to 2018
# Choose a representative period with good data quality
timezone_utc = timezone.utc

# Start with a single shorter period for debugging
# Use a winter period to capture heating behavior (reheat valve active)
# start_time = [datetime(2014, 1, 15, 0, 0, tzinfo=timezone_utc)]
# end_time = [datetime(2018, 1, 18, 0, 0, tzinfo=timezone_utc)]  # 3 days

step_size = 600  # 15 minutes (matching the typical HVAC control interval)

# Uncomment below for multiple periods (winter + summer):
start_time = [
    datetime(2016, 1, 15, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2016, 7, 15, 0, 0, tzinfo=timezone_utc),   # Summer - cooling
]
end_time = [
    datetime(2016, 1, 22, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2016, 7, 22, 0, 0, tzinfo=timezone_utc),   # 1 week summer
]

# ==========================================================================
# SENSOR UUID MAPPINGS FROM MORTAR bldg34
# These UUIDs are from the ref:hasTimeseriesId in the TTL file
# ==========================================================================

# Room RM1100 sensors
ROOM_SENSORS = {
    'zone_temp': '414e0dac-a298-4494-b071-7eeaaa21eb69',           # Zone_Air_Temp
    'zone_temp_setpoint': '854e50f1-e47f-4658-b586-f83343ec1b7d',  # Zone_Air_Temp_Setpoint
    'damper_position': '95fadb2b-3e3f-478c-94bf-0b9cbb0fdc33',     # Zone_Air_Damper_Command
    'reheat_valve': 'dcac3ea2-5698-4bcb-886d-1e4bd0295444',        # Zone_Reheat_Valve_Command
}

# ==========================================================================
# BUILD IDENTIFICATION MODEL
# ==========================================================================

print("="*80)
print("BUILDING CONTROLLER IDENTIFICATION MODEL FOR MORTAR RM1100")
print("="*80)

# Create sensors using tb.SensorSystem with real stream UUIDs
print("\nCreating sensor systems...")

# from fahrenheit to celsius
transformation = lambda x: (x - 32) * 5/9

# Primary sensor: Zone temperature
zone_temp_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS['zone_temp'],
    id="zone_temp_sensor",
    dbconfig=db_config,
    transformation=transformation
)

# Setpoint sensor
zone_temp_setpoint_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS['zone_temp_setpoint'],
    id="zone_temp_setpoint_sensor",
    dbconfig=db_config,
    transformation=transformation
)

# Transform damper/valve positions from percentage (0-100) to fraction (0-1)
transformation = lambda x: x / 100.0

# Actuators: Damper position and Reheat valve
damper_actuator = tb.SensorSystem(
    uuid=ROOM_SENSORS['damper_position'],
    id="damper_actuator",
    dbconfig=db_config,
    transformation=transformation
)

reheat_valve_actuator = tb.SensorSystem(
    uuid=ROOM_SENSORS['reheat_valve'],
    id="reheat_valve_actuator",
    dbconfig=db_config,
    transformation=transformation
)

actuator_sensors = [damper_actuator, reheat_valve_actuator]
print(f"  Created {len(actuator_sensors)} actuator sensors")
print(f"    - Damper position (cooling control)")
print(f"    - Reheat valve (heating control)")

# ==========================================================================
# CREATE CONTROLLER IDENTIFICATION SYSTEM
# ==========================================================================

print("\nCreating controller identification system...")

# Sensors that could potentially be used for feedback
sensors = [zone_temp_sensor]

# Setpoints that could be used
setpoints = [zone_temp_setpoint_sensor]

# Create controller with 2 actuators (damper + reheat valve)
controller = tb.ControllerIdentificationTorchSystem(
    n_sensors=len(sensors),
    n_setpoints=len(setpoints),
    n_actuators=len(actuator_sensors),  # 2 actuators
    id="identified_vav_controller",
)

print(controller.summary())

print(f"  Created controller with {len(actuator_sensors)} actuators")
print(f"  Will learn from {len(sensors)} sensor(s) and {len(setpoints)} setpoint(s)")

# ==========================================================================
# BUILD COMPLETE MODEL
# ==========================================================================

print("\nBuilding complete identification model...")

model = tb.Model(id="mortar_rm1100_identification_model")

# Add sensors
model.add_component(zone_temp_sensor)
model.add_component(zone_temp_setpoint_sensor)

# Add actuator sensors
for actuator in actuator_sensors:
    model.add_component(actuator)

# Add controller
model.add_component(controller)

# Connect sensors to controller (feedback)
for i, sensor in enumerate(sensors):
    model.add_connection(sensor, controller, "measuredValue", "sensorValue", input_port_index=i)

# Connect setpoints to controller
for i, setpoint in enumerate(setpoints):
    model.add_connection(setpoint, controller, "measuredValue", "setpointValue", input_port_index=i)

# Connect controller outputs to actuators
for i, actuator in enumerate(actuator_sensors):
    model.add_connection(controller, actuator, "inputSignal", "measuredValue", output_port_index=i)

print("  Model connections established")

# Load component data
print("\nLoading component data from database...")
model.load()

# Verify data was loaded for all sensors
print("\nVerifying loaded data:")
all_sensors = [zone_temp_sensor, zone_temp_setpoint_sensor, 
               damper_actuator, reheat_valve_actuator]
for sensor in all_sensors:
    if hasattr(sensor, 'df') and sensor.df is not None:
        print(f"  {sensor.id}: {len(sensor.df)} rows, range: {sensor.df.index.min()} to {sensor.df.index.max()}")
    else:
        print(f"  {sensor.id}: No data loaded (df is None or missing)")

# ==========================================================================
# RUN INITIAL SIMULATION
# ==========================================================================

print("\n" + "="*80)
print("RUNNING INITIAL SIMULATION (BEFORE IDENTIFICATION)")
print("="*80)




# Create simulator
simulator = tb.Simulator(model)


# simulator.set_simulation_timesteps(start_time=start_time, end_time=end_time, step_size=step_size)
# model.initialize(start_time=start_time, end_time=end_time, step_size=step_size) ###################

##############

simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

# Get initial predictions and actual values
initial_predictions = []
actual_values = []

print("\nInitial predictions vs actual:")
for i, actuator in enumerate(actuator_sensors):
    # Get predictions from ALL simulation periods
    pred = actuator.input["measuredValue"].history(i_c=0).detach().numpy().T
    actual = actuator.time_series_input.values[:,:,0].detach().numpy().T
    
    initial_predictions.append(pred)
    actual_values.append(actual)
    
    # Compute MAE per simulation
    mae_per_sim = np.mean(np.abs(pred - actual), axis=1)
    mae_avg = np.mean(mae_per_sim)
    
    print(f"  Actuator {i} ({actuator.id}):")
    print(f"    Shape: {pred.shape}")
    for j, mae in enumerate(mae_per_sim):
        print(f"    Simulation {j} MAE: {mae:.4f}")
    print(f"    Average MAE: {mae_avg:.4f}")


###############

# Plot initial results
print("\nPlotting initial predictions...")

zone_temp_data = zone_temp_sensor.time_series_input.values[:,:,0].detach().numpy().T
zone_setpoint_data = zone_temp_setpoint_sensor.time_series_input.values[:,:,0].detach().numpy().T

actuator_names = ["Damper Position", "Reheat Valve"]

for i in range(len(actuator_sensors)):
    entry = [
        tb.plot.Entry(actual_values[i], label=f"Actual {actuator_names[i]}"), 
        tb.plot.Entry(initial_predictions[i], label=f"Initial Prediction"),
        tb.plot.Entry(zone_temp_data, label=f"Zone Temperature", axis=2),
        tb.plot.Entry(zone_setpoint_data, label=f"Zone Setpoint", axis=2, linestyle="--")
    ]
    tb.plot.plot(
        simulator.date_time_steps, 
        entry, 
        title=f"{actuator_names[i]} ({actuator_sensors[i].id}): Initial Predictions vs Actual", 
        ylabel_1axis="Position (0-1)",
        ylabel_2axis="Temperature (°F)"
    )

plt.show()

# ==========================================================================
# SETUP ESTIMATOR WITH PARAMETERS
# ==========================================================================

print("\n" + "="*80)
print("SETTING UP PARAMETER ESTIMATION")
print("="*80)

# Get parameters from the controller
parameters = model.components['identified_vav_controller'].get_estimator_parameters()

print(f"\n  Total parameters to estimate: {len(parameters)}")

# Debug: Print parameter mapping
print("\n  [DEBUG] Parameter mapping (theta index -> parameter):")
theta_idx = 0
for comp, attr, x0, lb, ub, *_ in parameters:
    if isinstance(x0, (list, np.ndarray)):
        n_vals = len(x0)
    else:
        n_vals = 1
    print(f"    theta[{theta_idx}:{theta_idx+n_vals}] -> {attr} (x0={x0}, lb={lb}, ub={ub})")
    theta_idx += n_vals

# Count parameter types
n_alpha = sum(1 for p in parameters if 'alpha' in p[1])
n_beta = sum(1 for p in parameters if 'beta' in p[1])
n_gamma = sum(1 for p in parameters if 'gamma' in p[1])
n_ctrl = sum(1 for p in parameters if 'candidate' in p[1])

print(f"\n  - Alpha (candidate selection): {n_alpha} ({len(actuator_sensors)} actuators × {len(controller.candidate_controller_classes)} candidates)")
print(f"\n  - Beta (sensor selection): {n_beta} ({len(actuator_sensors)} actuators × {len(sensors)} sensors)")
print(f"  - Gamma (setpoint selection): {n_gamma} ({len(actuator_sensors)} actuators × {len(setpoints)} setpoints)")
print(f"  - Controller PID parameters: {n_ctrl}")

# Setup measurements: all actuator sensors with measurement uncertainty
measurements = []
for actuator in actuator_sensors:
    measurements.append((actuator, 0.02))  # 2% measurement uncertainty

print(f"\n  Measurements: {len(measurements)} actuators")

# Print initial weights
print("\n  Initial weights BEFORE estimation:")
for a in range(len(actuator_sensors)):
    alpha_vals = controller._get_alpha_vector(a)
    beta_vals = controller._get_beta_vector(a)
    gamma_vals = controller._get_gamma_vector(a)
    print(f"    Actuator {a} ({actuator_sensors[a].id}):")
    print(f"      Alpha: {[f'{v:.3f}' for v in alpha_vals]}")
    print(f"      Beta:  {[f'{v:.3f}' for v in beta_vals]}")
    print(f"      Gamma: {[f'{v:.3f}' for v in gamma_vals]}")

# Print initial controller parameters
print("\n  Initial PID parameters BEFORE estimation:")
for a in range(len(actuator_sensors)):
    ctrl = controller._get_candidate(a, 0)
    print(f"    Actuator {a} ({actuator_names[a]}):")
    print(f"      kp = {ctrl.kp.get().item():.6f}")
    print(f"      Ti = {ctrl.Ti.get().item():.6f}")
    print(f"      Td = {ctrl.Td.get().item():.6f}")

# ==========================================================================
# RUN ESTIMATION WITH REGULARIZATION
# ==========================================================================

print("\n" + "="*80)
print("RUNNING PARAMETER ESTIMATION")
print("="*80)

estimator = tb.Estimator(simulator)

# Optimization options
options = {
    "maxiter": 1000,
    "ftol": 1e-12,
    "disp": True,
}

print("\n  Using SLSQP optimizer with automatic differentiation")
print("  Regularization: λ=0.01 (binarization penalty)")
print(f"  Max iterations: {options['maxiter']}")

# Add debug wrapper to print objective and gradients during optimization
_orig_obj_ad = estimator._obj_ad
_orig_jac_ad = estimator._jac_ad
_orig_obj = estimator._obj
_debug_iter = [0]

from twin4build.utils.rgetattr import rgetattr
# Build parameter names for debug output
_param_names = []
for comp, attr, x0, lb, ub, *_ in parameters:
    a = rgetattr(comp, attr)
    if isinstance(a, (list, np.ndarray, torch.Tensor)):
        for i in range(a.shape[0]):
            _param_names.append(f"{attr}[{i}]")
    else:
        _param_names.append(attr)

def _debug_obj(theta_tensor, output="scalar"):
    """Wrap _obj to print controller params AFTER they're set."""
    result = _orig_obj(theta_tensor, output)
    
    # Only print on first iteration
    if _debug_iter[0] <= 1000000:
        # Print predictions vs actual data
        print(f"\n  [PREDICTIONS vs ACTUAL (first 20 timesteps per simulation)]:")
        for i, actuator in enumerate(actuator_sensors):
            pred = actuator.input["measuredValue"].history(i_c=0)
            actual = actuator.time_series_input.values[:,:,0]
            
            # Show first simulation, first 20 timesteps
            if pred.shape[0] >= 20:
                print(f"    Actuator {i} (sim 0): pred[:20] = {pred[:20, 0]}")
                print(f"    Actuator {i} (sim 0): actual[:20] = {actual[:20, 0]}")
            
            # Compute MSE per simulation
            mse_per_sim = torch.mean((pred - actual)**2, axis=0)
            print(f"    Actuator {i} MSE per sim: {mse_per_sim}")
    
    return result

def _debug_obj_ad(theta, output="scalar"):
    result = _orig_obj_ad(theta, output)
    _debug_iter[0] += 1
    print(f"\n[DEBUG iter {_debug_iter[0]}] obj = {result:.6f}")
    
    # Print theta with parameter names (only first 10 to avoid clutter)
    print(f"  theta (showing first 10 of {len(theta)} params):")
    for i, (name, val) in enumerate(zip(_param_names[:10], theta[:10])):
        print(f"    {name:25s} = {val:10.6f}")
    return result

def _debug_jac_ad(theta, output="scalar"):
    result = _orig_jac_ad(theta, output)
    grad_norm = np.linalg.norm(result)
    print(f"  |grad| = {grad_norm:.6f}")
    
    # Print gradients (only first 10 to avoid clutter)
    print(f"  grad (showing first 10 of {len(result)} params):")
    for i, (name, val) in enumerate(zip(_param_names[:10], result[:10])):
        print(f"    {name:25s} = {val:10.6f}")
    return result

estimator._obj = _debug_obj
estimator._obj_ad = _debug_obj_ad
estimator._jac_ad = _debug_jac_ad

# Run estimation
result = estimator.estimate(
    start_time=start_time,
    end_time=end_time,
    step_size=step_size,
    parameters=parameters,
    measurements=measurements,
    n_warmup=0,
    method=("scipy", "SLSQP", "ad"),
    options=options,
)

# Restore original methods
estimator._obj = _orig_obj
estimator._obj_ad = _orig_obj_ad
estimator._jac_ad = _orig_jac_ad

print("\n  Optimization complete!")
print(f"  Final objective value: {result['final_objective']:.6f}")
print(f"  Success: {result['success']}")
print(f"  Message: {result['message']}")

# ==========================================================================
# FINAL SIMULATION
# ==========================================================================

print("\n" + "="*80)
print("RUNNING FINAL SIMULATION (AFTER IDENTIFICATION)")
print("="*80)

simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

# Get final predictions
final_predictions = []

print("\nFinal predictions vs actual:")
for i, actuator in enumerate(actuator_sensors):
    # Get predictions from ALL simulation periods
    pred = actuator.input["measuredValue"].history(i_c=0).detach().numpy().T
    
    final_predictions.append(pred)
    
    # Compute MAE per simulation
    mae_per_sim = np.mean(np.abs(pred - actual_values[i]), axis=1)
    mae_avg = np.mean(mae_per_sim)
    
    print(f"  Actuator {i} ({actuator.id}):")
    for j, mae in enumerate(mae_per_sim):
        print(f"    Simulation {j} MAE: {mae:.4f}")
    print(f"    Average MAE: {mae_avg:.4f}")

# ==========================================================================
# ANALYZE IDENTIFIED CONTROLLER
# ==========================================================================

print("\n" + "="*80)
print("IDENTIFIED CONTROLLER STRUCTURE")
print("="*80)

sensor_names = ["zone_temp_sensor"]
setpoint_names = ["zone_temp_setpoint_sensor"]

print("\nBETA WEIGHTS (Sensor Selection):")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_names[a]}):")
    beta_vals = controller._get_beta_vector(a)
    for s in range(len(sensors)):
        beta = beta_vals[s].item()
        sensor_name = sensor_names[s]
        selected = " <-- SELECTED" if beta > 0.5 else ""
        print(f"    β_{a},{s} ({sensor_name}): {beta:.4f}{selected}")

print("\nGAMMA WEIGHTS (Setpoint Selection):")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_names[a]}):")
    gamma_vals = controller._get_gamma_vector(a)
    for s in range(len(setpoints)):
        gamma = gamma_vals[s].item()
        sp_name = setpoint_names[s]
        selected = " <-- SELECTED" if gamma > 0.5 else ""
        print(f"    γ_{a},{s} ({sp_name}): {gamma:.4f}{selected}")

print("\nIDENTIFIED PID PARAMETERS:")
print("(Note: PID covers full control range - high Kp with low Ti approximates on-off)")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_names[a]}):")
    
    # Access the PID controller for this actuator
    ctrl = controller._get_candidate(a, 0)  # First (and only) candidate controller
    kp = ctrl.kp.get().item()
    ti = ctrl.Ti.get().item()
    td = ctrl.Td.get().item()
    
    print(f"    Kp = {kp:.6f}")
    print(f"    Ti = {ti:.6f} seconds")
    print(f"    Td = {td:.6f} seconds")
    
    # Interpret the control type
    if ti > 100 and td < 0.1:
        control_type = "Pure P (proportional only)"
    elif ti < 100 and td < 0.1:
        control_type = "PI (proportional-integral)"
    elif kp > 1.0 and ti < 5.0:
        control_type = "Aggressive PI (near on-off behavior)"
    else:
        control_type = "PID (full 3-term control)"
    
    print(f"    Type: {control_type}")

# ==========================================================================
# PLOT RESULTS
# ==========================================================================

print("\n" + "="*80)
print("GENERATING FINAL COMPARISON PLOTS")
print("="*80)

# Refresh data for plotting
zone_temp_data = zone_temp_sensor.time_series_input.values[:,:,0].detach().numpy().T
zone_setpoint_data = zone_temp_setpoint_sensor.time_series_input.values[:,:,0].detach().numpy().T

# Plot final results with all three: actual, initial, and identified
for i in range(len(actuator_sensors)):
    # Compute average MAE across all simulations
    mae_initial = np.mean(np.abs(initial_predictions[i] - actual_values[i]))
    mae_final = np.mean(np.abs(final_predictions[i] - actual_values[i]))
    
    entry = [
        tb.plot.Entry(actual_values[i], label=f"Actual {actuator_names[i]}"),
        tb.plot.Entry(initial_predictions[i], label=f"Initial Prediction"),
        tb.plot.Entry(final_predictions[i], label=f"Identified Prediction"),
        tb.plot.Entry(zone_temp_data, label=f"Zone Temperature", axis=2),
        tb.plot.Entry(zone_setpoint_data, label=f"Zone Setpoint", axis=2, linestyle="--")
    ]
    
    tb.plot.plot(
        simulator.date_time_steps, 
        entry, 
        title=f"{actuator_names[i]}: Initial MAE={mae_initial:.4f}, Final MAE={mae_final:.4f}", 
        ylabel_1axis="Position (0-1)",
        ylabel_2axis="Temperature (°F)"
    )

plt.show()

print("\n" + "="*80)
print("CONTROLLER IDENTIFICATION COMPLETE")
print("="*80)
print("\nSummary:")
print(f"  Optimization success: {result['success']}")
print(f"  Final objective: {result['final_objective']:.6f}")
print(f"\n  Average improvement per actuator:")
for i in range(len(actuator_sensors)):
    mae_initial = np.mean(np.abs(initial_predictions[i] - actual_values[i]))
    mae_final = np.mean(np.abs(final_predictions[i] - actual_values[i]))
    improvement = (mae_initial - mae_final) / mae_initial * 100 if mae_initial > 0 else 0
    print(f"    {actuator_names[i]}: {improvement:.1f}% improvement (MAE: {mae_initial:.4f} → {mae_final:.4f})")

print("\n" + "="*80)
print("VAV CONTROL INTERPRETATION")
print("="*80)
print("""
Expected VAV Control Logic:
  Damper Position:
    - Controls cooling by modulating airflow
    - When zone temp > setpoint → open damper (more cold air)
    - When zone temp < setpoint → close to minimum (maintain ventilation)
    
  Reheat Valve:
    - Controls heating by modulating hot water/steam flow
    - When zone temp < setpoint → open valve (add heat)
    - When zone temp > setpoint → close valve
    - Often coupled with damper at minimum position

The identified parameters reveal:
  - Which sensors/setpoints actually drive each actuator (beta/gamma weights)
  - The PID gains that best explain the observed control behavior
  - Whether the control is primarily P, PI, or full PID
""")
