"""
Controller Identification for Room 10c340b0_0e7b FCUs

This script identifies the control logic for the two FCUs serving this room by:
- Using real building data from the database
- Testing PID controllers (which cover the full control range including on-off behavior)
- Learning which sensors and setpoints drive each valve
- Identifying PID parameters (Kp, Ti, Td)

Note: PID controllers can represent the full spectrum of control:
- Pure P: Ti → ∞, Td = 0
- PI: Ti finite, Td = 0 (most common for HVAC)
- On-Off approximation: High Kp with low Ti
- Full PID: All three terms active

FCU 1 (135e5147): 2 valves (likely heating + cooling)
FCU 2 (3458632f): 2 valves (likely heating + cooling)
"""

# Standard library imports
import os
from datetime import datetime, timezone
from dateutil import tz
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

# Database configuration
db_config = {
    "table_name": "bts_site_a",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

# Time range for identification (use periods with good data)
# Using lists to handle multiple time periods for heating and cooling
# start_time = [datetime(2022, 2, 1, 21, 10, tzinfo=timezone.utc),
#               datetime(2022, 7, 11, 18, tzinfo=timezone.utc)]  # Cooling, heating
# end_time = [datetime(2022, 2, 2, 20, tzinfo=timezone.utc),
#             datetime(2022, 7, 12, 18, tzinfo=timezone.utc)]  # Cooling, heating

from dateutil.zoneinfo import get_zonefile_instance
zonenames = list(get_zonefile_instance().zones)
# print(zonenames)



timezone = ZoneInfo("Australia/Canberra")
# timezone = timezone.utc

start_time = [datetime(2022, 7, 11, 18, 0, tzinfo=timezone)]  # Cooling, heating
end_time = [datetime(2022, 7, 12, 18, tzinfo=timezone)]  # Cooling, heating
step_size = 600  # 10 minutes for both periods


# ==========================================================================
# STREAM IDs FROM ANALYSIS
# ==========================================================================

# Room temperature sensor (shared between room and FCU1)
ROOM_TEMP_STREAM = "352e0a5b_240d_4cf1_97eb_81e8fb49e1ee"

# FCU 1 (135e5147_00e4_43a4_89cd_b6937633e0dd)
FCU1 = {
    'id': '135e5147_00e4_43a4_89cd_b6937633e0dd',
    'label': 'FCU1',
    'temp_setpoint': "ad7d7e3a_39b3_450d_8326_84b23a18f7b7",
    'valve_pos_1': "abe58af3_798a_4072_9944_f18bb5b93e5a",  # Actuator 0
    'valve_pos_2': "86ca8910_e873_4b3d_8c30_c0c448bd7165",  # Actuator 1
}

# FCU 2 (3458632f_4ce7_496d_a9cf_c7280ada9250)
FCU2 = {
    'id': '3458632f_4ce7_496d_a9cf_c7280ada9250',
    'label': 'FCU2',
    'temp_setpoint': "9d21e517_0ae6_48fe_b131_74422db7999a",
    'valve_pos_1': "f5b1e556_105d_4057_9c88_490997c51481",  # Actuator 2
    'valve_pos_2': "a26a33b8_b2de_4eb2_a884_850cb7b65413",  # Actuator 3
}

# ==========================================================================
# BUILD IDENTIFICATION MODEL
# ==========================================================================

print("="*80)
print("BUILDING CONTROLLER IDENTIFICATION MODEL")
print("="*80)

# Create sensors using tb.SensorSystem with real stream IDs
print("\nCreating sensor systems...")

# Primary sensor: Room temperature
room_temp_sensor = tb.SensorSystem(
    uuid=ROOM_TEMP_STREAM,
    id="room_temp_sensor",
    dbconfig=db_config
)

# Setpoint sensors
fcu1_setpoint_sensor = tb.SensorSystem(
    uuid=FCU1['temp_setpoint'],
    id="fcu1_setpoint",
    dbconfig=db_config
)

fcu2_setpoint_sensor = tb.SensorSystem(
    uuid=FCU2['temp_setpoint'],
    id="fcu2_setpoint",
    dbconfig=db_config
)


transformation = lambda x: x/100.0 # Convert to 1 from percentage

# Actuators: 4 valves total (2 per FCU)
actuator_0 = tb.SensorSystem(uuid=FCU1['valve_pos_1'], id="fcu1_valve1_actuator", dbconfig=db_config, transformation=transformation)
actuator_1 = tb.SensorSystem(uuid=FCU1['valve_pos_2'], id="fcu1_valve2_actuator", dbconfig=db_config, transformation=transformation)
actuator_2 = tb.SensorSystem(uuid=FCU2['valve_pos_1'], id="fcu2_valve1_actuator", dbconfig=db_config, transformation=transformation)
actuator_3 = tb.SensorSystem(uuid=FCU2['valve_pos_2'], id="fcu2_valve2_actuator", dbconfig=db_config, transformation=transformation)

actuator_sensors = [actuator_0, actuator_1, actuator_2, actuator_3]

print(f"  Created {len(actuator_sensors)} actuator sensors")


# ==========================================================================
# CREATE CONTROLLER IDENTIFICATION SYSTEM
# ==========================================================================

print("\nCreating controller identification system...")

# Prepare sensor and setpoint lists
sensors = [room_temp_sensor]
setpoints = [fcu1_setpoint_sensor, fcu2_setpoint_sensor]

# Create controller with 4 actuators (2 valves per FCU)
controller = tb.ControllerIdentificationTorchSystem(
    n_sensors=len(sensors),
    n_setpoints=len(setpoints),
    n_actuators=len(actuator_sensors),  # 4 valves total
    id="identified_controller",
)

print(controller.summary())

print(f"  Created controller with {len(actuator_sensors)} actuators")
print(f"  Will learn from {len(sensors)} sensor(s) and {len(setpoints)} setpoint(s)")

# ==========================================================================
# BUILD COMPLETE MODEL
# ==========================================================================

print("\nBuilding complete identification model...")

model = tb.Model(id="fcu_identification_model")

# Add sensors
model.add_component(room_temp_sensor)
model.add_component(fcu1_setpoint_sensor)
model.add_component(fcu2_setpoint_sensor)

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
model.load()

# ==========================================================================
# RUN INITIAL SIMULATION
# ==========================================================================

print("\n" + "="*80)
print("RUNNING INITIAL SIMULATION (BEFORE IDENTIFICATION)")
print("="*80)

# Create simulator
simulator = tb.Simulator(model)

simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

# Get initial predictions and actual values
initial_predictions = []
actual_values = []

print("\nInitial predictions vs actual:")
for i, actuator in enumerate(actuator_sensors):
    # Get predictions from ALL simulation periods
    # history(i_c=0) returns (n_timesteps, n_simulations), transpose to (n_simulations, n_timesteps)
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

# Plot initial results
print("\nPlotting initial predictions...")

room_temp_data = room_temp_sensor.time_series_input.values[:,:,0].detach().numpy().T

for i in range(len(actuator_sensors)):
    entry = [
        tb.plot.Entry(actual_values[i], label=f"Actual Valve Position"), 
        tb.plot.Entry(initial_predictions[i], label=f"Initial Prediction"),
        tb.plot.Entry(room_temp_data, label=f"Room Temperature", axis=2)
    ]
    tb.plot.plot(
        simulator.date_time_steps, 
        entry, 
        title=f"Actuator {i} ({actuator_sensors[i].id}): Initial Predictions vs Actual", 
        ylabel_1axis="Valve Position (%)",
        ylabel_2axis="Temperature (°C)"
    )

plt.show()

# ==========================================================================
# SETUP ESTIMATOR WITH PARAMETERS
# ==========================================================================

print("\n" + "="*80)
print("SETTING UP PARAMETER ESTIMATION")
print("="*80)

# Get parameters from the controller
parameters = model.components['identified_controller'].get_estimator_parameters()

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
    print(f"    Actuator {a}:")
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
        # print(f"\n  [CONTROLLER PARAMS after theta applied (iter {_debug_iter[0]})]:")
        # # Per-actuator weights
        # for a in range(len(actuator_sensors)):
        #     alpha = controller._get_alpha_vector(a)
        #     beta = controller._get_beta_vector(a)
        #     gamma = controller._get_gamma_vector(a)
        #     print(f"    Actuator {a} ({actuator_sensors[a].id}):")
        #     print(f"      alpha: {alpha}")
        #     print(f"      beta:  {beta}")
        #     print(f"      gamma: {gamma}")
        
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
    for i, (name, val) in enumerate(zip(_param_names, theta)):
        print(f"    {name:25s} = {val:10.6f}")
    return result

def _debug_jac_ad(theta, output="scalar"):
    result = _orig_jac_ad(theta, output)
    grad_norm = np.linalg.norm(result)
    print(f"  |grad| = {grad_norm:.6f}")
    
    # Print gradients (only first 10 to avoid clutter)
    print(f"  grad (showing first 10 of {len(result)} params):")
    for i, (name, val) in enumerate(zip(_param_names, result)):
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
    # regularization_lambda=0.01,  # Binarization penalty: P(x) = x(1-x)
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
    # history(i_c=0) returns (n_timesteps, n_simulations), transpose to (n_simulations, n_timesteps)
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

print("\nBETA WEIGHTS (Sensor Selection):")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_sensors[a].id}):")
    beta_vals = controller._get_beta_vector(a)
    for s in range(len(sensors)):
        beta = beta_vals[s].item()
        sensor_name = sensors[s].id
        selected = " <-- SELECTED" if beta > 0.5 else ""
        print(f"    β_{a},{s} ({sensor_name}): {beta:.4f}{selected}")

print("\nGAMMA WEIGHTS (Setpoint Selection):")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_sensors[a].id}):")
    gamma_vals = controller._get_gamma_vector(a)
    for s in range(len(setpoints)):
        gamma = gamma_vals[s].item()
        sp_name = setpoints[s].id
        selected = " <-- SELECTED" if gamma > 0.5 else ""
        print(f"    γ_{a},{s} ({sp_name}): {gamma:.4f}{selected}")

print("\nIDENTIFIED PID PARAMETERS:")
print("(Note: PID covers full control range - high Kp with low Ti approximates on-off)")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_sensors[a].id}):")
    
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

# Plot final results with all three: actual, initial, and identified
for i in range(len(actuator_sensors)):
    # Compute average MAE across all simulations
    mae_initial = np.mean(np.abs(initial_predictions[i] - actual_values[i]))
    mae_final = np.mean(np.abs(final_predictions[i] - actual_values[i]))
    
    entry = [
        tb.plot.Entry(actual_values[i], label=f"Actual Valve Position"),
        tb.plot.Entry(initial_predictions[i], label=f"Initial Prediction"),
        tb.plot.Entry(final_predictions[i], label=f"Identified Prediction"),
        tb.plot.Entry(room_temp_data, label=f"Room Temperature", axis=2)
    ]
    
    tb.plot.plot(
        simulator.date_time_steps, 
        entry, 
        title=f"Actuator {i} ({actuator_sensors[i].id}): Initial MAE={mae_initial:.4f}, Final MAE={mae_final:.4f}", 
        ylabel_1axis="Valve Position (%)",
        ylabel_2axis="Temperature (°C)"
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
    improvement = (mae_initial - mae_final) / mae_initial * 100
    print(f"    Actuator {i}: {improvement:.1f}% improvement (MAE: {mae_initial:.4f} → {mae_final:.4f})")