"""
Controller Identification for Mortar bldg1 Room RM107A

This script identifies the control logic for the VAV box serving room RM107A by:
- Using real building data from the Mortar bldg1 database
- Testing PID controllers (which cover the full control range including on-off behavior)
- Learning which sensors and setpoints drive the damper and reheat valve
- Identifying PID parameters (Kp, Ti, Td)

Control Logic in VAV Systems:
- Zone temperature too high → increase damper (more cooling air)
- Zone temperature too low → increase reheat valve (more heating)
- Damper position often has minimum flow requirement
- Reheat valve typically uses PI control for zone temperature

Building 1 Room RM107A (served by AHU01):
- Zone temperature sensor
- Zone air control temperature sensor
- Zone temperature setpoint
- Damper position command
- Reheat valve command
- Supply air flow measurement
- Percent air flow
- Supply air temperature (VAV discharge)

Data range: 2016-06-03 to 2017-12-18
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

# Database configuration for Mortar bldg1
db_config = {
    "table_name": "mortar_bldg1",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

# Time range for identification
# The mortar bldg1 dataset has data from 2016-06-03 to 2017-12-18
# Choose representative periods with good data quality
timezone_utc = ZoneInfo("America/Los_Angeles")

step_size = 600  # 10 minutes (matching the typical HVAC control interval)

# Use winter + summer periods to capture both heating and cooling behavior
start_time = [
    datetime(2017, 1, 16, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 17, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 18, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 19, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 20, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 21, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 22, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 23, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 24, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 25, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 26, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 27, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 28, 0, 0, tzinfo=timezone_utc),   # Winter - heating
    datetime(2017, 1, 29, 0, 0, tzinfo=timezone_utc),   # Winter - heating
]
end_time = [
    datetime(2017, 1, 17, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 18, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 19, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 20, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 21, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 22, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 23, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 24, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 25, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 26, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 27, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 28, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 29, 0, 0, tzinfo=timezone_utc),   # 1 week winter
    datetime(2017, 1, 30, 0, 0, tzinfo=timezone_utc),   # 1 week winter
]

# ==========================================================================
# SENSOR UUID MAPPINGS FROM MORTAR bldg1
# These UUIDs are from the ref:hasTimeseriesId in the TTL file
# Room RM107A sensors (served by AHU01)
# ==========================================================================

ROOM_SENSORS = {
    'zone_temp': 'a2b6510f-cf4f-4edd-a080-b8f4b35968d9',              # Zone_Air_Temp
    'zone_control_temp': '59b93fef-a0ab-4f2d-a036-01c62bfa8a4a',      # Zone_Air_Control_Temp
    'zone_temp_setpoint': '2cb39f2b-27e0-4611-a663-2de371007ff7',     # Zone_Air_Temp_Setpoint
    'damper_position': '13954408-3b78-4483-8b18-dc0471207943',        # Zone_Air_Damper_Command
    'reheat_valve': 'be8ce19d-5e81-4f43-be16-8d95366d2d1a',           # Zone_Reheat_Valve_Command
    'supply_air_flow': '037993e1-31fc-4212-aaf1-8465a9481bf8',        # Zone_Supply_Air_Flow
    'percent_air_flow': '778b01e9-8022-4134-a29c-1b9d0106328e',       # Zone_Percent_Air_Flow
    'supply_air_temp': '6ff31387-db42-48a8-a675-2876e9d95639',        # Zone_Supply_Air_Temp (VAV discharge)
}

# ==========================================================================
# BUILD IDENTIFICATION MODEL
# ==========================================================================

print("="*80)
print("BUILDING CONTROLLER IDENTIFICATION MODEL FOR MORTAR bldg1 RM107A")
print("="*80)

# Create sensors using tb.SensorSystem with real stream UUIDs
print("\nCreating sensor systems...")

# Temperature transformation: Fahrenheit to Celsius
transformation_temp = lambda x: (x - 32) * 5/9

# Primary sensor: Zone temperature
zone_temp_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS['zone_temp'],
    id="zone_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp
)

# Zone air control temperature (alternate feedback sensor)
zone_control_temp_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS['zone_control_temp'],
    id="zone_control_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp
)

# Setpoint sensor
zone_temp_setpoint_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS['zone_temp_setpoint'],
    id="zone_temp_setpoint_sensor",
    dbconfig=db_config,
    transformation=transformation_temp
)

# Transform damper/valve positions from percentage (0-100) to fraction (0-1)
transformation_pct = lambda x: x / 100.0

# Actuators: Damper position and Reheat valve
damper_actuator = tb.SensorSystem(
    uuid=ROOM_SENSORS['damper_position'],
    id="damper_actuator",
    dbconfig=db_config,
    transformation=transformation_pct
)

reheat_valve_actuator = tb.SensorSystem(
    uuid=ROOM_SENSORS['reheat_valve'],
    id="reheat_valve_actuator",
    dbconfig=db_config,
    transformation=transformation_pct
)

# Percent air flow sensor (used by cascade controller's B-loop via beta_b weights)
# Already measured as 0-100%, just normalize to 0-1 fraction
supply_air_flow_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS['percent_air_flow'],
    id="supply_air_flow_sensor",
    dbconfig=db_config,
    transformation=transformation_pct
)

actuator_sensors = [damper_actuator, reheat_valve_actuator]
print(f"  Created {len(actuator_sensors)} actuator sensors")
print(f"    - Damper position (cooling control)")
print(f"    - Reheat valve (heating control)")

# ==========================================================================
# CREATE SCHEDULE SWITCH CONTROLLER
# ==========================================================================

print("\nCreating schedule switch controllers...")

# Per-actuator schedule gates -- each learns when its channel is active
# 24 per-hour weights + 7 per-day weights (all start at 0.5 = undecided)
schedule_switch_damper = tb.ScheduleSwitchControllerTorchSystem(
    hour_weights=[0.5] * 24,
    day_weights=[0.5] * 7,
    id="damper_schedule",
)
schedule_switch_reheat = tb.ScheduleSwitchControllerTorchSystem(
    hour_weights=[0.5] * 24,
    day_weights=[0.5] * 7,
    id="reheat_schedule",
)
print("  Created schedule switch: damper_schedule")
print("  Created schedule switch: reheat_schedule")

# ==========================================================================
# CREATE CONTROLLER IDENTIFICATION SYSTEM
# ==========================================================================

print("\nCreating controller identification system...")

# Sensors that could potentially be used for feedback
# Both temperature and flow are in the same pool -- beta selects for A-loop,
# beta_b selects for cascade B-loop
sensors = [zone_temp_sensor, supply_air_flow_sensor]

# Setpoints that could be used
setpoints = [zone_temp_setpoint_sensor]

# Create controller with 2 actuators (damper + reheat valve)
# Default candidates include PID (reverse), PID (non-reverse), and Cascade PID.
# Cascade B-loop selects from the same sensor pool via beta_b weights.
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

model = tb.Model(id="mortar_bldg1_rm107a_identification_model")

# Add sensors (all in same pool -- beta selects A-loop feedback, beta_b selects B-loop)
for sensor in sensors:
    model.add_component(sensor)
model.add_component(zone_temp_setpoint_sensor)

# Add actuator sensors
for actuator in actuator_sensors:
    model.add_component(actuator)

# Add controller
model.add_component(controller)

# Add schedule switches
model.add_component(schedule_switch_damper)
model.add_component(schedule_switch_reheat)

# Connect sensors to controller (feedback -- same pool for PID and cascade)
for i, sensor in enumerate(sensors):
    model.add_connection(sensor, controller, "measuredValue", "sensorValue", input_port_index=i)

# Connect setpoints to controller
for i, setpoint in enumerate(setpoints):
    model.add_connection(setpoint, controller, "measuredValue", "setpointValue", input_port_index=i)

# Connect: controller -> schedule_switch_damper -> damper actuator (output 0)
model.add_connection(controller, schedule_switch_damper, "inputSignal", "inputSignal", output_port_index=0)
model.add_connection(schedule_switch_damper, damper_actuator, "inputSignal", "measuredValue")

# Connect: controller -> schedule_switch_reheat -> reheat actuator (output 1)
model.add_connection(controller, schedule_switch_reheat, "inputSignal", "inputSignal", output_port_index=1)
model.add_connection(schedule_switch_reheat, reheat_valve_actuator, "inputSignal", "measuredValue")

print("  Model connections established")
print("  Signal flow: controller --[damper]--> damper_schedule --> damper_actuator")
print("               controller --[reheat]--> reheat_schedule --> reheat_valve_actuator")

# Load component data
print("\nLoading component data from database...")
model.load()

# Verify data was loaded for all sensors
print("\nVerifying loaded data:")
all_sensors = [zone_temp_sensor, supply_air_flow_sensor, zone_temp_setpoint_sensor, 
               damper_actuator, reheat_valve_actuator]
for sensor in all_sensors:
    if hasattr(sensor, 'df') and sensor.df is not None:
        print(f"  {sensor.id}: {len(sensor.df)} rows, range: {sensor.df.index.min()} to {sensor.df.index.max()}")
    else:
        print(f"  {sensor.id}: No data loaded (df is None or missing)")

# ==========================================================================
# SETUP PARAMETERS WITH x0 START GUESSES
# ==========================================================================

print("\n" + "="*80)
print("SETTING UP PARAMETERS (x0 start guesses)")
print("="*80)

from twin4build.utils.rgetattr import rgetattr

# Get parameters from the controller AND both schedule switches
parameters = model.components['identified_vav_controller'].get_estimator_parameters()
parameters += schedule_switch_damper.get_estimator_parameters()
parameters += schedule_switch_reheat.get_estimator_parameters()

# parameters_ = []
# for comp, attr, x0, lb, ub in parameters:
#     if attr == "alpha_1":
#         x0 = [0, 1]  # EXACT: On-Off selected for actuator 1
#     elif "kp" in attr:
#         x0 = 0.001  # EXACT: 0.001
#     elif "Ti" in attr:
#         x0 = 8.0  # EXACT: 8.0
#     elif "Td" in attr:
#         x0 = 0.0  # EXACT: 0
#     parameters_.append((comp, attr, x0, lb, ub))

# parameters = parameters_

print(f"\n  Total parameters to estimate: {len(parameters)}")

# Apply x0 values to the model so initial simulation uses the start guesses
print("\n  Applying x0 start guesses to model parameters...")
for p in parameters:
    comp, attr, x0, lb, ub = p[:5]
    param = rgetattr(comp, attr)
    if isinstance(x0, (list, np.ndarray)):
        param.set(torch.tensor(x0, dtype=torch.float64), normalized=False)
    else:
        param.set(torch.tensor(x0, dtype=torch.float64), normalized=False)
    print(f"    {attr} = {x0}")

# ==========================================================================
# RUN INITIAL SIMULATION (with x0 start guesses applied)
# ==========================================================================

print("\n" + "="*80)
print("RUNNING INITIAL SIMULATION (with x0 start guesses)")
print("="*80)

# Create simulator
simulator = tb.Simulator(model)

simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

# Get initial predictions and actual values
initial_predictions = []
actual_values = []

actuator_names = ["Damper Position", "Reheat Valve"]

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

# Plot initial results
print("\nPlotting initial predictions (x0 start guess)...")

zone_temp_data = zone_temp_sensor.time_series_input.values[:,:,0].detach().numpy().T
zone_setpoint_data = zone_temp_setpoint_sensor.time_series_input.values[:,:,0].detach().numpy().T

for i in range(len(actuator_sensors)):
    entry = [
        tb.plot.Entry(actual_values[i], label=f"Actual {actuator_names[i]}"), 
        tb.plot.Entry(initial_predictions[i], label=f"Prediction (x0 start guess)"),
        tb.plot.Entry(zone_temp_data, label=f"Zone Temperature", axis=2),
        tb.plot.Entry(zone_setpoint_data, label=f"Zone Setpoint", axis=2, linestyle="--")
    ]
    tb.plot.plot(
        simulator.date_time_steps, 
        entry, 
        title=f"{actuator_names[i]} ({actuator_sensors[i].id}): x0 Start Guess vs Actual", 
        ylabel_1axis="Position (0-1)",
        ylabel_2axis="Temperature (°C)"
    )

plt.show()

# ==========================================================================
# SETUP ESTIMATOR WITH PARAMETERS
# ==========================================================================

print("\n" + "="*80)
print("SETTING UP PARAMETER ESTIMATION")
print("="*80)

print(f"\n  Total parameters to estimate: {len(parameters)}")

# Debug: Print parameter mapping
print("\n  [DEBUG] Parameter mapping (theta index -> parameter):")
theta_idx = 0
for p in parameters:
    comp, attr, x0, lb, ub = p[:5]
    if isinstance(x0, (list, np.ndarray)):
        n_vals = len(x0)
    else:
        n_vals = 1
    grp = p[6] if len(p) > 6 else None
    print(f"    theta[{theta_idx}:{theta_idx+n_vals}] -> {attr} (x0={x0}, lb={lb}, ub={ub}, group={grp})")
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

# ── Lambda scheduling (continuation method for binarization penalty) ──
#
# Each entry: (lambda, options)
#
# lambda (λ): binarization penalty weight.  Applied to both schedule weights
#   (from ScheduleSwitchController) and selection weights (alpha, beta, gamma
#   from ControllerIdentification).  The penalty P(x) = x(1-x) pushes weights
#   toward 0 or 1 for crisp binary decisions.
#
lambda_schedule = [
    # --- Smooth exploration: no penalty ---
    (0.0,   {"maxiter": 200, "disp": True}),   # Phase 1: pure fit, no penalty
    # --- Mild push toward binary ---
    (0.001, {"maxiter": 100, "disp": True}),   # Phase 2: gentle push
    # --- Stronger push ---
    (0.01,  {"maxiter": 100, "disp": True}),   # Phase 3: stronger push
    # --- Final: crisp binary ---
    (0.1,   {"maxiter": 50,  "disp": True}),   # Phase 4: hard binary
    (1,   {"maxiter": 50,  "disp": True}),   # Phase 4: hard binary
]

options = {
    "disp": True,
}

print("\n  Using SLSQP with lambda scheduling (binarization penalty)")
print(f"  Phases: {len(lambda_schedule)}")
for i, entry in enumerate(lambda_schedule):
    lam = entry[0]
    opts = entry[1] if len(entry) > 1 else {}
    mi = opts.get("maxiter", "default") if opts else "default"
    print(f"    Phase {i+1}: λ={lam}, maxiter={mi}")

# ── Progress wrapper: prints eval-by-eval improvements ──
import time as _time
_orig_obj_ad = estimator._obj_ad
_orig_jac_ad = estimator._jac_ad
_debug_iter = [0]
_best_obj = [float("inf")]
_last_print_time = [_time.time()]

def _debug_obj_ad(theta, output="scalar"):
    result = _orig_obj_ad(theta, output)
    _debug_iter[0] += 1
    now = _time.time()
    is_new_best = result < _best_obj[0]
    # Read diagnostics from estimator (set in _obj)
    rmse = getattr(estimator, '_last_rmse', float('nan'))
    pen = getattr(estimator, '_last_penalty', 0.0)
    lam = getattr(estimator, '_regularization_lambda', 0.0)
    pen_str = f"  λ·pen={lam*pen:.4f}" if lam > 0 else ""
    if is_new_best:
        _best_obj[0] = result
        print(f"  [eval {_debug_iter[0]:5d}] obj={result:.4f}  RMSE={rmse:.4f}{pen_str}  (best)")
        _last_print_time[0] = now
    elif now - _last_print_time[0] > 15.0:
        print(f"  [eval {_debug_iter[0]:5d}] obj={result:.4f}  RMSE={rmse:.4f}{pen_str}  (best={_best_obj[0]:.4f})")
        _last_print_time[0] = now
    return result

def _debug_jac_ad(theta, output="scalar"):
    result = _orig_jac_ad(theta, output)
    grad_norm = np.linalg.norm(result)
    print(f"    [jac {_debug_iter[0]:5d}] |grad| = {grad_norm:.4f}")
    return result

estimator._obj_ad = _debug_obj_ad
estimator._jac_ad = _debug_jac_ad

# Run estimation with lambda scheduling
result = estimator.estimate(
    start_time=start_time,
    end_time=end_time,
    step_size=step_size,
    parameters=parameters,
    measurements=measurements,
    n_warmup=0,
    method=("scipy", "SLSQP", "ad"),
    options=options,
    lambda_schedule=lambda_schedule,
)

# Restore original methods
estimator._obj_ad = _orig_obj_ad
estimator._jac_ad = _orig_jac_ad

print("\n  Optimization complete!")
print(f"  Total evaluations: {_debug_iter[0]}")
print(f"  Final objective value: {result['final_objective']:.6f}")
print(f"  Success: {result['success']}")
print(f"  Message: {result['message']}")

# ==========================================================================
# PRINT ALL ESTIMATED PARAMETER VALUES
# ==========================================================================

print("\n" + "="*80)
print("ALL ESTIMATED PARAMETER VALUES")
print("="*80)

# Group parameters by component for clearer output
from collections import OrderedDict
_param_groups_display = OrderedDict()
for p in parameters:
    comp, attr, x0, lb, ub = p[:5]
    if comp.id not in _param_groups_display:
        _param_groups_display[comp.id] = []
    _param_groups_display[comp.id].append((comp, attr, x0, lb, ub))

print(f"\n  Total parameters: {len(parameters)}")
print(f"  Components: {len(_param_groups_display)}")

for comp_id, group in _param_groups_display.items():
    print(f"\n  --- {comp_id} ({len(group)} parameters) ---")
    for comp, attr, x0, lb, ub, *_ in group:
        param = rgetattr(comp, attr)
        val = param.get().detach()

        # Format current value
        if val.numel() == 1:
            val_str = f"{val.item():.6f}"
        else:
            val_str = "[" + ", ".join(f"{v:.4f}" for v in val.flatten().tolist()) + "]"

        # Format x0
        if isinstance(x0, (list, np.ndarray)):
            x0_str = "[" + ", ".join(f"{v:.4f}" for v in np.atleast_1d(x0)) + "]"
        else:
            x0_str = f"{x0:.6f}"

        print(f"    {attr:30s}  x0={x0_str:>20s}  ->  {val_str}")

print()

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
    output_min = ctrl.output_min.get().item()
    output_max = ctrl.output_max.get().item()
    
    print(f"    Kp = {kp:.6f}")
    print(f"    Ti = {ti:.6f} seconds")
    print(f"    Td = {td:.6f} seconds")
    print(f"    output_min = {output_min:.6f}")
    print(f"    output_max = {output_max:.6f}")
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

print("\nIDENTIFIED SCHEDULE PARAMETERS (24h x 7d matrix):")
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for sched in [schedule_switch_damper, schedule_switch_reheat]:
    print(f"\n  {sched.id}:")
    # Print as a compact 24×7 matrix
    header = "       " + "  ".join(f"{dn:>5s}" for dn in day_names)
    print(header)
    for h in range(24):
        row_vals = []
        for d in range(7):
            val = sched._get_schedule_weight(h, d).get().item()
            row_vals.append(val)
        row_str = "  ".join(f"{v:5.2f}" for v in row_vals)
        print(f"    {h:02d}:00  {row_str}")

    override_val = sched.override_value.get().item()
    print(f"    override_value: {override_val:.4f}")

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
    improvement = (mae_initial - mae_final) / mae_initial * 100 if mae_initial > 0 else 0
    print(f"    {actuator_names[i]}: {improvement:.1f}% improvement (MAE: {mae_initial:.4f} → {mae_final:.4f})")

print("\n" + "="*80)
print("VAV CONTROL INTERPRETATION FOR bldg1 RM107A")
print("="*80)
print("""
Expected VAV Control Logic (Room RM107A, AHU01):
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

Building 1 notes:
  - Enhanced instrumentation includes VAV discharge temperature
  - Zone Air Control Temp may differ from Zone Air Temp
  - Supply air flow and percent flow available for cross-validation
""")
