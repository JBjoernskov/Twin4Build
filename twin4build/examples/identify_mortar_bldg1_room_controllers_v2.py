"""
Controller Identification for Mortar bldg1 Room RM107A (v2 - no schedule switch)

This script identifies the control logic for the VAV box serving room RM107A by:
- Using real building data from the Mortar bldg1 database
- Testing PID, Cascade PID, and SAT-Compensated controllers
- Learning which sensors and setpoints drive the damper and reheat valve
- Identifying controller parameters

Key difference from v1:
- No ScheduleSwitchController -- controller connects directly to actuators
- Good x0 initial guesses based on VAV control domain knowledge:
    * Damper: Cascade PID (A=temperature, B=flow) or SAT-Compensated
    * Reheat: PID reverse on temperature

Candidate Controllers:
  0: PID (reverse)       -- error-based, reverse acting
  1: PID (non-reverse)   -- error-based, direct acting
  2: Cascade PID         -- temp→flow→damper
  3: SAT-Compensated     -- linear mapping from AHU supply air temp

Sensor Pool:
  0: Zone temperature
  1: Percent air flow
  2: AHU supply air temperature (for SAT-compensated candidate)

Building 1 Room RM107A (served by AHU01):
- Zone temperature sensor
- Zone air control temperature sensor
- Zone temperature setpoint
- Damper position command
- Reheat valve command
- Supply air flow measurement
- Percent air flow
- Supply air temperature (VAV discharge)
- AHU supply air temperature (AHU01)

Data range: 2016-06-03 to 2017-12-18
"""

# Standard library imports
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Local application imports
import twin4build as tb
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.utils.data_loaders.load import load_from_database

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
timezone = ZoneInfo("America/Los_Angeles")

step_size = 600  # 10 minutes (matching the typical HVAC control interval)

# Use winter periods to capture heating behavior
start_time = [
    datetime(2017, 1, 16, 0, 0, tzinfo=timezone),
]
end_time = [
    datetime(2017, 1, 30, 0, 0, tzinfo=timezone),
]
# start_time = [
#     datetime(2017, 1, 16, 7, 0, tzinfo=timezone),
#     datetime(2017, 1, 17, 7, 0, tzinfo=timezone),
#     datetime(2017, 1, 18, 7, 0, tzinfo=timezone),
#     datetime(2017, 1, 28, 7, 0, tzinfo=timezone),
#     datetime(2017, 1, 29, 7, 0, tzinfo=timezone),
# ]
# end_time = [
#     datetime(2017, 1, 16, 12, 0, tzinfo=timezone),
#     datetime(2017, 1, 17, 12, 0, tzinfo=timezone),
#     datetime(2017, 1, 18, 12, 0, tzinfo=timezone),
#     datetime(2017, 1, 28, 12, 0, tzinfo=timezone),
#     datetime(2017, 1, 29, 12, 0, tzinfo=timezone),
# ]

# ==========================================================================
# SENSOR UUID MAPPINGS FROM MORTAR bldg1
# These UUIDs are from the ref:hasTimeseriesId in the TTL file
# Room RM107A sensors (served by AHU01)
# ==========================================================================

ROOM_SENSORS = {
    "zone_temp": "a2b6510f-cf4f-4edd-a080-b8f4b35968d9",  # Zone_Air_Temp
    "zone_control_temp": "59b93fef-a0ab-4f2d-a036-01c62bfa8a4a",  # Zone_Air_Control_Temp
    "zone_temp_setpoint": "2cb39f2b-27e0-4611-a663-2de371007ff7",  # Zone_Air_Temp_Setpoint
    "damper_position": "13954408-3b78-4483-8b18-dc0471207943",  # Zone_Air_Damper_Command
    "reheat_valve": "be8ce19d-5e81-4f43-be16-8d95366d2d1a",  # Zone_Reheat_Valve_Command
    "supply_air_flow": "037993e1-31fc-4212-aaf1-8465a9481bf8",  # Zone_Supply_Air_Flow
    "percent_air_flow": "778b01e9-8022-4134-a29c-1b9d0106328e",  # Zone_Percent_Air_Flow
    "supply_air_temp": "6ff31387-db42-48a8-a675-2876e9d95639",  # Zone_Supply_Air_Temp (VAV discharge)
}

# ==========================================================================
# BUILD IDENTIFICATION MODEL
# ==========================================================================

print("=" * 80)
print("BUILDING CONTROLLER IDENTIFICATION MODEL FOR MORTAR bldg1 RM107A (v2)")
print("  - No schedule switch controllers")
print("  - Domain-informed x0 initial guesses")
print("=" * 80)

# Create sensors using tb.SensorSystem with real stream UUIDs
print("\nCreating sensor systems...")

# Temperature transformation: Fahrenheit to Celsius
transformation_temp = lambda x: (x - 32) * 5 / 9

# Primary sensor: Zone temperature (sensor index 0)
zone_temp_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS["zone_temp"],
    id="zone_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,
)

# Zone air control temperature (alternate feedback sensor)
zone_control_temp_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS["zone_control_temp"],
    id="zone_control_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,
)

# Setpoint sensor
zone_temp_setpoint_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS["zone_temp_setpoint"],
    id="zone_temp_setpoint_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,
)

# Transform damper/valve positions from percentage (0-100) to fraction (0-1)
transformation_pct = lambda x: x / 100.0

# Actuators: Damper position and Reheat valve
damper_actuator = tb.SensorSystem(
    uuid=ROOM_SENSORS["damper_position"],
    id="damper_actuator",
    dbconfig=db_config,
    transformation=transformation_pct,
)

reheat_valve_actuator = tb.SensorSystem(
    uuid=ROOM_SENSORS["reheat_valve"],
    id="reheat_valve_actuator",
    dbconfig=db_config,
    transformation=transformation_pct,
)

# Percent air flow sensor (sensor index 1)
# Used by cascade controller's B-loop via beta_b weights
# Already measured as 0-100%, normalize to 0-1 fraction
supply_air_flow_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS["percent_air_flow"],
    id="supply_air_flow_sensor",
    dbconfig=db_config,
    transformation=transformation_pct,
)

# AHU supply air temperature sensor (sensor index 2)
# Used by SAT-compensated controller candidate; also useful for diagnostics
ahu_supply_air_temp_sensor = tb.SensorSystem(
    uuid="469e6e6f-c54b-4a58-a5e5-fae1442e04bd",  # AHU01 Supply_Air_Temp (bldg1)
    id="ahu_supply_air_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,
)

actuator_sensors = [damper_actuator, reheat_valve_actuator]
print(f"  Created {len(actuator_sensors)} actuator sensors")
print(f"    - Damper position (cooling control)")
print(f"    - Reheat valve (heating control)")

# ==========================================================================
# CREATE CONTROLLER IDENTIFICATION SYSTEM
# ==========================================================================

print("\nCreating controller identification system...")

# Sensor pool: index 0 = zone_temp, index 1 = percent_air_flow, index 2 = AHU SAT
# beta selects A-loop feedback, beta_b selects cascade B-loop feedback
# SAT-compensated candidate uses beta to select AHU SAT (sensor 2)
sensors = [zone_temp_sensor, supply_air_flow_sensor, ahu_supply_air_temp_sensor]

# Setpoints: index 0 = zone_temp_setpoint
setpoints = [zone_temp_setpoint_sensor]

# Create controller with 2 actuators (damper + reheat valve)
# Default candidates: PID (reverse), PID (non-reverse), Cascade PID, SAT-Compensated
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
# BUILD COMPLETE MODEL (no schedule switches!)
# ==========================================================================

print("\nBuilding complete identification model (direct controller→actuator)...")

model = tb.Model(id="mortar_bldg1_rm107a_identification_v2")

# Add sensors
for sensor in sensors:
    model.add_component(sensor)
model.add_component(zone_temp_setpoint_sensor)

# Add actuator sensors
for actuator in actuator_sensors:
    model.add_component(actuator)

# Add controller
model.add_component(controller)

# Connect sensors to controller (feedback)
for i, sensor in enumerate(sensors):
    model.add_connection(
        sensor, controller, "measuredValue", "sensorValue", input_port_index=i
    )

# Connect setpoints to controller
for i, setpoint in enumerate(setpoints):
    model.add_connection(
        setpoint, controller, "measuredValue", "setpointValue", input_port_index=i
    )

# Connect controller directly to actuators (no schedule switch!)
# output_port_index 0 = damper, 1 = reheat
model.add_connection(
    controller, damper_actuator, "inputSignal", "measuredValue", output_port_index=0
)
model.add_connection(
    controller,
    reheat_valve_actuator,
    "inputSignal",
    "measuredValue",
    output_port_index=1,
)

print("  Model connections established")
print("  Signal flow: controller --[damper]--> damper_actuator")
print("               controller --[reheat]--> reheat_valve_actuator")

# Load component data
print("\nLoading component data from database...")
model.load()

# Verify data was loaded for all sensors
print("\nVerifying loaded data:")
all_sensors = [
    zone_temp_sensor,
    supply_air_flow_sensor,
    ahu_supply_air_temp_sensor,
    zone_temp_setpoint_sensor,
    damper_actuator,
    reheat_valve_actuator,
]
for sensor in all_sensors:
    if hasattr(sensor, "df") and sensor.df is not None:
        print(
            f"  {sensor.id}: {len(sensor.df)} rows, range: {sensor.df.index.min()} to {sensor.df.index.max()}"
        )
    else:
        print(f"  {sensor.id}: No data loaded (df is None or missing)")

# ==========================================================================
# SETUP PARAMETERS WITH DOMAIN-INFORMED x0 START GUESSES
# ==========================================================================

print("\n" + "=" * 80)
print("SETTING UP PARAMETERS WITH DOMAIN-INFORMED x0")
print("=" * 80)

print(
    """
  Domain knowledge for VAV box RM107A:
  
  Candidates: 0=PID(rev), 1=PID(non-rev), 2=Cascade PID, 3=SAT-Compensated
  Sensors: 0=zone_temp, 1=percent_air_flow, 2=AHU_supply_air_temp
  Setpoints: 0=zone_temp_setpoint

  Damper (actuator 0) = Cascade PID or SAT-Compensated:
    Cascade: A-loop uses temperature, B-loop uses flow
    SAT-Comp: linear mapping from AHU SAT to damper position
    → alpha_0 = [0, 0, 0.5, 0.5]   (explore cascade vs SAT-compensated)
    → beta_0  = [1, 0, 0]           (A-loop uses temperature = sensor 0)
    → beta_b_0 = [0, 1, 0]          (B-loop uses flow = sensor 1)
    → gamma_0 = [1]                 (uses temp setpoint)
  
  Reheat valve (actuator 1) = PID reverse:
    When temp < setpoint → open valve (add heat)
    → alpha_1 = [1, 0, 0, 0]        (select PID reverse = candidate 0)
    → beta_1  = [1, 0, 0]           (uses temperature = sensor 0)
    → beta_b_1 = [0.33, 0.33, 0.34] (irrelevant, cascade not selected)
    → gamma_1 = [1]                 (uses temp setpoint)
"""
)

# Local application imports
from twin4build.utils.rgetattr import rgetattr

# Get parameters from the controller only (no schedule switches)
parameters = model.components["identified_vav_controller"].get_estimator_parameters()

# Override x0 with domain-informed values
parameters_updated = []
for p in parameters:
    comp, attr, x0, lb, ub = p[:5]
    rest = p[5:]

    # --- Selection weight overrides ---
    if attr == "alpha_0":
        # Damper: explore cascade vs SAT-compensated
        x0 = [0.0, 0.0, 0.0, 1.0]
    elif attr == "alpha_1":
        # Reheat: select PID reverse (candidate 0)
        x0 = [1.0, 0.0, 0.0, 0.0]
    elif attr == "beta_0":
        # Damper A-loop: temperature (sensor 0) for cascade; AHU SAT (sensor 2) for SAT-comp
        # Start at temp since cascade and SAT-comp both get beta-weighted signal
        x0 = [0, 0.0, 1]
    elif attr == "beta_1":
        # Reheat: temperature (sensor 0)
        x0 = [1.0, 0.0, 0.0]
    elif attr == "gamma_0":
        # Damper: uses setpoint
        x0 = [1.0]
    elif attr == "gamma_1":
        # Reheat: uses setpoint
        x0 = [1.0]
    elif attr == "beta_b_0":
        # Damper B-loop: flow (sensor 1)
        x0 = [0.0, 1.0, 0.0]
    elif attr == "beta_b_1":
        # Reheat: irrelevant (cascade not selected), leave uniform
        x0 = [0.33, 0.33, 0.34]

    # --- Cascade PID parameter overrides for damper (actuator 0, candidate 2) ---
    elif attr == "candidate_0_2.ctrl_a.kp":
        x0 = 0.1  # Outer loop: moderate proportional gain
    elif attr == "candidate_0_2.ctrl_a.Ti":
        x0 = 20.0  # Outer loop: slow integral (temperature is slow)
    elif attr == "candidate_0_2.ctrl_a.output_min":
        x0 = 0.2  # Minimum flow fraction (~20% of design)
    elif attr == "candidate_0_2.ctrl_a.output_max":
        x0 = 1.0  # Maximum flow fraction
    elif attr == "candidate_0_2.ctrl_b.kp":
        x0 = 0.5  # Inner loop: faster proportional gain
    elif attr == "candidate_0_2.ctrl_b.Ti":
        x0 = 5.0  # Inner loop: faster integral (flow responds quickly)
    elif attr == "candidate_0_2.ctrl_b.output_min":
        x0 = 0.0  # Damper can close fully

    # --- SAT-compensated cascade overrides for damper (actuator 0, candidate 3) ---
    # ctrl_a = SAT linear rule (SAT → flow setpoint)
    elif attr == "candidate_0_3.ctrl_a.base_position":
        x0 = 0.3  # 30% flow at design SAT
    elif attr == "candidate_0_3.ctrl_a.sat_design":
        x0 = 13.0  # ~55°F design supply air temp
    elif attr == "candidate_0_3.ctrl_a.gain":
        x0 = 0.05  # +5% flow setpoint per °C above design
    elif attr == "candidate_0_3.ctrl_a.output_min":
        x0 = 0.1  # Minimum flow setpoint
    elif attr == "candidate_0_3.ctrl_a.output_max":
        x0 = 1.0  # Maximum flow setpoint
    # ctrl_b = PID (flow error → damper position)
    elif attr == "candidate_0_3.ctrl_b.kp":
        x0 = 0.5  # Inner loop: fast response to flow error
    elif attr == "candidate_0_3.ctrl_b.Ti":
        x0 = 5.0  # Inner loop: moderate integral time
    elif attr == "candidate_0_3.ctrl_b.output_min":
        x0 = 0.0  # Damper can close fully

    # --- PID reverse parameter overrides for reheat (actuator 1, candidate 0) ---
    elif attr == "candidate_1_0.kp":
        x0 = 0.1  # Moderate proportional gain
    elif attr == "candidate_1_0.Ti":
        x0 = 10.0  # PI control with reasonable integral time
    elif attr == "candidate_1_0.output_min":
        x0 = 0.0  # Valve can close fully

    parameters_updated.append((comp, attr, x0, lb, ub) + tuple(rest))

parameters = parameters_updated

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


# Remove alpha, beta, gamma, beta_b parameters
parameters = [
    p
    for p in parameters
    if not p[1].startswith("alpha_")
    and not p[1].startswith("beta_")
    and not p[1].startswith("gamma_")
]

# ==========================================================================
# RUN INITIAL SIMULATION (with x0 start guesses applied)
# ==========================================================================

print("\n" + "=" * 80)
print("RUNNING INITIAL SIMULATION (with x0 start guesses)")
print("=" * 80)

# Create simulator
simulator = tb.Simulator(model)

simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

# Get initial predictions and actual values
initial_predictions = []
actual_values = []

actuator_names = ["Damper Position", "Reheat Valve"]

print("\nInitial predictions vs actual:")
for i, actuator in enumerate(actuator_sensors):
    pred = actuator.input["measuredValue"].history(i_c=0).detach().numpy().T
    actual = actuator.time_series_input.values[:, :, 0].detach().numpy().T

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

zone_temp_data = zone_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
zone_setpoint_data = (
    zone_temp_setpoint_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
supply_air_temp_data = (
    ahu_supply_air_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
for i in range(len(actuator_sensors)):
    entry = [
        tb.plot.Entry(actual_values[i], label=f"Actual {actuator_names[i]}"),
        tb.plot.Entry(initial_predictions[i], label=f"Prediction (x0 start guess)"),
        tb.plot.Entry(zone_temp_data, label=f"Zone Temperature", axis=2),
        tb.plot.Entry(
            zone_setpoint_data, label=f"Zone Setpoint", axis=2, linestyle="--"
        ),
        tb.plot.Entry(
            supply_air_temp_data, label=f"Supply Air Temp", axis=2, linestyle="-."
        ),
    ]
    tb.plot.plot(
        simulator.date_time_steps,
        entry,
        title=f"{actuator_names[i]} ({actuator_sensors[i].id}): x0 Start Guess vs Actual",
        ylabel_1axis="Position (0-1)",
        ylabel_2axis="Temperature (°C)",
    )

plt.show()

# ==========================================================================
# SETUP ESTIMATOR WITH PARAMETERS
# ==========================================================================

print("\n" + "=" * 80)
print("SETTING UP PARAMETER ESTIMATION")
print("=" * 80)

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
    print(
        f"    theta[{theta_idx}:{theta_idx+n_vals}] -> {attr} (x0={x0}, lb={lb}, ub={ub}, group={grp})"
    )
    theta_idx += n_vals

# Count parameter types
n_alpha = sum(1 for p in parameters if "alpha" in p[1])
n_beta = sum(1 for p in parameters if "beta" in p[1])
n_gamma = sum(1 for p in parameters if "gamma" in p[1])
n_ctrl = sum(1 for p in parameters if "candidate" in p[1])

print(
    f"\n  - Alpha (candidate selection): {n_alpha} ({len(actuator_sensors)} actuators × {len(controller.candidate_controller_classes)} candidates)"
)
print(
    f"\n  - Beta (sensor selection): {n_beta} ({len(actuator_sensors)} actuators × {len(sensors)} sensors)"
)
print(
    f"  - Gamma (setpoint selection): {n_gamma} ({len(actuator_sensors)} actuators × {len(setpoints)} setpoints)"
)
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
    if controller._has_cascade:
        beta_b_vals = controller._get_beta_b_vector(a)
        print(f"      Beta_b: {[f'{v:.3f}' for v in beta_b_vals]}")

# Print initial controller parameters
print("\n  Initial controller parameters BEFORE estimation:")
for a in range(len(actuator_sensors)):
    print(f"    Actuator {a} ({actuator_names[a]}):")
    for c in range(controller.n_candidates):
        ctrl = controller._get_candidate(a, c)
        print(f"      Candidate {c} ({ctrl.__class__.__name__}):")
        if hasattr(ctrl, "ctrl_a"):
            for sub_name in ("ctrl_a", "ctrl_b"):
                sub = getattr(ctrl, sub_name)
                print(f"        {sub_name} ({sub.__class__.__name__}):")
                if hasattr(sub, "kp"):
                    print(
                        f"          kp={sub.kp.get().item():.6f}, Ti={sub.Ti.get().item():.6f}, Td={sub.Td.get().item():.6f}"
                    )
                    print(
                        f"          output_min={sub.output_min.get().item():.4f}, output_max={sub.output_max.get().item():.4f}, isReverse={sub.isReverse}"
                    )
                if hasattr(sub, "base_position"):
                    print(
                        f"          base_position={sub.base_position.get().item():.6f}, sat_design={sub.sat_design.get().item():.6f}, gain={sub.gain.get().item():.6f}"
                    )
                    print(
                        f"          output_min={sub.output_min.get().item():.4f}, output_max={sub.output_max.get().item():.4f}"
                    )
        elif hasattr(ctrl, "base_position"):
            print(
                f"        base_position={ctrl.base_position.get().item():.6f}, sat_design={ctrl.sat_design.get().item():.6f}, gain={ctrl.gain.get().item():.6f}"
            )
            print(
                f"          output_min={ctrl.output_min.get().item():.4f}, output_max={ctrl.output_max.get().item():.4f}"
            )
        else:
            print(
                f"        kp={ctrl.kp.get().item():.6f}, Ti={ctrl.Ti.get().item():.6f}, Td={ctrl.Td.get().item():.6f}"
            )
            print(
                f"        output_min={ctrl.output_min.get().item():.4f}, output_max={ctrl.output_max.get().item():.4f}, isReverse={ctrl.isReverse}"
            )

# ==========================================================================
# RUN ESTIMATION WITH REGULARIZATION
# ==========================================================================

print("\n" + "=" * 80)
print("RUNNING PARAMETER ESTIMATION")
print("=" * 80)

estimator = tb.Estimator(simulator)

# Lambda scheduling (continuation method for binarization penalty)
lambda_schedule = [
    # --- Smooth exploration: no penalty ---
    (0.0, {"maxiter": 200, "disp": True}),  # Phase 1: pure fit, no penalty
    # --- Mild push toward binary ---
    # (0.001, {"maxiter": 100, "disp": True}),   # Phase 2: gentle push
    # # --- Stronger push ---
    # (0.01,  {"maxiter": 100, "disp": True}),   # Phase 3: stronger push
    # # --- Final: crisp binary ---
    # (0.1,   {"maxiter": 50,  "disp": True}),   # Phase 4: hard binary
    # (1,     {"maxiter": 50,  "disp": True}),   # Phase 5: crisp binary
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

# Standard library imports
# Progress wrapper: prints eval-by-eval improvements
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
    rmse = getattr(estimator, "_last_rmse", float("nan"))
    pen = getattr(estimator, "_last_penalty", 0.0)
    lam = getattr(estimator, "_regularization_lambda", 0.0)
    pen_str = f"  λ·pen={lam*pen:.4f}" if lam > 0 else ""
    if is_new_best:
        _best_obj[0] = result
        print(
            f"  [eval {_debug_iter[0]:5d}] obj={result:.4f}  RMSE={rmse:.4f}{pen_str}  (best)"
        )
        _last_print_time[0] = now
    elif now - _last_print_time[0] > 15.0:
        print(
            f"  [eval {_debug_iter[0]:5d}] obj={result:.4f}  RMSE={rmse:.4f}{pen_str}  (best={_best_obj[0]:.4f})"
        )
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

print("\n" + "=" * 80)
print("ALL ESTIMATED PARAMETER VALUES")
print("=" * 80)

# Standard library imports
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

print("\n" + "=" * 80)
print("RUNNING FINAL SIMULATION (AFTER IDENTIFICATION)")
print("=" * 80)

simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

# Get final predictions
final_predictions = []

print("\nFinal predictions vs actual:")
for i, actuator in enumerate(actuator_sensors):
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

print("\n" + "=" * 80)
print("IDENTIFIED CONTROLLER STRUCTURE")
print("=" * 80)

sensor_names = [
    "zone_temp (sensor 0)",
    "percent_air_flow (sensor 1)",
    "AHU_SAT (sensor 2)",
]
setpoint_names = ["zone_temp_setpoint (setpoint 0)"]

print("\nALPHA WEIGHTS (Candidate Selection):")
candidate_names = [
    "PID (reverse)",
    "PID (non-reverse)",
    "Cascade PID",
    "SAT-Compensated",
]
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_names[a]}):")
    alpha_vals = controller._get_alpha_vector(a)
    for c in range(controller.n_candidates):
        alpha = alpha_vals[c].item()
        selected = " <-- SELECTED" if alpha > 0.5 else ""
        print(f"    α_{a},{c} ({candidate_names[c]}): {alpha:.4f}{selected}")

print("\nBETA WEIGHTS (A-loop Sensor Selection):")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_names[a]}):")
    beta_vals = controller._get_beta_vector(a)
    for s in range(len(sensors)):
        beta = beta_vals[s].item()
        selected = " <-- SELECTED" if beta > 0.5 else ""
        print(f"    β_{a},{s} ({sensor_names[s]}): {beta:.4f}{selected}")

if controller._has_cascade:
    print("\nBETA_B WEIGHTS (Cascade B-loop Sensor Selection):")
    for a in range(len(actuator_sensors)):
        print(f"\n  Actuator {a} ({actuator_names[a]}):")
        beta_b_vals = controller._get_beta_b_vector(a)
        for s in range(len(sensors)):
            beta_b = beta_b_vals[s].item()
            selected = " <-- SELECTED" if beta_b > 0.5 else ""
            print(f"    β_b_{a},{s} ({sensor_names[s]}): {beta_b:.4f}{selected}")

print("\nGAMMA WEIGHTS (Setpoint Selection):")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_names[a]}):")
    gamma_vals = controller._get_gamma_vector(a)
    for s in range(len(setpoints)):
        gamma = gamma_vals[s].item()
        selected = " <-- SELECTED" if gamma > 0.5 else ""
        print(f"    γ_{a},{s} ({setpoint_names[s]}): {gamma:.4f}{selected}")

print("\nIDENTIFIED CONTROLLER PARAMETERS:")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_names[a]}):")
    alpha_vals = controller._get_alpha_vector(a)

    for c in range(controller.n_candidates):
        alpha = alpha_vals[c].item()
        ctrl = controller._get_candidate(a, c)
        selected_str = " *** SELECTED ***" if alpha > 0.5 else " (not selected)"
        print(f"\n    Candidate {c} ({candidate_names[c]}) α={alpha:.4f}{selected_str}")

        if hasattr(ctrl, "ctrl_a"):
            for sub_name in ("ctrl_a", "ctrl_b"):
                sub = getattr(ctrl, sub_name)
                print(f"      {sub_name} ({sub.__class__.__name__}):")
                if hasattr(sub, "kp"):
                    print(
                        f"        kp={sub.kp.get().item():.6f}, Ti={sub.Ti.get().item():.6f}, Td={sub.Td.get().item():.6f}"
                    )
                    print(
                        f"        output_min={sub.output_min.get().item():.4f}, output_max={sub.output_max.get().item():.4f}, isReverse={sub.isReverse}"
                    )
                if hasattr(sub, "base_position"):
                    print(
                        f"        base_position={sub.base_position.get().item():.6f}, sat_design={sub.sat_design.get().item():.6f}, gain={sub.gain.get().item():.6f}"
                    )
                    print(
                        f"        output_min={sub.output_min.get().item():.4f}, output_max={sub.output_max.get().item():.4f}"
                    )
        elif hasattr(ctrl, "base_position"):
            # SAT-compensated controller (standalone)
            print(
                f"      base_position={ctrl.base_position.get().item():.6f}, sat_design={ctrl.sat_design.get().item():.6f}, gain={ctrl.gain.get().item():.6f}"
            )
            print(
                f"      output_min={ctrl.output_min.get().item():.4f}, output_max={ctrl.output_max.get().item():.4f}"
            )
        else:
            print(
                f"      kp={ctrl.kp.get().item():.6f}, Ti={ctrl.Ti.get().item():.6f}, Td={ctrl.Td.get().item():.6f}"
            )
            print(
                f"      output_min={ctrl.output_min.get().item():.4f}, output_max={ctrl.output_max.get().item():.4f}, isReverse={ctrl.isReverse}"
            )

# ==========================================================================
# PLOT RESULTS
# ==========================================================================

print("\n" + "=" * 80)
print("GENERATING FINAL COMPARISON PLOTS")
print("=" * 80)

# Refresh data for plotting
zone_temp_data = zone_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
zone_setpoint_data = (
    zone_temp_setpoint_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)

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
        tb.plot.Entry(
            zone_setpoint_data, label=f"Zone Setpoint", axis=2, linestyle="--"
        ),
    ]

    tb.plot.plot(
        simulator.date_time_steps,
        entry,
        title=f"{actuator_names[i]}: Initial MAE={mae_initial:.4f}, Final MAE={mae_final:.4f}",
        ylabel_1axis="Position (0-1)",
        ylabel_2axis="Temperature (°C)",
    )

plt.show()

print("\n" + "=" * 80)
print("CONTROLLER IDENTIFICATION COMPLETE")
print("=" * 80)
print("\nSummary:")
print(f"  Optimization success: {result['success']}")
print(f"  Final objective: {result['final_objective']:.6f}")
print(f"\n  Average improvement per actuator:")
for i in range(len(actuator_sensors)):
    mae_initial = np.mean(np.abs(initial_predictions[i] - actual_values[i]))
    mae_final = np.mean(np.abs(final_predictions[i] - actual_values[i]))
    improvement = (
        (mae_initial - mae_final) / mae_initial * 100 if mae_initial > 0 else 0
    )
    print(
        f"    {actuator_names[i]}: {improvement:.1f}% improvement (MAE: {mae_initial:.4f} → {mae_final:.4f})"
    )

print("\n" + "=" * 80)
print("VAV CONTROL INTERPRETATION FOR bldg1 RM107A (v2)")
print("=" * 80)
print(
    """
Candidate Control Hypotheses (Room RM107A, AHU01):

  Damper Position -- Hypothesis A: SAT-Compensated (rule-based):
    - Damper position = f(AHU supply air temperature)
    - All rooms move in unison when AHU SAT changes
    - output = clamp(base + gain * (SAT - sat_design), min, max)

  Damper Position -- Hypothesis B: Cascade PID:
    A-loop (outer): Temperature error → intermediate flow setpoint
    B-loop (inner): Flow error → damper position

  Reheat Valve (PID reverse):
    - When zone temp < setpoint → open valve (add heat)
    - When zone temp > setpoint → close valve

The identified parameters reveal:
  - Which controller type best fits each actuator (alpha weights)
  - Which sensors drive each control loop (beta/gamma/beta_b weights)
  - The PID/SAT-compensated parameters that explain the observed behavior
"""
)
