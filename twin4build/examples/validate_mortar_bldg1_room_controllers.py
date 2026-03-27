"""
Validation Script for Mortar bldg1 Room RM107A Controller Identification

Loads the latest estimation results from identify_mortar_bldg1_room_controllers.py
and runs a longer simulation on unseen data to test generalization.

Steps:
  1. Rebuild the same model (same components, connections, and parameter structure)
  2. Find and load the latest estimation result pickle
  3. Apply the identified parameters to the model
  4. Simulate over a longer / different time period
  5. Plot and compare predictions vs actual data
"""

# Standard library imports
import glob
import os
import pickle
from datetime import datetime, timezone

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Local application imports
import twin4build as tb
from twin4build.utils.data_loaders.load import load_from_database
from twin4build.utils.rgetattr import rgetattr

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

timezone_utc = timezone.utc
step_size = 600  # 10 minutes

# --- Validation time range ---
# Use a period DIFFERENT from what was used for identification
# Identification used: 2017-01-16 to 2017-01-19
# Validation: a longer window covering weeks before and after
start_time = [
    datetime(2017, 1, 9, 0, 0, tzinfo=timezone_utc),  # Week before training
    datetime(2017, 1, 19, 0, 0, tzinfo=timezone_utc),  # Continues after training
]
end_time = [
    datetime(2017, 1, 16, 0, 0, tzinfo=timezone_utc),  # Up to training start
    datetime(2017, 2, 26, 0, 0, tzinfo=timezone_utc),  # Week after training
]

# ==========================================================================
# SENSOR UUID MAPPINGS (same as identification script)
# ==========================================================================

ROOM_SENSORS = {
    "zone_temp": "a2b6510f-cf4f-4edd-a080-b8f4b35968d9",
    "zone_control_temp": "59b93fef-a0ab-4f2d-a036-01c62bfa8a4a",
    "zone_temp_setpoint": "2cb39f2b-27e0-4611-a663-2de371007ff7",
    "damper_position": "13954408-3b78-4483-8b18-dc0471207943",
    "reheat_valve": "be8ce19d-5e81-4f43-be16-8d95366d2d1a",
    "supply_air_flow": "037993e1-31fc-4212-aaf1-8465a9481bf8",
    "percent_air_flow": "778b01e9-8022-4134-a29c-1b9d0106328e",
    "supply_air_temp": "6ff31387-db42-48a8-a675-2876e9d95639",
}

# ==========================================================================
# REBUILD THE SAME MODEL
# ==========================================================================

print("=" * 80)
print("VALIDATION: Mortar bldg1 RM107A Controller Identification")
print("=" * 80)

print("\nRebuilding model (same structure as identification)...")

transformation_temp = lambda x: (x - 32) * 5 / 9
transformation_pct = lambda x: x / 100.0

zone_temp_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS["zone_temp"],
    id="zone_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,
)
zone_temp_setpoint_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS["zone_temp_setpoint"],
    id="zone_temp_setpoint_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,
)
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

actuator_sensors = [damper_actuator, reheat_valve_actuator]
actuator_names = ["Damper Position", "Reheat Valve"]

# Schedule switches (one per actuator channel)
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

sensors = [zone_temp_sensor]
setpoints = [zone_temp_setpoint_sensor]

controller = tb.ControllerIdentificationTorchSystem(
    n_sensors=len(sensors),
    n_setpoints=len(setpoints),
    n_actuators=len(actuator_sensors),
    id="identified_vav_controller",
)

# --- Build model (MUST use the same id for get_dir to find the pickle) ---
model = tb.Model(id="mortar_bldg1_rm107a_identification_model")

model.add_component(zone_temp_sensor)
model.add_component(zone_temp_setpoint_sensor)
for actuator in actuator_sensors:
    model.add_component(actuator)
model.add_component(controller)
model.add_component(schedule_switch_damper)
model.add_component(schedule_switch_reheat)

for i, sensor in enumerate(sensors):
    model.add_connection(
        sensor, controller, "measuredValue", "sensorValue", input_port_index=i
    )
for i, setpoint in enumerate(setpoints):
    model.add_connection(
        setpoint, controller, "measuredValue", "setpointValue", input_port_index=i
    )

# controller -> schedule_switch_damper -> damper_actuator
model.add_connection(
    controller,
    schedule_switch_damper,
    "inputSignal",
    "inputSignal",
    output_port_index=0,
)
model.add_connection(
    schedule_switch_damper, damper_actuator, "inputSignal", "measuredValue"
)

# controller -> schedule_switch_reheat -> reheat_valve_actuator
model.add_connection(
    controller,
    schedule_switch_reheat,
    "inputSignal",
    "inputSignal",
    output_port_index=1,
)
model.add_connection(
    schedule_switch_reheat, reheat_valve_actuator, "inputSignal", "measuredValue"
)

print("  Model rebuilt with same structure and connections")

# Load component data
print("\nLoading component data from database...")
model.load()

# ==========================================================================
# FIND AND LOAD LATEST ESTIMATION RESULT
# ==========================================================================

print("\n" + "=" * 80)
print("LOADING IDENTIFIED PARAMETERS")
print("=" * 80)

# Find the latest estimation result pickle
results_dir, _ = model.get_dir(folder_list=["model_parameters", "estimation_results"])
pickle_files = sorted(glob.glob(os.path.join(results_dir, "*.pickle")))

if not pickle_files:
    raise FileNotFoundError(
        f"No estimation results found in:\n  {results_dir}\n"
        "Run identify_mortar_bldg1_room_controllers.py first."
    )

latest_pickle = pickle_files[-1]  # sorted by timestamp -> last is latest
print(f"\n  Found {len(pickle_files)} estimation result(s)")
print(f"  Loading latest: {os.path.basename(latest_pickle)}")

# Use the built-in method to load the pickle AND apply all parameters to the model
model.load_estimation_result(filename=latest_pickle)

# Also load the result dict for metadata display
with open(latest_pickle, "rb") as f:
    est_result = pickle.load(f)

print(f"  Optimization success: {est_result['success']}")
print(f"  Final objective: {est_result['final_objective']:.6f}")
print(f"  Iterations: {est_result['iterations']}")
print("  All identified parameters applied via model.load_estimation_result()")

# ==========================================================================
# PRINT IDENTIFIED PARAMETERS
# ==========================================================================

print("\n" + "=" * 80)
print("IDENTIFIED PARAMETERS (loaded from estimation)")
print("=" * 80)

# PID parameters
for a in range(len(actuator_sensors)):
    ctrl = controller._get_candidate(a, 0)
    kp = ctrl.kp.get().item()
    ti = ctrl.Ti.get().item()
    td = ctrl.Td.get().item()
    out_min = ctrl.output_min.get().item()
    out_max = ctrl.output_max.get().item()
    print(f"\n  Actuator {a} ({actuator_names[a]}):")
    print(f"    Kp = {kp:.6f}")
    print(f"    Ti = {ti:.6f}")
    print(f"    Td = {td:.6f}")
    print(f"    output_min = {out_min:.6f}")
    print(f"    output_max = {out_max:.6f}")

# Schedule parameters
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for sched in [schedule_switch_damper, schedule_switch_reheat]:
    print(f"\n  {sched.id}:")
    header = "       " + "  ".join(f"{dn:>5s}" for dn in day_names)
    print(header)
    for h in range(24):
        row_vals = [sched._get_schedule_weight(h, d).get().item() for d in range(7)]
        row_str = "  ".join(f"{v:5.2f}" for v in row_vals)
        print(f"    {h:02d}:00  {row_str}")
    override_val = sched.override_value.get().item()
    print(f"    override_value: {override_val:.4f}")

# Selection weights
sensor_names = ["zone_temp_sensor"]
setpoint_names = ["zone_temp_setpoint_sensor"]
print("\n  Selection weights:")
for a in range(len(actuator_sensors)):
    alpha = controller._get_alpha_vector(a)
    beta = controller._get_beta_vector(a)
    gamma = controller._get_gamma_vector(a)
    print(f"    Actuator {a} ({actuator_names[a]}):")
    print(f"      Alpha: {[f'{v:.4f}' for v in alpha]}")
    print(f"      Beta:  {[f'{v:.4f}' for v in beta]}")
    print(f"      Gamma: {[f'{v:.4f}' for v in gamma]}")

# ==========================================================================
# RUN VALIDATION SIMULATION
# ==========================================================================

print("\n" + "=" * 80)
print("RUNNING VALIDATION SIMULATION")
print("=" * 80)

n_periods = len(start_time)
total_days = sum(
    (et - st).total_seconds() / 86400 for st, et in zip(start_time, end_time)
)
print(f"\n  {n_periods} simulation period(s), {total_days:.0f} days total:")
for j, (st, et) in enumerate(zip(start_time, end_time)):
    days = (et - st).total_seconds() / 86400
    print(
        f"    Period {j}: {st.strftime('%Y-%m-%d')} to {et.strftime('%Y-%m-%d')} ({days:.0f} days)"
    )

simulator = tb.Simulator(model)
simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

# ==========================================================================
# EXTRACT RESULTS
# ==========================================================================

predictions = []
actual_values = []

print("\nValidation predictions vs actual:")
for i, actuator in enumerate(actuator_sensors):
    pred = actuator.input["measuredValue"].history(i_c=0).detach().numpy().T
    actual = actuator.time_series_input.values[:, :, 0].detach().numpy().T

    predictions.append(pred)
    actual_values.append(actual)

    mae_per_sim = np.mean(np.abs(pred - actual), axis=1)
    mae_avg = np.mean(mae_per_sim)

    print(f"  Actuator {i} ({actuator.id}):")
    print(f"    Shape: {pred.shape}")
    for j, mae in enumerate(mae_per_sim):
        print(f"    Period {j} MAE: {mae:.4f}")
    print(f"    Average MAE: {mae_avg:.4f}")

# ==========================================================================
# PLOT VALIDATION RESULTS
# ==========================================================================

print("\n" + "=" * 80)
print("GENERATING VALIDATION PLOTS")
print("=" * 80)

zone_temp_data = zone_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
zone_setpoint_data = (
    zone_temp_setpoint_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)

for i in range(len(actuator_sensors)):
    mae = np.mean(np.abs(predictions[i] - actual_values[i]))

    entry = [
        tb.plot.Entry(actual_values[i], label=f"Actual {actuator_names[i]}"),
        tb.plot.Entry(predictions[i], label=f"Identified Model"),
        tb.plot.Entry(zone_temp_data, label="Zone Temperature", axis=2),
        tb.plot.Entry(
            zone_setpoint_data, label="Zone Setpoint", axis=2, linestyle="--"
        ),
    ]

    tb.plot.plot(
        simulator.date_time_steps,
        entry,
        title=f"VALIDATION - {actuator_names[i]}: MAE={mae:.4f}",
        ylabel_1axis="Position (0-1)",
        ylabel_2axis="Temperature (°C)",
    )

plt.show()

# ==========================================================================
# SUMMARY
# ==========================================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

# Training info from estimation result
train_start = est_result["start_time"]
train_end = est_result["end_time"]
print("\n  Training periods:")
for j, (ts, te) in enumerate(zip(train_start, train_end)):
    print(f"    {ts.strftime('%Y-%m-%d %H:%M')} to {te.strftime('%Y-%m-%d %H:%M')}")

print(f"\n  Validation periods:")
for j, (st, et) in enumerate(zip(start_time, end_time)):
    print(f"    {st.strftime('%Y-%m-%d %H:%M')} to {et.strftime('%Y-%m-%d %H:%M')}")

print(f"\n  Results per actuator:")
for i in range(len(actuator_sensors)):
    mae = np.mean(np.abs(predictions[i] - actual_values[i]))
    print(f"    {actuator_names[i]}: MAE = {mae:.4f}")

print("\n  (Compare these MAE values to training MAE to assess generalization)")
print("  If validation MAE is similar to training MAE → good generalization")
print("  If validation MAE is much higher → possible overfitting")
