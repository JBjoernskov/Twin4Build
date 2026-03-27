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

# Database configuration
db_config = {
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

# Time range for identification (use a week with good data)
start_time = datetime(2022, 1, 1)
end_time = datetime(2023, 1, 8)
step_size = 600  # 1 minute

# ==========================================================================
# STREAM IDs FROM ANALYSIS
# ==========================================================================

# Room temperature sensor (shared between room and FCU1)
ROOM_TEMP_STREAM = "352e0a5b_240d_4cf1_97eb_81e8fb49e1ee"

# FCU 1 (135e5147_00e4_43a4_89cd_b6937633e0dd)
FCU1 = {
    "id": "135e5147_00e4_43a4_89cd_b6937633e0dd",
    "label": "FCU1",
    "temp_setpoint": "ad7d7e3a_39b3_450d_8326_84b23a18f7b7",
    "valve_pos_1": "abe58af3_798a_4072_9944_f18bb5b93e5a",  # Actuator 0
    "valve_pos_2": "86ca8910_e873_4b3d_8c30_c0c448bd7165",  # Actuator 1
}

# FCU 2 (3458632f_4ce7_496d_a9cf_c7280ada9250)
FCU2 = {
    "id": "3458632f_4ce7_496d_a9cf_c7280ada9250",
    "label": "FCU2",
    "temp_setpoint": "9d21e517_0ae6_48fe_b131_74422db7999a",
    "valve_pos_1": "f5b1e556_105d_4057_9c88_490997c51481",  # Actuator 2
    "valve_pos_2": "a26a33b8_b2de_4eb2_a884_850cb7b65413",  # Actuator 3
}

# ==========================================================================
# DATA LOADING
# ==========================================================================


def load_stream(stream_id, label):
    """Load a single stream from the database"""
    print(f"  Loading {label}: {stream_id[:20]}...")
    try:
        df = load_from_database(
            table_name="bts_site_a",
            sensor_id=stream_id,
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            resample=True,
            resample_method="linear",
            clip=True,
            cache=True,
            tz="UTC",
            **db_config,
        )
        if df is not None and len(df) > 0:
            print(f"    Loaded {len(df)} points")
            return df
        else:
            print(f"    No data returned")
            return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


print("=" * 80)
print("LOADING REAL BUILDING DATA")
print("=" * 80)

# Load room temperature (shared sensor)
print("\nROOM TEMPERATURE:")
room_temp_df = load_stream(ROOM_TEMP_STREAM, "Room Temperature")

# Load FCU1 data
print(f"\n{FCU1['label']} STREAMS:")
fcu1_setpoint_df = load_stream(FCU1["temp_setpoint"], "Setpoint")
fcu1_valve1_df = load_stream(FCU1["valve_pos_1"], "Valve 1")
fcu1_valve2_df = load_stream(FCU1["valve_pos_2"], "Valve 2")

# Load FCU2 data
print(f"\n{FCU2['label']} STREAMS:")
fcu2_setpoint_df = load_stream(FCU2["temp_setpoint"], "Setpoint")
fcu2_valve1_df = load_stream(FCU2["valve_pos_1"], "Valve 1")
fcu2_valve2_df = load_stream(FCU2["valve_pos_2"], "Valve 2")

# Verify we have data
if room_temp_df is None or fcu1_valve1_df is None or fcu1_valve2_df is None:
    raise ValueError("Missing critical data streams!")

# ==========================================================================
# PREPARE DATA FOR IDENTIFICATION
# ==========================================================================

print("\n" + "=" * 80)
print("PREPARING DATA FOR IDENTIFICATION")
print("=" * 80)


# Extract values and ensure consistent length
def prepare_sensor_data(df, label):
    """Convert DataFrame to numpy array with datetime index"""
    if df is None:
        return None, None

    if isinstance(df, pd.Series):
        return df.index, df.values

    if len(df.columns) == 0:
        return df.index, df.values

    try:
        return df.index, df.iloc[:, 0].values
    except:
        return df.index, df.values


# Get all data
room_temp_idx, room_temp_vals = prepare_sensor_data(room_temp_df, "room_temp")
fcu1_sp_idx, fcu1_sp_vals = prepare_sensor_data(fcu1_setpoint_df, "fcu1_setpoint")
fcu1_v1_idx, fcu1_v1_vals = prepare_sensor_data(fcu1_valve1_df, "fcu1_valve1")
fcu1_v2_idx, fcu1_v2_vals = prepare_sensor_data(fcu1_valve2_df, "fcu1_valve2")
fcu2_sp_idx, fcu2_sp_vals = prepare_sensor_data(fcu2_setpoint_df, "fcu2_setpoint")
fcu2_v1_idx, fcu2_v1_vals = prepare_sensor_data(fcu2_valve1_df, "fcu2_valve1")
fcu2_v2_idx, fcu2_v2_vals = prepare_sensor_data(fcu2_valve2_df, "fcu2_valve2")

# Use the room temperature index as reference
timestamps = room_temp_idx
n_steps = len(timestamps)

print(f"  Time range: {timestamps[0]} to {timestamps[-1]}")
print(f"  Number of timesteps: {n_steps}")
print(
    f"  Room temp range: [{np.min(room_temp_vals):.2f}, {np.max(room_temp_vals):.2f}]°C"
)

# ==========================================================================
# BUILD IDENTIFICATION MODEL
# ==========================================================================

print("\n" + "=" * 80)
print("BUILDING CONTROLLER IDENTIFICATION MODEL")
print("=" * 80)

# Create sensors using tb.SensorSystem with real stream IDs
print("\nCreating sensor systems...")

# Primary sensor: Room temperature
room_temp_sensor = tb.SensorSystem(uuid=ROOM_TEMP_STREAM, id="room_temp_sensor")

# Setpoint sensors
fcu1_setpoint_sensor = tb.SensorSystem(uuid=FCU1["temp_setpoint"], id="fcu1_setpoint")

if fcu2_sp_vals is not None:
    fcu2_setpoint_sensor = tb.SensorSystem(
        uuid=FCU2["temp_setpoint"], id="fcu2_setpoint"
    )
else:
    fcu2_setpoint_sensor = None

# Actuators: 4 valves total (2 per FCU)
actuator_sensors = []

actuator_0 = tb.SensorSystem(uuid=FCU1["valve_pos_1"], id="fcu1_valve1_actuator")
actuator_1 = tb.SensorSystem(uuid=FCU1["valve_pos_2"], id="fcu1_valve2_actuator")
actuator_2 = tb.SensorSystem(uuid=FCU2["valve_pos_1"], id="fcu2_valve1_actuator")
actuator_3 = tb.SensorSystem(uuid=FCU2["valve_pos_2"], id="fcu2_valve2_actuator")

actuator_sensors = [actuator_0, actuator_1, actuator_2, actuator_3]

print(f"  Created {len(actuator_sensors)} actuator sensors")

# ==========================================================================
# SETUP CONTROLLER CANDIDATES
# ==========================================================================

print("\nSetting up controller candidates...")

# Candidate controller types to test
# PID can cover the full range including:
# - Pure P control (Ti → ∞, Td = 0)
# - PI control (Ti finite, Td = 0)
# - PD control (Ti → ∞, Td finite)
# - PID control (all parameters active)
# - On-Off behavior (high Kp, low Ti approximates bang-bang)
controller_classes = [
    PIDControllerSystem,  # Universal controller
]

print(f"  Using PID controller (covers full control range including on-off)")

# ==========================================================================
# CREATE COMBINED IDENTIFICATION CONTROLLER
# ==========================================================================

print("\nCreating controller identification system...")

# Prepare sensor and setpoint lists
sensors = [room_temp_sensor]  # List of sensor systems
setpoints = [fcu1_setpoint_sensor]  # List of setpoint systems
if fcu2_setpoint_sensor is not None:
    setpoints.append(fcu2_setpoint_sensor)


# Create controller with 4 actuators (2 valves per FCU)
controller = tb.ControllerIdentificationTorchSystem(
    n_sensors=len(sensors),
    n_setpoints=len(setpoints),
    n_actuators=len(actuator_sensors),  # 4 valves total
    id="identified_controller",
)

print(f"  Created controller with {len(actuator_sensors)} actuators")
print(f"  Will learn from {len(sensors)} sensor(s) and {len(setpoints)} setpoint(s)")
print(f"  Using {len(controller_classes)} controller candidate type(s)")

# ==========================================================================
# BUILD COMPLETE MODEL
# ==========================================================================

print("\nBuilding complete identification model...")

model = tb.Model(id="fcu_identification_model")

# Add sensors
model.add_component(room_temp_sensor)
model.add_component(fcu1_setpoint_sensor)
if fcu2_setpoint_sensor is not None:
    model.add_component(fcu2_setpoint_sensor)

# Add actuator sensors
for actuator in actuator_sensors:
    model.add_component(actuator)

# Add controller
model.add_component(controller)

# Connect sensors to controller (feedback)
for i, sensor in enumerate(sensors):
    model.add_connection(
        sensor, controller, "measuredValue", f"sensorValue", input_port_index=i
    )

# Connect setpoints to controller
for i, setpoint in enumerate(setpoints):
    model.add_connection(
        setpoint, controller, "measuredValue", f"setpointValue", input_port_index=i
    )

# Connect controller outputs to actuators
for i, actuator in enumerate(actuator_sensors):
    model.serialize()
    model.visualize()
    model.add_connection(
        controller, actuator, f"inputSignal", "measuredValue", output_port_index=i
    )

print("  Model connections established")

# ==========================================================================
# LOAD REAL DATA INTO MODEL
# ==========================================================================

print("\nLoading real data into model components...")

# Load sensors
for sensor in sensors:
    sensor.load_data(startTime=start_time, endTime=end_time, stepSize=step_size)

# Load setpoints
for setpoint in setpoints:
    setpoint.load_data(startTime=start_time, endTime=end_time, stepSize=step_size)

# Load actual actuator values (valve positions)
for actuator in actuator_sensors:
    actuator.load_data(startTime=start_time, endTime=end_time, stepSize=step_size)

print("  Data loaded successfully")

# ==========================================================================
# RUN INITIAL SIMULATION
# ==========================================================================

print("\n" + "=" * 80)
print("RUNNING INITIAL SIMULATION (BEFORE IDENTIFICATION)")
print("=" * 80)

model.reset()
model.simulate(stepSize=step_size)

# Get initial predictions
initial_predictions = []
for i, actuator in enumerate(actuator_sensors):
    pred = actuator.input["inputSignal"].history(i_s=0, i_c=0).detach().numpy()
    initial_predictions.append(pred)
    print(
        f"  Actuator {i}: Initial MAE = {np.mean(np.abs(pred - actuator.input['measuredValue'].history(i_s=0, i_c=0).detach().numpy())):.4f}"
    )

# ==========================================================================
# SETUP OPTIMIZATION
# ==========================================================================

print("\n" + "=" * 80)
print("SETTING UP OPTIMIZATION")
print("=" * 80)

# Collect parameters to optimize
trainable_params = []

# Add controller parameters
for param in controller.parameters():
    if param.requires_grad:
        trainable_params.append(param)

print(f"  Total trainable parameters: {len(trainable_params)}")
print(f"  Total parameter count: {sum(p.numel() for p in trainable_params)}")

# Setup optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

print(f"  Using Adam optimizer with lr={learning_rate}")

# ==========================================================================
# OPTIMIZATION LOOP
# ==========================================================================

print("\n" + "=" * 80)
print("RUNNING IDENTIFICATION (OPTIMIZATION)")
print("=" * 80)

n_epochs = 200
loss_history = []

print(f"  Training for {n_epochs} epochs...")
print()

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Reset and simulate
    model.reset()
    result = model.simulate(stepSize=step_size)

    # Compute loss: MSE between predicted and actual valve positions
    total_loss = 0.0
    for i, actuator in enumerate(actuator_sensors):
        pred = actuator.input["inputSignal"].history(i_s=0, i_c=0)
        actual = actuator.input["measuredValue"].history(i_s=0, i_c=0)

        loss = torch.mean((pred - actual) ** 2)
        total_loss += loss

    # Backpropagate
    total_loss.backward()
    optimizer.step()

    loss_history.append(total_loss.item())

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}: Loss = {total_loss.item():.6f}")

print("\n  Optimization complete!")

# ==========================================================================
# FINAL SIMULATION
# ==========================================================================

print("\n" + "=" * 80)
print("RUNNING FINAL SIMULATION (AFTER IDENTIFICATION)")
print("=" * 80)

model.reset()
result = model.simulate(stepSize=step_size)

# Get final predictions and actuals
final_predictions = []
actual_values = []

for i, actuator in enumerate(actuator_sensors):
    pred = actuator.input["inputSignal"].history(i_s=0, i_c=0).detach().numpy()
    actual = actuator.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()

    final_predictions.append(pred)
    actual_values.append(actual)

    mae = np.mean(np.abs(pred - actual))
    print(f"  Actuator {i} ({actuator.id}): Final MAE = {mae:.4f}")

# ==========================================================================
# ANALYZE IDENTIFIED CONTROLLER
# ==========================================================================

print("\n" + "=" * 80)
print("IDENTIFIED CONTROLLER STRUCTURE")
print("=" * 80)

print("\nBETA WEIGHTS (Sensor Selection):")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_sensors[a].id}):")
    for s in range(len(sensors)):
        beta = controller.beta_weights[a, s].item()
        sensor_name = sensors[s].id
        selected = " <-- SELECTED" if beta > 0.5 else ""
        print(f"    β_{a},{s} ({sensor_name}): {beta:.4f}{selected}")

print("\nGAMMA WEIGHTS (Setpoint Selection):")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_sensors[a].id}):")
    for s in range(len(setpoints)):
        gamma = controller.gamma_weights[a, s].item()
        sp_name = setpoints[s].id
        selected = " <-- SELECTED" if gamma > 0.5 else ""
        print(f"    γ_{a},{s} ({sp_name}): {gamma:.4f}{selected}")

print("\nIDENTIFIED PID PARAMETERS:")
print("(Note: PID covers full control range - high Kp with low Ti approximates on-off)")
for a in range(len(actuator_sensors)):
    print(f"\n  Actuator {a} ({actuator_sensors[a].id}):")

    # Access the PID controller for this actuator
    ctrl = controller.controllers[a][0]  # First (and only) candidate controller
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

print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

# Convert timestamps to hours for plotting
time_hours = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]

# Create figure with subplots for each valve
fig, axes = plt.subplots(
    len(actuator_sensors) + 1,
    1,
    figsize=(16, 4 * (len(actuator_sensors) + 1)),
    sharex=True,
)

# Plot loss history
ax_loss = axes[0]
ax_loss.plot(loss_history, "b-", linewidth=1.5)
ax_loss.set_ylabel("Loss (MSE)")
ax_loss.set_title("Optimization Loss History")
ax_loss.grid(True, alpha=0.3)
ax_loss.set_yscale("log")

# Plot each actuator
for i in range(len(actuator_sensors)):
    ax = axes[i + 1]

    ax.plot(
        time_hours, actual_values[i], "g-", alpha=0.7, linewidth=1.5, label="Actual"
    )
    ax.plot(
        time_hours,
        initial_predictions[i],
        "b--",
        alpha=0.5,
        linewidth=1,
        label="Initial",
    )
    ax.plot(time_hours, final_predictions[i], "m-", linewidth=1.5, label="Identified")

    ax.set_ylabel(f"Actuator {i}\n({actuator_sensors[i].id})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    mae = np.mean(np.abs(final_predictions[i] - actual_values[i]))
    ax.set_title(f"Actuator {i}: MAE = {mae:.4f}")

axes[-1].set_xlabel("Time (hours)")

plt.tight_layout()
plt.savefig(
    "/mnt/user-data/outputs/fcu_identification_results.png",
    dpi=150,
    bbox_inches="tight",
)
print("\nPlot saved to: /mnt/user-data/outputs/fcu_identification_results.png")

plt.show()

print("\n" + "=" * 80)
print("CONTROLLER IDENTIFICATION COMPLETE")
print("=" * 80)
