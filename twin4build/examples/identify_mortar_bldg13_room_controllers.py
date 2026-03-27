"""
Controller Identification for Mortar bldg13 (no schedule switch)

This script identifies the control logic for a VAV box in bldg13 by:
- Using real building data from the Mortar bldg13 database
- Testing PID, Cascade PID, and SAT-Compensated cascade controllers
- Learning which sensors and setpoints drive the damper and reheat valve
- Identifying PID parameters (Kp, Ti, Td) and SAT-rule parameters

Toggle SELECTED_ROOM near the top of the script to switch between rooms.
All 12 rooms (RM01_1, RM01_2, RM11_1-RM11_3, RM12_1-RM12_2, RM13_1-RM13_2,
RM21, RM22, RM31) are pre-configured with their sensor UUIDs.

Key features:
- No ScheduleSwitchController -- controller connects directly to actuators
- 4 candidate controllers:
    0: PID (reverse)
    1: PID (non-reverse)
    2: Cascade PID (A=temperature PID, B=flow PID)
    3: SAT-Compensated cascade (A=SAT linear rule, B=flow PID)
- 3 sensors in pool:
    0: zone temperature
    1: percent air flow
    2: AHU supply air temperature
- Good x0 initial guesses based on VAV control domain knowledge:
    * Damper: explore Cascade PID vs SAT-Compensated cascade
    * Reheat: PID reverse on temperature

Control Logic in VAV Systems:
- Damper is a cascade controller:
    A-loop (outer): temperature error → flow setpoint (Cascade PID)
      OR: SAT → min airflow setpoint (SAT-compensated)
    B-loop (inner): flow error → damper position
- Reheat valve: PID on temperature error

Available sensors per room (all served by AHU01):
- Zone temperature sensor
- Zone temperature setpoint
- Damper position command
- Reheat valve command
- Supply air flow measurement (CFM)
- Percent air flow
- Supply air temperature (VAV discharge) -- not available for all rooms
- AHU supply air temperature (shared across all rooms)

Data range: TBD (adjust once ingestion is confirmed)
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

# Database configuration for Mortar bldg13
db_config = {
    "table_name": "mortar_bldg13",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

# Time range for identification
# Adjust once bldg13 data ingestion is confirmed and data range is known
timezone_utc = ZoneInfo("America/Los_Angeles")

step_size = 600  # 10 minutes (matching the typical HVAC control interval)

# Use winter periods to capture heating behavior
start_time = [
    datetime(2016, 9, 22, 8, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 2, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 3, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 4, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 5, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 6, 0, 0, tzinfo=timezone_utc),
    # datetime(2016, 1, 1, 0, 0, tzinfo=timezone_utc),
]
end_time = [
    datetime(2016, 9, 22, 16, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 3, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 4, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 5, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 6, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 7, 0, 0, tzinfo=timezone_utc),
    # datetime(2017, 1, 1, 0, 0, tzinfo=timezone_utc),
]

# ==========================================================================
# SENSOR UUID MAPPINGS FROM MORTAR bldg13
# These UUIDs are from the ref:hasTimeseriesId in the bldg13.ttl file
# All rooms served by AHU01. supply_air_temp is None for rooms without it.
# ==========================================================================

ROOMS = {
    "RM01_1": {  # Floor 0
        "zone_temp": "6c2edb86-2117-41be-91d8-af5a2423af57",
        "zone_temp_setpoint": "05d2947d-cfc8-4972-8cbf-29bef31576ae",
        "damper_position": "338d4128-1174-4bd3-a46c-7e13fdf4d41a",
        "reheat_valve": "7444a931-7138-4918-990b-99ffadb25644",
        "supply_air_flow": "6799fb2d-3476-42a4-a227-6bb5dc0e2b61",
        "percent_air_flow": "0a9b1084-b457-4ce8-a126-70fed9e5f659",
        "supply_air_temp": "0c1aada9-8fde-4dd8-a6db-881957ceab46",
    },
    "RM01_2": {  # Floor 0
        "zone_temp": "15293b97-bf40-4f7e-b4ab-fd024cf5396c",
        "zone_temp_setpoint": "1e4dfb4e-ce87-4080-8a54-9dc7ceffd066",
        "damper_position": "ebe9c5eb-f2cc-41f7-9366-7f866d97884e",
        "reheat_valve": "90936253-ed06-4a26-845c-0d5188899eac",
        "supply_air_flow": "6b3584d7-0064-446d-ae1b-49b5e65f3a5e",
        "percent_air_flow": "0db4126d-1a14-4379-b4a7-61960bdd8b18",
        "supply_air_temp": "52db6807-5a4e-4371-80f5-3ea0199c904c",
    },
    "RM11_1": {  # Floor 1
        "zone_temp": "23a74139-b96e-4c1e-9c2f-d59f002aba5a",
        "zone_temp_setpoint": "5a0cfc00-d4be-4656-b034-ff4921c2966e",
        "damper_position": "e98cba81-51b7-44b4-be8b-d724d8ba1d22",
        "reheat_valve": "53c486ee-b6e1-48c8-866e-8a062d91fa82",
        "supply_air_flow": "be533cf0-ebc2-4529-8cb0-938f11a634d3",
        "percent_air_flow": "1dfb5487-dd0e-4397-98e1-91963655949a",
        "supply_air_temp": "17f64b73-33ba-432b-a91f-d19c9f4ab928",
    },
    "RM11_2": {  # Floor 1 (no supply_air_temp)
        "zone_temp": "90da68ec-eafe-4730-9818-9baebd5c6dd2",
        "zone_temp_setpoint": "d67fb4db-579a-4208-b5ad-ae1e704f5b73",
        "damper_position": "51cf3b6c-aa59-44d2-aa60-9abfe2fd80c1",
        "reheat_valve": "06983740-f4a8-4e1e-921b-586c992d1cb6",
        "supply_air_flow": "4e1a9da8-3322-453a-9bfd-45badb4619bf",
        "percent_air_flow": "4e3bdf97-e584-4683-af09-8c73e2d803a8",
        "supply_air_temp": None,
    },
    "RM11_3": {  # Floor 1
        "zone_temp": "5353d78d-cef9-4ff1-aca6-a7576bba6680",
        "zone_temp_setpoint": "5dfdab3c-dc73-4cfe-8968-bc694d33bdc0",
        "damper_position": "d7c7c96b-7fed-4005-87dc-17dcbdd88d5c",
        "reheat_valve": "37c10674-e2ce-4344-ba3e-ed470dde4b58",
        "supply_air_flow": "97438420-f272-43a2-b110-fcb04f2dd772",
        "percent_air_flow": "619dfff8-667d-41f5-be0f-9165a7095ea4",
        "supply_air_temp": "1ef61828-32a3-475f-b495-58c2e27ea199",
    },
    "RM12_1": {  # Floor 1 (no supply_air_temp)
        "zone_temp": "e5d68e70-5caa-4228-9218-06fe00759ff4",
        "zone_temp_setpoint": "69cc42c2-34e8-40d2-9c0c-c378464d3864",
        "damper_position": "23876ee1-0d47-405e-8dbc-f3336c0c42b0",
        "reheat_valve": "bdba4982-0373-42a1-aca0-9eebdf7017a8",
        "supply_air_flow": "b976b8ae-c349-4abd-84d4-b23ce3fe7432",
        "percent_air_flow": "b73acb8c-345f-4394-a4ce-91269d09da8a",
        "supply_air_temp": None,
    },
    "RM12_2": {  # Floor 1
        "zone_temp": "75f83f2d-2ba3-4d89-9a58-8fb1971cdad1",
        "zone_temp_setpoint": "7b97a66d-c9a8-4cb1-8d89-2362b90df241",
        "damper_position": "6e662022-da07-44e1-85b1-185eff60660c",
        "reheat_valve": "6f613dc0-8cac-477b-b640-dd96ad6f6dc2",
        "supply_air_flow": "22470a08-79b1-42fa-a95b-fab7df3fcede",
        "percent_air_flow": "3fe9ac15-6d7f-49bc-ac1e-26bde054434e",
        "supply_air_temp": "7e94b002-0e82-4cdc-a0cd-610ead301c94",
    },
    "RM13_1": {  # Floor 1 (no supply_air_temp)
        "zone_temp": "1954cc8b-5715-4558-80dd-be97cfc57f07",
        "zone_temp_setpoint": "31b23b76-544e-4488-bce7-e139ee33e108",
        "damper_position": "f771ce69-4a68-40a2-85e3-d9593bbf0f20",
        "reheat_valve": "2230870e-839f-4584-ad49-1c5897dedd6e",
        "supply_air_flow": "b21e9e31-08d7-44b0-9eb1-62cddb42b6f8",
        "percent_air_flow": "5ac5465a-c50e-4998-ac11-908ed5cbb631",
        "supply_air_temp": None,
    },
    "RM13_2": {  # Floor 1
        "zone_temp": "24b2b322-c6f5-4ff2-9075-494b2b8812fb",
        "zone_temp_setpoint": "ac734ca0-5bbc-4991-863e-98697dcbbf23",
        "damper_position": "0be785d7-7d79-47d1-80bd-354dba9e7610",
        "reheat_valve": "18162af8-bd69-424e-8922-901c45af8c58",
        "supply_air_flow": "368f7afe-009f-4a25-978d-778452e5c6e2",
        "percent_air_flow": "d36f1e69-ebca-49a9-be43-c11f6060b384",
        "supply_air_temp": "13fa8768-1a1e-41de-80d3-189ffe1042c5",
    },
    "RM21": {  # Floor 2 (no supply_air_temp)
        "zone_temp": "7d722ee2-3894-43d8-8775-9533d817bb93",
        "zone_temp_setpoint": "ec6a4310-99d3-4728-a754-36c5c7e22f6f",
        "damper_position": "220e2d90-4d91-4399-84d9-96ccfd3f0a6b",
        "reheat_valve": "f5695126-50a4-4292-a910-84f18e26d510",
        "supply_air_flow": "11a63c28-b046-4d46-ae3f-43d636b6bdd1",
        "percent_air_flow": "e45e7ab7-72f1-448a-bc9e-ef5706b00788",
        "supply_air_temp": None,
    },
    "RM22": {  # Floor 2 (no supply_air_temp)
        "zone_temp": "223cccca-a251-47c4-86de-1fff7dd67655",
        "zone_temp_setpoint": "f02dc3de-2b0b-4c87-b701-677ae42b7760",
        "damper_position": "7098433c-2287-4bf6-b09c-aec3d6644f7a",
        "reheat_valve": "da8f56a8-85ad-4847-90e4-05b371afb3fa",
        "supply_air_flow": "670c02e8-8c1e-44a5-af98-87caffb7fba7",
        "percent_air_flow": "69f67351-9c55-4c77-b35b-7e19c5d30abb",
        "supply_air_temp": None,
    },
    "RM31": {  # Floor 3
        "zone_temp": "f6acb13f-5edb-432d-9f45-3705ef394148",
        "zone_temp_setpoint": "faaa645e-e341-4dcb-adb1-972034eae34f",
        "damper_position": "4a3e7cef-93d0-4888-9aed-9de72ce6c3b8",
        "reheat_valve": "3117c067-f51f-4fb7-a158-3ad4a62c838e",
        "supply_air_flow": "285bb781-c4fc-4a8b-9ca9-24cb91b761de",
        "percent_air_flow": "7cb06a8f-6f72-47ad-87dc-39f88ed83f51",
        "supply_air_temp": "02a8a3a8-e040-4e3d-ae73-5a14794ce64f",
    },
}

# ========== TOGGLE ROOM HERE ==========
SELECTED_ROOM = "RM13_1"  # RM11_1
# =======================================

ROOM_SENSORS = ROOMS[SELECTED_ROOM]
# Room-level supply_air_temp is unreliable; AHU SAT is used instead

# ==========================================================================
# BUILD IDENTIFICATION MODEL
# ==========================================================================

print("=" * 80)
print(f"BUILDING CONTROLLER IDENTIFICATION MODEL FOR MORTAR bldg13 {SELECTED_ROOM}")
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

# Extra sensors for plotting context (not used in controller identification)
# Supply air flow in CFM (raw measurement)
supply_air_flow_cfm_sensor = tb.SensorSystem(
    uuid=ROOM_SENSORS["supply_air_flow"],
    id="supply_air_flow_cfm_sensor",
    dbconfig=db_config,
)

# AHU-level sensors (shared across all rooms, not room-specific)
AHU_SENSORS = {
    "supply_air_temp": "77c50c34-1387-4ce4-a527-153fd143704e",  # AHU01 Supply_Air_Temp
    "supply_air_temp_setpoint": "cee1c63a-54c2-490e-a7a4-7200fa93b270",  # AHU01 Supply_Air_Temp_Setpoint
    "cooling_coil_valve": "40ba4519-46e9-4d8f-ad24-8a9ee9095c5d",  # AHU01 CCV (Cooling Coil Valve)
    "cooling_valve_output": "61704369-134c-4022-bb21-ef2368c91eb1",  # AHU01 Cooling Valve Output
    "heating_valve_output": "2a800b32-492e-47f3-943a-db0aa985c316",  # AHU01 Heating Valve Output
    "outside_air_temp": "436a91fd-e4fe-486f-9a63-2e00303d6188",  # AHU01 Outside Air Temp
    "return_air_temp": "53acd0c7-37e5-4cb7-93c0-a5e4283ad61f",  # AHU01 Return Air Temp
}

ahu_supply_air_temp_sensor = tb.SensorSystem(
    uuid=AHU_SENSORS["supply_air_temp"],
    id="ahu_supply_air_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,
)

ahu_supply_air_temp_sp_sensor = tb.SensorSystem(
    uuid=AHU_SENSORS["supply_air_temp_setpoint"],
    id="ahu_supply_air_temp_sp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,
)

ahu_ccv_sensor = tb.SensorSystem(
    uuid=AHU_SENSORS["cooling_coil_valve"],
    id="ahu_ccv_sensor",
    dbconfig=db_config,
    transformation=transformation_pct,  # percentage to fraction (0-1)
)

ahu_cooling_valve_sensor = tb.SensorSystem(
    uuid=AHU_SENSORS["cooling_valve_output"],
    id="ahu_cooling_valve_sensor",
    dbconfig=db_config,
    transformation=transformation_pct,  # percentage to fraction (0-1)
)

ahu_heating_valve_sensor = tb.SensorSystem(
    uuid=AHU_SENSORS["heating_valve_output"],
    id="ahu_heating_valve_sensor",
    dbconfig=db_config,
    transformation=transformation_pct,  # percentage to fraction (0-1)
)

ahu_outside_air_temp_sensor = tb.SensorSystem(
    uuid=AHU_SENSORS["outside_air_temp"],
    id="ahu_outside_air_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,  # Fahrenheit to Celsius
)

ahu_return_air_temp_sensor = tb.SensorSystem(
    uuid=AHU_SENSORS["return_air_temp"],
    id="ahu_return_air_temp_sensor",
    dbconfig=db_config,
    transformation=transformation_temp,  # Fahrenheit to Celsius
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
# SAT-compensated cascade candidate uses beta to select AHU SAT (sensor 2)
sensors = [zone_temp_sensor, supply_air_flow_sensor, ahu_supply_air_temp_sensor]

# Setpoints: index 0 = zone_temp_setpoint
setpoints = [zone_temp_setpoint_sensor]

# Create controller with 2 actuators (damper + reheat valve)
# Default candidates: PID (reverse), PID (non-reverse), Cascade PID, SAT-Compensated cascade
controller = tb.ControllerIdentificationTorchSystem(
    n_sensors=len(sensors),
    n_setpoints=len(setpoints),
    n_actuators=len(actuator_sensors),  # 2 actuators
    id=f"identified_vav_controller_bldg13_{SELECTED_ROOM}",
)

print(controller.summary())

print(f"  Created controller with {len(actuator_sensors)} actuators")
print(f"  Will learn from {len(sensors)} sensor(s) and {len(setpoints)} setpoint(s)")

# ==========================================================================
# BUILD COMPLETE MODEL (no schedule switches!)
# ==========================================================================

print("\nBuilding complete identification model (direct controller→actuator)...")

model = tb.Model(id=f"mortar_bldg13_{SELECTED_ROOM.lower()}_identification")

# Add sensors
for sensor in sensors:
    model.add_component(sensor)
model.add_component(zone_temp_setpoint_sensor)

# Add actuator sensors
for actuator in actuator_sensors:
    model.add_component(actuator)

# Add extra sensors for plotting context
model.add_component(supply_air_flow_cfm_sensor)
model.add_component(ahu_supply_air_temp_sensor)
model.add_component(ahu_supply_air_temp_sp_sensor)
model.add_component(ahu_ccv_sensor)
model.add_component(ahu_cooling_valve_sensor)
model.add_component(ahu_heating_valve_sensor)
model.add_component(ahu_outside_air_temp_sensor)
model.add_component(ahu_return_air_temp_sensor)

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
    zone_temp_setpoint_sensor,
    damper_actuator,
    reheat_valve_actuator,
    supply_air_flow_cfm_sensor,
    ahu_supply_air_temp_sensor,
    ahu_supply_air_temp_sp_sensor,
    ahu_ccv_sensor,
    ahu_cooling_valve_sensor,
    ahu_heating_valve_sensor,
    ahu_outside_air_temp_sensor,
    ahu_return_air_temp_sensor,
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
    f"""
  Domain knowledge for VAV box {SELECTED_ROOM}:
  
  Candidates: 0=PID(rev), 1=PID(non-rev), 2=Cascade PID, 3=SAT-Compensated cascade
  Sensors: 0=zone_temp, 1=percent_air_flow, 2=AHU_supply_air_temp
  Setpoints: 0=zone_temp_setpoint

  Damper (actuator 0) = Cascade PID (zone-controlled):
    Damper correlation analysis (operating hours only) shows dampers are
    NOT centrally driven by AHU SAT — they operate independently per zone.
    Cascade PID: A-loop uses temperature, B-loop uses flow
    → alpha_0 = [0, 0, 1, 0]       (select cascade PID = candidate 2)
    → beta_0  = [1, 0, 0]          (A-loop uses temperature = sensor 0)
    → beta_b_0 = [0, 1, 0]         (B-loop uses flow = sensor 1)
    → gamma_0 = [1]                (uses temp setpoint)
  
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
parameters = model.components[
    f"identified_vav_controller_bldg13_{SELECTED_ROOM}"
].get_estimator_parameters()

# Override x0 with domain-informed values
parameters_updated = []
for p in parameters:
    comp, attr, x0, lb, ub = p[:5]
    rest = p[5:]

    # --- Selection weight overrides ---
    if attr == "alpha_0":
        # Damper: favor cascade PID (dampers are zone-controlled, not SAT-driven)
        x0 = [0.0, 0.0, 1.0, 0.0]
    elif attr == "alpha_1":
        # Reheat: select PID reverse (candidate 0)
        x0 = [1.0, 0.0, 0.0, 0.0]
    elif attr == "beta_0":
        # Damper A-loop: temperature (sensor 0) for cascade PID
        x0 = [1.0, 0.0, 0.0]
    elif attr == "beta_1":
        # Reheat: temperature (sensor 0)
        x0 = [1.0, 0.0, 0.0]
    elif attr == "gamma_0":
        # Damper: uses temp setpoint
        x0 = [1.0]
    elif attr == "gamma_1":
        # Reheat: uses temp setpoint
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
        x0 = 50  # Outer loop: slow integral (temperature is slow)
    elif attr == "candidate_0_2.ctrl_a.output_min":
        x0 = 0.0  # Minimum flow fraction (~20% of design)
    elif attr == "candidate_0_2.ctrl_a.output_max":
        x0 = 1.0  # Maximum flow fraction
    elif attr == "candidate_0_2.ctrl_b.kp":
        x0 = 0.5  # Inner loop: faster proportional gain
    elif attr == "candidate_0_2.ctrl_b.Ti":
        x0 = 50  # Inner loop: faster integral (flow responds quickly)
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

simulator.set_simulation_timesteps(
    start_time=start_time, end_time=end_time, step_size=[step_size]
)
model.initialize(start_time=start_time, end_time=end_time, step_size=[step_size])
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
    # print(f"    Shape: {pred.shape}")
    for j, mae in enumerate(mae_per_sim):
        print(f"    Simulation {j} MAE: {mae:.4f}")
    print(f"    Average MAE: {mae_avg:.4f}")

# Plot initial results
print("\nPlotting initial predictions (x0 start guess)...")

zone_temp_data = zone_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
zone_setpoint_data = (
    zone_temp_setpoint_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
percent_air_flow_data = (
    supply_air_flow_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
supply_air_flow_cfm_data = (
    supply_air_flow_cfm_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
ahu_sat_plot_data = (
    ahu_supply_air_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)

temp_error_data = zone_setpoint_data - zone_temp_data

for i in range(len(actuator_sensors)):
    entry = [
        tb.plot.Entry(actual_values[i], label=f"Actual {actuator_names[i]}"),
        tb.plot.Entry(initial_predictions[i], label=f"Prediction (x0 start guess)"),
        tb.plot.Entry(
            percent_air_flow_data, label=f"Percent Air Flow (0-1)", linestyle="-."
        ),
        tb.plot.Entry(zone_temp_data, label=f"Zone Temperature", axis=2),
        tb.plot.Entry(
            zone_setpoint_data, label=f"Zone Setpoint", axis=2, linestyle="--"
        ),
        tb.plot.Entry(
            temp_error_data, label=f"Temp Error (SP - actual)", axis=2, linestyle=":"
        ),
        tb.plot.Entry(
            ahu_sat_plot_data, label=f"AHU Supply Air Temp", axis=2, linestyle="-."
        ),
    ]
    tb.plot.plot(
        simulator.date_time_steps,
        entry,
        title=f"{actuator_names[i]} ({actuator_sensors[i].id}): x0 Start Guess vs Actual",
        ylabel_1axis="Position / Flow (0-1)",
        ylabel_2axis="Temperature (°C)",
        show=True,
    )

# Separate plot for supply air flow in CFM
entry_cfm = [
    tb.plot.Entry(supply_air_flow_cfm_data, label="Supply Air Flow (CFM)"),
    tb.plot.Entry(percent_air_flow_data, label="Percent Air Flow (0-1)", axis=2),
]
tb.plot.plot(
    simulator.date_time_steps,
    entry_cfm,
    title=f"Supply Air Flow (CFM) - {SELECTED_ROOM} Initial",
    ylabel_1axis="Flow (CFM)",
    ylabel_2axis="Percent Air Flow (0-1)",
)

# ==========================================================================
# PLOT CASCADE CONTROLLER INTERNALS (after x0 simulation)
# ==========================================================================

print("\nPlotting cascade controller internal signals (x0 start guess)...")

# Get the cascade controller for actuator 0 (damper) = candidate 2
cascade_ctrl = controller._get_candidate(0, 2)

# A-loop: setpoint (weighted zone temp setpoint), feedback (weighted zone temp), output (intermediate flow setpoint)
cascade_a_setpoint = (
    cascade_ctrl.ctrl_a.input["setpointValue"].history(i_c=0).detach().numpy().T
)
cascade_a_feedback = (
    cascade_ctrl.ctrl_a.input["actualValue"].history(i_c=0).detach().numpy().T
)
cascade_a_output = (
    cascade_ctrl.ctrl_a.output["inputSignal"].history(i_c=0).detach().numpy().T
)

# B-loop: setpoint (A-loop output = desired flow), feedback (weighted percent air flow), output (damper position)
cascade_b_setpoint = (
    cascade_ctrl.ctrl_b.input["setpointValue"].history(i_c=0).detach().numpy().T
)
cascade_b_feedback = (
    cascade_ctrl.ctrl_b.input["actualValue"].history(i_c=0).detach().numpy().T
)
cascade_b_output = (
    cascade_ctrl.ctrl_b.output["inputSignal"].history(i_c=0).detach().numpy().T
)

# Compute A-loop error: setpoint - feedback (temperature error driving the cascade)
cascade_a_error = cascade_a_setpoint - cascade_a_feedback

# Plot A-loop internals: temperature loop
entry_a = [
    tb.plot.Entry(cascade_a_setpoint, label="A-loop Setpoint (weighted zone temp SP)"),
    tb.plot.Entry(cascade_a_feedback, label="A-loop Feedback (weighted zone temp)"),
    tb.plot.Entry(
        cascade_a_error, label="A-loop Error (SP - feedback)", linestyle="-."
    ),
    tb.plot.Entry(
        cascade_a_output, label="A-loop Output (intermediate flow SP)", axis=2
    ),
]
tb.plot.plot(
    simulator.date_time_steps,
    entry_a,
    title="Cascade Damper - A-loop (Temperature) Internals [x0]",
    ylabel_1axis="Temperature (°C)",
    ylabel_2axis="Flow Setpoint (0-1)",
)

# Plot B-loop internals: flow loop
entry_b = [
    tb.plot.Entry(cascade_b_setpoint, label="B-loop Setpoint (A-loop output)"),
    tb.plot.Entry(cascade_b_feedback, label="B-loop Feedback (weighted % air flow)"),
    tb.plot.Entry(cascade_b_output, label="B-loop Output (damper position)", axis=2),
    tb.plot.Entry(
        actual_values[0], label="Actual Damper Position", axis=2, linestyle="--"
    ),
]
tb.plot.plot(
    simulator.date_time_steps,
    entry_b,
    title="Cascade Damper - B-loop (Flow) Internals [x0]",
    ylabel_1axis="Flow (0-1)",
    ylabel_2axis="Damper Position (0-1)",
)

# Plot full cascade overview: inputs → intermediate → output
entry_cascade = [
    tb.plot.Entry(cascade_a_setpoint, label="Zone Temp Setpoint (A-input)"),
    tb.plot.Entry(cascade_a_feedback, label="Zone Temp (A-feedback)"),
    tb.plot.Entry(cascade_a_output, label="A→B Flow SP (intermediate)", axis=2),
    tb.plot.Entry(
        cascade_b_feedback, label="% Air Flow (B-feedback)", axis=2, linestyle="-."
    ),
    tb.plot.Entry(
        cascade_b_output, label="Damper Cmd (B-output)", axis=2, linestyle="--"
    ),
]
tb.plot.plot(
    simulator.date_time_steps,
    entry_cascade,
    title="Cascade Damper - Full Signal Flow [x0]",
    ylabel_1axis="Temperature (°C)",
    ylabel_2axis="Position / Flow (0-1)",
)

# plt.show()

# ==========================================================================
# DIAGNOSTIC: WHAT DOES THE DAMPER ACTUALLY TRACK?
# ==========================================================================

print("\n" + "=" * 80)
print(f"DAMPER CONTROL DIAGNOSTIC FOR bldg13 {SELECTED_ROOM}")
print("=" * 80)

# Extract AHU-level data
ahu_sat_data = (
    ahu_supply_air_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
ahu_sat_sp_data = (
    ahu_supply_air_temp_sp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)

# Use first simulation for diagnostics (flatten from (n_sim, n_t) to (n_t,))
damper_flat = actual_values[0][0]  # damper position (0-1)
reheat_flat = actual_values[1][0]  # reheat valve (0-1)
zone_temp_flat = zone_temp_data[0]  # zone temp (°
zone_sp_flat = zone_setpoint_data[0]  # zone temp setpoint (°C)
temp_error_flat = zone_sp_flat - zone_temp_flat  # SP - actual
pct_flow_flat = percent_air_flow_data[0]  # percent air flow (0-1)
cfm_flat = supply_air_flow_cfm_data[0]  # supply air flow (CFM)
ahu_sat_flat = ahu_sat_data[0]  # AHU supply air temp (°C)
ahu_sat_sp_flat = ahu_sat_sp_data[0]  # AHU supply air temp SP (°C)

n_pts = len(damper_flat)
time_color = np.arange(n_pts)  # color by time index

# --- 1. Scatter plots: damper vs each signal ---
print("\n  Generating scatter plots (damper position vs each signal)...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f"Damper Position vs Potential Driving Signals - {SELECTED_ROOM}", fontsize=14
)

scatter_signals = [
    (zone_temp_flat, "Zone Temperature (°C)"),
    (zone_sp_flat, "Zone Temp Setpoint (°C)"),
    (temp_error_flat, "Temp Error: SP - actual (°C)"),
    (pct_flow_flat, "Percent Air Flow (0-1)"),
    (cfm_flat, "Supply Air Flow (CFM)"),
    (ahu_sat_flat, "AHU Supply Air Temp (°C)"),
]

for ax, (signal, label) in zip(axes.flat, scatter_signals):
    sc = ax.scatter(signal, damper_flat, c=time_color, cmap="viridis", s=10, alpha=0.7)
    ax.set_xlabel(label)
    ax.set_ylabel("Damper Position (0-1)")
    # Compute and show Pearson correlation
    corr = np.corrcoef(signal, damper_flat)[0, 1]
    ax.set_title(f"r = {corr:.3f}")

fig.colorbar(sc, ax=axes.ravel().tolist(), label="Time Step Index", shrink=0.6)
fig.tight_layout()

# --- 2. Damper mode analysis ---
print("  Generating damper mode analysis...")

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle(
    f"Temperature Error Distribution by Damper Mode - {SELECTED_ROOM}", fontsize=14
)

mode_min = damper_flat < 0.3
mode_mod = (damper_flat >= 0.3) & (damper_flat <= 0.9)
mode_max = damper_flat > 0.9

modes = [
    (mode_min, f"Minimum (damper < 0.3)\nn={np.sum(mode_min)}"),
    (mode_mod, f"Modulating (0.3-0.9)\nn={np.sum(mode_mod)}"),
    (mode_max, f"Full Open (damper > 0.9)\nn={np.sum(mode_max)}"),
]

for ax, (mask, label) in zip(axes2, modes):
    if np.sum(mask) > 0:
        ax.hist(temp_error_flat[mask], bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Error = 0")
        mean_err = np.mean(temp_error_flat[mask])
        ax.axvline(
            mean_err,
            color="green",
            linestyle="-",
            alpha=0.7,
            label=f"Mean = {mean_err:.2f}",
        )
        ax.legend(fontsize=8)
    ax.set_title(label)
    ax.set_xlabel("Temp Error: SP - actual (°C)")
    ax.set_ylabel("Count")

fig2.tight_layout()

# --- 3. Cross-correlation with lag ---
print("  Computing cross-correlations...")


def cross_corr(x, y, max_lag=10):
    """Compute normalized cross-correlation for lags -max_lag to +max_lag."""
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            cc[i] = np.mean(x[: n - lag] * y[lag:])
        else:
            cc[i] = np.mean(x[-lag:] * y[: n + lag])
    return lags, cc


max_lag = 12  # 12 steps = 2 hours at 10-min intervals

xcorr_signals = [
    (temp_error_flat, "Temp Error (SP-actual)"),
    (pct_flow_flat, "Percent Air Flow"),
    (cfm_flat, "Supply Air Flow (CFM)"),
    (ahu_sat_flat, "AHU Supply Air Temp"),
]

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8))
fig3.suptitle(
    f"Cross-Correlation: Damper Position vs Signals (lag in steps of {step_size}s) - {SELECTED_ROOM}",
    fontsize=13,
)

for ax, (signal, label) in zip(axes3.flat, xcorr_signals):
    lags, cc = cross_corr(damper_flat, signal, max_lag=max_lag)
    colors = ["tab:red" if abs(c) == max(abs(cc)) else "tab:blue" for c in cc]
    ax.bar(lags, cc, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"Lag (steps, 1 step = {step_size}s)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(f"vs {label}")
    ax.axhline(0, color="black", linewidth=0.5)
    best_lag = lags[np.argmax(np.abs(cc))]
    best_cc = cc[np.argmax(np.abs(cc))]
    ax.annotate(
        f"peak: lag={best_lag}, r={best_cc:.3f}",
        xy=(best_lag, best_cc),
        fontsize=9,
        ha="center",
        xytext=(0, 10),
        textcoords="offset points",
    )

fig3.tight_layout()

# --- 4. Overlay: damper + flow + temp error + AHU SAT ---
print("  Generating overlay diagnostic plot...")

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig4.suptitle(f"Damper Control Diagnostic Overview - {SELECTED_ROOM}", fontsize=14)

time_axis = np.arange(n_pts)

# Top: damper, percent air flow, reheat on left; temp error on right
ax4a.plot(time_axis, damper_flat, label="Damper Position", linewidth=1.5)
ax4a.plot(time_axis, pct_flow_flat, label="Percent Air Flow", linewidth=1, alpha=0.8)
ax4a.plot(
    time_axis, reheat_flat, label="Reheat Valve", linewidth=1, linestyle="--", alpha=0.7
)
ax4a.set_ylabel("Position / Flow (0-1)")
ax4a.legend(loc="upper left", fontsize=9)
ax4a_r = ax4a.twinx()
ax4a_r.plot(
    time_axis,
    temp_error_flat,
    color="red",
    linewidth=1,
    alpha=0.6,
    label="Temp Error (SP-actual)",
)
ax4a_r.axhline(0, color="red", linestyle=":", alpha=0.3)
ax4a_r.set_ylabel("Temp Error (°C)")
ax4a_r.legend(loc="upper right", fontsize=9)
ax4a.set_title("Actuators & Flow vs Temperature Error")

# Bottom: damper on left; zone temp, setpoint, AHU SAT on right
ax4b.plot(time_axis, damper_flat, label="Damper Position", linewidth=1.5)
ax4b.set_ylabel("Damper Position (0-1)")
ax4b.legend(loc="upper left", fontsize=9)
ax4b_r = ax4b.twinx()
ax4b_r.plot(
    time_axis, zone_temp_flat, color="tab:orange", label="Zone Temp", linewidth=1
)
ax4b_r.plot(
    time_axis,
    zone_sp_flat,
    color="tab:orange",
    linestyle="--",
    label="Zone Temp SP",
    linewidth=1,
)
ax4b_r.plot(
    time_axis, ahu_sat_flat, color="tab:green", label="AHU Supply Air Temp", linewidth=1
)
ax4b_r.plot(
    time_axis,
    ahu_sat_sp_flat,
    color="tab:green",
    linestyle="--",
    label="AHU SAT Setpoint",
    linewidth=1,
    alpha=0.7,
)
ax4b_r.set_ylabel("Temperature (°C)")
ax4b_r.legend(loc="upper right", fontsize=9)
ax4b.set_title("Damper vs Temperatures (Zone & AHU)")
ax4b.set_xlabel(f"Time Step ({step_size}s intervals)")

fig4.tight_layout()

# --- Print summary ---
print("\n  Pearson correlations (damper position vs signal):")
for signal, label in scatter_signals:
    corr = np.corrcoef(signal, damper_flat)[0, 1]
    print(f"    {label:35s}  r = {corr:+.4f}")

print(f"\n  Damper mode breakdown ({n_pts} timesteps):")
print(f"    Minimum (< 0.3):    {np.sum(mode_min):4d} ({100*np.mean(mode_min):.1f}%)")
print(f"    Modulating (0.3-0.9):{np.sum(mode_mod):4d} ({100*np.mean(mode_mod):.1f}%)")
print(f"    Full open (> 0.9):  {np.sum(mode_max):4d} ({100*np.mean(mode_max):.1f}%)")

# plt.show()

# ==========================================================================
# AHU CONTROL DIAGNOSTIC
# ==========================================================================

print("\n" + "=" * 80)
print(f"AHU CONTROL DIAGNOSTIC FOR bldg13 AHU01")
print("=" * 80)

# Extract AHU-level data (all flattened to first simulation)
ahu_ccv_data = ahu_ccv_sensor.time_series_input.values[:, :, 0].detach().numpy().T
ahu_cooling_valve_data = (
    ahu_cooling_valve_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
ahu_heating_valve_data = (
    ahu_heating_valve_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
ahu_oat_data = (
    ahu_outside_air_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
ahu_rat_data = (
    ahu_return_air_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)

ahu_ccv_flat = ahu_ccv_data[0]
ahu_cooling_valve_flat = ahu_cooling_valve_data[0]
ahu_heating_valve_flat = ahu_heating_valve_data[0]
ahu_oat_flat = ahu_oat_data[0]
ahu_rat_flat = ahu_rat_data[0]
ahu_sat_error_flat = ahu_sat_sp_flat - ahu_sat_flat  # SAT tracking error

# --- Plot A: AHU SAT Control Loop (time-series overlay) ---
print("\n  [Plot A] AHU SAT Control Loop time-series overlay...")

fig_a, (ax_a1, ax_a2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig_a.suptitle("AHU01 Supply Air Temperature Control Loop", fontsize=14)

# Top: AHU SAT vs Setpoint (left), SAT tracking error (right)
ax_a1.plot(
    time_axis,
    ahu_sat_flat,
    label="AHU Supply Air Temp",
    linewidth=1.5,
    color="tab:blue",
)
ax_a1.plot(
    time_axis,
    ahu_sat_sp_flat,
    label="AHU SAT Setpoint",
    linewidth=1.5,
    color="tab:blue",
    linestyle="--",
)
ax_a1.set_ylabel("Temperature (°C)")
ax_a1.legend(loc="upper left", fontsize=9)
ax_a1.set_title("AHU Supply Air Temp vs Setpoint")
ax_a1_r = ax_a1.twinx()
ax_a1_r.plot(
    time_axis,
    ahu_sat_error_flat,
    color="red",
    linewidth=1,
    alpha=0.7,
    label="SAT Error (SP - actual)",
)
ax_a1_r.axhline(0, color="red", linestyle=":", alpha=0.3)
ax_a1_r.set_ylabel("SAT Error (°C)")
ax_a1_r.legend(loc="upper right", fontsize=9)

# Bottom: AHU valves (left), Outside Air Temp (right)
ax_a2.plot(
    time_axis,
    ahu_ccv_flat,
    label="CCV (Cooling Coil Valve)",
    linewidth=1.2,
    color="tab:cyan",
)
ax_a2.plot(
    time_axis,
    ahu_cooling_valve_flat,
    label="Cooling Valve Output",
    linewidth=1.2,
    color="tab:blue",
    linestyle="--",
)
ax_a2.plot(
    time_axis,
    ahu_heating_valve_flat,
    label="Heating Valve Output",
    linewidth=1.2,
    color="tab:red",
)
ax_a2.set_ylabel("Valve Position (0-1)")
ax_a2.set_ylim(-0.05, 1.05)
ax_a2.legend(loc="upper left", fontsize=9)
ax_a2.set_title("AHU Valves & Outside Air Temperature")
ax_a2_r = ax_a2.twinx()
ax_a2_r.plot(
    time_axis, ahu_oat_flat, color="tab:green", linewidth=1.2, label="Outside Air Temp"
)
ax_a2_r.plot(
    time_axis,
    ahu_rat_flat,
    color="tab:orange",
    linewidth=1.2,
    label="Return Air Temp",
    linestyle="-.",
)
ax_a2_r.set_ylabel("Temperature (°C)")
ax_a2_r.legend(loc="upper right", fontsize=9)
ax_a2.set_xlabel(f"Time Step ({step_size}s intervals)")

fig_a.tight_layout()

# --- Plot B: Scatter matrix -- damper vs AHU signals ---
print("  [Plot B] Scatter matrix: damper position vs AHU signals...")

fig_b, axes_b = plt.subplots(2, 3, figsize=(16, 10))
fig_b.suptitle(f"Damper Position vs AHU Signals - {SELECTED_ROOM}", fontsize=14)

ahu_scatter_signals = [
    (ahu_sat_flat, "AHU Supply Air Temp (°C)"),
    (ahu_sat_sp_flat, "AHU SAT Setpoint (°C)"),
    (ahu_sat_error_flat, "AHU SAT Error: SP - actual (°C)"),
    (ahu_ccv_flat, "CCV (Cooling Coil Valve, 0-1)"),
    (ahu_cooling_valve_flat, "Cooling Valve Output (0-1)"),
    (ahu_oat_flat, "Outside Air Temp (°C)"),
]

for ax, (signal, label) in zip(axes_b.flat, ahu_scatter_signals):
    sc = ax.scatter(signal, damper_flat, c=time_color, cmap="viridis", s=10, alpha=0.7)
    ax.set_xlabel(label)
    ax.set_ylabel("Damper Position (0-1)")
    corr = np.corrcoef(signal, damper_flat)[0, 1]
    ax.set_title(f"r = {corr:.3f}")

fig_b.colorbar(sc, ax=axes_b.ravel().tolist(), label="Time Step Index", shrink=0.6)
fig_b.tight_layout()

# --- Plot C: Cross-correlation -- damper vs AHU signals ---
print("  [Plot C] Cross-correlation: damper vs AHU signals...")

ahu_xcorr_signals = [
    (ahu_sat_flat, "AHU Supply Air Temp"),
    (ahu_sat_error_flat, "AHU SAT Error (SP-actual)"),
    (ahu_ccv_flat, "CCV (Cooling Coil Valve)"),
    (ahu_cooling_valve_flat, "Cooling Valve Output"),
    (ahu_oat_flat, "Outside Air Temp"),
    (ahu_heating_valve_flat, "Heating Valve Output"),
]

fig_c, axes_c = plt.subplots(2, 3, figsize=(18, 9))
fig_c.suptitle(
    f"Cross-Correlation: Damper Position vs AHU Signals (lag in steps of {step_size}s) - {SELECTED_ROOM}",
    fontsize=13,
)

for ax, (signal, label) in zip(axes_c.flat, ahu_xcorr_signals):
    lags, cc = cross_corr(damper_flat, signal, max_lag=max_lag)
    colors = ["tab:red" if abs(c) == max(abs(cc)) else "tab:blue" for c in cc]
    ax.bar(lags, cc, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"Lag (steps, 1 step = {step_size}s)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(f"vs {label}")
    ax.axhline(0, color="black", linewidth=0.5)
    best_lag = lags[np.argmax(np.abs(cc))]
    best_cc = cc[np.argmax(np.abs(cc))]
    ax.annotate(
        f"peak: lag={best_lag}, r={best_cc:.3f}",
        xy=(best_lag, best_cc),
        fontsize=8,
        ha="center",
        xytext=(0, 10),
        textcoords="offset points",
    )

fig_c.tight_layout()

# --- Plot D: AHU Valve Sequencing ---
print("  [Plot D] AHU valve sequencing visualization...")

fig_d, ax_d = plt.subplots(1, 1, figsize=(16, 6))
ax_d.set_title("AHU01 Valve Sequencing & Supply Air Temperature", fontsize=14)

ax_d.fill_between(
    time_axis,
    0,
    ahu_heating_valve_flat,
    alpha=0.3,
    color="tab:red",
    label="Heating Valve",
)
ax_d.fill_between(
    time_axis,
    0,
    ahu_cooling_valve_flat,
    alpha=0.3,
    color="tab:blue",
    label="Cooling Valve Output",
)
ax_d.plot(
    time_axis,
    ahu_ccv_flat,
    linewidth=1.2,
    color="tab:cyan",
    label="CCV (Cooling Coil Valve)",
)
ax_d.plot(time_axis, ahu_heating_valve_flat, linewidth=1, color="tab:red", alpha=0.7)
ax_d.plot(time_axis, ahu_cooling_valve_flat, linewidth=1, color="tab:blue", alpha=0.7)
ax_d.set_ylabel("Valve Position (0-1)")
ax_d.set_ylim(-0.05, 1.05)
ax_d.legend(loc="upper left", fontsize=9)

ax_d_r = ax_d.twinx()
ax_d_r.plot(
    time_axis,
    ahu_sat_flat,
    color="tab:green",
    linewidth=1.5,
    label="AHU Supply Air Temp",
)
ax_d_r.plot(
    time_axis,
    ahu_sat_sp_flat,
    color="tab:green",
    linewidth=1.5,
    linestyle="--",
    label="AHU SAT Setpoint",
)
ax_d_r.plot(
    time_axis,
    ahu_oat_flat,
    color="tab:orange",
    linewidth=1,
    linestyle="-.",
    label="Outside Air Temp",
)
ax_d_r.set_ylabel("Temperature (°C)")
ax_d_r.legend(loc="upper right", fontsize=9)

ax_d.set_xlabel(f"Time Step ({step_size}s intervals)")
fig_d.tight_layout()

# --- Plot E: Multi-room damper comparison vs AHU SAT ---
print("  [Plot E] Multi-room damper comparison vs AHU SAT...")

# Pick 3 comparison rooms (different from SELECTED_ROOM)
_comparison_rooms = [r for r in ROOMS.keys() if r != SELECTED_ROOM][:3]
print(f"    Comparing {SELECTED_ROOM} with: {_comparison_rooms}")

fig_e, ax_e = plt.subplots(1, 1, figsize=(16, 7))
ax_e.set_title(f"Multi-Room Damper Positions vs AHU Supply Air Temp", fontsize=14)

# Plot selected room's damper
ax_e.plot(
    time_axis,
    damper_flat,
    linewidth=1.8,
    label=f"{SELECTED_ROOM} Damper (selected)",
    color="black",
)

# Load comparison rooms' dampers directly from database
_cmap_rooms = plt.cm.Set2
for _ri, _rname in enumerate(_comparison_rooms):
    _ruuid = ROOMS[_rname]["damper_position"]
    try:
        _rdf = load_from_database(
            start_time=start_time[0],
            end_time=end_time[0],
            step_size=step_size,
            table_name=db_config["table_name"],
            sensor_id=_ruuid,
            db_host=db_config["db_host"],
            db_port=db_config["db_port"],
            db_name=db_config["db_name"],
            db_user=db_config["db_user"],
            db_password=db_config["db_password"],
        )
        if _rdf is not None and len(_rdf) > 0:
            _rvals = _rdf.iloc[:, 0].values / 100.0  # percentage to fraction
            # Align to same length as time_axis
            _rvals = _rvals[:n_pts]
            ax_e.plot(
                np.arange(len(_rvals)),
                _rvals,
                linewidth=1,
                alpha=0.7,
                color=_cmap_rooms(_ri),
                label=f"{_rname} Damper",
            )
        else:
            print(f"    WARNING: No damper data for {_rname}")
    except Exception as _ex:
        print(f"    WARNING: Could not load damper data for {_rname}: {_ex}")

ax_e.set_ylabel("Damper Position (0-1)")
ax_e.set_ylim(-0.05, 1.05)
ax_e.legend(loc="upper left", fontsize=9, ncol=2)

ax_e_r = ax_e.twinx()
ax_e_r.plot(
    time_axis,
    ahu_sat_flat,
    color="tab:green",
    linewidth=1.5,
    label="AHU Supply Air Temp",
)
ax_e_r.plot(
    time_axis,
    ahu_sat_sp_flat,
    color="tab:green",
    linewidth=1.5,
    linestyle="--",
    label="AHU SAT Setpoint",
)
ax_e_r.set_ylabel("Temperature (°C)")
ax_e_r.legend(loc="upper right", fontsize=9)

ax_e.set_xlabel(f"Time Step ({step_size}s intervals)")
fig_e.tight_layout()

# --- AHU Diagnostic Summary ---
print("\n" + "-" * 60)
print("  AHU CONTROL DIAGNOSTIC SUMMARY")
print("-" * 60)

print("\n  Pearson correlations (damper position vs AHU signal):")
for signal, label in ahu_scatter_signals:
    corr = np.corrcoef(signal, damper_flat)[0, 1]
    print(f"    {label:40s}  r = {corr:+.4f}")

print(f"\n  AHU SAT tracking error statistics:")
print(f"    Mean:  {np.mean(ahu_sat_error_flat):+.3f} °C")
print(f"    Std:   {np.std(ahu_sat_error_flat):.3f} °C")
print(f"    Max:   {np.max(np.abs(ahu_sat_error_flat)):.3f} °C")

# Valve activity breakdown
_htg_active = ahu_heating_valve_flat > 0.05
_clg_active = ahu_cooling_valve_flat > 0.05
_ccv_active = ahu_ccv_flat > 0.05
_both_off = (~_htg_active) & (~_clg_active)

print(f"\n  AHU valve activity breakdown ({n_pts} timesteps):")
print(
    f"    Heating valve active (> 5%):  {np.sum(_htg_active):4d} ({100*np.mean(_htg_active):.1f}%)"
)
print(
    f"    Cooling valve active (> 5%):  {np.sum(_clg_active):4d} ({100*np.mean(_clg_active):.1f}%)"
)
print(
    f"    CCV active (> 5%):            {np.sum(_ccv_active):4d} ({100*np.mean(_ccv_active):.1f}%)"
)
print(
    f"    Both heating & cooling off:   {np.sum(_both_off):4d} ({100*np.mean(_both_off):.1f}%)"
)

print(f"\n  AHU valve mean positions (when active):")
if np.sum(_htg_active) > 0:
    print(
        f"    Heating valve (when active): {np.mean(ahu_heating_valve_flat[_htg_active]):.3f}"
    )
if np.sum(_clg_active) > 0:
    print(
        f"    Cooling valve (when active): {np.mean(ahu_cooling_valve_flat[_clg_active]):.3f}"
    )
if np.sum(_ccv_active) > 0:
    print(f"    CCV (when active):           {np.mean(ahu_ccv_flat[_ccv_active]):.3f}")

print(f"\n  Temperature summary:")
print(
    f"    Outside Air Temp:  mean={np.mean(ahu_oat_flat):.1f}°C, min={np.min(ahu_oat_flat):.1f}°C, max={np.max(ahu_oat_flat):.1f}°C"
)
print(
    f"    Return Air Temp:   mean={np.mean(ahu_rat_flat):.1f}°C, min={np.min(ahu_rat_flat):.1f}°C, max={np.max(ahu_rat_flat):.1f}°C"
)
print(
    f"    AHU Supply Air:    mean={np.mean(ahu_sat_flat):.1f}°C, min={np.min(ahu_sat_flat):.1f}°C, max={np.max(ahu_sat_flat):.1f}°C"
)
print(
    f"    AHU SAT Setpoint:  mean={np.mean(ahu_sat_sp_flat):.1f}°C, min={np.min(ahu_sat_sp_flat):.1f}°C, max={np.max(ahu_sat_sp_flat):.1f}°C"
)

# plt.show()

# ==========================================================================
# CASCADE CONTROL DIAGNOSTIC: IS THE DAMPER CASCADE-CONTROLLED?
# ==========================================================================

print("\n" + "=" * 80)
print(f"CASCADE CONTROL DIAGNOSTIC FOR bldg13 {SELECTED_ROOM}")
print("=" * 80)
print(
    """
  Testing whether the damper uses cascade control (temp→flow→damper) or
  direct PID (temp→damper). The strong AHU SAT correlation masks the
  zone-level signal, so we use four targeted diagnostics:

  Test 1: Partial correlation (remove AHU SAT confound)
  Test 2: Inner-loop tightness (damper vs airflow)
  Test 3: Conditional analysis (AHU-SAT-stable windows)
  Test 4: Airflow setpoint inference (flow vs temp error bins)
"""
)

# --- Test 1: Partial Correlation ---
# Remove the AHU SAT effect from both damper and temp error via linear
# regression, then check if the residuals still correlate.
print("  TEST 1: Partial Correlation (controlling for AHU SAT)")
print("  " + "-" * 55)

# Regress AHU SAT out of damper position
_coeffs_d = np.polyfit(ahu_sat_flat, damper_flat, 1)
damper_resid = damper_flat - np.polyval(_coeffs_d, ahu_sat_flat)

# Regress AHU SAT out of temperature error
_coeffs_te = np.polyfit(ahu_sat_flat, temp_error_flat, 1)
temp_error_resid = temp_error_flat - np.polyval(_coeffs_te, ahu_sat_flat)

# Regress AHU SAT out of percent airflow
_coeffs_pf = np.polyfit(ahu_sat_flat, pct_flow_flat, 1)
pct_flow_resid = pct_flow_flat - np.polyval(_coeffs_pf, ahu_sat_flat)

# Partial correlations
r_partial_damper_temperr = np.corrcoef(damper_resid, temp_error_resid)[0, 1]
r_partial_damper_flow = np.corrcoef(damper_resid, pct_flow_resid)[0, 1]
r_raw_damper_temperr = np.corrcoef(damper_flat, temp_error_flat)[0, 1]
r_raw_damper_flow = np.corrcoef(damper_flat, pct_flow_flat)[0, 1]

print(f"    Raw correlation:     damper vs temp_error  r = {r_raw_damper_temperr:+.4f}")
print(
    f"    Partial correlation: damper vs temp_error  r = {r_partial_damper_temperr:+.4f}  (AHU SAT removed)"
)
print(f"    Raw correlation:     damper vs pct_flow    r = {r_raw_damper_flow:+.4f}")
print(
    f"    Partial correlation: damper vs pct_flow    r = {r_partial_damper_flow:+.4f}  (AHU SAT removed)"
)

if abs(r_partial_damper_temperr) > 0.2:
    print(
        f"\n    --> RESULT: Significant partial correlation ({r_partial_damper_temperr:+.3f})."
    )
    print(f"        Zone temp error DOES drive the damper beyond AHU SAT effect.")
    print(f"        This supports cascade control (outer temp loop is active).")
else:
    print(
        f"\n    --> RESULT: Weak partial correlation ({r_partial_damper_temperr:+.3f})."
    )
    print(f"        After removing AHU SAT, temp error has little residual effect.")
    print(f"        Cascade outer loop may be weak or inactive.")

# Plot residuals
fig_t1, axes_t1 = plt.subplots(1, 3, figsize=(18, 5))
fig_t1.suptitle(
    f"Test 1: Partial Correlation (AHU SAT effect removed) - {SELECTED_ROOM}",
    fontsize=13,
)

# Scatter: raw damper vs temp error
axes_t1[0].scatter(
    temp_error_flat, damper_flat, c=time_color, cmap="viridis", s=10, alpha=0.7
)
axes_t1[0].set_xlabel("Temp Error: SP - actual (°C)")
axes_t1[0].set_ylabel("Damper Position (0-1)")
axes_t1[0].set_title(f"Raw: r = {r_raw_damper_temperr:.3f}")

# Scatter: residual damper vs residual temp error
axes_t1[1].scatter(
    temp_error_resid, damper_resid, c=time_color, cmap="viridis", s=10, alpha=0.7
)
axes_t1[1].set_xlabel("Temp Error Residual (AHU SAT removed)")
axes_t1[1].set_ylabel("Damper Residual (AHU SAT removed)")
axes_t1[1].set_title(f"Partial: r = {r_partial_damper_temperr:.3f}")

# Scatter: residual damper vs residual pct flow
axes_t1[2].scatter(
    pct_flow_resid, damper_resid, c=time_color, cmap="viridis", s=10, alpha=0.7
)
axes_t1[2].set_xlabel("Pct Airflow Residual (AHU SAT removed)")
axes_t1[2].set_ylabel("Damper Residual (AHU SAT removed)")
axes_t1[2].set_title(f"Partial: r = {r_partial_damper_flow:.3f}")

fig_t1.tight_layout()

# --- Test 2: Inner-Loop Tightness (Damper vs Airflow) ---
# In cascade, the inner B-loop actively regulates airflow, so damper and
# airflow should be extremely tightly correlated (R² > 0.9). In direct
# PID the relationship is looser due to duct pressure variations.
print("\n  TEST 2: Inner-Loop Tightness (Damper vs Airflow)")
print("  " + "-" * 55)

r_damper_flow = np.corrcoef(damper_flat, pct_flow_flat)[0, 1]
r2_damper_flow = r_damper_flow**2

# Fit a linear model for residual analysis
_coeffs_df = np.polyfit(damper_flat, pct_flow_flat, 1)
pct_flow_predicted = np.polyval(_coeffs_df, damper_flat)
flow_residuals = pct_flow_flat - pct_flow_predicted
flow_resid_std = np.std(flow_residuals)

print(
    f"    Damper vs Pct Airflow:  r = {r_damper_flow:+.4f},  R² = {r2_damper_flow:.4f}"
)
print(f"    Linear fit residual std: {flow_resid_std:.4f}")

if r2_damper_flow > 0.90:
    print(
        f"\n    --> RESULT: Very tight relationship (R²={r2_damper_flow:.3f} > 0.90)."
    )
    print(
        f"        The inner flow loop is actively correcting, consistent with cascade."
    )
elif r2_damper_flow > 0.70:
    print(f"\n    --> RESULT: Moderately tight relationship (R²={r2_damper_flow:.3f}).")
    print(f"        Suggestive of cascade, but not conclusive.")
else:
    print(f"\n    --> RESULT: Loose relationship (R²={r2_damper_flow:.3f} < 0.70).")
    print(f"        Weaker evidence for an active inner flow loop.")

fig_t2, axes_t2 = plt.subplots(1, 2, figsize=(14, 5))
fig_t2.suptitle(f"Test 2: Inner-Loop Tightness - {SELECTED_ROOM}", fontsize=13)

# Scatter: damper vs pct flow
axes_t2[0].scatter(
    damper_flat, pct_flow_flat, c=time_color, cmap="viridis", s=10, alpha=0.7
)
_x_line = np.linspace(np.min(damper_flat), np.max(damper_flat), 100)
axes_t2[0].plot(
    _x_line,
    np.polyval(_coeffs_df, _x_line),
    "r-",
    linewidth=2,
    label=f"Linear fit (R²={r2_damper_flow:.3f})",
)
axes_t2[0].set_xlabel("Damper Position (0-1)")
axes_t2[0].set_ylabel("Percent Airflow (0-1)")
axes_t2[0].set_title(f"Damper vs Airflow: R² = {r2_damper_flow:.3f}")
axes_t2[0].legend(fontsize=9)

# Residual histogram
axes_t2[1].hist(flow_residuals, bins=30, edgecolor="black", alpha=0.7)
axes_t2[1].axvline(0, color="red", linestyle="--", alpha=0.5)
axes_t2[1].set_xlabel("Flow Residual (actual - linear fit)")
axes_t2[1].set_ylabel("Count")
axes_t2[1].set_title(f"Residual Distribution (std={flow_resid_std:.4f})")

fig_t2.tight_layout()

# --- Test 3: Conditional Analysis (AHU-SAT-stable windows) ---
# Find windows where AHU SAT has low variance. In those windows, the
# AHU SAT confound is minimized, so we can see the zone-level signal.
print("\n  TEST 3: Conditional Analysis (AHU-SAT-stable windows)")
print("  " + "-" * 55)

# Use a rolling window to find stable AHU SAT periods
_window_size = 6  # 6 steps = 1 hour at 10-min intervals
_rolling_std = pd.Series(ahu_sat_flat).rolling(_window_size, center=True).std().values

# Threshold: bottom 25th percentile of rolling std
_valid_mask = ~np.isnan(_rolling_std)
_std_threshold = np.nanpercentile(_rolling_std, 25)
_stable_mask = _valid_mask & (_rolling_std <= _std_threshold)

n_stable = np.sum(_stable_mask)
print(f"    Window size: {_window_size} steps ({_window_size * step_size}s)")
print(
    f"    SAT stability threshold: rolling_std <= {_std_threshold:.3f}°C (25th percentile)"
)
print(f"    Stable timesteps: {n_stable} / {n_pts} ({100*n_stable/n_pts:.1f}%)")

# Initialize with NaN (overwritten if enough stable data exists)
r_stable_damper_temperr = float("nan")
r_stable_damper_flow = float("nan")
r_stable_damper_sat = float("nan")

if n_stable >= 10:
    r_stable_damper_temperr = np.corrcoef(
        damper_flat[_stable_mask], temp_error_flat[_stable_mask]
    )[0, 1]
    r_stable_damper_flow = np.corrcoef(
        damper_flat[_stable_mask], pct_flow_flat[_stable_mask]
    )[0, 1]
    r_stable_damper_sat = np.corrcoef(
        damper_flat[_stable_mask], ahu_sat_flat[_stable_mask]
    )[0, 1]

    print(f"\n    Correlations during AHU-SAT-stable windows:")
    print(
        f"      damper vs temp_error:  r = {r_stable_damper_temperr:+.4f}  (zone-level signal)"
    )
    print(f"      damper vs pct_flow:    r = {r_stable_damper_flow:+.4f}  (inner loop)")
    print(
        f"      damper vs AHU SAT:     r = {r_stable_damper_sat:+.4f}  (should be weaker)"
    )

    if abs(r_stable_damper_temperr) > abs(r_raw_damper_temperr) + 0.05:
        print(f"\n    --> RESULT: Temp error correlation STRONGER in stable windows")
        print(
            f"        ({r_stable_damper_temperr:+.3f} vs {r_raw_damper_temperr:+.3f})."
        )
        print(f"        Zone temp error signal emerges when AHU SAT noise is removed.")
        print(f"        This supports cascade control with an outer temperature loop.")
    elif abs(r_stable_damper_temperr) > 0.2:
        print(
            f"\n    --> RESULT: Moderate temp error correlation in stable windows ({r_stable_damper_temperr:+.3f})."
        )
        print(f"        Some evidence for zone-level temperature control.")
    else:
        print(
            f"\n    --> RESULT: Weak temp error correlation even in stable windows ({r_stable_damper_temperr:+.3f})."
        )
        print(f"        Zone temperature may not be the primary damper driver.")

    # Plot
    fig_t3, axes_t3 = plt.subplots(1, 3, figsize=(18, 5))
    fig_t3.suptitle(
        f"Test 3: Conditional Analysis (AHU-SAT-stable windows) - {SELECTED_ROOM}",
        fontsize=13,
    )

    # Highlight stable windows on a time series
    axes_t3[0].plot(time_axis, ahu_sat_flat, linewidth=1, alpha=0.5, label="AHU SAT")
    axes_t3[0].fill_between(
        time_axis,
        np.min(ahu_sat_flat),
        np.max(ahu_sat_flat),
        where=_stable_mask,
        alpha=0.3,
        color="green",
        label="Stable windows",
    )
    axes_t3[0].set_xlabel(f"Time Step ({step_size}s)")
    axes_t3[0].set_ylabel("AHU SAT (°C)")
    axes_t3[0].set_title(f"Stable AHU SAT windows ({n_stable} pts)")
    axes_t3[0].legend(fontsize=9)

    # Scatter: damper vs temp error in stable windows
    axes_t3[1].scatter(
        temp_error_flat[_stable_mask],
        damper_flat[_stable_mask],
        c=time_color[_stable_mask],
        cmap="viridis",
        s=15,
        alpha=0.7,
    )
    axes_t3[1].set_xlabel("Temp Error: SP - actual (°C)")
    axes_t3[1].set_ylabel("Damper Position (0-1)")
    axes_t3[1].set_title(f"Stable windows: r = {r_stable_damper_temperr:.3f}")

    # Scatter: damper vs pct flow in stable windows
    axes_t3[2].scatter(
        pct_flow_flat[_stable_mask],
        damper_flat[_stable_mask],
        c=time_color[_stable_mask],
        cmap="viridis",
        s=15,
        alpha=0.7,
    )
    axes_t3[2].set_xlabel("Percent Airflow (0-1)")
    axes_t3[2].set_ylabel("Damper Position (0-1)")
    axes_t3[2].set_title(f"Stable windows: r = {r_stable_damper_flow:.3f}")

    fig_t3.tight_layout()
else:
    print("\n    --> RESULT: Not enough stable timesteps for meaningful analysis.")

# --- Test 4: Airflow Setpoint Inference ---
# In cascade, the outer loop computes a flow setpoint from temp error.
# Bin data by temp error and check if mean airflow shows a monotonic
# staircase pattern (min flow → max flow as cooling demand increases).
print("\n  TEST 4: Airflow Setpoint Inference (flow vs temp error bins)")
print("  " + "-" * 55)

# Create temperature error bins
_n_bins = 8
_te_bins = np.linspace(
    np.percentile(temp_error_flat, 2), np.percentile(temp_error_flat, 98), _n_bins + 1
)
_bin_centers = []
_bin_mean_flow = []
_bin_std_flow = []
_bin_mean_damper = []
_bin_counts = []

for _bi in range(_n_bins):
    _mask_bin = (temp_error_flat >= _te_bins[_bi]) & (
        temp_error_flat < _te_bins[_bi + 1]
    )
    _n_in_bin = np.sum(_mask_bin)
    _bin_counts.append(_n_in_bin)
    _bin_centers.append((_te_bins[_bi] + _te_bins[_bi + 1]) / 2)
    if _n_in_bin > 2:
        _bin_mean_flow.append(np.mean(pct_flow_flat[_mask_bin]))
        _bin_std_flow.append(np.std(pct_flow_flat[_mask_bin]))
        _bin_mean_damper.append(np.mean(damper_flat[_mask_bin]))
    else:
        _bin_mean_flow.append(np.nan)
        _bin_std_flow.append(np.nan)
        _bin_mean_damper.append(np.nan)

_bin_centers = np.array(_bin_centers)
_bin_mean_flow = np.array(_bin_mean_flow)
_bin_std_flow = np.array(_bin_std_flow)
_bin_mean_damper = np.array(_bin_mean_damper)
_bin_counts = np.array(_bin_counts)

# Check monotonicity: in cascade with reverse-acting outer loop,
# more negative temp error (zone too warm) should lead to higher airflow
_valid_bins = ~np.isnan(_bin_mean_flow)
_valid_flow = _bin_mean_flow[_valid_bins]
_valid_centers = _bin_centers[_valid_bins]
if len(_valid_flow) >= 3:
    # Compute rank correlation (Spearman) for monotonicity
    # Third party imports
    from scipy.stats import spearmanr as _spearmanr

    _rho, _pval = _spearmanr(_valid_centers, _valid_flow)
    print(
        f"    Spearman rank correlation (temp_error bins vs mean airflow): rho = {_rho:+.4f}, p = {_pval:.4f}"
    )

    # Also check the flow range across bins
    _flow_range = np.max(_valid_flow) - np.min(_valid_flow)
    print(
        f"    Airflow range across bins: {_flow_range:.4f} (min={np.min(_valid_flow):.4f}, max={np.max(_valid_flow):.4f})"
    )

    if abs(_rho) > 0.7 and _pval < 0.05:
        _direction = "increases" if _rho > 0 else "decreases"
        print(
            f"\n    --> RESULT: Strong monotonic trend (rho={_rho:+.3f}, p={_pval:.4f})."
        )
        print(f"        Airflow {_direction} with temp error (SP-actual).")
        print(f"        This is consistent with cascade control: outer loop maps")
        print(f"        temperature error to an airflow setpoint.")
    elif abs(_rho) > 0.4:
        print(f"\n    --> RESULT: Moderate monotonic trend (rho={_rho:+.3f}).")
        print(f"        Some evidence of a temperature→flow mapping.")
    else:
        print(f"\n    --> RESULT: Weak/no monotonic trend (rho={_rho:+.3f}).")
        print(f"        No clear evidence of a temperature→flow setpoint mapping.")
else:
    _rho = np.nan
    print("    Not enough valid bins for monotonicity analysis.")

fig_t4, axes_t4 = plt.subplots(1, 3, figsize=(18, 5))
fig_t4.suptitle(f"Test 4: Airflow Setpoint Inference - {SELECTED_ROOM}", fontsize=13)

# Bar chart: mean airflow per temp error bin
_bar_colors = ["tab:blue" if c > 5 else "tab:gray" for c in _bin_counts]
axes_t4[0].bar(
    _bin_centers,
    _bin_mean_flow,
    width=(_te_bins[1] - _te_bins[0]) * 0.8,
    yerr=_bin_std_flow,
    color=_bar_colors,
    edgecolor="black",
    capsize=3,
    alpha=0.7,
)
axes_t4[0].set_xlabel("Temp Error: SP - actual (°C)")
axes_t4[0].set_ylabel("Mean Percent Airflow (0-1)")
axes_t4[0].set_title(
    f"Mean Airflow per Temp Error Bin (rho={_rho:+.3f})"
    if not np.isnan(_rho)
    else "Mean Airflow per Temp Error Bin"
)
# Add count labels
for _xi, _yi, _ci in zip(_bin_centers, _bin_mean_flow, _bin_counts):
    if not np.isnan(_yi):
        axes_t4[0].annotate(
            f"n={_ci}",
            xy=(_xi, _yi),
            fontsize=7,
            ha="center",
            xytext=(0, 8),
            textcoords="offset points",
        )

# Bar chart: mean damper per temp error bin
axes_t4[1].bar(
    _bin_centers,
    _bin_mean_damper,
    width=(_te_bins[1] - _te_bins[0]) * 0.8,
    color="tab:orange",
    edgecolor="black",
    alpha=0.7,
)
axes_t4[1].set_xlabel("Temp Error: SP - actual (°C)")
axes_t4[1].set_ylabel("Mean Damper Position (0-1)")
axes_t4[1].set_title("Mean Damper per Temp Error Bin")

# 2D density: temp error vs airflow (to see if there's a "setpoint line")
axes_t4[2].scatter(
    temp_error_flat, pct_flow_flat, c=ahu_sat_flat, cmap="coolwarm", s=8, alpha=0.5
)
cb = fig_t4.colorbar(axes_t4[2].collections[0], ax=axes_t4[2], shrink=0.8)
cb.set_label("AHU SAT (°C)")
axes_t4[2].set_xlabel("Temp Error: SP - actual (°C)")
axes_t4[2].set_ylabel("Percent Airflow (0-1)")
axes_t4[2].set_title("Airflow vs Temp Error (colored by AHU SAT)")

fig_t4.tight_layout()

# --- Cascade Diagnostic Summary ---
print("\n" + "-" * 60)
print("  CASCADE CONTROL DIAGNOSTIC SUMMARY")
print("-" * 60)
print(
    f"""
  Test 1 (Partial Correlation):
    damper vs temp_error | AHU SAT :  r = {r_partial_damper_temperr:+.4f}
    damper vs pct_flow   | AHU SAT :  r = {r_partial_damper_flow:+.4f}

  Test 2 (Inner-Loop Tightness):
    damper vs pct_flow:  R² = {r2_damper_flow:.4f}

  Test 3 (Conditional / SAT-stable windows):
    damper vs temp_error (stable):  r = {r_stable_damper_temperr:+.4f}
    damper vs pct_flow   (stable):  r = {r_stable_damper_flow:+.4f}

  Test 4 (Airflow Setpoint Inference):
    Spearman rho (temp_error bins vs mean flow): {f'{_rho:+.4f}' if not np.isnan(_rho) else 'N/A'}
"""
)

# Overall verdict
_cascade_score = 0
if abs(r_partial_damper_temperr) > 0.2:
    _cascade_score += 1
if r2_damper_flow > 0.85:
    _cascade_score += 1
if not np.isnan(r_stable_damper_temperr) and abs(r_stable_damper_temperr) > 0.2:
    _cascade_score += 1
if not np.isnan(_rho) and abs(_rho) > 0.5:
    _cascade_score += 1

print(f"  CASCADE EVIDENCE SCORE: {_cascade_score} / 4")
if _cascade_score >= 3:
    print("  --> STRONG evidence for cascade control (temp→flow→damper)")
elif _cascade_score >= 2:
    print("  --> MODERATE evidence for cascade control")
elif _cascade_score >= 1:
    print("  --> WEAK evidence for cascade control")
else:
    print(
        "  --> NO evidence for cascade control; damper may use direct PID or another scheme"
    )

# plt.show()

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
print("\n  Initial PID parameters BEFORE estimation:")
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
        elif hasattr(ctrl, "kp"):
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
    (
        0.0,
        {"maxiter": 100, "ftol": 1e-12, "disp": True},
    ),  # Phase 1: pure fit, no penalty
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
    "AHU_supply_air_temp (sensor 2)",
]
setpoint_names = ["zone_temp_setpoint (setpoint 0)"]

print("\nALPHA WEIGHTS (Candidate Selection):")
candidate_names = [
    "PID (reverse)",
    "PID (non-reverse)",
    "Cascade PID",
    "SAT-Compensated cascade",
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
        elif hasattr(ctrl, "kp"):
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
percent_air_flow_data = (
    supply_air_flow_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
supply_air_flow_cfm_data = (
    supply_air_flow_cfm_sensor.time_series_input.values[:, :, 0].detach().numpy().T
)
ahu_sat_plot_data = (
    ahu_supply_air_temp_sensor.time_series_input.values[:, :, 0].detach().numpy().T
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
        tb.plot.Entry(
            percent_air_flow_data, label=f"Percent Air Flow (0-1)", linestyle="-."
        ),
        tb.plot.Entry(zone_temp_data, label=f"Zone Temperature", axis=2),
        tb.plot.Entry(
            zone_setpoint_data, label=f"Zone Setpoint", axis=2, linestyle="--"
        ),
        tb.plot.Entry(
            ahu_sat_plot_data, label=f"AHU Supply Air Temp", axis=2, linestyle="-."
        ),
    ]

    tb.plot.plot(
        simulator.date_time_steps,
        entry,
        title=f"{actuator_names[i]}: Initial MAE={mae_initial:.4f}, Final MAE={mae_final:.4f}",
        ylabel_1axis="Position / Flow (0-1)",
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
print(f"VAV CONTROL INTERPRETATION FOR bldg13 {SELECTED_ROOM}")
print("=" * 80)
print(
    f"""
Expected VAV Control Logic (Room {SELECTED_ROOM}, AHU01):

  Damper Position (Cascade PID):
    A-loop (outer): Temperature error → intermediate flow setpoint
      - Setpoint: zone temperature setpoint
      - Feedback: zone temperature
      - When zone temp deviates from setpoint → adjusts desired airflow
    B-loop (inner): Flow error → damper position
      - Setpoint: A-loop output (desired flow fraction)
      - Feedback: percent air flow measurement
      - Tracks the flow setpoint by modulating damper

  Reheat Valve (PID reverse):
    - When zone temp < setpoint → open valve (add heat)
    - When zone temp > setpoint → close valve
    - Reverse acting: error = setpoint - actual

The identified parameters reveal:
  - Which controller type best fits each actuator (alpha weights)
  - Which sensors drive each control loop (beta/gamma/beta_b weights)
  - The PID gains that best explain the observed control behavior
"""
)
