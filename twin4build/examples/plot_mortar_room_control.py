#!/usr/bin/env python3
"""
Mortar Building Room Temperature Control Visualization

This script visualizes how temperature is controlled in a specific room,
showing zone temperature, setpoints, valve positions, damper positions,
and air flow to understand the control dynamics.

Usage:
    python plot_mortar_room_control.py [--room RM1100] [--days 7] [--start 2016-01-01]
"""

# Standard library imports
import os

os.environ["DISABLE_AUTORESET_PRINT"] = "1"
# Standard library imports
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Third party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local application imports
# Local application imports - twin4build data loader
from twin4build.utils.data_loaders.load import load_from_database

# ==========================================================================
# CONFIGURATION
# ==========================================================================

# Database configuration
DB_CONFIG = {
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
    "table_name": "mortar_bldg34",
}

# Default table name for Mortar bldg34
TABLE_NAME = "mortar_bldg34"

# Time range for analysis
START_TIME = datetime(2013, 1, 1, tzinfo=timezone.utc)
END_TIME = datetime(2017, 2, 1, tzinfo=timezone.utc)

# Room and AHU configuration
ROOM_ID = "RM1100"
AHU_ID = "AHU01"

# ==========================================================================
# SENSOR UUID MAPPINGS (ref:hasTimeseriesId from bldg34.ttl)
# These UUIDs are stored in the 'uuid' column of the database
# ==========================================================================

# Room RM1100 sensors - UUIDs from TTL file
ROOM_SENSORS = {
    "zone_temp": "414e0dac-a298-4494-b071-7eeaaa21eb69",  # Zone_Air_Temp
    "zone_temp_setpoint": "854e50f1-e47f-4658-b586-f83343ec1b7d",  # Zone_Air_Temp_Setpoint
    "damper_position": "95fadb2b-3e3f-478c-94bf-0b9cbb0fdc33",  # Zone_Air_Damper_Command
    "reheat_valve": "dcac3ea2-5698-4bcb-886d-1e4bd0295444",  # Zone_Reheat_Valve_Command
    "supply_air_flow": "f10234e9-f76c-46aa-ab53-87978645d646",  # Zone_Supply_Air_Flow
    "heating_mode": "a8862381-a247-435c-ae9e-5fc3a4aa2c47",  # Heating_Mode
}

# AHU01 sensors - UUIDs from TTL file
AHU_SENSORS = {
    "supply_air_temp": "130d0d7b-cdd2-4b65-bfa9-812ea5379484",  # Supply_Air_Temp
    "supply_air_temp_setpoint": "ebc9e0ec-e24d-4770-86f1-5b532e662d45",  # Supply_Air_Temp_Setpoint
    "outside_air_temp": "6dd892cb-ef6c-4334-83c7-91d300843c72",  # Outside_Air_Temp
    "return_air_temp": "e8bccd6c-fb18-4aab-9b86-09893d47cf2f",  # Return_Air_Temp
    "cooling_valve": "361fccbb-f5d7-4862-90b6-32da3c973369",  # Cooling_Valve_Output
}


# ==========================================================================
# DATA LOADING FUNCTIONS
# ==========================================================================


def query_sensor_names(table_name: str, pattern: str = None) -> list:
    """Query available sensor names from the database."""
    if pattern:
        sql = f"SELECT DISTINCT name FROM {table_name} WHERE name LIKE '%{pattern}%' ORDER BY name;"
    else:
        sql = f"SELECT DISTINCT name FROM {table_name} ORDER BY name LIMIT 200;"

    cmd = f'docker exec timescaledb psql -U postgres -t -A -c "{sql}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0 and result.stdout.strip():
        return [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
    return []


def get_time_range(table_name: str) -> tuple:
    """Get the available time range in the database."""
    sql = f"SELECT MIN(time), MAX(time) FROM {table_name};"
    cmd = f'docker exec timescaledb psql -U postgres -t -A -F"," -c "{sql}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split(",")
        if len(parts) == 2:
            return pd.to_datetime(parts[0]), pd.to_datetime(parts[1])
    return None, None


def find_sensor_name(table_name: str, room_id: str, sensor_pattern: str) -> str:
    """Find the exact sensor name matching room and sensor type."""
    sensors = query_sensor_names(table_name, room_id)

    for s in sensors:
        if sensor_pattern.lower() in s.lower():
            # Additional filtering for specific cases
            if sensor_pattern == "Zone_Air_Temperature_Sensor":
                if "Setpoint" not in s:
                    return s
            elif sensor_pattern == "Zone_Air_Temperature_Setpoint":
                if "Setpoint" in s:
                    return s
            elif sensor_pattern == "Command" and "Reheat" in s and "Valve" in s:
                return s
            elif sensor_pattern not in [
                "Zone_Air_Temperature_Sensor",
                "Zone_Air_Temperature_Setpoint",
                "Command",
            ]:
                return s

    return None


def load_stream(
    sensor_name: str,
    start_time: datetime,
    end_time: datetime,
    table_name: str = TABLE_NAME,
    step_size: int = 900,
):
    """Load a single sensor stream from the database using twin4build loader."""
    print(f"  Loading: {sensor_name[:60]}...")
    try:
        df = load_from_database(
            table_name=table_name,
            sensor_id=sensor_name,
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            resample=True,
            resample_method="linear",
            clip=True,
            cache=True,
            # Uses 'uuid' column by default - stores ref:hasTimeseriesId from TTL
            **DB_CONFIG,
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


def get_values_safe(df):
    """Safely extract values and index from DataFrame or Series."""
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


def fetch_room_data(
    start_time: datetime, end_time: datetime, table_name: str = TABLE_NAME
) -> dict:
    """Fetch all relevant data for the configured room and AHU using UUID mappings."""

    data = {}

    # Load room sensors using UUID mappings
    print(f"\nLoading room sensors for {ROOM_ID}...")
    for key, uuid in ROOM_SENSORS.items():
        df = load_stream(uuid, start_time, end_time, table_name)
        if df is not None:
            data[key] = df

    # Load AHU sensors using UUID mappings
    print(f"\nLoading AHU sensors for {AHU_ID}...")
    for key, uuid in AHU_SENSORS.items():
        df = load_stream(uuid, start_time, end_time, table_name)
        if df is not None:
            data[f"ahu_{key}"] = df

    return data


def plot_room_control(data: dict, room_id: str, ahu_id: str, save_path: Path = None):
    """Create visualization of room temperature control."""

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"Room {room_id} Temperature Control (AHU: {ahu_id})\nBuilding: bldg34 (Mortar Dataset)",
        fontsize=14,
        fontweight="bold",
    )

    colors = {
        "zone_temp": "#e74c3c",  # Red
        "setpoint": "#2ecc71",  # Green
        "supply_temp": "#3498db",  # Blue
        "outside_temp": "#f39c12",  # Orange
        "damper": "#9b59b6",  # Purple
        "valve": "#1abc9c",  # Teal
        "flow": "#34495e",  # Dark gray
        "heating": "#e67e22",  # Dark orange
    }

    # Plot 1: Temperatures
    ax1 = axes[0]
    ax1.set_ylabel("Temperature (°F)", fontsize=10)
    ax1.set_title("Zone Temperature vs Setpoint", fontsize=11)

    if "zone_temp" in data:
        idx, vals = get_values_safe(data["zone_temp"])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["zone_temp"],
                label="Zone Temp",
                linewidth=1,
                alpha=0.8,
            )

    if "zone_temp_setpoint" in data:
        idx, vals = get_values_safe(data["zone_temp_setpoint"])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["setpoint"],
                label="Setpoint",
                linewidth=1.5,
                linestyle="--",
            )

    if "ahu_supply_air_temp" in data:
        idx, vals = get_values_safe(data["ahu_supply_air_temp"])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["supply_temp"],
                label="AHU Supply Temp",
                linewidth=1,
                alpha=0.6,
            )

    if "ahu_outside_air_temp" in data or "outside_temp" in data:
        key = (
            "ahu_outside_air_temp" if "ahu_outside_air_temp" in data else "outside_temp"
        )
        idx, vals = get_values_safe(data[key])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["outside_temp"],
                label="Outside Temp",
                linewidth=1,
                alpha=0.5,
            )

    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Damper Position
    ax2 = axes[1]
    ax2.set_ylabel("Position (%)", fontsize=10)
    ax2.set_title("VAV Damper Position", fontsize=11)

    if "damper_position" in data:
        idx, vals = get_values_safe(data["damper_position"])
        if idx is not None:
            ax2.plot(
                idx, vals, color=colors["damper"], label="Damper Position", linewidth=1
            )
            ax2.fill_between(idx, 0, vals, alpha=0.3, color=colors["damper"])

    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Reheat Valve & Heating Mode
    ax3 = axes[2]
    ax3.set_ylabel("Position/Mode (%)", fontsize=10)
    ax3.set_title("Reheat Valve Command & Heating Mode", fontsize=11)

    if "reheat_valve" in data:
        idx, vals = get_values_safe(data["reheat_valve"])
        if idx is not None:
            ax3.plot(
                idx, vals, color=colors["valve"], label="Reheat Valve", linewidth=1
            )
            ax3.fill_between(idx, 0, vals, alpha=0.3, color=colors["valve"])

    if "heating_mode" in data:
        idx, vals = get_values_safe(data["heating_mode"])
        if idx is not None:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(
                idx,
                vals,
                color=colors["heating"],
                label="Heating Mode",
                linewidth=1,
                alpha=0.7,
            )
            ax3_twin.set_ylabel("Heating Mode", fontsize=10, color=colors["heating"])
            ax3_twin.tick_params(axis="y", labelcolor=colors["heating"])

    ax3.set_ylim(0, 105)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Supply Air Flow
    ax4 = axes[3]
    ax4.set_ylabel("Flow (CFM)", fontsize=10)
    ax4.set_title("Zone Supply Air Flow", fontsize=11)
    ax4.set_xlabel("Time", fontsize=10)

    if "supply_air_flow" in data:
        idx, vals = get_values_safe(data["supply_air_flow"])
        if idx is not None:
            ax4.plot(
                idx, vals, color=colors["flow"], label="Supply Air Flow", linewidth=1
            )
            ax4.fill_between(idx, 0, vals, alpha=0.2, color=colors["flow"])

    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Plot saved to: {save_path}")

    plt.show()


def list_rooms_and_sensors(table_name: str = TABLE_NAME):
    """List available rooms and their sensors."""
    print("\nQuerying available sensors...")

    sensors = query_sensor_names(table_name)

    # Find rooms (look for ZONE patterns with RM)
    rooms = set()
    for s in sensors:
        if "ZONE" in s and "RM" in s:
            parts = s.split(".")
            for p in parts:
                if p.startswith("RM"):
                    rooms.add(p)

    print(f"\n{'='*60}")
    print(f"Available Rooms in {table_name}:")
    print(f"{'='*60}")
    for room in sorted(rooms):
        print(f"  - {room}")

    print(f"\n{'='*60}")
    print(f"Sample Sensors (first 30):")
    print(f"{'='*60}")
    for s in sensors[:30]:
        print(f"  - {s}")

    return sorted(rooms)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize room temperature control from Mortar bldg34 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  Edit the configuration section at the top of the script:
  - START_TIME, END_TIME: Time range for analysis
  - ROOM_ID, AHU_ID: Room and AHU labels for plot titles
  - ROOM_SENSORS: UUID mappings for room sensors (from TTL ref:hasTimeseriesId)
  - AHU_SENSORS: UUID mappings for AHU sensors (from TTL ref:hasTimeseriesId)

Examples:
  # List available sensors in the database
  python plot_mortar_room_control.py --list
  
  # Plot using configured defaults
  python plot_mortar_room_control.py
  
  # Override time range via CLI
  python plot_mortar_room_control.py --start 2016-06-01 --end 2016-07-01
  
  # Save plot to file
  python plot_mortar_room_control.py --save room_control.png
        """,
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides START_TIME config.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Overrides END_TIME config.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available rooms and sensors"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save plot to file (e.g., room_control.png)",
    )
    parser.add_argument(
        "--table",
        type=str,
        default=TABLE_NAME,
        help=f"Database table name (default: {TABLE_NAME})",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Mortar Building Room Temperature Control Visualization")
    print("=" * 80)

    # Check if table exists
    sql = f"SELECT COUNT(*) FROM {args.table};"
    cmd = f'docker exec timescaledb psql -U postgres -t -A -c "{sql}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0 or not result.stdout.strip():
        print(f"ERROR: Table '{args.table}' not found or empty.")
        print("Make sure to run the ingestion first:")
        print(
            f"  python unified_ingest_streaming.py --dataset mortar --building bldg34"
        )
        return

    record_count = int(result.stdout.strip())
    print(f"Table {args.table}: {record_count:,} records")

    # List mode
    if args.list:
        list_rooms_and_sensors(args.table)
        return

    # Get time range from database
    min_time, max_time = get_time_range(args.table)
    if min_time and max_time:
        print(f"Data range: {min_time} to {max_time}")

    # Determine time range for query (CLI overrides config)
    if args.start:
        start_time = pd.to_datetime(args.start).tz_localize("UTC")
    else:
        start_time = START_TIME

    if args.end:
        end_time = pd.to_datetime(args.end).tz_localize("UTC")
    else:
        end_time = END_TIME

    print(f"\nFetching data for room {ROOM_ID} (AHU: {AHU_ID})")
    print(f"Time range: {start_time} to {end_time}")

    # Fetch data using configured UUID mappings
    data = fetch_room_data(start_time, end_time, args.table)

    if not data:
        print("\nNo data found. Check UUID mappings in configuration.")
        return

    print(f"\n{'='*80}")
    print("Creating visualization...")
    print("=" * 80)

    # Create plot
    save_path = Path(args.save) if args.save else None
    plot_room_control(data, ROOM_ID, AHU_ID, save_path)


if __name__ == "__main__":
    main()
