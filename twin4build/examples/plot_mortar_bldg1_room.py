#!/usr/bin/env python3
"""
Mortar Building 1 Room Temperature Control Visualization

This script visualizes how temperature is controlled in a specific room,
showing zone temperature, setpoints, valve positions, damper positions,
and air flow to understand the control dynamics.

Building 1 has enhanced instrumentation including:
- Zone Supply Air Temperature (at VAV discharge)
- Zone Air Control Temperature
- Zone Percent Air Flow

Usage:
    python plot_mortar_bldg1_room.py
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
}

# Default table name for Mortar bldg1
TABLE_NAME = "mortar_bldg1"

# Time range for analysis (bldg1 data: 2016-06-03 to 2017-12-18)
START_TIME = datetime(2016, 6, 15, tzinfo=timezone.utc)
END_TIME = datetime(2017, 12, 15, tzinfo=timezone.utc)

# Room and AHU configuration
ROOM_ID = "RM107A"
AHU_ID = "AHU01"

# ==========================================================================
# SENSOR UUID MAPPINGS (ref:hasTimeseriesId from bldg1.ttl)
# These UUIDs are stored in the 'uuid' column of the database
# ==========================================================================

# Room RM107A sensors - UUIDs from TTL file (8 sensors - very well instrumented!)
ROOM_SENSORS = {
    "zone_temp": "a2b6510f-cf4f-4edd-a080-b8f4b35968d9",  # Zone_Air_Temp
    "zone_control_temp": "59b93fef-a0ab-4f2d-a036-01c62bfa8a4a",  # Zone_Air_Control_Temp
    "zone_temp_setpoint": "2cb39f2b-27e0-4611-a663-2de371007ff7",  # Zone_Air_Temp_Setpoint
    "damper_position": "13954408-3b78-4483-8b18-dc0471207943",  # Zone_Air_Damper_Command
    "reheat_valve": "be8ce19d-5e81-4f43-be16-8d95366d2d1a",  # Zone_Reheat_Valve_Command
    "supply_air_flow": "037993e1-31fc-4212-aaf1-8465a9481bf8",  # Zone_Supply_Air_Flow
    "percent_air_flow": "778b01e9-8022-4134-a29c-1b9d0106328e",  # Zone_Percent_Air_Flow
    "supply_air_temp": "6ff31387-db42-48a8-a675-2876e9d95639",  # Zone_Supply_Air_Temp (VAV discharge!)
}

# AHU01 sensors - UUIDs from TTL file
AHU_SENSORS = {
    "supply_air_temp": "469e6e6f-c54b-4a58-a5e5-fae1442e04bd",  # Supply_Air_Temp
    "supply_air_temp_setpoint": "0355ffc0-a20f-4e2d-aec3-1febf7536e26",  # Supply_Air_Temp_Setpoint
    "outside_air_temp": "0d3446d0-d237-44b8-a6f8-0d804a0bd83a",  # Outside_Air_Temp
    "return_air_temp": "e0e463f3-1526-4ab5-972b-01e4a3d9cba9",  # Return_Air_Temp
    "supply_air_pressure": "220b1728-6017-42be-9ded-daec6206342e",  # Supply_Air_Pressure
    "cooling_valve": "7c54d613-bec5-4a5d-84a8-e83e97990809",  # Cooling_Valve_Output
    "ccv": "c632b4c5-8744-4540-aa70-9b2d095cf6c3",  # CCV (Cooling Coil Valve)
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

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"Room {room_id} Temperature Control (AHU: {ahu_id})\nBuilding: bldg1 (Mortar Dataset)",
        fontsize=14,
        fontweight="bold",
    )

    colors = {
        "zone_temp": "#e74c3c",  # Red
        "control_temp": "#c0392b",  # Dark red
        "setpoint": "#2ecc71",  # Green
        "vav_supply_temp": "#9b59b6",  # Purple - VAV discharge temp
        "ahu_supply_temp": "#3498db",  # Blue
        "outside_temp": "#f39c12",  # Orange
        "damper": "#8e44ad",  # Purple
        "valve": "#1abc9c",  # Teal
        "flow": "#34495e",  # Dark gray
        "percent_flow": "#7f8c8d",  # Gray
    }

    # Plot 1: Temperatures (Zone, Setpoint, VAV Supply, AHU Supply, Outside)
    ax1 = axes[0]
    ax1.set_ylabel("Temperature (°F)", fontsize=10)
    ax1.set_title("Zone Temperature vs Setpoint & Supply Temperatures", fontsize=11)

    if "zone_temp" in data:
        idx, vals = get_values_safe(data["zone_temp"])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["zone_temp"],
                label="Zone Temp",
                linewidth=1.2,
                alpha=0.9,
            )

    if "zone_control_temp" in data:
        idx, vals = get_values_safe(data["zone_control_temp"])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["control_temp"],
                label="Control Temp",
                linewidth=1,
                alpha=0.6,
                linestyle=":",
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

    # VAV discharge temperature - key for understanding reheat effect!
    if "supply_air_temp" in data:
        idx, vals = get_values_safe(data["supply_air_temp"])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["vav_supply_temp"],
                label="VAV Supply Temp",
                linewidth=1.2,
                alpha=0.8,
            )

    if "ahu_supply_air_temp" in data:
        idx, vals = get_values_safe(data["ahu_supply_air_temp"])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["ahu_supply_temp"],
                label="AHU Supply Temp",
                linewidth=1,
                alpha=0.5,
            )

    if "ahu_outside_air_temp" in data:
        idx, vals = get_values_safe(data["ahu_outside_air_temp"])
        if idx is not None:
            ax1.plot(
                idx,
                vals,
                color=colors["outside_temp"],
                label="Outside Temp",
                linewidth=1,
                alpha=0.4,
            )

    ax1.legend(loc="upper right", fontsize=8, ncol=2)
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

    # Plot 3: Reheat Valve
    ax3 = axes[2]
    ax3.set_ylabel("Position (%)", fontsize=10)
    ax3.set_title("Reheat Valve Command", fontsize=11)

    if "reheat_valve" in data:
        idx, vals = get_values_safe(data["reheat_valve"])
        if idx is not None:
            ax3.plot(
                idx, vals, color=colors["valve"], label="Reheat Valve", linewidth=1
            )
            ax3.fill_between(idx, 0, vals, alpha=0.3, color=colors["valve"])

    ax3.set_ylim(0, 105)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Supply Air Flow (CFM and %)
    ax4 = axes[3]
    ax4.set_ylabel("Flow (CFM)", fontsize=10)
    ax4.set_title("Zone Supply Air Flow", fontsize=11)

    if "supply_air_flow" in data:
        idx, vals = get_values_safe(data["supply_air_flow"])
        if idx is not None:
            ax4.plot(
                idx,
                vals,
                color=colors["flow"],
                label="Supply Air Flow (CFM)",
                linewidth=1,
            )
            ax4.fill_between(idx, 0, vals, alpha=0.2, color=colors["flow"])

    # Add percent flow on secondary axis
    if "percent_air_flow" in data:
        ax4_twin = ax4.twinx()
        idx, vals = get_values_safe(data["percent_air_flow"])
        if idx is not None:
            ax4_twin.plot(
                idx,
                vals,
                color=colors["percent_flow"],
                label="% of Design",
                linewidth=1,
                alpha=0.7,
                linestyle="--",
            )
            ax4_twin.set_ylabel("Flow (%)", fontsize=10, color=colors["percent_flow"])
            ax4_twin.tick_params(axis="y", labelcolor=colors["percent_flow"])
            ax4_twin.set_ylim(0, 105)

    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: AHU Cooling Valve
    ax5 = axes[4]
    ax5.set_ylabel("Position (%)", fontsize=10)
    ax5.set_title("AHU Cooling Coil Valve", fontsize=11)
    ax5.set_xlabel("Time", fontsize=10)

    if "ahu_cooling_valve" in data:
        idx, vals = get_values_safe(data["ahu_cooling_valve"])
        if idx is not None:
            ax5.plot(idx, vals, color="#3498db", label="Cooling Valve", linewidth=1)
            ax5.fill_between(idx, 0, vals, alpha=0.2, color="#3498db")

    ax5.set_ylim(0, 105)
    ax5.legend(loc="upper right", fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Format x-axis
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax5.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha="right")

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
        description="Visualize room temperature control from Mortar bldg1 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  Edit the configuration section at the top of the script:
  - START_TIME, END_TIME: Time range for analysis
  - ROOM_ID, AHU_ID: Room and AHU labels for plot titles
  - ROOM_SENSORS: UUID mappings for room sensors (from TTL ref:hasTimeseriesId)
  - AHU_SENSORS: UUID mappings for AHU sensors (from TTL ref:hasTimeseriesId)

Available rooms in bldg1 (served by AHU01 or AHU02):
  AHU01: RM107A, RM107B, RM115, RM120
  AHU02: RM100, RM103, RM110, RM112

Examples:
  # List available sensors in the database
  python plot_mortar_bldg1_room.py --list
  
  # Plot using configured defaults
  python plot_mortar_bldg1_room.py
  
  # Override time range via CLI
  python plot_mortar_bldg1_room.py --start 2016-06-01 --end 2016-07-01
  
  # Save plot to file
  python plot_mortar_bldg1_room.py --save room_control.png
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
    print("Mortar Building 1 Room Temperature Control Visualization")
    print("=" * 80)

    # Check if table exists
    sql = f"SELECT COUNT(*) FROM {args.table};"
    cmd = f'docker exec timescaledb psql -U postgres -t -A -c "{sql}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0 or not result.stdout.strip():
        print(f"ERROR: Table '{args.table}' not found or empty.")
        print("Make sure to run the ingestion first:")
        print(f"  python unified_ingest_streaming.py --dataset mortar --building bldg1")
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
