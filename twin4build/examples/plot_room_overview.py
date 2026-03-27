"""
Plot overview of temperature, valve position, and damper position for all rooms
in the SensorSystem folder. Helps identify good candidate rooms for parameter
estimation and optimization by visualizing thermal dynamics.
"""

import os
import json
import datetime
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import tz

from twin4build.utils.data_loaders.load import load_from_spreadsheet

SENSOR_SYSTEM_DIR = os.path.join(os.path.dirname(__file__), "SensorSystem")
DATA_ROOT = r"C:\Users\jabj\Documents\python\Twin4build-Case-Studies\DP37\model\data"

TIMEZONE = tz.gettz("Europe/Copenhagen")
START_TIME = datetime.datetime(2023, 10, 1, 0, 0, 0, tzinfo=TIMEZONE)
END_TIME = datetime.datetime(2024, 2, 1, 0, 0, 0, tzinfo=TIMEZONE)
STEP_SIZE = 600

SENSOR_TYPES = ["temperature_sensor", "valve_position_sensor", "damper_position_sensor"]
SENSOR_LABELS = {
    "temperature_sensor": "Temperature",
    "valve_position_sensor": "Valve Position [%]",
    "damper_position_sensor": "Damper Position [%]",
}


def resolve_csv_path(json_path):
    relative = json_path.lstrip("/")
    if relative.startswith("data/"):
        relative = relative[len("data/"):]
    return os.path.join(DATA_ROOT, relative)


def load_sensor_csv(csv_path, datecolumn, valuecolumn, time_shift_hours=0):
    """Load CSV using load_from_spreadsheet with the datecolumn/valuecolumn from JSON."""
    df = load_from_spreadsheet(
        filename=csv_path, datecolumn=datecolumn, valuecolumn=valuecolumn,
        step_size=STEP_SIZE, start_time=START_TIME, end_time=END_TIME, cache=False,
    )
    if time_shift_hours != 0:
        df.index = df.index + pd.Timedelta(hours=time_shift_hours)
    return df

TEMP_SHIFT_HOURS = 0


def load_sensor_configs():
    rooms = defaultdict(dict)
    for fname in os.listdir(SENSOR_SYSTEM_DIR):
        if not fname.endswith(".json"):
            continue
        parts = fname.replace(".json", "").split("_", 1)
        if len(parts) != 2:
            continue
        room_id, sensor_type = parts[0], parts[1]
        if sensor_type not in SENSOR_TYPES:
            continue

        with open(os.path.join(SENSOR_SYSTEM_DIR, fname)) as f:
            config = json.load(f)

        csv_relpath = config["readings"].get("filename")
        if csv_relpath is None:
            continue

        csv_path = resolve_csv_path(csv_relpath)
        if not os.path.isfile(csv_path):
            print(f"  WARNING: CSV not found for {room_id}/{sensor_type}: {csv_path}")
            continue

        rooms[room_id][sensor_type] = {
            "csv_path": csv_path,
            "datecolumn": config["readings"]["datecolumn"],
            "valuecolumn": config["readings"]["valuecolumn"],
        }
    return rooms


def main():
    rooms = load_sensor_configs()

    complete_rooms = {
        rid: sensors
        for rid, sensors in rooms.items()
        if all(st in sensors for st in SENSOR_TYPES)
    }
    room_ids = sorted(complete_rooms.keys())

    if not room_ids:
        print("No rooms found with all three sensor types.")
        return

    print(f"Found {len(room_ids)} complete rooms: {', '.join(room_ids)}")

    n_rooms = len(room_ids)
    n_cols = 3
    n_rows = (n_rooms + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    colors = {"temperature_sensor": "tab:red", "valve_position_sensor": "tab:blue", "damper_position_sensor": "tab:green"}

    for i, room_id in enumerate(room_ids):
        ax = axes[i]
        ax_twin = ax.twinx()
        sensors = complete_rooms[room_id]
        for sensor_type in SENSOR_TYPES:
            info = sensors[sensor_type]
            try:
                shift = TEMP_SHIFT_HOURS if sensor_type == "temperature_sensor" else 0
                df = load_sensor_csv(
                    info["csv_path"],
                    info["datecolumn"],
                    info["valuecolumn"],
                    time_shift_hours=shift,
                )
                target_ax = ax if sensor_type == "temperature_sensor" else ax_twin
                target_ax.plot(
                    df.index, df.values,
                    linewidth=0.6, alpha=0.8,
                    color=colors[sensor_type],
                    label=SENSOR_LABELS[sensor_type],
                )
            except Exception as e:
                print(f"  ERROR loading {room_id}/{sensor_type}: {e}")

        ax.set_title(room_id, fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature", fontsize=9, color="tab:red")
        ax_twin.set_ylabel("Position [%]", fontsize=9, color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:red", labelsize=8)
        ax_twin.tick_params(axis="y", labelcolor="tab:blue", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        if i == 0:
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=7)

    for j in range(len(room_ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Room Thermal Dynamics Overview (Oct 2023 - Jan 2024)",
        fontsize=14, fontweight="bold",
    )
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "room_overview.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.show()
    print("Saved to room_overview.png")


if __name__ == "__main__":
    main()
