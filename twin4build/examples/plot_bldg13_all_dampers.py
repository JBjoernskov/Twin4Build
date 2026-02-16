"""
Plot damper position for ALL rooms in bldg13 AHU01.

Loads damper position data directly from the database for all 12 rooms
and overlays them with AHU supply air temperature to investigate whether
damper control is centrally coordinated or individually zone-controlled.

If dampers across all rooms move in unison, it suggests central (AHU-level)
control or a shared driving signal (e.g., AHU SAT reset).
If dampers move independently, zone-level control is dominant.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from psycopg2.extras import RealDictCursor

# ==========================================================================
# CONFIGURATION
# ==========================================================================

db_config = {
    "table_name": "mortar_bldg13",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

timezone_local = ZoneInfo("America/Los_Angeles")

# Time range to plot (adjust as needed)
START_TIME = datetime(2016, 9, 22, 0, 0, tzinfo=timezone_local)
END_TIME = datetime(2016, 9, 23, 0, 0, tzinfo=timezone_local)

# Resample interval
RESAMPLE = "10min"

# Operating hours filter (only analyze data within these hours)
OPERATING_HOUR_START = 9   # 09:00
OPERATING_HOUR_END = 23    # 23:00

# ==========================================================================
# ROOM AND AHU SENSOR UUIDs
# ==========================================================================

ROOMS = {
    'RM01_1': {'damper_position': '338d4128-1174-4bd3-a46c-7e13fdf4d41a', 'floor': 0},
    'RM01_2': {'damper_position': 'ebe9c5eb-f2cc-41f7-9366-7f866d97884e', 'floor': 0},
    'RM11_1': {'damper_position': 'e98cba81-51b7-44b4-be8b-d724d8ba1d22', 'floor': 1},
    'RM11_2': {'damper_position': '51cf3b6c-aa59-44d2-aa60-9abfe2fd80c1', 'floor': 1},
    'RM11_3': {'damper_position': 'd7c7c96b-7fed-4005-87dc-17dcbdd88d5c', 'floor': 1},
    'RM12_1': {'damper_position': '23876ee1-0d47-405e-8dbc-f3336c0c42b0', 'floor': 1},
    'RM12_2': {'damper_position': '6e662022-da07-44e1-85b1-185eff60660c', 'floor': 1},
    'RM13_1': {'damper_position': 'f771ce69-4a68-40a2-85e3-d9593bbf0f20', 'floor': 1},
    'RM13_2': {'damper_position': '0be785d7-7d79-47d1-80bd-354dba9e7610', 'floor': 1},
    'RM21':   {'damper_position': '220e2d90-4d91-4399-84d9-96ccfd3f0a6b', 'floor': 2},
    'RM22':   {'damper_position': '7098433c-2287-4bf6-b09c-aec3d6644f7a', 'floor': 2},
    'RM31':   {'damper_position': '4a3e7cef-93d0-4888-9aed-9de72ce6c3b8', 'floor': 3},
}

AHU_SENSORS = {
    'AHU Supply Air Temp':     '77c50c34-1387-4ce4-a527-153fd143704e',
    'AHU SAT Setpoint':        'cee1c63a-54c2-490e-a7a4-7200fa93b270',
    'AHU Outside Air Temp':    '436a91fd-e4fe-486f-9a63-2e00303d6188',
    'AHU Cooling Valve Output':'61704369-134c-4022-bb21-ef2368c91eb1',
    'AHU Heating Valve Output':'2a800b32-492e-47f3-943a-db0aa985c316',
}

# ==========================================================================
# DATA LOADING
# ==========================================================================

def load_sensor(conn, uuid, start_time, end_time, resample="10min"):
    """Load a single sensor's data from the database, resample, and return a Series."""
    query = """
        SELECT time, value
        FROM {table}
        WHERE uuid = %s AND time >= %s AND time <= %s
        ORDER BY time
    """.format(table=db_config["table_name"])

    df = pd.read_sql(query, conn, params=(uuid, start_time, end_time))
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    series = df["value"].resample(resample).mean().interpolate(method="linear")
    return series


def main():
    conn_string = (
        f"host={db_config['db_host']} "
        f"port={db_config['db_port']} "
        f"dbname={db_config['db_name']} "
        f"user={db_config['db_user']} "
        f"password={db_config['db_password']}"
    )

    print(f"Connecting to database...")
    conn = psycopg2.connect(conn_string)

    # --- Load all damper positions ---
    print(f"\nLoading damper positions for {len(ROOMS)} rooms...")
    damper_data = {}
    for room_name, room_info in ROOMS.items():
        series = load_sensor(conn, room_info['damper_position'], START_TIME, END_TIME, RESAMPLE)
        if series is not None and len(series) > 0:
            damper_data[room_name] = series / 100.0  # percentage to fraction
            print(f"  {room_name}: {len(series)} points")
        else:
            print(f"  {room_name}: NO DATA")

    # --- Load AHU sensors ---
    print(f"\nLoading AHU sensors...")
    ahu_data = {}
    temp_transform = lambda x: (x - 32) * 5/9  # F to C
    for label, uuid in AHU_SENSORS.items():
        series = load_sensor(conn, uuid, START_TIME, END_TIME, RESAMPLE)
        if series is not None and len(series) > 0:
            if "Temp" in label:
                series = series.apply(temp_transform)
            elif "Valve" in label:
                series = series / 100.0
            ahu_data[label] = series
            print(f"  {label}: {len(series)} points")
        else:
            print(f"  {label}: NO DATA")

    conn.close()

    # --- Filter to operating hours only ---
    print(f"\nFiltering to operating hours {OPERATING_HOUR_START:02d}:00 – {OPERATING_HOUR_END:02d}:00...")
    for room_name in list(damper_data.keys()):
        s = damper_data[room_name]
        mask = (s.index.hour >= OPERATING_HOUR_START) & (s.index.hour < OPERATING_HOUR_END)
        damper_data[room_name] = s[mask]
    for label in list(ahu_data.keys()):
        s = ahu_data[label]
        mask = (s.index.hour >= OPERATING_HOUR_START) & (s.index.hour < OPERATING_HOUR_END)
        ahu_data[label] = s[mask]

    if not damper_data:
        print("\nERROR: No damper data loaded. Check database and UUIDs.")
        return

    # ======================================================================
    # PLOT 1: All damper positions on one chart + AHU SAT
    # ======================================================================
    print("\n--- Plot 1: All damper positions overlay ---")

    fig1, ax1 = plt.subplots(figsize=(18, 8))
    ax1.set_title("All Room Damper Positions (bldg13 AHU01)", fontsize=14)

    # Color by floor
    floor_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red'}
    for room_name, series in damper_data.items():
        floor = ROOMS[room_name]['floor']
        ax1.plot(series.index, series.values, linewidth=1, alpha=0.7,
                 color=floor_colors[floor], label=room_name)

    ax1.set_ylabel("Damper Position (0-1)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Overlay AHU SAT on right axis
    ax1r = ax1.twinx()
    if 'AHU Supply Air Temp' in ahu_data:
        ax1r.plot(ahu_data['AHU Supply Air Temp'].index, ahu_data['AHU Supply Air Temp'].values,
                  color='black', linewidth=2, linestyle='-', label="AHU Supply Air Temp")
    if 'AHU SAT Setpoint' in ahu_data:
        ax1r.plot(ahu_data['AHU SAT Setpoint'].index, ahu_data['AHU SAT Setpoint'].values,
                  color='black', linewidth=2, linestyle='--', label="AHU SAT Setpoint")
    ax1r.set_ylabel("Temperature (°C)")
    ax1r.legend(loc="upper right", fontsize=9)

    ax1.set_xlabel("Time")
    fig1.tight_layout()

    # ======================================================================
    # PLOT 2: Damper positions by floor (subplots)
    # ======================================================================
    print("--- Plot 2: Damper positions by floor ---")

    floors = sorted(set(r['floor'] for r in ROOMS.values()))
    fig2, axes2 = plt.subplots(len(floors), 1, figsize=(18, 4 * len(floors)), sharex=True)
    fig2.suptitle("Damper Positions by Floor (bldg13 AHU01)", fontsize=14)

    if len(floors) == 1:
        axes2 = [axes2]

    for ax, floor in zip(axes2, floors):
        floor_rooms = {k: v for k, v in damper_data.items() if ROOMS[k]['floor'] == floor}
        for room_name, series in floor_rooms.items():
            ax.plot(series.index, series.values, linewidth=1.2, alpha=0.8, label=room_name)

        ax.set_ylabel("Damper (0-1)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Floor {floor} ({len(floor_rooms)} rooms)", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)

        # AHU SAT on right axis
        axr = ax.twinx()
        if 'AHU Supply Air Temp' in ahu_data:
            axr.plot(ahu_data['AHU Supply Air Temp'].index, ahu_data['AHU Supply Air Temp'].values,
                     color='black', linewidth=1.5, linestyle='-', alpha=0.5, label="AHU SAT")
        axr.set_ylabel("°C")
        if floor == floors[0]:
            axr.legend(loc="upper right", fontsize=8)

    axes2[-1].set_xlabel("Time")
    fig2.tight_layout()

    # ======================================================================
    # PLOT 3: Pairwise correlation matrix
    # ======================================================================
    print("--- Plot 3: Pairwise correlation matrix ---")

    # Align all damper series to a common time index
    damper_df = pd.DataFrame(damper_data)
    damper_df = damper_df.dropna()

    if len(damper_df) > 5:
        corr_matrix = damper_df.corr()

        fig3, ax3 = plt.subplots(figsize=(10, 8))
        im = ax3.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax3.set_xticks(range(len(corr_matrix.columns)))
        ax3.set_yticks(range(len(corr_matrix.columns)))
        ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
        ax3.set_yticklabels(corr_matrix.columns, fontsize=9)

        # Annotate with correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                val = corr_matrix.iloc[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax3.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color)

        fig3.colorbar(im, ax=ax3, shrink=0.8, label="Pearson r")
        ax3.set_title("Pairwise Damper Position Correlation (bldg13 AHU01)", fontsize=13)
        fig3.tight_layout()
    else:
        print("  Not enough aligned data for correlation matrix")

    # ======================================================================
    # PLOT 4: Mean & spread of all dampers + AHU context
    # ======================================================================
    print("--- Plot 4: Mean damper + AHU context ---")

    if len(damper_df) > 5:
        damper_mean = damper_df.mean(axis=1)
        damper_std = damper_df.std(axis=1)
        damper_min = damper_df.min(axis=1)
        damper_max = damper_df.max(axis=1)

        fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        fig4.suptitle("Aggregate Damper Behavior & AHU Context (bldg13 AHU01)", fontsize=14)

        # Top: mean damper with min/max envelope
        ax4a.plot(damper_mean.index, damper_mean.values, linewidth=2, color='tab:blue', label="Mean Damper")
        ax4a.fill_between(damper_mean.index, damper_min.values, damper_max.values,
                          alpha=0.2, color='tab:blue', label="Min-Max Range")
        ax4a.fill_between(damper_mean.index, (damper_mean - damper_std).values,
                          (damper_mean + damper_std).values,
                          alpha=0.3, color='tab:blue', label="Mean +/- 1 Std")
        ax4a.set_ylabel("Damper Position (0-1)")
        ax4a.set_ylim(-0.05, 1.05)
        ax4a.legend(loc="upper left", fontsize=9)
        ax4a.grid(True, alpha=0.3)
        ax4a.set_title("Mean Damper Position with Spread Across All Rooms")

        # Overlay AHU SAT
        ax4a_r = ax4a.twinx()
        if 'AHU Supply Air Temp' in ahu_data:
            ax4a_r.plot(ahu_data['AHU Supply Air Temp'].index, ahu_data['AHU Supply Air Temp'].values,
                        color='black', linewidth=1.5, label="AHU SAT")
        if 'AHU SAT Setpoint' in ahu_data:
            ax4a_r.plot(ahu_data['AHU SAT Setpoint'].index, ahu_data['AHU SAT Setpoint'].values,
                        color='black', linewidth=1.5, linestyle='--', label="AHU SAT SP")
        ax4a_r.set_ylabel("Temperature (°C)")
        ax4a_r.legend(loc="upper right", fontsize=9)

        # Bottom: AHU valves + OAT
        if 'AHU Cooling Valve Output' in ahu_data:
            ax4b.plot(ahu_data['AHU Cooling Valve Output'].index,
                      ahu_data['AHU Cooling Valve Output'].values,
                      linewidth=1.2, color='tab:blue', label="Cooling Valve")
        if 'AHU Heating Valve Output' in ahu_data:
            ax4b.plot(ahu_data['AHU Heating Valve Output'].index,
                      ahu_data['AHU Heating Valve Output'].values,
                      linewidth=1.2, color='tab:red', label="Heating Valve")
        ax4b.set_ylabel("Valve Position (0-1)")
        ax4b.set_ylim(-0.05, 1.05)
        ax4b.legend(loc="upper left", fontsize=9)
        ax4b.grid(True, alpha=0.3)
        ax4b.set_title("AHU Valve Positions & Outside Air Temperature")

        ax4b_r = ax4b.twinx()
        if 'AHU Outside Air Temp' in ahu_data:
            ax4b_r.plot(ahu_data['AHU Outside Air Temp'].index,
                        ahu_data['AHU Outside Air Temp'].values,
                        color='tab:green', linewidth=1.5, label="Outside Air Temp")
        ax4b_r.set_ylabel("Temperature (°C)")
        ax4b_r.legend(loc="upper right", fontsize=9)

        ax4b.set_xlabel("Time")
        fig4.tight_layout()

    # ======================================================================
    # PRINT SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("DAMPER CORRELATION SUMMARY")
    print("="*70)

    if len(damper_df) > 5:
        # Off-diagonal correlations
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        off_diag = corr_matrix.values[mask]

        print(f"\n  Rooms loaded:       {len(damper_data)} / {len(ROOMS)}")
        print(f"  Aligned timesteps:  {len(damper_df)}")
        print(f"\n  Pairwise damper correlations (off-diagonal):")
        print(f"    Mean:   {np.mean(off_diag):+.4f}")
        print(f"    Median: {np.median(off_diag):+.4f}")
        print(f"    Min:    {np.min(off_diag):+.4f}")
        print(f"    Max:    {np.max(off_diag):+.4f}")
        print(f"    Std:    {np.std(off_diag):.4f}")

        n_high = np.sum(off_diag > 0.7)
        n_total = len(off_diag)
        print(f"\n  Pairs with r > 0.7: {n_high} / {n_total} ({100*n_high/n_total:.1f}%)")

        if np.mean(off_diag) > 0.7:
            print("\n  --> STRONG inter-room correlation: dampers move in unison.")
            print("      This strongly suggests a shared driving signal (AHU SAT reset)")
            print("      rather than independent zone-level control.")
        elif np.mean(off_diag) > 0.4:
            print("\n  --> MODERATE inter-room correlation: partially synchronized.")
            print("      Some shared AHU influence, but individual zones also differ.")
        else:
            print("\n  --> WEAK inter-room correlation: dampers act independently.")
            print("      Zone-level control is dominant.")

        # Correlation of mean damper with AHU SAT
        if 'AHU Supply Air Temp' in ahu_data:
            ahu_aligned = ahu_data['AHU Supply Air Temp'].reindex(damper_mean.index).interpolate()
            valid = ~(damper_mean.isna() | ahu_aligned.isna())
            if valid.sum() > 5:
                r_mean_sat = np.corrcoef(damper_mean[valid], ahu_aligned[valid])[0, 1]
                print(f"\n  Mean damper vs AHU SAT:  r = {r_mean_sat:+.4f}")

    plt.show()


if __name__ == "__main__":
    main()
