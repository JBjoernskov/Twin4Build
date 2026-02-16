"""
Data extract for Mortar bldg13 RM11_1.

Extracts all sensor data related to room RM11_1 including:
- Room-level sensors (zone temp, setpoint, damper, reheat valve, airflow, etc.)
- AHU01-level sensors (supply air temp, valves, outside air temp, return air temp)

Each sensor's time series is saved as a CSV file named by its database UUID.
A metadata CSV (index.csv) maps UUIDs to human-readable descriptions.
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

# ==========================================================================
# CONFIGURATION
# ==========================================================================

db_config = {
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

TABLE_NAME = "mortar_bldg13"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data_extract_bldg13_RM11_1")

# ==========================================================================
# SENSOR UUID MAPPINGS
# ==========================================================================

# Room-level sensors for RM11_1
ROOM_SENSORS = {
    "RM11_1_zone_temp": "23a74139-b96e-4c1e-9c2f-d59f002aba5a",
    "RM11_1_zone_temp_setpoint": "5a0cfc00-d4be-4656-b034-ff4921c2966e",
    "RM11_1_damper_position": "e98cba81-51b7-44b4-be8b-d724d8ba1d22",
    "RM11_1_reheat_valve": "53c486ee-b6e1-48c8-866e-8a062d91fa82",
    "RM11_1_supply_air_flow": "be533cf0-ebc2-4529-8cb0-938f11a634d3",
    "RM11_1_percent_air_flow": "1dfb5487-dd0e-4397-98e1-91963655949a",
    "RM11_1_supply_air_temp": "17f64b73-33ba-432b-a91f-d19c9f4ab928",
}

# AHU01-level sensors (shared across all rooms)
AHU_SENSORS = {
    "AHU01_supply_air_temp": "77c50c34-1387-4ce4-a527-153fd143704e",
    "AHU01_supply_air_temp_setpoint": "cee1c63a-54c2-490e-a7a4-7200fa93b270",
    "AHU01_cooling_coil_valve": "40ba4519-46e9-4d8f-ad24-8a9ee9095c5d",
    "AHU01_cooling_valve_output": "61704369-134c-4022-bb21-ef2368c91eb1",
    "AHU01_heating_valve_output": "2a800b32-492e-47f3-943a-db0aa985c316",
    "AHU01_outside_air_temp": "436a91fd-e4fe-486f-9a63-2e00303d6188",
    "AHU01_return_air_temp": "53acd0c7-37e5-4cb7-93c0-a5e4283ad61f",
}

# Combine all sensors
ALL_SENSORS = {**ROOM_SENSORS, **AHU_SENSORS}


# ==========================================================================
# EXTRACTION
# ==========================================================================

def extract_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn_string = (
        f"host={db_config['db_host']} "
        f"port={db_config['db_port']} "
        f"dbname={db_config['db_name']} "
        f"user={db_config['db_user']} "
        f"password={db_config['db_password']}"
    )

    print(f"Connecting to database: {db_config['db_host']}:{db_config['db_port']}/{db_config['db_name']}")
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Verify table exists
    cursor.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);",
        (TABLE_NAME,),
    )
    if not cursor.fetchone()["exists"]:
        print(f"ERROR: Table '{TABLE_NAME}' does not exist in the database.")
        conn.close()
        sys.exit(1)

    index_rows = []  # For the metadata index file

    for description, uuid in ALL_SENSORS.items():
        print(f"  Extracting: {description:45s}  uuid={uuid} ... ", end="", flush=True)

        query = """
            SELECT time, value
            FROM {table}
            WHERE uuid = %s
            ORDER BY time
        """.format(table=TABLE_NAME)

        cursor.execute(query, (uuid,))
        rows = cursor.fetchall()

        if not rows:
            print(f"NO DATA")
            index_rows.append({
                "uuid": uuid,
                "description": description,
                "rows": 0,
                "time_min": None,
                "time_max": None,
            })
            continue

        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUTPUT_DIR, f"{uuid}.csv")
        df.to_csv(csv_path, index=False)

        print(f"{len(df)} rows  [{df['time'].min()} .. {df['time'].max()}]")

        index_rows.append({
            "uuid": uuid,
            "description": description,
            "rows": len(df),
            "time_min": str(df["time"].min()),
            "time_max": str(df["time"].max()),
        })

    conn.close()

    # Write metadata index
    index_df = pd.DataFrame(index_rows)
    index_path = os.path.join(OUTPUT_DIR, "index.csv")
    index_df.to_csv(index_path, index=False)
    print(f"\nMetadata index written to: {index_path}")
    print(f"All CSV files saved in:    {OUTPUT_DIR}")
    print(f"Total sensors extracted:   {len(ALL_SENSORS)}")


if __name__ == "__main__":
    extract_data()
