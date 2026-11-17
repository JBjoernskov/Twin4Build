"""
Ingest Sacramento weather data (2009-2023) from CSV into the mortar_bldg1 PostgreSQL table.

The CSV has a 3-row metadata header, then columns:
  time, temperature_2m (°C), shortwave_radiation (W/m²)

Times in the CSV are in America/Los_Angeles (UTC-7, no DST adjustment in this file).
They are converted to UTC before insertion.

Sensor UUIDs used:
  outdoor_temperature  -> "sacramento_outdoor_temperature"
  solar_radiation      -> "sacramento_solar_radiation"
"""

from datetime import timezone, timedelta

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

CSV_PATH = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\sacramento_weather_2009_2023.csv"

DB_CONFIG = {
    "table_name": "mortar_bldg1",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

SENSOR_IDS = {
    "temperature_2m (°C)": ("sacramento_outdoor_temperature", "Outside_Air_Temperature_Sensor"),
    "shortwave_radiation (W/m²)": ("sacramento_solar_radiation", "Solar_Radiance_Sensor"),
}

# UTC offset from the CSV metadata header: utc_offset_seconds = -25200  (= -7 h)
UTC_OFFSET = timedelta(seconds=-25200)


def load_csv(path: str) -> pd.DataFrame:
    # Skip the 3-line metadata block (rows 0-2), then read data
    df = pd.read_csv(path, skiprows=3)
    # Parse timestamps as naive — the building sensor data in this database is stored
    # as local time (America/Los_Angeles) without proper UTC conversion, so we match
    # that convention here (no tz conversion).
    df["time"] = pd.to_datetime(df["time"])
    # Attach UTC label so psycopg2 stores them as TIMESTAMPTZ matching the DB convention
    df["time"] = df["time"].dt.tz_localize("UTC")
    return df


def build_rows(df: pd.DataFrame) -> list[tuple]:
    rows = []
    for col, (sensor_id, name) in SENSOR_IDS.items():
        subset = df[["time", col]].dropna()
        for ts, val in zip(subset["time"], subset[col]):
            rows.append((ts, sensor_id, name, float(val)))
    return rows


def ensure_table(cursor, table_name: str):
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            time  TIMESTAMPTZ NOT NULL,
            uuid  TEXT        NOT NULL,
            value DOUBLE PRECISION
        );
        """
    )


def ingest(rows: list[tuple], db_config: dict):
    table = db_config["table_name"]
    conn = psycopg2.connect(
        host=db_config["db_host"],
        port=db_config["db_port"],
        dbname=db_config["db_name"],
        user=db_config["db_user"],
        password=db_config["db_password"],
    )
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_table(cur, table)
                # Delete existing weather rows so re-runs are idempotent
                sensor_ids = [sid for sid, _name in SENSOR_IDS.values()]
                cur.execute(
                    f"DELETE FROM {table} WHERE uuid = ANY(%s)",
                    (sensor_ids,),
                )
                deleted = cur.rowcount
                print(f"Deleted {deleted} existing weather rows.")

                execute_values(
                    cur,
                    f"INSERT INTO {table} (time, uuid, name, value) VALUES %s",
                    rows,
                    page_size=10_000,
                )
                print(f"Inserted {len(rows)} rows into '{table}'.")
    finally:
        conn.close()


if __name__ == "__main__":
    print(f"Reading {CSV_PATH} ...")
    df = load_csv(CSV_PATH)
    print(f"  Loaded {len(df)} timesteps ({df['time'].iloc[0]} to {df['time'].iloc[-1]})")

    rows = build_rows(df)
    print(f"  Built {len(rows)} (time, uuid, value) rows for insertion.")

    ingest(rows, DB_CONFIG)
    print("Done.")
    print()
    print("Sensor IDs written to the database:")
    for col, (sid, name) in SENSOR_IDS.items():
        print(f"  uuid={sid!r}, name={name!r}  <-  {col}")
