# Standard library imports
import configparser
import os

# Third party imports
import numpy as np
import pandas as pd
from dateutil.parser import parse
from dateutil.tz import gettz

# Local application imports
from twin4build.utils.mkdir_in_root import mkdir_in_root


def parseDateStr(s):
    if s != "":
        try:
            return np.datetime64(parse(s))
        except ValueError:
            return np.datetime64("NaT")
    else:
        return np.datetime64("NaT")


def sample_from_df(
    df,
    datecolumn=0,
    valuecolumn=None,
    stepSize=None,
    start_time=None,
    end_time=None,
    resample=True,
    resample_method="linear",
    clip=True,
    tz="Europe/Copenhagen",
    preserve_order=True,
):
    r"""
    Sample and process time series data from a DataFrame with various resampling options.
    
    This function processes time series data with support for resampling, timezone conversion,
    and data clipping. It handles both constant and linear resampling methods.

    Mathematical Formulation
    -----------------------

    1. Time Series Resampling:
       a) Constant Resampling:
          For each time step :math:`t`:

          .. math::

             y(t) = y(t_{last})

          where :math:`t_{last}` is the last available data point before :math:`t`

       b) Linear Resampling:
          For each time step :math:`t`:

          .. math::

             y(t) = y(t_1) + \frac{t - t_1}{t_2 - t_1} \cdot (y(t_2) - y(t_1))

          where:
          - :math:`t_1` is the last available data point before :math:`t`
          - :math:`t_2` is the first available data point after :math:`t`

    2. Time Zone Conversion:
       For a time :math:`t` in timezone :math:`TZ_1`:

       .. math::

          t_{TZ_2} = t_{TZ_1} + \Delta TZ

       where:
       - :math:`t_{TZ_2}` is the time in target timezone
       - :math:`\Delta TZ` is the time difference between timezones

    3. Data Clipping:
       For a time series :math:`y(t)`:

       .. math::

          y_{clipped}(t) = \begin{cases}
          y(t) & \text{if } t_{start} \leq t < t_{end} \\
          \text{undefined} & \text{otherwise}
          \end{cases}

    Args:
        df (pandas.DataFrame): Input DataFrame with time series data
        datecolumn (int): Column index containing datetime information
        valuecolumn (int, optional): Column index containing values to process
        stepSize (int, optional): Time step size in seconds for resampling
        start_time (datetime, optional): Start time for data extraction
        end_time (datetime, optional): End time for data extraction
        resample (bool): Whether to resample data to regular intervals
        resample_method (str): Resampling method ("linear" or "constant")
        clip (bool): Whether to clip data to specified time range
        tz (str): Timezone for data processing
        preserve_order (bool): Whether to preserve original data order

    Returns:
        pandas.DataFrame: Processed DataFrame with resampled time series data
    """
    assert datecolumn != valuecolumn, "datecolumn and valuecolumn cannot be the same"
    df = df.rename(columns={df.columns.to_list()[datecolumn]: "datetime"})

    for i, column in enumerate(df.columns.to_list()):
        if column != "datetime" and valuecolumn is None:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )  # Remove string entries
        elif i == valuecolumn:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )  # Remove string entries

    df["datetime"] = pd.to_datetime(df["datetime"])  # ), format=format)
    if df["datetime"].apply(lambda x: x.tzinfo is not None).any():
        has_tz = True
        df["datetime"] = df["datetime"].apply(lambda x: x.tz_convert("UTC"))
    else:
        has_tz = False

    df = df.set_index(pd.DatetimeIndex(df["datetime"]))
    df = df.drop(columns=["datetime"])

    if preserve_order and has_tz == False:
        # Detect if dates are reverse
        diff_seconds = df.index.to_series().diff().dt.total_seconds()
        frac_neg = np.sum(diff_seconds < 0) / diff_seconds.size
        if frac_neg >= 0.95:
            df = df.iloc[::-1]
        elif frac_neg > 0.05 and frac_neg < 0.95:
            raise Exception(
                '"preserve_order" is true, but the datetime order cannot be determined.'
            )
    else:
        df = df.sort_index()

    df = df.dropna(how="all")

    # Check if the first index is timezone aware
    if df.index[0].tzinfo is None:
        df = df.tz_localize(gettz(tz), ambiguous="infer", nonexistent="NaT")
    else:
        df = df.tz_convert(gettz(tz))

    # Duplicate dates can occur either due to measuring/logging malfunctions
    # or due to change of daylight saving time where an hour occurs twice in fall.
    df = df.groupby(level=0).mean()

    if start_time.tzinfo is None:
        start_time = start_time.astimezone(tz=gettz(tz))
    if end_time.tzinfo is None:
        end_time = end_time.astimezone(tz=gettz(tz))

    if resample:
        allowable_resample_methods = ["constant", "linear"]
        assert (
            resample_method in allowable_resample_methods
        ), f"resample_method \"{resample_method}\" is not valid. The options are: {', '.join(allowable_resample_methods)}"
        if resample_method == "constant":
            df = df.resample(f"{stepSize}s", origin=start_time).ffill().bfill()
        elif resample_method == "linear":
            oidx = df.index
            nidx = pd.date_range(start_time, end_time, freq=f"{stepSize}s")
            df = df.reindex(oidx.union(nidx)).interpolate("index").reindex(nidx)

    if clip:
        df = df[
            (df.index >= start_time) & (df.index < end_time)
        ]  # Exclude end time for similar behavior as normal python slicing

    return df


def load_from_spreadsheet(
    filename,
    datecolumn=0,
    valuecolumn=None,
    stepSize=None,
    start_time=None,
    end_time=None,
    resample=True,
    clip=True,
    cache=True,
    cache_root=None,
    tz="Europe/Copenhagen",
    preserve_order=True,
):
    """
    This function loads a spead either in .csv or .xlsx format.
    The datetime should in the first column - timezone-naive inputs are localized as "tz", while timezone-aware inputs are converted to "tz".
    All data except for datetime column is converted to numeric data.

    tz: can be "UTC+2", "GMT-8" (no trailing zeros) or timezone name "Europe/Copenhagen"

    preserve_order: If True, the order of rows in the spreadsheet are important in order to resolve DST when timezone information is not available

    PRINT THE FOLLOWING TO SEE AVAILABLE NAMES:
    from dateutil.zoneinfo import getzoneinfofile_stream, ZoneInfoFile
    print(ZoneInfoFile(getzoneinfofile_stream()).zones.keys())
    """
    name, file_extension = os.path.splitext(filename)

    if cache:
        # Check if file is cached
        startTime_str = start_time.strftime("%d-%m-%Y %H-%M-%S")
        endTime_str = end_time.strftime("%d-%m-%Y %H-%M-%S")
        cached_filename = f"name({os.path.basename(name)})_stepSize({str(stepSize)})_startTime({startTime_str})_endTime({endTime_str})_cached.pickle"
        cached_filename, isfile = mkdir_in_root(
            folder_list=["generated_files", "cached_data"],
            filename=cached_filename,
            root=cache_root,
        )
    if cache and os.path.isfile(cached_filename):
        df = pd.read_pickle(cached_filename)
    else:
        with open(filename, "rb") as filehandler:
            if file_extension == ".csv":
                df = pd.read_csv(filehandler, low_memory=False)  # , parse_dates=[0])
            elif file_extension == ".xlsx":
                df = pd.read_excel(filehandler)
            else:
                raise Exception(f"Invalid file extension: {file_extension}")

        if valuecolumn is not None:
            valuename = df.columns[valuecolumn]
        df = sample_from_df(
            df,
            datecolumn,
            stepSize=stepSize,
            start_time=start_time,
            end_time=end_time,
            resample=resample,
            clip=clip,
            tz=tz,
            preserve_order=preserve_order,
        )

        if valuecolumn is not None:
            df = df[valuename]

        if cache:
            df.to_pickle(cached_filename)

    return df


def load_database_config(config_file=None, section="timescaledb"):
    """
    Load TimescaleDB configuration from file or environment variables.

    This function loads database configuration from either:
    1. An INI configuration file (if provided)
    2. Environment variables (as fallback)

    Configuration Priority:
    1. Function parameters (highest priority)
    2. INI file configuration
    3. Environment variables (lowest priority)

    Args:
        config_file (str, optional): Path to INI configuration file. If None, only
            environment variables are used.
        section (str, optional): Section name in INI file. Defaults to "timescaledb".

    Returns:
        dict: Database configuration dictionary with keys:
            - host: Database host address
            - port: Database port number
            - name: Database name
            - user: Database username
            - password: Database password (None if not set)

    Environment Variables:
        The following environment variables can be used as fallbacks:
        - TIMESCALEDB_HOST: Database host (default: "localhost")
        - TIMESCALEDB_PORT: Database port (default: 5432)
        - TIMESCALEDB_NAME: Database name (default: "postgres")
        - TIMESCALEDB_USER: Database username (default: "postgres")
        - TIMESCALEDB_PASSWORD: Database password (default: None)

    Example INI file format:
        [timescaledb]
        host = localhost
        port = 5432
        name = postgres
        user = postgres
        password = mypassword

    Example:
        >>> config = load_database_config("database.ini")
        >>> print(config)
        {'host': 'localhost', 'port': 5432, 'name': 'postgres', 'user': 'postgres', 'password': 'mypassword'}
    """
    config = {
        "host": "localhost",
        "port": 5432,
        "name": "postgres",
        "user": "postgres",
        "password": None,
    }

    # Load from INI file if provided
    if config_file and os.path.exists(config_file):
        try:
            parser = configparser.ConfigParser()
            parser.read(config_file)

            if parser.has_section(section):
                if parser.has_option(section, "host"):
                    config["host"] = parser.get(section, "host")
                if parser.has_option(section, "port"):
                    config["port"] = parser.getint(section, "port")
                if parser.has_option(section, "name"):
                    config["name"] = parser.get(section, "name")
                if parser.has_option(section, "user"):
                    config["user"] = parser.get(section, "user")
                if parser.has_option(section, "password"):
                    config["password"] = parser.get(section, "password")
        except Exception as e:
            print(f"Warning: Could not load configuration from {config_file}: {e}")

    # Override with environment variables
    config["host"] = os.getenv("TIMESCALEDB_HOST", config["host"])
    config["port"] = int(os.getenv("TIMESCALEDB_PORT", str(config["port"])))
    config["name"] = os.getenv("TIMESCALEDB_NAME", config["name"])
    config["user"] = os.getenv("TIMESCALEDB_USER", config["user"])
    config["password"] = os.getenv("TIMESCALEDB_PASSWORD", config["password"])

    return config


def load_from_database(
    building_name,
    sensor_name=None,
    sensor_uuid=None,
    stepSize=None,
    start_time=None,
    end_time=None,
    resample=True,
    resample_method="linear",
    clip=True,
    cache=True,
    cache_root=None,
    tz="Europe/Copenhagen",
    preserve_order=True,
    config_file=None,
    section="timescaledb",
    db_host=None,
    db_port=None,
    db_name=None,
    db_user=None,
    db_password=None,
):
    r"""
    Load time series data from TimescaleDB database for building sensor data.

    This function connects to a TimescaleDB database and loads sensor data from tables
    with the naming convention `data_{building_name}`. The database schema should have
    columns: time (TIMESTAMPTZ), uuid (TEXT), name (TEXT), and value (FLOAT).

    Mathematical Formulation
    -----------------------

    The database query is formulated as:

    .. math::

        Q(t) = \begin{cases}
        \{y(t) : \text{name} = s_{name} \land \text{uuid} = s_{uuid}\} & \text{if } s_{name}, s_{uuid} \text{ specified} \\
        \{y(t) : \text{name} = s_{name}\} & \text{if only } s_{name} \text{ specified} \\
        \{y(t) : \text{uuid} = s_{uuid}\} & \text{if only } s_{uuid} \text{ specified} \\
        \{y(t)\} & \text{otherwise (all sensors)}
        \end{cases}

    where:
       - :math:`Q(t)` is the query result at time :math:`t`
       - :math:`s_{name}` is the sensor name filter
       - :math:`s_{uuid}` is the sensor UUID filter
       - :math:`y(t)` represents sensor values at time :math:`t`

    For multiple sensors, data is transformed from long to wide format:

    .. math::

        \mathbf{Y}(t) = \begin{bmatrix}
        y_1(t) & y_2(t) & \cdots & y_n(t)
        \end{bmatrix}

    where:
       - :math:`\mathbf{Y}(t)` is the wide-format data matrix at time :math:`t`
       - :math:`y_i(t)` is the value of sensor :math:`i` at time :math:`t`
       - :math:`n` is the number of sensors

    The function supports the same resampling and timezone conversion as the spreadsheet
    loader, following the mathematical formulations defined in the module docstring.

    Configuration
    -------------

    Database connection can be configured through multiple methods (in order of priority):
    1. Function parameters (highest priority)
    2. Configuration file (INI format)
    3. Environment variables (lowest priority)

    Args:
       building_name (str): Name of the building (e.g., "bldg1", "bldg10"). The function will query
           the table named `data_{building_name}`.
       sensor_name (str, optional): Name of the sensor to filter by (e.g., "temperature", "humidity", "CO2").
           If provided, only data from sensors with this name will be returned.
           If None, data from all sensors will be returned.
       sensor_uuid (str, optional): UUID of the sensor to filter by. If provided, only data from sensors with
           this UUID will be returned. Can be used alone or in combination with sensor_name.
       stepSize (int, optional): Time step size in seconds for resampling (e.g., 300 for 5-minute intervals).
           Required if resample=True. Ignored if resample=False.
       start_time (datetime, optional): Start time for data extraction. If timezone-naive, will be localized to 'tz'.
           If None, no lower time bound is applied.
       end_time (datetime, optional): End time for data extraction (exclusive). If timezone-naive, will be localized to 'tz'.
           If None, no upper time bound is applied.
       resample (bool, optional): Whether to resample the data to regular time intervals. If True, requires
           stepSize, start_time, and end_time to be provided. Defaults to True.
       resample_method (str, optional): Resampling method to use when resample=True:
           - "linear": Linear interpolation between data points
           - "constant": Forward-fill with backward-fill for gaps
           Defaults to "linear".
       clip (bool, optional): Whether to clip data to the specified start_time and end_time range.
           If False, all available data within the time range will be returned. Defaults to True.
       cache (bool, optional): Whether to cache the results in pickle files for faster subsequent loads.
           Cache files are stored in generated_files/cached_data/ directory. Defaults to True.
       cache_root (str, optional): Root directory for cache files. If None, uses the default Twin4Build cache location.
       tz (str, optional): Timezone for data processing. Can be timezone name (e.g., "Europe/Copenhagen"),
           UTC offset (e.g., "UTC+2", "GMT-8"), or "UTC". Defaults to "Europe/Copenhagen".
       config_file (str, optional): Path to INI configuration file for database settings.
           If provided, database connection parameters will be loaded from this file.
           See load_database_config() for file format details.
       section (str, optional): Section of the INI file to use for database settings.
       db_host (str, optional): Database host address. Overrides config file and environment variables.
       db_port (int, optional): Database port number. Overrides config file and environment variables.
       db_name (str, optional): Database name. Overrides config file and environment variables.
       db_user (str, optional): Database username. Overrides config file and environment variables.
       db_password (str, optional): Database password. Overrides config file and environment variables.

    Returns:
       pandas.DataFrame: DataFrame with time series data. The index is a DatetimeIndex with timezone
           information. Columns represent different sensors (if multiple sensors are
           selected) or a single column named after the sensor (if single sensor).

           For multiple sensors, the DataFrame will have a wide format with each
           sensor as a separate column. For a single sensor, the DataFrame will have
           a single column named after the sensor.

    Raises:
       Exception: If the specified table does not exist in the database, if database connection fails,
           if invalid resample_method is provided, or if required parameters are missing for resampling.

    Example:
       Basic usage with configuration file:

       .. code-block:: python

          from datetime import datetime, timezone
          start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
          end_time = datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
          df = load_from_database("bldg1", start_time=start_time, end_time=end_time, config_file="database.ini")

       Load specific sensor with explicit connection parameters:

       .. code-block:: python

          df = load_from_database(
              building_name="bldg1",
              sensor_name="temperature",
              start_time=start_time,
              end_time=end_time,
              stepSize=300,
              db_host="192.168.1.100",
              db_port=5433,
              db_user="myuser",
              db_password="mypassword"
          )

    Note:
       The function requires psycopg2 to be installed for PostgreSQL connectivity.
       Database tables should follow the naming convention: data_{building_name}.
       Database schema should have columns: time, uuid, name, value.
       Timezone handling follows the same logic as load_from_spreadsheet.
       Caching uses the same mechanism as load_from_spreadsheet for consistency.
       For large datasets, consider using sensor_name or sensor_uuid filters to reduce
       memory usage and improve performance.
    """
    # Third party imports
    import psycopg2
    from psycopg2.extras import RealDictCursor

    # Load database configuration
    db_config = load_database_config(config_file, section)

    # Override with function parameters if provided
    if db_host is not None:
        db_config["host"] = db_host
    if db_port is not None:
        db_config["port"] = db_port
    if db_name is not None:
        db_config["name"] = db_name
    if db_user is not None:
        db_config["user"] = db_user
    if db_password is not None:
        db_config["password"] = db_password

    # Handle caching
    if cache:
        startTime_str = (
            start_time.strftime("%d-%m-%Y %H-%M-%S") if start_time else "None"
        )
        endTime_str = end_time.strftime("%d-%m-%Y %H-%M-%S") if end_time else "None"
        sensor_filter = (
            f"sensor_{sensor_name}_{sensor_uuid}"
            if sensor_name or sensor_uuid
            else "all_sensors"
        )
        cached_filename = f"db_{building_name}_{sensor_filter}_stepSize({str(stepSize)})_startTime({startTime_str})_endTime({endTime_str})_cached.pickle"
        cached_filename, isfile = mkdir_in_root(
            folder_list=["generated_files", "cached_data"],
            filename=cached_filename,
            root=cache_root,
        )

        if os.path.isfile(cached_filename):
            return pd.read_pickle(cached_filename)

    # Build connection string
    if db_config["password"]:
        conn_string = f"host={db_config['host']} port={db_config['port']} dbname={db_config['name']} user={db_config['user']} password={db_config['password']}"
    else:
        conn_string = f"host={db_config['host']} port={db_config['port']} dbname={db_config['name']} user={db_config['user']}"

    try:
        # Connect to database
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Build query
        table_name = f"data_{building_name}"

        # Check if table exists
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """,
            (table_name,),
        )

        if not cursor.fetchone()["exists"]:
            raise Exception(f"Table {table_name} does not exist in the database")

        # Build WHERE clause
        where_conditions = []
        params = []

        if sensor_name:
            where_conditions.append("name = %s")
            params.append(sensor_name)

        if sensor_uuid:
            where_conditions.append("uuid = %s")
            params.append(sensor_uuid)

        if start_time:
            where_conditions.append("time >= %s")
            params.append(start_time)

        if end_time:
            where_conditions.append("time < %s")
            params.append(end_time)

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Execute query
        query = f"""
            SELECT time, uuid, name, value 
            FROM {table_name} 
            WHERE {where_clause}
            ORDER BY time
        """

        print(f"Executing query: {query}")
        print(f"Query parameters: {params}")

        cursor.execute(query, params)
        rows = cursor.fetchall()

        print(f"Query returned {len(rows)} rows")
        if len(rows) > 0:
            print(f"Sample row: {rows[0]}")
        else:
            print("No rows returned from query")
            # Debug: Check what sensor names exist in the database
            try:
                cursor.execute(f"SELECT DISTINCT name FROM {table_name} ORDER BY name")
                existing_sensors = cursor.fetchall()
                print(
                    f"Available sensors in database: {[row['name'] for row in existing_sensors]}"
                )
            except Exception as e:
                print(f"Could not check existing sensors: {e}")

        if "conn" in locals():
            conn.close()

    except Exception as e:
        if "conn" in locals():
            conn.close()
        print(f"Error loading data from database: {e}")
        raise

    if not rows:
        print(f"No data found for building {building_name}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Make the columns the sensor names (if there are multiple sensors)
    # This is necessary for the sample_from_df function as it expects all columns to be numeric for groupby operations
    df = df.pivot_table(index="time", columns="name", values="value", aggfunc="mean")
    df = df.reset_index()

    # Use the existing sample_from_df function for consistent processing
    df = sample_from_df(
        df,
        datecolumn=0,  # datetime is now the index
        valuecolumn=3,
        stepSize=stepSize,
        start_time=start_time,
        end_time=end_time,
        resample=resample,
        resample_method=resample_method,
        clip=clip,
        tz=tz,
        preserve_order=preserve_order,
    )

    # Cache the result
    if cache:
        df.to_pickle(cached_filename)

    return df
