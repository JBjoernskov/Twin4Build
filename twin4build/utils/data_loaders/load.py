# Standard library imports
import configparser
import datetime
import hashlib
import os

# Third party imports
import numpy as np
import pandas as pd
import psycopg2
from dateutil.parser import parse
from dateutil.tz import gettz
from psycopg2.extras import RealDictCursor

# Local application imports
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.logger import LOGGER


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
    date_column=0,
    value_column=None,
    step_size=None,
    start_time=None,
    end_time=None,
    resample=True,
    resample_method="linear",
    clip=True,
    tz=datetime.timezone.utc,
    preserve_order=True,
):
    r"""
    Sample and process time series data from a DataFrame with various resampling options.
    
    This function processes time series data with support for resampling, timezone conversion,
    and data clipping. It handles both constant and linear resampling methods.

    Mathematical Formulation
    ------------------------

    **1. Time Series Resampling**

    Constant resampling -- for each time step :math:`t`:

    .. math::

       y(t) = y(t_{last})

    where :math:`t_{last}` is the last available data point before :math:`t`.

    Linear resampling -- for each time step :math:`t`:

    .. math::

       y(t) = y(t_1) + \frac{t - t_1}{t_2 - t_1} \cdot (y(t_2) - y(t_1))

    where :math:`t_1` is the last available data point before :math:`t` and
    :math:`t_2` is the first available data point after :math:`t`.

    **2. Time Zone Conversion**

    For a time :math:`t` in timezone :math:`TZ_1`:

    .. math::

       t_{TZ_2} = t_{TZ_1} + \Delta TZ

    where :math:`t_{TZ_2}` is the time in the target timezone and
    :math:`\Delta TZ` is the time difference between timezones.

    **3. Data Clipping**

    For a time series :math:`y(t)`:

       .. math::

          y_{clipped}(t) = \begin{cases}
          y(t) & \text{if } t_{start} \leq t < t_{end} \\
          \text{undefined} & \text{otherwise}
          \end{cases}

    Args:
        df (pandas.DataFrame): Input DataFrame with time series data
        datecolumn (int): Column index containing date_time information
        valuecolumn (int, optional): Column index containing values to process
        step_size (int, optional): Time step size in seconds for resampling
        start_time (date_time, optional): Start time for data extraction
        end_time (date_time, optional): End time for data extraction
        resample (bool): Whether to resample data to regular intervals
        resample_method (str): Resampling method ("linear" or "constant")
        clip (bool): Whether to clip data to specified time range
        tz (str): Timezone for data processing
        preserve_order (bool): Whether to preserve original data order

    Returns:
        pandas.DataFrame: Processed DataFrame with resampled time series data
    """
    assert date_column != value_column, "date_column and value_column cannot be the same"
    df = df.rename(columns={df.columns.to_list()[date_column]: "date_time"})

    LOGGER.task("Starting dataframe sampling")
    LOGGER.add_level()
    try:
        LOGGER.config("Start time: %s", start_time)
        LOGGER.config("End time: %s", end_time)
        LOGGER.config("Step size: %s", step_size)

        for i, column in enumerate(df.columns.to_list()):
            if column != "date_time" and value_column is None:
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                )  # Remove string entries
            elif i == value_column:
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                )  # Remove string entries

        df["date_time"] = pd.to_datetime(df["date_time"])  # ), format=format)
        if df["date_time"].apply(lambda x: x.tzinfo is not None).any():
            has_tz = True
            df["date_time"] = df["date_time"].apply(lambda x: x.tz_convert("UTC"))
            LOGGER.info(
                "Dataframe has timezone information %s, converting to UTC",
                df["date_time"].apply(lambda x: x.tzinfo).unique(),
            )

        else:
            has_tz = False
            LOGGER.info("Dataframe has no timezone information")

        df = df.set_index(pd.DatetimeIndex(df["date_time"]))
        df = df.drop(columns=["date_time"])

        if preserve_order and has_tz == False:
            # Detect if dates are reverse
            diff_seconds = df.index.to_series().diff().dt.total_seconds()
            # Remove NaN values from the calculation
            diff_seconds_valid = diff_seconds.dropna()
            if len(diff_seconds_valid) > 0:
                frac_neg = np.sum(diff_seconds_valid < 0) / len(diff_seconds_valid)
                if frac_neg >= 0.95:
                    df = df.iloc[::-1]
                elif frac_neg > 0.05 and frac_neg < 0.95:
                    raise Exception(
                        '"preserve_order" is true, but the date_time order cannot be determined.'
                    )
        else:
            df = df.sort_index()

        df = df.dropna(how="all")

        LOGGER.task("Enforcing timezone awareness")
        LOGGER.add_level()

        # Check if the first index is timezone aware
        if df.index[0].tzinfo is None:
            df = df.tz_localize(tz, ambiguous="infer", nonexistent="NaT")
            LOGGER.config("Localized index timezone: %s", tz)
        else:
            LOGGER.info(
                "Index has timezone %s, converting to %s",
                df.index[0].tzinfo,
                tz,
            )
            df = df.tz_convert(tz)
        LOGGER.remove_level()
           

        # Duplicate dates can occur either due to measuring/logging malfunctions
        # or due to change of daylight saving time where an hour occurs twice in fall.
        df = df.groupby(level=0).mean(numeric_only=True)

        if start_time.tzinfo is None:
            start_time = start_time.astimezone(tz=tz)
            LOGGER.info(
                "Start time had no timezone, converted to %s",
                tz,
            )
        if end_time.tzinfo is None:
            end_time = end_time.astimezone(tz=tz)
            LOGGER.info("End time had no timezone, converted to %s", tz)

        if resample:
            allowable_resample_methods = ["constant", "linear"]
            assert (
                resample_method in allowable_resample_methods
            ), f"resample_method \"{resample_method}\" is not valid. The options are: {', '.join(allowable_resample_methods)}"
            if resample_method == "constant":
                df = df.resample(f"{step_size}s", origin=start_time).ffill().bfill()
            elif resample_method == "linear":
                oidx = df.index
                nidx = pd.date_range(start_time, end_time, freq=f"{step_size}s")
                df = df.reindex(oidx.union(nidx)).interpolate("index").reindex(nidx)
                df = df.ffill().bfill()

        if clip:
            df = df[
                (df.index >= start_time) & (df.index < end_time)
            ]  # Exclude end time for similar behavior as normal python slicing

    except Exception:
        LOGGER.remove_level()
        LOGGER.error("Starting dataframe sampling", change_status=True)
        raise

    LOGGER.remove_level()
    LOGGER.ok("Starting dataframe sampling", change_status=True)
    return df


def load_from_spreadsheet(
    filename,
    date_column=0,
    value_column=None,
    step_size=None,
    start_time=None,
    end_time=None,
    resample=True,
    clip=True,
    cache=True,
    cache_root=None,
    preserve_order=True,
):
    """
    This function loads a spead either in .csv or .xlsx format.
    The date_time should in the first column - timezone-naive inputs are localized as "tz", while timezone-aware inputs are converted to "tz".
    All data except for date_time column is converted to numeric data.

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
        col_suffix = f"_col({value_column})" if value_column is not None else ""
        # The key MUST distinguish files that share a basename.  Two different
        # CSVs with the same name -- e.g. examples/estimator_example/ and
        # examples/full_workflow_example/ both ship a
        # "damper_position_sensor.csv" -- otherwise collide, and the second
        # one silently reads the first one's cached values.  That failure is
        # invisible: no error, no warning, just wrong data (it cost a
        # calibration debugging session -- the damper measurements loaded as
        # all-zeros and the estimator dutifully switched the ventilation
        # branch off to fit them).
        #
        # Include the resolved directory (hashed, to keep the name short and
        # filesystem-safe) and the source file's size + mtime, so that editing
        # or replacing a CSV in place also invalidates the entry.
        try:
            _stat = os.stat(filename)
            _stamp = f"{_stat.st_size}-{int(_stat.st_mtime)}"
        except OSError:  # unreadable/missing -- fall back to path only
            _stamp = "nostat"
        _dir_key = hashlib.sha1(
            os.path.normcase(os.path.abspath(os.path.dirname(filename))).encode(
                "utf-8", "replace"
            )
        ).hexdigest()[:12]
        cached_filename = f"name({os.path.basename(name)})_dir({_dir_key})_src({_stamp})_stepSize({str(step_size)})_start_time({startTime_str})_end_time({endTime_str}){col_suffix}_cached.pickle"
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

        if value_column is not None:
            valuename = df.columns[value_column]

        df = sample_from_df(
            df,
            date_column=date_column,
            value_column=value_column,
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            resample=resample,
            clip=clip,
            tz=start_time.tzinfo,
            preserve_order=preserve_order,
        )

        if value_column is not None:
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
            LOGGER.warning(
                "Could not load configuration from %s: %s.",
                config_file,
                e,
            )

    # Override with environment variables
    config["host"] = os.getenv("TIMESCALEDB_HOST", config["host"])
    config["port"] = int(os.getenv("TIMESCALEDB_PORT", str(config["port"])))
    config["name"] = os.getenv("TIMESCALEDB_NAME", config["name"])
    config["user"] = os.getenv("TIMESCALEDB_USER", config["user"])
    config["password"] = os.getenv("TIMESCALEDB_PASSWORD", config["password"])

    return config


def load_from_database(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step_size: int,
    resample=True,
    resample_method="linear",
    clip=True,
    cache=True,
    cache_root=None,
    preserve_order=True,
    table_name=None,
    sensor_id=None,
    time_column="time",
    id_column="uuid",
    value_column="value",
    db_host=None,
    db_port=None,
    db_name=None,
    db_user=None,
    db_password=None,
):
    r"""
    Load time series data from TimescaleDB database for building sensor data.

    This function connects to a TimescaleDB database and loads sensor data from tables.
    By default, the database schema should have columns: time (TIMESTAMPTZ), id (TEXT), 
    and value (FLOAT). However, column names are fully configurable via the
    time_column, id_column, and value_column parameters.

    Mathematical Formulation
    ------------------------

    The database query is formulated as:

    .. math::

        Q(t) = \begin{cases}
        \{y(t) : \text{id} = s_{id}\} & \text{if } s_{id} \text{ specified} \\
        \{y(t)\} & \text{otherwise (all sensors)}
        \end{cases}

    where:
       - :math:`Q(t)` is the query result at time :math:`t`
       - :math:`s_{id}` is the sensor ID filter
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
       table_name (str): Name of the table (e.g., "bldg1", "bldg10"). The function will query
           the table named `table_name`.
       sensor_id (str, optional): ID of the sensor to filter by. If provided, only data from sensors with
           this ID will be returned. If None, data from all sensors will be returned.
       step_size (int, optional): Time step size in seconds for resampling (e.g., 300 for 5-minute intervals).
           Required if resample=True. Ignored if resample=False.
       start_time (date_time, optional): Start time for data extraction. If timezone-naive, will be localized to 'tz'.
           If None, no lower time bound is applied.
       end_time (date_time, optional): End time for data extraction (exclusive). If timezone-naive, will be localized to 'tz'.
           If None, no upper time bound is applied.
       resample (bool, optional): Whether to resample the data to regular time intervals. If True, requires
           step_size, start_time, and end_time to be provided. Defaults to True.
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
       time_column (str, optional): Name of the timestamp column in the database table. Defaults to "time".
       id_column (str, optional): Name of the sensor ID column in the database table. Defaults to "id".
       value_column (str, optional): Name of the value column in the database table. Defaults to "value".
       db_host (str, optional): Database host address. Overrides config file and environment variables.
       db_port (int, optional): Database port number. Overrides config file and environment variables.
       db_name (str, optional): Database name. Overrides config file and environment variables.
       db_user (str, optional): Database username. Overrides config file and environment variables.
       db_password (str, optional): Database password. Overrides config file and environment variables.

    Returns:
       pandas.DataFrame: DataFrame with time series data. The index is a date_timeIndex with timezone
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

          from date_time import datetime, timezone
          start_time = date_time(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
          end_time = date_time(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
          df = load_from_database("bldg1", start_time=start_time, end_time=end_time, config_file="database.ini")

       Load specific sensor with explicit connection parameters:

       .. code-block:: python

          df = load_from_database(
              table_name="bldg1",
              sensor_id="sensor_12345",
              start_time=start_time,
              end_time=end_time,
              step_size=300,
              db_host="192.168.1.100",
              db_port=5433,
              db_user="myuser",
              db_password="mypassword"
          )

    Note:
       The function requires psycopg2 to be installed for PostgreSQL connectivity.
       Database schema should have columns: time, id, value (configurable).
       Timezone handling follows the same logic as load_from_spreadsheet.
       Caching uses the same mechanism as load_from_spreadsheet for consistency.
       For large datasets, consider using sensor_id filter to reduce
       memory usage and improve performance.
    """
    # Third party imports
    assert isinstance(
        start_time, datetime.datetime
    ), "start_time must be a datetime object"
    assert isinstance(end_time, datetime.datetime), "end_time must be a datetime object"
    assert isinstance(step_size, int), "step_size must be an integer"
    assert isinstance(resample, bool), "resample must be a boolean"
    assert isinstance(resample_method, str), "resample_method must be a string"
    assert isinstance(clip, bool), "clip must be a boolean"
    assert isinstance(cache, bool), "cache must be a boolean"
    assert isinstance(
        cache_root, (str, type(None))
    ), "cache_root must be a string or None"
    assert isinstance(preserve_order, bool), "preserve_order must be a boolean"
    assert isinstance(table_name, str), "table_name must be a string"
    assert isinstance(sensor_id, str), "sensor_id must be a string"
    assert isinstance(time_column, str), "time_column must be a string"
    assert isinstance(id_column, str), "id_column must be a string"
    assert isinstance(value_column, str), "value_column must be a string"
    assert isinstance(db_host, str), "db_host must be a string"
    assert isinstance(db_port, int), "db_port must be an integer"
    assert isinstance(db_name, str), "db_name must be a string"
    assert isinstance(db_user, str), "db_user must be a string"
    assert isinstance(db_password, str), "db_password must be a string"

    # Handle caching
    if cache:
        startTime_str = (
            start_time.strftime("%d-%m-%Y %H-%M-%S") if start_time else "None"
        )
        endTime_str = end_time.strftime("%d-%m-%Y %H-%M-%S") if end_time else "None"
        sensor_filter = f"sensor_{sensor_id}" if sensor_id else "all_sensors"
        cached_filename = f"db_{table_name}_{sensor_filter}_stepSize({str(step_size)})_start_time({startTime_str})_end_time({endTime_str})_cached.pickle"
        cached_filename, isfile = mkdir_in_root(
            folder_list=["generated_files", "cached_data"],
            filename=cached_filename,
            root=cache_root,
        )

        if os.path.isfile(cached_filename):
            return pd.read_pickle(cached_filename)

    # Build connection string
    if db_password:
        conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"
    else:
        conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user}"

    try:
        # Connect to database
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Build full table name with schema if provided

        # Check if table exists
        # if schema:
        #     cursor.execute(
        #         """
        #         SELECT EXISTS (
        #             SELECT FROM information_schema.tables
        #             WHERE table_schema = %s AND table_name = %s
        #         );
        #     """,
        #         (schema, table_name),
        #     )
        # else:
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

        # Get data for +/- 3 steps of buffer
        buffer = datetime.timedelta(seconds=3 * step_size)
        query_start_time = start_time - buffer
        query_end_time = end_time + buffer

        where_conditions.append(f"{id_column} = %s")
        params.append(sensor_id)

        where_conditions.append(f"{time_column} >= %s")
        params.append(query_start_time)

        where_conditions.append(f"{time_column} < %s")
        params.append(query_end_time)
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Execute query
        query = f"""
            SELECT {time_column}, {value_column} 
            FROM {table_name} 
            WHERE {where_clause}
            ORDER BY {time_column}
        """

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if len(rows) > 0:
            pass
        else:
            q = query % tuple(params)
            LOGGER.warning("No rows returned from query:\n%s.", q)
            # # Debug: Check what sensor IDs exist in the database
            # try:
            #     cursor.execute(
            #         f"SELECT DISTINCT {id_column} FROM {full_table_name} ORDER BY {id_column}"
            #     )
            #     existing_sensors = cursor.fetchall()
            #     print(
            #         f"Available sensor IDs in database: {[row[id_column] for row in existing_sensors]}"
            #     )
            # except Exception as e:
            #     print(f"Could not check existing sensors: {e}")

        if "conn" in locals():
            conn.close()

    except Exception as e:
        if "conn" in locals():
            conn.close()
        LOGGER.error("Error loading data from database: %s.", e)
        raise

    if not rows:
        LOGGER.warning("No data found for table %s.", table_name)
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Rename columns to standard names for sample_from_df
    df = df.rename(columns={time_column: "date_time", value_column: "value"})

    # Use the existing sample_from_df function for consistent processing
    # Pass valuecolumn to match the behavior of load_from_spreadsheet
    df = sample_from_df(
        df,
        date_column=0,  # date_time column
        value_column=1,  # value column (consistent with load_from_spreadsheet)
        step_size=step_size,
        start_time=start_time,
        end_time=end_time,
        resample=resample,
        resample_method=resample_method,
        clip=clip,
        tz=start_time.tzinfo,
        preserve_order=preserve_order,
    )

    df = df["value"]

    # Cache the result
    if cache:
        df.to_pickle(cached_filename)

    return df
