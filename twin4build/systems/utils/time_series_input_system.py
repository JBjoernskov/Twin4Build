# Standard library imports
import datetime
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.data_loaders.load import (
    load_from_database,
    load_from_spreadsheet,
    sample_from_df,
)
from twin4build.utils.deprecation import deprecate_args
from twin4build.utils.get_main_dir import get_main_dir


class TimeSeriesInputSystem(core.System):
    """A system for reading and processing time series data from files or DataFrames.

    This component provides functionality to handle time series data inputs, either from
    CSV files or pandas DataFrames. It supports automatic file path resolution and
    caching of processed data for improved performance.

    Args:
        df: Input dataframe containing time series data. Must have date_time index and value column.
        filename: Path to the CSV file. Can be absolute or relative to cache_root. If relative, will try both current directory and cache_root.
        datecolumn: Index of the date column (0-based). Defaults to 0.
        valuecolumn: Index of the value column (0-based). Defaults to 1.
        use_spreadsheet: Whether to use a spreadsheet for input. Defaults to False.
        use_database: Whether to use a database for input. Defaults to False.
        uuid: UUID for database operations.
        dbconfig: Database configuration parameters.
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        filename: Optional[str] = None,
        datecolumn: int = 0,
        valuecolumn: int = 1,
        use_spreadsheet: bool = False,
        use_database: bool = False,
        uuid: Optional[str] = None,
        dbconfig: Optional[Dict[str, Any]] = None,
        cache: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """Initialize the TimeSeriesInputSystem.

        Args:
            df: Input dataframe containing time series data. Must have date_time index and value column.
            filename: Path to the CSV file. Can be absolute or relative to cache_root. If relative, will try both current directory and cache_root.
            datecolumn: Index of the date column (0-based). Defaults to 0.
            valuecolumn: Index of the value column (0-based). Defaults to 1.
            use_spreadsheet: Whether to use a spreadsheet for input. Defaults to False.
            use_database: Whether to use a database for input. Defaults to False.
            uuid: UUID for database operations.
            dbconfig: Database configuration parameters.
            **kwargs: Additional keyword arguments passed to parent System class.

        Raises:
            AssertionError: If neither df nor filename is provided.
            ValueError: If the specified file cannot be found in any of the search paths.
        """
        # Handle deprecated camelCase arguments
        deprecated_args = ["useSpreadsheet", "useDatabase"]
        new_args = ["use_spreadsheet", "use_database"]
        positions = [None, None]
        value_map = deprecate_args(deprecated_args, new_args, positions, kwargs)
        use_spreadsheet = value_map.get("use_spreadsheet", use_spreadsheet)
        use_database = value_map.get("use_database", use_database)

        assert (
            use_spreadsheet == False or use_database == False
        ), "use_spreadsheet and use_database cannot both be True."
        super().__init__(**kwargs)
        assert (
            df is not None or filename is not None
        ), 'Either "df" or "filename" must be provided as argument.'

        # # Store attributes as private variables
        # if isinstance(df, pd.DataFrame):
        #     df = [df]
        # else:
        #     assert isinstance(df, [list, type(None)]), "df must be a pandas DataFrame or a list of pandas DataFrames"
        #     if df is None:
        #         df = []
        self._df_init = df
        self.df = []
        self._use_spreadsheet = use_spreadsheet
        self._use_database = use_database
        self._filename = filename
        self._datecolumn = datecolumn
        self._valuecolumn = valuecolumn
        self._uuid = uuid
        self._dbconfig = dbconfig
        self._cached_initialize_arguments = []
        self._cache_root = get_main_dir()
        self.cache = cache

        # Define inputs and outputs as private variables
        self._input = {}
        self._output = {"value": tps.Scalar(is_leaf=True)}

        if filename is not None:
            if os.path.isfile(filename):  # Absolute or relative was provided
                self._filename = filename
            else:  # Check if relative path to root was provided
                filename = filename.lstrip("/\\")
                filename_ = os.path.join(self.cache_root, filename)
                if os.path.isfile(filename_) == False:
                    raise (
                        ValueError(
                            f'Neither one of the following filenames exist: \n"{filename}"\n{filename_}'
                        )
                    )
                self._filename = filename_

        self._config = {
            "parameters": [],
            "spreadsheet": ["filename", "datecolumn", "valuecolumn"],
            "database": ["uuid", "dbconfig"],
        }

    @property
    def config(self):
        """
        Get the configuration of the TimeSeriesInputSystem.

        Returns:
            dict: The configuration dictionary.
        """
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the time series input system.

        Returns:
            dict: Dictionary containing input ports (empty for leaf systems)
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the time series input system.

        Returns:
            dict: Dictionary containing output ports:
                - "value": Time series values [units depend on data]
        """
        return self._output

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """
        Get the processed input data containing time series values.
        """
        return self._df

    @df.setter
    def df(self, value: Optional[pd.DataFrame]) -> None:
        """
        Set the processed input data containing time series values.
        """
        self._df = value

    @property
    def filename(self) -> Optional[str]:
        """
        Get the path to the input CSV file (absolute or relative to root).
        """
        return self._filename

    @filename.setter
    def filename(self, value: Optional[str]) -> None:
        """
        Set the path to the input CSV file (absolute or relative to root).
        """
        self._filename = value

    @property
    def datecolumn(self) -> int:
        """
        Get the index of the date/time column (0-based).
        """
        return self._datecolumn

    @datecolumn.setter
    def datecolumn(self, value: int) -> None:
        """
        Set the index of the date/time column (0-based).
        """
        self._datecolumn = value

    @property
    def valuecolumn(self) -> int:
        """
        Get the index of the value column (0-based).
        """
        return self._valuecolumn

    @valuecolumn.setter
    def valuecolumn(self, value: int) -> None:
        """
        Set the index of the value column (0-based).
        """
        self._valuecolumn = value

    @property
    def use_spreadsheet(self) -> bool:
        """
        Get whether to use a spreadsheet for input.
        """
        return self._use_spreadsheet

    @use_spreadsheet.setter
    def use_spreadsheet(self, value: bool) -> None:
        """
        Set whether to use a spreadsheet for input.
        """
        self._use_spreadsheet = value

    @property
    def use_database(self) -> bool:
        """
        Get whether to use a database for input.
        """
        return self._use_database

    @use_database.setter
    def use_database(self, value: bool) -> None:
        """
        Set whether to use a database for input.
        """
        self._use_database = value

    # ==================== Deprecated Properties (camelCase) ====================

    @property
    def useSpreadsheet(self) -> bool:
        """Deprecated: Use use_spreadsheet instead."""
        warnings.warn(
            "Property 'useSpreadsheet' is deprecated. Use 'use_spreadsheet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_spreadsheet

    @useSpreadsheet.setter
    def useSpreadsheet(self, value: bool) -> None:
        warnings.warn(
            "Property 'useSpreadsheet' is deprecated. Use 'use_spreadsheet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_spreadsheet = value

    @property
    def useDatabase(self) -> bool:
        """Deprecated: Use use_database instead."""
        warnings.warn(
            "Property 'useDatabase' is deprecated. Use 'use_database' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_database

    @useDatabase.setter
    def useDatabase(self, value: bool) -> None:
        warnings.warn(
            "Property 'useDatabase' is deprecated. Use 'use_database' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_database = value

    @property
    def uuid(self) -> Optional[str]:
        """
        Get the UUID for database operations.
        """
        return self._uuid

    @uuid.setter
    def uuid(self, value: Optional[str]) -> None:
        """
        Set the UUID for database operations.
        """
        self._uuid = value

    @property
    def dbconfig(self) -> Optional[Dict[str, Any]]:
        """
        Get the database configuration parameters.
        """
        return self._dbconfig

    @dbconfig.setter
    def dbconfig(self, value: Optional[Dict[str, Any]]) -> None:
        """
        Set the database configuration parameters.
        """
        self._dbconfig = value

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
    ) -> None:
        """
        Initialize the TimeSeriesInputSystem.

        Args:
            start_time (datetime.datetime): Start time for the simulation.
            end_time (datetime.datetime): End time for the simulation.
            step_size (int): Step size for the simulation.
        """
        #
        if len(self._cached_initialize_arguments) > 0 and len(
            self._cached_initialize_arguments
        ) == len(
            start_time
        ):  # Only check first element of tuple for length as all elements are the same length
            is_cached = all(
                start_time_ == c[0] and end_time_ == c[1] and step_size_ == c[2]
                for start_time_, end_time_, step_size_, c in zip(
                    start_time, end_time, step_size, self._cached_initialize_arguments
                )
            )
        else:
            is_cached = False

        if is_cached == False:
            self.df = []
            self._cached_initialize_arguments = []
            for start_time_, end_time_, step_size_ in zip(
                start_time, end_time, step_size
            ):
                # if (start_time_, end_time_, step_size_) not in self._cached_initialize_arguments:

                if self._df_init is None:
                    if self.use_spreadsheet:
                        df = load_from_spreadsheet(
                            self.filename,
                            self._datecolumn,
                            self._valuecolumn,
                            step_size=step_size_,
                            start_time=start_time_,
                            end_time=end_time_,
                            cache_root=self._cache_root,
                            cache=self.cache,
                        )
                    elif self.use_database:
                        df = load_from_database(
                            config=self.dbconfig,
                            sensor_id=self.uuid,
                            step_size=step_size_,
                            start_time=start_time_,
                            end_time=end_time_,
                            cache_root=self._cache_root,
                            cache=self.cache,
                        )
                else:
                    df_ = self._df_init.copy()
                    df_.reset_index(inplace=True)
                    df = sample_from_df(
                        df_,
                        datecolumn=0,
                        valuecolumn=1,
                        step_size=step_size_,
                        start_time=start_time_,
                        end_time=end_time_,
                    )
                    valuename = df.columns[
                        0
                    ]  # The value column is the first column as we set index to datecolumn
                    df = df[valuename]

                self._cached_initialize_arguments.append(
                    (start_time_, end_time_, step_size_)
                )
                self.df.append(df)

        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        values = np.empty((len(self.df), max_timesteps))
        values.fill(
            0
        )  # Before we used nan, but this caused issues with the optimizer when the optimizer tried to compute the gradient of the loss function.
        for batch_index, df in enumerate(self.df):
            size = len(df.index)
            # OLD: Only fill actual data, leave rest as 0
            # values[batch_index,:size] = df.values

            # NEW: Fill actual data, then forward-fill (repeat last value) for extended timesteps
            values[batch_index, :size] = df.values
            if size < max_timesteps:
                # Forward-fill: repeat the last value for extended timesteps
                values[batch_index, size:] = df.values[-1]

        assert not np.isnan(values).any(), "Values contain NaN."

        self.n_timesteps = max_timesteps
        self.batch_size = len(start_time)
        # Convert values from (n_s, n_t) to time-first (n_t, n_s, n_c) where n_c=1
        # First transpose to (n_t, n_s), then unsqueeze to (n_t, n_s, 1)
        self.values = torch.tensor(values, dtype=torch.float64).T.unsqueeze(-1)  # (n_t, n_s, 1)
        self.output["value"].initialize(
            n_t=max_timesteps,
            n_s=len(start_time),
            n_c=1,
            values=self.values,
        )

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
        simulator: Optional[core.Simulator] = None,
    ) -> None:
        """
        Perform a single timestep for the TimeSeriesInputSystem.

        Args:
            second_time (int, optional): Current simulation time in seconds.
            date_time (date_time, optional): Current simulation time as a date_time object.
            step_size (int, optional): Step size for the simulation.
        """
        self.output["value"]._set(i_t=step_index)
