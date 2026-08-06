# Standard library imports
import datetime
import os
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
        cache: Whether to cache loaded/resampled data on disk for faster
            re-initialization. Defaults to True.
        transformation: Optional function applied to the loaded values
            (e.g. unit conversion). Defaults to None.
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        filename: Optional[str] = None,
        date_column: int = 0,
        value_column: int = 1,
        source: Optional[str] = None,
        use_spreadsheet: bool = False,
        use_database: bool = False,
        uuid: Optional[str] = None,
        dbconfig: Optional[Dict[str, Any]] = None,
        cache: Optional[bool] = True,
        transformation: Optional[callable] = None,
        **kwargs,
    ) -> None:
        """Initialize the TimeSeriesInputSystem.

        Args:
            df: Input dataframe containing time series data. Must have date_time index and value column.
            filename: Path to the CSV file. Can be absolute or relative to cache_root. If relative, will try both current directory and cache_root.
            date_column: Index of the date column (0-based). Defaults to 0.
            value_column: Index of the value column (0-based). Defaults to 1.
            source: Preferred data-source selector: ``"spreadsheet"``, ``"database"``,
                ``"df"``, or ``None`` for auto-detect.
            use_spreadsheet / use_database: Legacy boolean source flags (deprecated;
                use ``source=``; removed in 2.1).
            uuid: UUID for database operations.
            dbconfig: Database configuration parameters.
            cache: Whether to cache loaded/resampled data on disk for faster
                re-initialization. Defaults to True.
            transformation: Optional function applied to the loaded values
                (e.g. unit conversion). Defaults to None.
            **kwargs: Additional keyword arguments passed to parent System class.

        Raises:
            AssertionError: If neither df nor filename is provided.
            ValueError: If the specified file cannot be found in any of the search paths.
        """
        from twin4build.utils.deprecation import deprecate_args, deprecate_name

        for legacy_key, new_key in (
            ("useSpreadsheet", "use_spreadsheet"),
            ("useDatabase", "use_database"),
        ):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` has been removed. Use `{new_key}` instead."
                )

        legacy = deprecate_args(
            ["datecolumn", "valuecolumn"],
            ["date_column", "value_column"],
            [None, None],
            kwargs,
        )
        date_column = legacy.get("date_column", date_column)
        value_column = legacy.get("value_column", value_column)

        if source is not None:
            source = str(source).lower()
            assert source in {
                "spreadsheet",
                "database",
                "df",
            }, f"source must be 'spreadsheet', 'database', or 'df', got {source!r}"
            use_spreadsheet = source == "spreadsheet"
            use_database = source == "database"
        elif use_spreadsheet or use_database:
            deprecate_name(
                "use_spreadsheet/use_database",
                "source='spreadsheet'|'database'|'df'",
            )

        assert (
            use_spreadsheet == False or use_database == False
        ), "use_spreadsheet and use_database cannot both be True."
        super().__init__(**kwargs)
        assert (
            df is not None or filename is not None or uuid is not None
        ), f'Either "df" or "filename" or "uuid" must be provided as argument.'

        self._df_init = df
        self.df = []
        self._use_spreadsheet = use_spreadsheet
        self._use_database = use_database
        self._filename = filename
        self._date_column = date_column
        self._value_column = value_column
        self._uuid = uuid
        self._dbconfig = dbconfig
        self._cached_initialize_arguments = []
        self._cache_root = get_main_dir()
        self._cache = cache
        self._transformation = transformation

        # Define inputs and outputs as private variables
        self._input = {}
        self._output = {"value": tps.Scalar(is_leaf=True)}

        if filename is not None:
            if os.path.isfile(filename):  # Absolute or relative was provided
                self._filename = filename
            else:  # Check if relative path to root was provided
                filename = filename.lstrip("/\\")
                filename_ = os.path.join(self._cache_root, filename)
                if os.path.isfile(filename_) == False:
                    raise (
                        ValueError(
                            f'Neither one of the following filenames exist: \n"{filename}"\n{filename_}'
                        )
                    )
                self._filename = filename_

        self._config = {
            "parameters": [],
            "spreadsheet": ["filename", "date_column", "value_column"],
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
    def date_column(self) -> int:
        """
        Get the index of the date/time column (0-based).
        """
        return self._date_column

    @date_column.setter
    def date_column(self, value: int) -> None:
        """
        Set the index of the date/time column (0-based).
        """
        self._date_column = value

    @property
    def value_column(self) -> int:
        """
        Get the index of the value column (0-based).
        """
        return self._value_column

    @value_column.setter
    def value_column(self, value: int) -> None:
        """
        Set the index of the value column (0-based).
        """
        self._value_column = value

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

        # Fast path: the window is unchanged AND the (fixed) value tensor is
        # already built -> skip the expensive rebuild.  ``initialize`` is
        # otherwise re-run on every ``model.initialize`` (i.e. every estimator
        # objective / gradient / constraint evaluation), and even on a data
        # cache-hit it would rebuild the value array and rescan for NaNs --
        # ~0.05 s each, thousands of times, the dominant estimation cost.  The
        # time-series "state" is just the data, which is constant for a fixed
        # window, so skipping that rebuild is safe.  The output PORT must still
        # be re-initialized: this may run inside a functorch grad transform
        # (jacrev of the estimation objective), and ``do_step`` writes the port
        # tensor in place -- mutating a tensor captured from OUTSIDE the
        # transform is an error, so the (cheap) port init recreates it inside.
        if (
            is_cached
            and getattr(self, "values", None) is not None
            and getattr(self, "batch_size", None) == len(start_time)
        ):
            self.output["value"].initialize(
                n_t=self.n_timesteps,
                n_s=self.batch_size,
                n_c=1,
                values=self.values,
            )
            return

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
                            self._date_column,
                            self._value_column,
                            step_size=step_size_,
                            start_time=start_time_,
                            end_time=end_time_,
                            cache_root=self._cache_root,
                            cache=self._cache,
                        )
                    elif self.use_database:
                        df = load_from_database(
                            step_size=step_size_,
                            start_time=start_time_,
                            end_time=end_time_,
                            cache_root=self._cache_root,
                            cache=self._cache,
                            sensor_id=self.uuid,
                            **self.dbconfig,
                        )
                else:
                    df_ = self._df_init.copy()
                    df_.reset_index(inplace=True)
                    df = sample_from_df(
                        df_,
                        date_column=0,
                        value_column=1,
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
                if self._transformation is not None:
                    df = df.apply(self._transformation)
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

        nan_mask = np.isnan(values)
        if nan_mask.any():
            nan_count = int(nan_mask.sum())
            nan_pct = nan_count / values.size * 100
            batch_indices = np.where(nan_mask.any(axis=1))[0]
            raise AssertionError(
                f"Values contain {nan_count} NaN(s) ({nan_pct:.1f}%) in "
                f"TimeSeriesInput '{self.id}' "
                f"(file: {self.filename}, batch indices: {batch_indices.tolist()}). "
                f"Check the source data for missing values in the queried date range."
            )

        self.n_timesteps = max_timesteps
        self.batch_size = len(start_time)
        # Convert values from (n_s, n_t) to time-first (n_t, n_s, n_c) where n_c=1
        # First transpose to (n_t, n_s), then unsqueeze to (n_t, n_s, 1)
        self.values = torch.tensor(values, dtype=tps.float_dtype()).T.unsqueeze(
            -1
        )  # (n_t, n_s, 1)
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
