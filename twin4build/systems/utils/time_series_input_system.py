# Standard library imports
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import pandas as pd
import numpy as np

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.data_loaders.load import load_from_database, load_from_spreadsheet, sample_from_df
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
        useSpreadsheet: Whether to use a spreadsheet for input. Defaults to False.
        useDatabase: Whether to use a database for input. Defaults to False.
        uuid: UUID for database operations.
        name: Name for database operations.
        dbconfig: Database configuration parameters.
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        filename: Optional[str] = None,
        datecolumn: int = 0,
        valuecolumn: int = 1,
        useSpreadsheet: bool = False,
        useDatabase: bool = False,
        uuid: Optional[str] = None,
        name: Optional[str] = None,
        dbconfig: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the TimeSeriesInputSystem.

        Args:
            df: Input dataframe containing time series data. Must have date_time index and value column.
            filename: Path to the CSV file. Can be absolute or relative to cache_root. If relative, will try both current directory and cache_root.
            datecolumn: Index of the date column (0-based). Defaults to 0.
            valuecolumn: Index of the value column (0-based). Defaults to 1.
            useSpreadsheet: Whether to use a spreadsheet for input. Defaults to False.
            useDatabase: Whether to use a database for input. Defaults to False.
            uuid: UUID for database operations.
            name: Name for database operations.
            dbconfig: Database configuration parameters.
            **kwargs: Additional keyword arguments passed to parent System class.

        Raises:
            AssertionError: If neither df nor filename is provided.
            ValueError: If the specified file cannot be found in any of the search paths.
        """
        assert (
            useSpreadsheet == False or useDatabase == False
        ), "useSpreadsheet and useDatabase cannot both be True."
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
        self._useSpreadsheet = useSpreadsheet
        self._useDatabase = useDatabase
        self._filename = filename
        self._datecolumn = datecolumn
        self._valuecolumn = valuecolumn
        self._uuid = uuid
        self._name = name
        self._dbconfig = dbconfig
        self._cached_initialize_arguments = []
        self._cache_root = get_main_dir()

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
            "database": ["uuid", "name", "dbconfig"],
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
    def useSpreadsheet(self) -> bool:
        """
        Get whether to use a spreadsheet for input.
        """
        return self._useSpreadsheet

    @useSpreadsheet.setter
    def useSpreadsheet(self, value: bool) -> None:
        """
        Set whether to use a spreadsheet for input.
        """
        self._useSpreadsheet = value

    @property
    def useDatabase(self) -> bool:
        """
        Get whether to use a database for input.
        """
        return self._useDatabase

    @useDatabase.setter
    def useDatabase(self, value: bool) -> None:
        """
        Set whether to use a database for input.
        """
        self._useDatabase = value

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
    def name(self) -> Optional[str]:
        """
        Get the name for database operations.
        """
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        """
        Set the name for database operations.
        """
        self._name = value

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
        if len(self._cached_initialize_arguments)>0 and len(self._cached_initialize_arguments) == len(start_time): # Only check first element of tuple for length as all elements are the same length
            is_cached = all(start_time_==c[0] and end_time_==c[1] and step_size_==c[2] for start_time_, end_time_, step_size_, c in zip(start_time, end_time, step_size, self._cached_initialize_arguments))
        else:
            is_cached = False



        if is_cached==False:
            self.df = []
            self._cached_initialize_arguments = []
            for start_time_, end_time_, step_size_ in zip(start_time, end_time, step_size):
                # if (start_time_, end_time_, step_size_) not in self._cached_initialize_arguments:

                if self._df_init is None:
                    if self.useSpreadsheet:
                        df = load_from_spreadsheet(
                            self.filename,
                            self.datecolumn,
                            self.valuecolumn,
                            step_size=step_size_,
                            start_time=start_time_,
                            end_time=end_time_,
                            cache_root=self._cache_root,
                        )
                    elif self.useDatabase:
                        df = load_from_database(
                            config=self.dbconfig,
                            sensor_uuid=self.uuid,
                            sensor_name=self.name,
                            step_size=step_size_,
                            start_time=start_time_,
                            end_time=end_time_,
                            cache_root=self._cache_root,
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
                    

                self._cached_initialize_arguments.append((start_time_, end_time_, step_size_))
                self.df.append(df)

        _, _, n_timesteps = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        values = np.empty((len(self.df), n_timesteps))
        values.fill(np.nan)
        for batch_index, df in enumerate(self.df):
            size = len(df.index)
            values[batch_index,:size] = df.values[:,0]

        self.output["value"].initialize(
                n_timesteps,
                batch_size=len(start_time),
                values=values,
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
        self.output["value"].set(step_index=step_index)
