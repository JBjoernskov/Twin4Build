# Standard library imports
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import pandas as pd

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.data_loaders.load import load_from_database, load_from_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir


class TimeSeriesInputSystem(core.System):
    """A system for reading and processing time series data from files or DataFrames.

    This component provides functionality to handle time series data inputs, either from
    CSV files or pandas DataFrames. It supports automatic file path resolution and
    caching of processed data for improved performance.
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
            df (Optional[pd.DataFrame]): Input dataframe containing time series data. Must have datetime index and value column.
            filename (Optional[str]): Path to the CSV file. Can be absolute or relative to cache_root. If relative, will try both current directory and cache_root.
            datecolumn (int): Index of the date column (0-based). Defaults to 0.
            valuecolumn (int): Index of the value column (0-based). Defaults to 1.
            useSpreadsheet (bool, optional): Whether to use a spreadsheet for input.
                Defaults to False.
            useDatabase (bool, optional): Whether to use a database for input.
                Defaults to False.
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

        # Store attributes as private variables
        self._df = df
        self._useSpreadsheet = useSpreadsheet
        self._useDatabase = useDatabase
        self._filename = filename
        self._datecolumn = datecolumn
        self._valuecolumn = valuecolumn
        self._uuid = uuid
        self._name = name
        self._dbconfig = dbconfig
        self._cached_initialize_arguments = None
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
            "parameters": {},
            "spreadsheet": {
                "filename": self.filename,
                "datecolumn": self.datecolumn,
                "valuecolumn": self.valuecolumn,
            },
            "database": {
                "uuid": self.uuid,
                "name": self.name,
                "dbconfig": self.dbconfig,
            },
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
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """
        Initialize the TimeSeriesInputSystem.

        Args:
            startTime (datetime.datetime): Start time for the simulation.
            endTime (datetime.datetime): End time for the simulation.
            stepSize (int): Step size for the simulation.
            simulator (core.Simulator): Simulator to be used for initialization.
        """
        if self.df is None or (
            self._cached_initialize_arguments != (startTime, endTime, stepSize)
            and self._cached_initialize_arguments is not None
        ):
            if self.useSpreadsheet:
                self.df = load_from_spreadsheet(
                    self.filename,
                    self.datecolumn,
                    self.valuecolumn,
                    stepSize=stepSize,
                    start_time=startTime,
                    end_time=endTime,
                    cache_root=self._cache_root,
                )
            elif self.useDatabase:
                self.df = load_from_database(
                    config=self.dbconfig,
                    sensor_uuid=self.uuid,
                    sensor_name=self.name,
                    stepSize=stepSize,
                    start_time=startTime,
                    end_time=endTime,
                    cache_root=self._cache_root,
                )

        self._cached_initialize_arguments = (startTime, endTime, stepSize)

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
        simulator: Optional[core.Simulator] = None,
    ) -> None:
        """
        Perform a single timestep for the TimeSeriesInputSystem.

        Args:
            secondTime (int, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation time as a datetime object.
            stepSize (int, optional): Step size for the simulation.
        """
        self.output["value"].set(self.df.values[stepIndex], stepIndex)
