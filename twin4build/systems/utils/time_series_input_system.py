import twin4build.core as core
import os
from twin4build.utils.data_loaders.load import load_from_spreadsheet, load_from_database
from twin4build.utils.get_main_dir import get_main_dir
from typing import Optional, Dict, List, Any, Union, Tuple
import pandas as pd
import twin4build.utils.types as tps

class TimeSeriesInputSystem(core.System):
    """A system for reading and processing time series data from files or DataFrames.
    
    This component provides functionality to handle time series data inputs, either from
    CSV files or pandas DataFrames. It supports automatic file path resolution and
    caching of processed data for improved performance.

    Attributes:
        df (pd.DataFrame): Processed input data containing time series values.
        filename (str): Path to the input CSV file (absolute or relative to root).
        datecolumn (int): Index of the date/time column (0-based). Defaults to 0.
        valuecolumn (int): Index of the value column (0-based). Defaults to 1.
        cached_initialize_arguments (Tuple[datetime.datetime, datetime.datetime, float]): Cached initialization parameters (startTime, endTime, stepSize).
        cache_root (str): Root directory for resolving relative paths and caching.
        df (pd.DataFrame): Processed and resampled time series data.
        stepIndex (int): Current step index in the time series.
    """

    def __init__(self,
                df: Optional[pd.DataFrame] = None,
                filename: Optional[str] = None,
                datecolumn: int = 0,
                valuecolumn: int = 1,
                useSpreadsheet: bool = False,
                useDatabase: bool = False,
                uuid: Optional[str] = None,
                name: Optional[str] = None,
                dbconfig: Optional[Dict[str, Any]] = None,
                **kwargs) -> None:
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
        assert (useSpreadsheet==False or useDatabase==False), "useSpreadsheet and useDatabase cannot both be True."
        super().__init__(**kwargs)
        assert df is not None or filename is not None, "Either \"df\" or \"filename\" must be provided as argument."
        self.df = df
        self.useSpreadsheet = useSpreadsheet
        self.useDatabase = useDatabase
        self.filename = filename
        self.datecolumn = datecolumn
        self.valuecolumn = valuecolumn
        self.uuid = uuid
        self.name = name
        self.dbconfig = dbconfig
        self.cached_initialize_arguments = None
        self.cache_root = get_main_dir()

        self.input = {}
        self.output = {"value": tps.Scalar(is_leaf=True)}
        

        if filename is not None:
            if os.path.isfile(filename): #Absolute or relative was provided
                self.filename = filename
            else: #Check if relative path to root was provided
                filename = filename.lstrip("/\\")
                filename_ = os.path.join(self.cache_root, filename)
                if os.path.isfile(filename_)==False:
                    raise(ValueError(f"Neither one of the following filenames exist: \n\"{filename}\"\n{filename_}"))
                self.filename = filename_
        
        self._config = {"parameters": {},
                        "spreadsheet": {"filename": self.filename,
                                     "datecolumn": self.datecolumn,
                                     "valuecolumn": self.valuecolumn},
                        "database": {"uuid": self.uuid,
                                     "name": self.name,
                                     "dbconfig": self.dbconfig}}

    @property
    def config(self):
        """
        Get the configuration of the TimeSeriesInputSystem.

        Returns:
            dict: The configuration dictionary.
        """
        return self._config

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    simulator=None):
        """
        Initialize the TimeSeriesInputSystem.

        Args:
            startTime (int, optional): Start time for the simulation.
            endTime (int, optional): End time for the simulation.
            stepSize (int, optional): Step size for the simulation.
            model (Model, optional): Model to be used for initialization.
        """
        if self.df is None or (self.cached_initialize_arguments!=(startTime, endTime, stepSize) and self.cached_initialize_arguments is not None):
            if self.useSpreadsheet:
                self.df = load_from_spreadsheet(self.filename, self.datecolumn, self.valuecolumn, stepSize=stepSize, start_time=startTime, end_time=endTime, cache_root=self.cache_root)
            elif self.useDatabase:
                self.df = load_from_database(config=self.dbconfig, sensor_uuid=self.uuid, sensor_name=self.name, stepSize=stepSize, start_time=startTime, end_time=endTime, cache_root=self.cache_root)

        self.stepIndex = 0
        self.cached_initialize_arguments = (startTime, endTime, stepSize)
        
    def do_step(self, 
                secondTime=None, 
                dateTime=None, 
                stepSize=None,
                stepIndex: Optional[int] = None,
                simulator: Optional[core.Simulator] = None):
        """
        Perform a single timestep for the TimeSeriesInputSystem.

        Args:
            secondTime (int, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation time as a datetime object.
            stepSize (int, optional): Step size for the simulation.
        """
        self.output["value"].set(self.df.values[self.stepIndex], stepIndex)
        self.stepIndex += 1
        
        