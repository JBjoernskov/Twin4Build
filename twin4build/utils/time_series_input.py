from twin4build.saref4syst.system import System
import os
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir
from pathlib import Path, PurePosixPath
from typing import Optional, Dict, List, Any, Union, Tuple
import pandas as pd
import datetime

class TimeSeriesInputSystem(System):
    """A system for reading and processing time series data from files or DataFrames.
    
    This component provides functionality to handle time series data inputs, either from
    CSV files or pandas DataFrames. It supports automatic file path resolution and
    caching of processed data for improved performance.

    Attributes:
        df (pd.DataFrame): Processed input data containing time series values.
        filename (str): Path to the input CSV file (absolute or relative to root).
        datecolumn (int): Index of the date/time column (0-based). Defaults to 0.
        valuecolumn (int): Index of the value column (0-based). Defaults to 1.
        cached_initialize_arguments (Tuple[datetime.datetime, datetime.datetime, float]): 
            Cached initialization parameters (startTime, endTime, stepSize).
        cache_root (str): Root directory for resolving relative paths and caching.
        physicalSystemReadings (pd.DataFrame): Processed and resampled time series data.
        stepIndex (int): Current step index in the time series.

    Example:
        ```python
        # Using a CSV file
        ts_system = TimeSeriesInputSystem(
            filename="data/temperatures.csv",
            datecolumn=0,
            valuecolumn=1
        )

        # Using a DataFrame
        ts_system = TimeSeriesInputSystem(
            df_input=existing_dataframe
        )
        ```
    """

    def __init__(self,
                df_input: Optional[pd.DataFrame] = None,
                filename: Optional[str] = None,
                datecolumn: int = 0,
                valuecolumn: int = 1,
                **kwargs) -> None:
        """Initialize the TimeSeriesInputSystem.

        Args:
            df_input (Optional[pd.DataFrame]): Input dataframe containing time series data.
                Must have datetime index and value column.
            filename (Optional[str]): Path to the CSV file. Can be absolute or relative
                to cache_root. If relative, will try both current directory and cache_root.
            datecolumn (int): Index of the date column (0-based). Defaults to 0.
            valuecolumn (int): Index of the value column (0-based). Defaults to 1.
            **kwargs: Additional keyword arguments passed to parent System class.

        Raises:
            AssertionError: If neither df_input nor filename is provided.
            ValueError: If the specified file cannot be found in any of the search paths.
        """
        super().__init__(**kwargs)
        assert df_input is not None or filename is not None, "Either \"df_input\" or \"filename\" must be provided as argument."
        self.df = df_input
        self.filename = filename
        self.cached_initialize_arguments = None
        self.cache_root = get_main_dir()
        

        if filename is not None:
            if os.path.isfile(filename): #Absolute or relative was provided
                self.filename = filename
            else: #Check if relative path to root was provided
                filename = filename.lstrip("/\\")
                filename_ = os.path.join(self.cache_root, filename)
                if os.path.isfile(filename_)==False:
                    raise(ValueError(f"Neither one of the following filenames exist: \n\"{filename}\"\n{filename_}"))
                self.filename = filename_
        self.datecolumn = datecolumn
        self.valuecolumn = valuecolumn
        self._config = {"parameters": {},
                        "readings": {"filename": self.filename,
                                     "datecolumn": self.datecolumn,
                                     "valuecolumn": self.valuecolumn}
                        }

    @property
    def config(self):
        """
        Get the configuration of the TimeSeriesInputSystem.

        Returns:
            dict: The configuration dictionary.
        """
        return self._config


    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        """
        Cache the initialization arguments.

        Args:
            startTime (int, optional): Start time for the simulation.
            endTime (int, optional): End time for the simulation.
            stepSize (int, optional): Step size for the simulation.
        """
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        """
        Initialize the TimeSeriesInputSystem.

        Args:
            startTime (int, optional): Start time for the simulation.
            endTime (int, optional): End time for the simulation.
            stepSize (int, optional): Step size for the simulation.
            model (Model, optional): Model to be used for initialization.
        """
        if self.df is None or (self.cached_initialize_arguments!=(startTime, endTime, stepSize) and self.cached_initialize_arguments is not None):
            self.df = load_spreadsheet(self.filename, self.datecolumn, self.valuecolumn, stepSize=stepSize, start_time=startTime, end_time=endTime, dt_limit=1200, cache_root=self.cache_root)
        self.physicalSystemReadings = self.df            
        self.stepIndex = 0
        self.cached_initialize_arguments = (startTime, endTime, stepSize)
        
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        """
        Perform a single timestep for the TimeSeriesInputSystem.

        Args:
            secondTime (int, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation time as a datetime object.
            stepSize (int, optional): Step size for the simulation.
        """
        key = list(self.output.keys())[0]
        self.output[key].set(self.physicalSystemReadings.values[self.stepIndex])
        self.stepIndex += 1
        
        