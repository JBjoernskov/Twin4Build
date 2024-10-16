from twin4build.saref4syst.system import System
import os
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir
from pathlib import Path, PurePosixPath

class TimeSeriesInputSystem(System):
    """
    This component models a generic dynamic input based on prescribed time series data.
    It extracts and samples the second column of a csv file given by "filename".

    Attributes:
        df_input (pd.DataFrame): Input dataframe containing time series data.
        filename (str): Path to the CSV file containing time series data.
        datecolumn (int): Index of the date column in the CSV file.
        valuecolumn (int): Index of the value column in the CSV file.
        cached_initialize_arguments (tuple): Cached arguments from the last initialization.
        cache_root (str): Root directory for caching.
    """
    def __init__(self,
                df_input=None,
                filename=None,
                datecolumn=0,
                valuecolumn=1,
                **kwargs):
        """
        Initialize the TimeSeriesInputSystem.

        Args:
            df_input (pd.DataFrame, optional): Input dataframe containing time series data.
            filename (str, optional): Path to the CSV file containing time series data.
            datecolumn (int, optional): Index of the date column in the CSV file. Defaults to 0.
            valuecolumn (int, optional): Index of the value column in the CSV file. Defaults to 1.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If neither df_input nor filename is provided.
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
        
        