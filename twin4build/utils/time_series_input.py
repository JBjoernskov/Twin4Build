from twin4build.saref4syst.system import System
import os
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.logger.Logging import Logging
from twin4build.utils.get_main_dir import get_main_dir
logger = Logging.get_logger("ai_logfile")

class TimeSeriesInputSystem(System):
    """
    This component models a generic dynamic input based on prescribed time series data.
    It extracts and samples the second column of a csv file given by "filename".
    """
    def __init__(self,
                df_input=None,
                filename=None,
                **kwargs):
        super().__init__(**kwargs)
        assert df_input is not None or filename is not None, "Either \"df_input\" or \"filename\" must be provided as argument."
        self.df = df_input
        logger.info("[Time Series Input] : Entered in Initialise Function")
        self.cached_initialize_arguments = None
        self.cache_root = get_main_dir()
        if os.path.isfile(filename): #Absolute or relative was provided
            self.filename = filename
        else: #Check if relative path to root was provided
            filename_ = os.path.join(self.cache_root, filename)
            if os.path.isfile(filename_)==False:
                raise(ValueError(f"Neither one of the following filenames exist: \n\"{filename}\"\n{filename_}"))
            self.filename = filename_
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        if self.df is None or self.cached_initialize_arguments!=(startTime, endTime, stepSize):
            self.df = load_spreadsheet(filename=self.filename, stepSize=stepSize, start_time=startTime, end_time=endTime, dt_limit=1200, cache_root=self.cache_root)
        self.physicalSystemReadings = self.df
            
        self.stepIndex = 0
        self.cached_initialize_arguments = (startTime, endTime, stepSize)
        logger.info("[Time Series Input] : Exited from Initialise Function")
        
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        key = list(self.output.keys())[0]
        self.output[key] = self.physicalSystemReadings.iloc[self.stepIndex, 0]
        self.stepIndex += 1
        
        