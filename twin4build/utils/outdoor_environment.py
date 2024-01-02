from twin4build.saref4syst.system import System
import numpy as np
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir
import pandas as pd
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

class OutdoorEnvironmentSystem(System):
    """
    This component represents the outdoor environment, i.e. outdoor temperature and global irraidation.
    Currently, it reads from 2 csv files containing weather data in the period 22-Nov-2021 to 02-Feb-2023.
    In the future, it should read from quantumLeap or a weather API. 
    """
    def __init__(self,
                 df_input=None,
                 filename=None,
                **kwargs):
        super().__init__(**kwargs)
        assert df_input is not None and filename is not None, "Either \"df_input\" or \"filename\" must be provided as argument."
        self.input = {}
        self.output = {"outdoorTemperature": None,
                       "globalIrradiation": None}
        self.df = df_input
        self.filename = filename
        self.cache_root = get_main_dir()
        # if df_input is not None:
            # data_collection = DataCollection(name="outdoor_environment", df=df_input, nan_interpolation_gap_limit=99999)
            # data_collection.interpolate_nans()
            # self.database = {}
            # self.database["outdoorTemperature"] = data_collection.clean_data_dict["outdoorTemperature"]
            # self.database["globalIrradiation"] = data_collection.clean_data_dict["globalIrradiation"]
            # nan_dates_outdoorTemperature = data_collection.time[np.isnan(self.database["outdoorTemperature"])]
            # nan_dates_globalIrradiation = data_collection.time[np.isnan(self.database["globalIrradiation"])]

            # if nan_dates_outdoorTemperature.size>0:
            #     message = f"outdoorTemperature data for OutdoorEnvironmentSystem object {self.id} contains NaN values at date {nan_dates_outdoorTemperature[0].strftime('%m/%d/%Y')}."
            #     logger.error(message)
            #     raise Exception(message)
            
            
            # if nan_dates_globalIrradiation.size>0:
            #     message = f"outdoorTemperature data for OutdoorEnvironmentSystem object {self.id} contains NaN values at date {nan_dates_globalIrradiation[0].strftime('%m/%d/%Y')}."
            #     logger.error(message)
            #     raise Exception(message)
    
    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):

        if self.df is None:
            self.df = load_spreadsheet(filename=self.filename, stepSize=stepSize, start_time=startTime, end_time=endTime, dt_limit=1200, cache_root=self.cache_root)
        
        required_keys = ["outdoorTemperature", "globalIrradiation"]
        is_included = np.array([key in np.array([self.df.columns]) for key in required_keys])
        assert np.all(is_included), f"The following required columns \"{', '.join(list(np.array(required_keys)[is_included==False]))}\" are not included in the provided weather file {self.filename}." 
        self.stepIndex = 0

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        # self.output["outdoorTemperature"] = self.database["outdoorTemperature"][self.stepIndex]
        # self.output["globalIrradiation"] = self.database["globalIrradiation"][self.stepIndex]
        self.output["outdoorTemperature"] = self.df["outdoorTemperature"][self.stepIndex]
        self.output["globalIrradiation"] = self.df["globalIrradiation"][self.stepIndex]
        self.stepIndex += 1