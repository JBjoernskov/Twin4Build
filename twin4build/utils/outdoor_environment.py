from twin4build.saref4syst.system import System
from twin4build.utils.uppath import uppath
import pickle
import numpy as np
import os
from twin4build.utils.data_loaders.load_from_file import load_from_file
import datetime
import pandas as pd
from twin4build.utils.preprocessing.data_collection import DataCollection


class OutdoorEnvironment(System):
    """
    This component represents the outdoor environment, i.e. outdoor temperature and global irraidation.
    Currently, it reads from 2 csv files containing weather data in the period 22-Nov-2021 to 02-Feb-2023.
    In the future, it should read from quantumLeap or a weather API. 
    """
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        

        self.input = {}
        self.output = {"outdoorTemperature": None, 
                       "globalIrradiation": None}
        
    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        self.database = {}
        format = "%m/%d/%Y %I:%M:%S %p"

        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "weather_DMI.csv")
        df_weather_DMI = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200) #From 
        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "weather_BMS.csv")
        df_weather_BMS = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200) #From local weather station at building roof
        # df_weather_BMS["outdoorTemperature"] = (df_weather_BMS["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius
        df_input = pd.DataFrame()
        df_input.insert(0, "Time", df_weather_BMS["Time stamp"])
        df_input.insert(1, "outdoorTemperature", df_weather_BMS["outdoorTemperature"])
        df_input.insert(2, "globalIrradiation", df_weather_DMI["globalIrradiation"])

        data_collection = DataCollection(name="outdoor_environment", df=df_input, nan_interpolation_gap_limit=99999)
        data_collection.interpolate_nans()
        self.database["outdoorTemperature"] = data_collection.clean_data_dict["outdoorTemperature"]
        self.database["globalIrradiation"] = data_collection.clean_data_dict["globalIrradiation"]
        nan_dates_outdoorTemperature = data_collection.time[np.isnan(self.database["outdoorTemperature"])]
        nan_dates_globalIrradiation = data_collection.time[np.isnan(self.database["globalIrradiation"])]

        if nan_dates_outdoorTemperature.size>0:
            message = f"outdoorTemperature data for OutdoorEnvironment object {self.id} contains NaN values at date {nan_dates_outdoorTemperature[0].strftime('%m/%d/%Y')}."
            raise Exception(message)
        
        if nan_dates_globalIrradiation.size>0:
            message = f"outdoorTemperature data for OutdoorEnvironment object {self.id} contains NaN values at date {nan_dates_globalIrradiation[0].strftime('%m/%d/%Y')}."
            raise Exception(message)
        self.stepIndex = 0

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["outdoorTemperature"] = self.database["outdoorTemperature"][self.stepIndex]
        self.output["globalIrradiation"] = self.database["globalIrradiation"][self.stepIndex]
        self.stepIndex += 1