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
    """
    def __init__(self,
                startPeriod=None,
                endPeriod=None,
                stepSize=None,
                **kwargs):
        super().__init__(**kwargs)
        
        self.database = {}
        # file_path = os.path.join(uppath(__file__, 2), "test", "data", "outdoor_air_temperature.pickle")
        # filehandler = open(file_path, 'rb')
        # data_dict = pickle.load(filehandler)
        # self.database["outdoorTemperature"] = data_dict["value"]

        # file_path = os.path.join(uppath(os.path.abspath(__file__), 2), "test", "data", "shortwave_radiation.pickle")
        # filehandler = open(file_path, 'rb')
        # data_dict = pickle.load(filehandler)
        # self.database["shortwaveRadiation"] = data_dict["value"]

        # file_path = os.path.join(uppath(os.path.abspath(__file__), 2), "test", "data", "longwave_radiation.pickle")
        # filehandler = open(file_path, 'rb')
        # data_dict = pickle.load(filehandler)
        # self.database["longwaveRadiation"] = data_dict["value"]

        # time = data_dict["time"]

        


        format = "%m/%d/%Y %I:%M:%S %p"

        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "weather_DMI.csv")
        df_weather_DMI = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200) #From 
        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "weather_BMS.csv")
        df_weather_BMS = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200) #From local weather station at building roof
        df_weather_BMS["outdoorTemperature"] = (df_weather_BMS["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius
        df_input = pd.DataFrame()
        df_input.insert(0, "Time", df_weather_BMS["Time stamp"])
        df_input.insert(1, "outdoorTemperature", df_weather_BMS["outdoorTemperature"])
        df_input.insert(2, "globalIrradiation", df_weather_DMI["globalIrradiation"])

        data_collection = DataCollection("outdoor_environment", df_input)
        data_collection.interpolate_nans()

        time = data_collection.time
        self.database["outdoorTemperature"] = data_collection.clean_data_dict["outdoorTemperature"]
        self.database["shortwaveRadiation"] = data_collection.clean_data_dict["globalIrradiation"]

        self.stepSizeIndex = 0
        
    def initialize(self):
        pass

    def do_step(self, time=None, stepSize=None):
        self.output["outdoorTemperature"] = self.database["outdoorTemperature"][self.stepSizeIndex]
        self.output["shortwaveRadiation"] = self.database["shortwaveRadiation"][self.stepSizeIndex]
        # self.output["longwaveRadiation"] = self.database["longwaveRadiation"][self.stepSizeIndex]
        self.stepSizeIndex += 1