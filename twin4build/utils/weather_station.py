from posixpath import dirname
from twin4build.saref4syst.system import System
import pickle
import numpy as np
import os

class WeatherStation(System):
    def __init__(self,
                startPeriod = None,
                endPeriod = None,
                **kwargs):
        super().__init__(**kwargs)
        
        self.database = {}

        uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

        print(uppath(__file__, 2))

        filehandler = open(os.path.join(uppath(__file__, 2), "test", "data", "outdoor_air_temperature.pickle"), 'rb')
        data_dict = pickle.load(filehandler)
        self.database["outdoorTemperature"] = data_dict["value"]

        filehandler = open(os.path.join(uppath(__file__, 2), "test", "data", "shortwave_radation.pickle"), 'rb')
        data_dict = pickle.load(filehandler)
        self.database["shortwaveRadiation"] = data_dict["value"]

        filehandler = open(os.path.join(uppath(__file__, 2), "test", "data", "longwave_radiation.pickle"), 'rb')
        data_dict = pickle.load(filehandler)
        self.database["longwaveRadiation"] = data_dict["value"]


        minute_vec = np.vectorize(lambda x: x.minute)(data_dict["time"]) == startPeriod.minute
        hour_vec = np.vectorize(lambda x: x.hour)(data_dict["time"]) == startPeriod.hour
        day_vec = np.vectorize(lambda x: x.day)(data_dict["time"]) == startPeriod.day
        month_vec = np.vectorize(lambda x: x.month)(data_dict["time"]) == startPeriod.month
        year_vec = np.vectorize(lambda x: x.year)(data_dict["time"]) == startPeriod.year
        bool_vec_acc = np.ones((year_vec.shape[0]), dtype=np.bool)
        bool_vec_acc = np.logical_and(bool_vec_acc, minute_vec)
        bool_vec_acc = np.logical_and(bool_vec_acc, hour_vec)
        bool_vec_acc = np.logical_and(bool_vec_acc, day_vec)
        bool_vec_acc = np.logical_and(bool_vec_acc, month_vec)
        bool_vec_acc = np.logical_and(bool_vec_acc, year_vec)
        start_idx = np.where(bool_vec_acc)[0][0]

        minute_vec = np.vectorize(lambda x: x.minute)(data_dict["time"]) == endPeriod.minute
        hour_vec = np.vectorize(lambda x: x.hour)(data_dict["time"]) == endPeriod.hour
        day_vec = np.vectorize(lambda x: x.day)(data_dict["time"]) == endPeriod.day
        month_vec = np.vectorize(lambda x: x.month)(data_dict["time"]) == endPeriod.month
        year_vec = np.vectorize(lambda x: x.year)(data_dict["time"]) == endPeriod.year
        bool_vec_acc = np.ones((year_vec.shape[0]), dtype=np.bool)
        bool_vec_acc = np.logical_and(bool_vec_acc, minute_vec)
        bool_vec_acc = np.logical_and(bool_vec_acc, hour_vec)
        bool_vec_acc = np.logical_and(bool_vec_acc, day_vec)
        bool_vec_acc = np.logical_and(bool_vec_acc, month_vec)
        bool_vec_acc = np.logical_and(bool_vec_acc, year_vec)
        end_idx = np.where(bool_vec_acc)[0][0]

        for key in self.database:
            value = self.database[key]
            value = value[start_idx:end_idx]
            self.database[key] = value

        self.timeStepIndex = 0
        


    def update_output(self):
        self.output["outdoorTemperature"] = self.database["outdoorTemperature"][self.timeStepIndex]
        self.output["shortwaveRadiation"] = self.database["shortwaveRadiation"][self.timeStepIndex]
        self.output["longwaveRadiation"] = self.database["longwaveRadiation"][self.timeStepIndex]
        self.timeStepIndex += 1