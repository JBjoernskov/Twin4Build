from twin4build.saref4syst.system import System
import pickle
import numpy as np
import datetime

class WeatherStation(System):
    def __init__(self,
                startPeriod = None,
                endPeriod = None,
                **kwargs):
        super().__init__(**kwargs)
        
        self.database = {}

        filehandler = open("HVAC_data_dict_OU44_OAT.pickle", 'rb')
        data_dict = pickle.load(filehandler)
        self.database["outdoorTemperature"] = data_dict["value"]

        filehandler = open("HVAC_data_dict__sw_radiation.pickle", 'rb')
        data_dict = pickle.load(filehandler)
        self.database["directRadiation"] = data_dict["value"]

        filehandler = open("HVAC_data_dict__lw_radiation.pickle", 'rb')
        data_dict = pickle.load(filehandler)
        self.database["diffuseRadiation"] = data_dict["value"]


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
        self.output["directRadiation"] = self.database["directRadiation"][self.timeStepIndex]
        self.output["diffuseRadiation"] = self.database["diffuseRadiation"][self.timeStepIndex]
        self.timeStepIndex += 1