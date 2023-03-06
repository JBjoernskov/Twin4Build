from twin4build.saref4syst.system import System
from twin4build.utils.uppath import uppath
import pickle
import numpy as np
import os
from twin4build.utils.data_loaders.load_from_file import load_from_file
import datetime
import pandas as pd
from twin4build.utils.preprocessing.data_collection import DataCollection


class TimeSeriesInput(System):
    """
    This component models a generic dynamic input based on prescribed time series data. 
    It extracts and samples the second column of a csv file given by "filename".
    """
    def __init__(self,
                filename=None,
                **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        
    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        format = "%m/%d/%Y %I:%M:%S %p"
        df = load_from_file(filename=self.filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)
        data_collection = DataCollection(name=self.id, df=df)
        data_collection.interpolate_nans()
        df = data_collection.get_dataframe()
        self.database = df.iloc[:,1]
        self.stepSizeIndex = 0

        assert np.any(np.isnan(self.database))==False, f"Loaded data for TimeSeriesInput object {self.id} contains NaN values."

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        key = list(self.output.keys())[0]
        self.output[key] = self.database[self.stepSizeIndex]
        self.stepSizeIndex += 1