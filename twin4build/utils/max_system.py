from twin4build.saref4syst.system import System
import numpy as np
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir
import pandas as pd
from twin4build.utils.preprocessing.data_collection import DataCollection

class MaxSystem(System):
    """
    If value>=threshold set to on_value else set to off_value
    """
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        
        self.input = {}
        self.output = {"value": None}
    
    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["value"] = max(self.input.values())