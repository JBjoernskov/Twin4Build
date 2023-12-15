from twin4build.saref4syst.system import System
import numpy as np
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir
import pandas as pd
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

class OnOffSystem(System):
    """
    If value>=threshold set to on_value else set to off_value
    """
    def __init__(self,
                 threshold=None,
                 is_on_value=None,
                 is_off_value=None,
                **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.is_off_value = is_off_value

        
        self.input = {"value": None,
                      "criteriaValue": None}
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
        if self.input["criteriaValue"]>=self.threshold:
            self.output["value"] = self.input["value"]
        else:
            self.output["value"] = self.is_off_value