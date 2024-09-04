from twin4build.saref4syst.system import System
import numpy as np
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir
import pandas as pd
import os
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes, Optional
import warnings

def get_signature_pattern():
    node0 = Node(cls=base.OutdoorEnvironment, id="<n<SUB>1</SUB>(OutdoorEnvironment)>")
    sp = SignaturePattern(ownedBy="OutdoorEnvironmentSystem")
    sp.add_modeled_node(node0)
    return sp

class OutdoorEnvironmentSystem(base.OutdoorEnvironment, System):
    sp = [get_signature_pattern()]
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
        if df_input is None and filename is None:
            warnings.warn("Neither \"df_input\" nor \"filename\" was provided as argument. The component will not be able to provide any output.")
            
        self.input = {}
        self.output = {"outdoorTemperature": None,
                       "globalIrradiation": None,
                       "outdoorCo2Concentration": None}
        self.filename = filename
        self.df = df_input
        self.cached_initialize_arguments = None
        self.cache_root = get_main_dir()
        self._config = {"parameters": {},
                        "readings": {"filename": filename,
                                     "datecolumn": None,
                                     "valuecolumn": None}
                        }
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
                    stepSize=None,
                    model=None):
        
        if self.filename is not None:
            if os.path.isfile(self.filename)==False: #Absolute or relative was provided
                #Check if relative path to root was provided
                filename_ = os.path.join(self.cache_root, self.filename)
                if os.path.isfile(filename_)==False:
                    raise(ValueError(f"Neither one of the following filenames exist: \n\"{self.filename}\"\n{filename_}"))
                self.filename = filename_

        if self.df is None or (self.cached_initialize_arguments!=(startTime, endTime, stepSize) and self.cached_initialize_arguments is not None):
            self.df = load_spreadsheet(filename=self.filename, stepSize=stepSize, start_time=startTime, end_time=endTime, dt_limit=1200, cache_root=self.cache_root)
        self.cached_initialize_arguments = (startTime, endTime, stepSize)
        required_keys = ["outdoorTemperature", "globalIrradiation"]
        is_included = np.array([key in np.array([self.df.columns]) for key in required_keys])
        assert np.all(is_included), f"The following required columns \"{', '.join(list(np.array(required_keys)[is_included==False]))}\" are not included in the provided weather file {self.filename}." 
        self.stepIndex = 0

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["outdoorTemperature"] = self.df["outdoorTemperature"].iloc[self.stepIndex]
        self.output["globalIrradiation"] = self.df["globalIrradiation"].iloc[self.stepIndex]
        self.output["outdoorCo2Concentration"] = 400
        self.stepIndex += 1