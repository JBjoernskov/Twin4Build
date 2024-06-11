from .valve import Valve
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.constants import Constants
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.flow.flow import Flow
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing


class ValvePumpFMUSystem(FMUComponent, Valve):
    def __init__(self,
                 mFlowValve_nominal=None,
                 mFlowPump_nominal=None,
                 dpCheckValve_nominal=None,
                 dpCoil_nominal=None,
                 dpPump=None,
                 dpValve_nominal=None,
                 dpSystem=None,
                 riseTime=None,
                **kwargs):
        super().__init__(self, **kwargs)
        self.start_time = 0
        fmu_filename = "valve_0wbypass_0full_0FMUmodel.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.mFlowValve_nominal = mFlowValve_nominal
        self.mFlowPump_nominal = mFlowPump_nominal
        self.dpCheckValve_nominal = dpCheckValve_nominal
        self.dpCoil_nominal = dpCoil_nominal

        self.dpPump = dpPump
        self.dpValve_nominal = dpValve_nominal
        self.dpSystem = dpSystem
        self.riseTime = riseTime 



        self.input = {"valvePosition": None}
        self.output = {"waterFlowRate": None,
                       "valvePosition": None}
        
        #Used in finite difference jacobian approximation for uncertainty analysis.
        self.inputLowerBounds = {"valvePosition": 0}
        self.inputUpperBounds = {"valvePosition": 1}

        self.FMUinputMap = {"valvePosition": "u"}
        self.FMUoutputMap = {"waterFlowRate": "m_flow"}
        self.FMUparameterMap = {"mFlowValve_nominal": "mFlowValve_nominal",
                                "flowCoefficient.hasValue": "Kv",
                                "mFlowPump_nominal": "mFlowPump_nominal",
                                "dpCheckValve_nominal": "dpCheckValve_nominal",
                                "dpCoil_nominal": "dpCoil_nominal",
                                "dpPump": "dpPump",
                                "dpValve_nominal": "dpValve_nominal",
                                "dpSystem": "dpSystem",
                                "riseTime": "riseTime"}
        
        self.input_conversion = {"valvePosition": do_nothing}
        self.output_conversion = {"waterFlowRate": do_nothing,
                                       "valvePosition": do_nothing}

        self.INITIALIZED = False
        self._config = {"parameters": list(self.FMUparameterMap.keys())}

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
        '''
            This function initializes the FMU component by setting the start_time and fmu_filename attributes, 
            and then sets the parameters for the FMU model.
        '''
        if self.INITIALIZED:
            self.reset()
        else:
            self.initialize_fmu()
            self.INITIALIZED = True
