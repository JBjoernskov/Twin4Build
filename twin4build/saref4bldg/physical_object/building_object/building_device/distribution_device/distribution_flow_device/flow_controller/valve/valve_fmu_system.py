from .valve import Valve
from typing import Union
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import os
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing


class ValveFMUSystem(FMUComponent, Valve):
    def __init__(self,
                 waterFlowRateMax=None,
                 dpFixed_nominal=None,
                **kwargs):
        Valve.__init__(self, **kwargs)
        self.start_time = 0
        fmu_filename = "Valve_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)
        self.waterFlowRateMax = waterFlowRateMax
        self.dpFixed_nominal = dpFixed_nominal

        self.input = {"valvePosition": None}
        self.output = {"waterFlowRate": None,
                       "valvePosition": None}
        
        #Used in finite difference jacobian approximation for uncertainty analysis.
        self.inputLowerBounds = {"valvePosition": 0}
        self.inputUpperBounds = {"valvePosition": 1}

        self.FMUinputMap = {"valvePosition": "u"}
        self.FMUoutputMap = {"waterFlowRate": "m_flow"}
        self.FMUparameterMap = {"waterFlowRateMax": "m_flow_nominal",
                                "flowCoefficient.hasValue": "Kv",
                                "dpFixed_nominal": "dpFixed_nominal"}
        
        self.input_unit_conversion = {"valvePosition": do_nothing}
        self.output_unit_conversion = {"waterFlowRate": do_nothing,
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
            FMUComponent.__init__(self, fmu_path=self.fmu_path, unzipdir=self.unzipdir)
            # Set self.INITIALIZED to True to call self.reset() for future calls to initialize().
            # This currently does not work with some FMUs, because the self.fmu.reset() function fails in some cases.
            self.INITIALIZED = True
