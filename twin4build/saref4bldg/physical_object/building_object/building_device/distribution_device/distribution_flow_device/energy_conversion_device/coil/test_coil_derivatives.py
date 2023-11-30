from .coil import Coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.fmu.fmu_component import FMUComponent
from twin4build.utils.constants import Constants
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.flow.flow import Flow
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing


class CoilSystem(FMUComponent, Coil):
    def __init__(self,
                **kwargs):
        Coil.__init__(self, **kwargs)
        self.start_time = 0
        # fmu_filename = "Coil.fmu"
        fmu_filename = "test_0coil_0derivatives.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        
        self.m1_flow_nominal = 1.5
        self.m2_flow_nominal = 10
        self.K = 1

        self.input = {"waterFlowRate": None,
                      "airFlowRate": None,
                      "inletWaterTemperature": None,
                      "inletAirTemperature": None}
        
        self.output = {"outletWaterTemperature": None, 
                       "outletAirTemperature": None}
        

        self.FMUinputMap = {"waterFlowRate": "inlet1.m_flow",
                        "airFlowRate": "inlet2.m_flow",
                        "inletWaterTemperature": "inlet1.forward.T",
                        "inletAirTemperature": "inlet2.forward.T"}
        
        self.FMUoutputMap = {"outletWaterTemperature": "outlet1.forward.T", 
                            "outletAirTemperature": "outlet2.forward.T"}

        self.FMUparameterMap = {"m1_flow_nominal": "m1_flow_nominal",
                                "m2_flow_nominal": "m2_flow_nominal",
                                "K": "K"}


        # self.FMUinput = {"waterFlowRate": "waterFlowRate",
        #                 "airFlowRate": "airFlowRate",
        #                 "inletWaterTemperature": "inletWaterTemperature",
        #                 "inletAirTemperature": "inletAirTemperature"}
        
        # self.FMUoutput = {"outletWaterTemperature": "outletWaterTemperature", 
        #                "outletAirTemperature": "outletAirTemperature"}
        
        self.input_unit_conversion = {"waterFlowRate": do_nothing,
                                      "airFlowRate": do_nothing,
                                      "inletWaterTemperature": to_degK_from_degC,
                                      "inletAirTemperature": to_degK_from_degC}
        
        self.output_unit_conversion = {"outletWaterTemperature": to_degC_from_degK,
                                      "outletAirTemperature": to_degC_from_degK}
        self.INITIALIZED = False

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
            FMUComponent.__init__(self, start_time=self.start_time, fmu_path=self.fmu_path)
            self.reset(set_parameters=False)
            # Set self.INITIALIZED to True to call self.reset() for future calls to initialize().
            # This currently does not work with some FMUs, because the self.fmu.reset() function fails in some cases.
            self.INITIALIZED = False

        


        