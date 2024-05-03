from .coil import Coil
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


class CoilSystem(FMUComponent, Coil):
    def __init__(self,
                **kwargs):
        Coil.__init__(self, **kwargs)
        self.start_time = 0
        fmu_filename = "DryCoilDiscretized_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)


        self.m1_flow_nominal = 1
        self.m2_flow_nominal = 1
        self.tau1 = 10
        self.tau2 = 20
        self.tau_m = 10
        self.T_a1_nominal = 45+273.15
        self.T_b1_nominal = 30+273.15
        self.T_a2_nominal = 12+273.15
        self.T_b2_nominal = 21+273.15

        self.input = {"waterFlowRate": None,
                      "airFlowRate": None,
                      "inletWaterTemperature": None,
                      "inletAirTemperature": None}
        
        self.outputMap = {"outletWaterTemperature": None, 
                       "outletAirTemperature": None}
        

        self.FMUinputMap = {"waterFlowRate": "waterFlowRate",
                        "airFlowRate": "airFlowRate",
                        "inletWaterTemperature": "inletWaterTemperature",
                        "inletAirTemperature": "inletAirTemperature"}
        
        self.FMUoutput = {"outletWaterTemperature": "outletWaterTemperature", 
                       "outletAirTemperature": "outletAirTemperature"}
        
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
        self.initialParameters = {"m1_flow_nominal": self.m1_flow_nominal,
                                    "m2_flow_nominal": self.m2_flow_nominal,
                                    "tau1": self.tau1,
                                    "tau2": self.tau2,
                                    "tau_m": self.tau_m,
                                    "UA_nominal": self.nominalUa.hasValue,
                                    "Q_flow_nominal": self.nominalSensibleCapacity.hasValue,
                                    "T_a1_nominal": self.T_a1_nominal,
                                    "T_b1_nominal": self.T_b1_nominal,
                                    "T_a2_nominal": self.T_a2_nominal,
                                    "T_b2_nominal": self.T_b2_nominal}
        
        if self.INITIALIZED:
            self.reset()
        else:
            self.initialize_fmu()
            self.INITIALIZED = False ###



        