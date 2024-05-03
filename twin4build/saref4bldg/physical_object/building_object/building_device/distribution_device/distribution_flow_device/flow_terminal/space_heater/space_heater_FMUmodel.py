from .space_heater import SpaceHeater
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


class SpaceHeaterSystem(FMUComponent, SpaceHeater):
    def __init__(self,
                 waterFlowRateMax=None,
                **kwargs):
        SpaceHeater.__init__(self, **kwargs)
        self.start_time = 0
        fmu_filename = "Radiator.FMU"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)
        self.waterFlowRateMax = waterFlowRateMax

        self.input = {"supplyWaterTemperature": None,
                      "waterFlowRate": None,
                      "indoorTemperature": None}
        self.output = {"outletWaterTemperature": None,
                       "PowerToRadiator": None,
                       "EnergyToRadiator": None}
        
        #Used in finite difference jacobian approximation for uncertainty analysis.
        self.inputLowerBounds = {"valvePosition": 0}
        self.inputUpperBounds = {"valvePosition": 1}


        self.FMUinputMap = {"supplyWaterTemperature": "supplyWaterTemperature",
                            "waterFlowRate": "waterFlowRate",
                            "indoorTemperature": "indoorTemperature"}
        self.FMUoutputMap = {"outletWaterTemperature": "outletWaterTemperature",
                            "PowerToRadiator": "PowerToRadiator",
                            "EnergyToRadiator": "EnergyToRadiator"}

        self.FMUparameterMap = {"self.outputCapacity.hasValue": "Q_flow_nominal",
                                "nominalRoomTemperature": "Kv",
                                "nominalSupplyTemperature": "dpFixed_nominal",
                                "nominalReturnTemperature": "dpFixed_nominal",
                                "nominalSupplyTemperature": "dpFixed_nominal"} ####################################################
        
        # parameters = {"Q_flow_nominal": self.outputCapacity.hasValue,
        #         "T_a_nominal": self.nominalSupplyTemperature,
        #         "T_b_nominal": self.nominalReturnTemperature,
        #         "Radiator.UAEle": 10}#0.70788274}
        
        # self.initialParameters = {"Q_flow_nominal": self.outputCapacity.hasValue,
        #                           "TAir_nominal": self.nominalRoomTemperature+273.15,
        #                             "T_a_nominal": self.nominalSupplyTemperature+273.15,
        #                             "T_b_nominal": self.nominalReturnTemperature+273.15,
        #                             "T_start": self.output["outletTemperature"]+273.15,
        
        self.input_unit_conversion = {"valvePosition": do_nothing}
        self.output_unit_conversion = {"waterFlowRate": do_nothing,
                                       "valvePosition": do_nothing}

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
            self.initialize_fmu()
            self.INITIALIZED = True


        