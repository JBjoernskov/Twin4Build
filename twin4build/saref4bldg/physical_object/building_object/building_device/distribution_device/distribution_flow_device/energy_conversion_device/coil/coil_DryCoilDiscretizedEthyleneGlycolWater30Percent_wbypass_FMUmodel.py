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
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, regularize
from twin4build.utils.context_signature.context_signature import ContextSignature, Node
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil import Coil
# import twin4build as tb
import twin4build.base as base

def get_context_signature():
    node1 = Node(cls=(base.Fan, base.Coil, base.AirToAirHeatRecovery))
    node2 = Node(cls=(base.Coil,))
    node3 = Node(cls=(base.Pump,))
    node4 = Node(cls=(base.Valve,))
    node5 = Node(cls=(base.Valve,))
    # node6 = Node(cls=(tb.Valve))
    node7 = Node(cls=(base.Controller,))
    node8 = Node(cls=(base.OpeningPosition,))
    cs = ContextSignature()
    cs.add_edge(node1, node2, "connectedBefore")
    cs.add_edge(node3, node2, "connectedBefore")
    cs.add_edge(node2, node4, "connectedBefore")
    cs.add_edge(node4, node3, "connectedBefore")
    cs.add_edge(node2, node5, "connectedBefore")
    cs.add_edge(node5, node8, "hasProperty")
    cs.add_edge(node7, node8, "actuatesProperty")
    cs.add_input("airFlow", node1)
    cs.add_input("inletAirTemperature", node1)
    cs.add_input("supplyWaterTemperature", node2)
    cs.add_input("valvePosition", node5)

    # cs.print_edges()
    # cs.print_inputs()
    return cs

class CoilSystem(FMUComponent, Coil):
    cs = get_context_signature()
    def __init__(self,
                m1_flow_nominal=None,
                m2_flow_nominal=None,
                tau1=None,
                tau2=None,
                tau_m=None,
                mFlowValve_nominal=None,
                flowCoefficient=None,
                mFlowPump_nominal=None,
                dpCheckValve_nominal=None,
                dp1_nominal=None,
                dpPump=None,
                dpValve_nominal=None,
                dpSystem=None,
                tau_w_inlet=None,
                tau_w_outlet=None,
                tau_air_outlet=None,
                **kwargs):
        Coil.__init__(self, **kwargs)
        self.start_time = 0
        fmu_filename = "coil_0wbypass_0FMUmodel.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.m1_flow_nominal = m1_flow_nominal
        self.m2_flow_nominal = m2_flow_nominal
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau_m = tau_m
        self.mFlowValve_nominal = mFlowValve_nominal
        self.flowCoefficient = flowCoefficient
        self.mFlowPump_nominal = mFlowPump_nominal
        self.dpCheckValve_nominal = dpCheckValve_nominal
        self.dp1_nominal = dp1_nominal
        self.dpPump = dpPump
        self.dpValve_nominal = dpValve_nominal
        self.dpSystem = dpSystem
        self.tau_w_inlet = tau_w_inlet
        self.tau_w_outlet = tau_w_outlet
        self.tau_air_outlet = tau_air_outlet

        self.input = {"valvePosition": None,
                      "airFlowRate": None,
                      "supplyWaterTemperature": None,
                      "inletAirTemperature": None}
        
        self.output = {"outletWaterTemperature": None, 
                       "outletAirTemperature": None,
                       "inletWaterTemperature": None,
                       "valvePosition": None}
        
        
        self.FMUinputMap = {"valvePosition": "u",
                            "airFlowRate": "inlet2.m_flow",
                            "supplyWaterTemperature": "supplyWaterTemperature",
                            "inletAirTemperature": "inlet2.forward.T"}
        
        self.FMUoutputMap = {"outletWaterTemperature": "outletWaterTemperature", 
                            "outletAirTemperature": "outletAirTemperature",
                            "inletWaterTemperature": "inletWaterTemperature",
                            "valvePosition": "u"}
        
        self.FMUparameterMap = {"m1_flow_nominal": "m1_flow_nominal",
                                "m2_flow_nominal": "m2_flow_nominal",
                                "tau1": "tau1",
                                "tau2": "tau2",
                                "tau_m": "tau_m",
                                "nominalUa.hasValue": "UA_nominal",
                                "mFlowValve_nominal": "mFlowValve_nominal",
                                "flowCoefficient": "Kv",
                                "mFlowPump_nominal": "mFlowPump_nominal",
                                "dpCheckValve_nominal": "dpCheckValve_nominal",
                                "dp1_nominal": "dp1_nominal",
                                "dpPump": "dpPump",
                                "dpSystem": "dpSystem",
                                "tau_w_inlet": "tau_w_inlet",
                                "tau_w_outlet": "tau_w_outlet",
                                "tau_air_outlet": "tau_air_outlet"}
        
        self.input_unit_conversion = {"valvePosition": do_nothing,
                                      "airFlowRate": regularize(0.01),
                                      "supplyWaterTemperature": to_degK_from_degC,
                                      "inletAirTemperature": to_degK_from_degC}
        
        self.output_unit_conversion = {"outletWaterTemperature": to_degC_from_degK,
                                      "outletAirTemperature": to_degC_from_degK,
                                      "inletWaterTemperature": to_degC_from_degK,
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
            FMUComponent.__init__(self, fmu_path=self.fmu_path, unzipdir=self.unzipdir)
            # Set self.INITIALIZED to True to call self.reset() for future calls to initialize().
            # This currently does not work with some FMUs, because the self.fmu.reset() function fails in some cases.
            self.INITIALIZED = True


        