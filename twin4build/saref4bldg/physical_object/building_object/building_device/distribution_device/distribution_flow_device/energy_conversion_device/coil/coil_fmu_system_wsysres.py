from .coil import Coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.constants import Constants
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import types
import sys
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.flow.flow import Flow
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, regularize
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes, Optional
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil import Coil
# import twin4build as tb
import twin4build.base as base

def get_signature_pattern():
    sp = SignaturePattern(ownedBy="CoilPumpValveFMUSystem")
    node0 = Node(cls=base.Meter, id="<n<SUB>1</SUB>(Meter)>")
    node1 = Node(cls=base.Coil, id="<n<SUB>2</SUB>(Coil)>")
    node2 = Node(cls=base.Pump, id="<n<SUB>3</SUB>(Pump)>")
    node3 = Node(cls=base.Valve, id="<n<SUB>4</SUB>(Valve)>")
    node4 = Node(cls=base.Valve, id="<n<SUB>5</SUB>(Valve)>")
    node5 = Node(cls=base.OpeningPosition, id="<n<SUB>5</SUB>(OpeningPosition)>")
    node6 = Node(cls=base.Controller, id="<n<SUB>6</SUB>(Controller)>")
    node7 = Node(cls=base.Sensor, id="<n<SUB>7</SUB>(Sensor)>")
    node8 = Node(cls=(base.Fan, base.AirToAirHeatRecovery, base.Coil), id="<n<SUB>8</SUB>(Fan, AirToAirHeatRecovery, Coil)>")
    node9 = Node(cls=base.PropertyValue, id="<n<SUB>9</SUB>(PropertyValue)>")
    node10 = Node(cls=(float), id="<n<SUB>10</SUB>(Float)>")
    node11 = Node(cls=base.NominalUa, id="<n<SUB>11</SUB>(nominalUa)>")
    node12 = Node(cls=base.PropertyValue, id="<n<SUB>12</SUB>(PropertyValue)>")
    node13 = Node(cls=(float), id="<n<SUB>13</SUB>(Float)>")
    node14 = Node(cls=base.FlowCoefficient, id="<n<SUB>14</SUB>(flowCoefficient)>")
    node15 = Node(cls=base.PropertyValue, id="<n<SUB>15</SUB>(PropertyValue)>")
    node16 = Node(cls=(float), id="<n<SUB>16</SUB>(Float)>")
    node17 = Node(cls=base.FlowCoefficient, id="<n<SUB>17</SUB>(flowCoefficient)>")
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node1, predicate="connectedBefore"))
    sp.add_edge(Exact(object=node1, subject=node3, predicate="connectedBefore"))
    sp.add_edge(Exact(object=node3, subject=node2, predicate="connectedBefore"))
    sp.add_edge(Exact(object=node1, subject=node4, predicate="connectedBefore"))
    sp.add_edge(Exact(object=node4, subject=node5, predicate="hasProperty"))
    sp.add_edge(Exact(object=node6, subject=node5, predicate="controls"))
    sp.add_edge(IgnoreIntermediateNodes(object=node2, subject=node1, predicate="connectedBefore"))
    sp.add_edge(IgnoreIntermediateNodes(object=node7, subject=node2, predicate="connectedBefore"))
    sp.add_edge(IgnoreIntermediateNodes(object=node8, subject=node1, predicate="connectedBefore"))
    sp.add_edge(Optional(object=node9, subject=node10, predicate="hasValue"))
    sp.add_edge(Optional(object=node9, subject=node11, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node1, subject=node9, predicate="hasPropertyValue"))
    sp.add_edge(Optional(object=node12, subject=node13, predicate="hasValue"))
    sp.add_edge(Optional(object=node12, subject=node14, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node4, subject=node12, predicate="hasPropertyValue"))
    sp.add_edge(Optional(object=node15, subject=node16, predicate="hasValue"))
    sp.add_edge(Optional(object=node15, subject=node17, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node3, subject=node15, predicate="hasPropertyValue"))
    sp.add_input("airFlowRate", node0)
    sp.add_input("inletAirTemperature", node8, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_input("supplyWaterTemperature", node7, "supplyWaterTemperature")
    sp.add_input("valvePosition", node6, "inputSignal")

    sp.add_parameter("nominalUa.hasValue", node10)
    sp.add_parameter("flowCoefficient.hasValue", node13)

    sp.add_modeled_node(node1)
    sp.add_modeled_node(node2)
    sp.add_modeled_node(node3)
    sp.add_modeled_node(node4)

    return sp

class CoilPumpValveFMUSystem(FMUComponent, Coil, base.Valve, base.Pump):
    sp = [get_signature_pattern()]
    def __init__(self,
                m1_flow_nominal=None,
                m2_flow_nominal=None,
                tau1=None,
                tau2=None,
                tau_m=None,
                mFlowValve_nominal=None,
                # flowCoefficient=None,
                mFlowPump_nominal=None,
                KvCheckValve=None,
                dp1_nominal=None,
                dpPump=None,
                dpValve_nominal=None,
                dpSystem=None,
                dpFixedSystem=None,
                tau_w_inlet=None,
                tau_w_outlet=None,
                tau_air_outlet=None,
                **kwargs):
        # Coil.__init__(self, **kwargs)
        # base.Valve.__init__(self, **kwargs)
        # base.Pump.__init__(self, **kwargs)
        # super(Coil, base.Valve, base.Pump).__init__(**kwargs)
        super().__init__(**kwargs)
        self.start_time = 0
        fmu_filename = "coil_0wbypass_0FMUmodel_wsysres.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.m1_flow_nominal = m1_flow_nominal
        self.m2_flow_nominal = m2_flow_nominal
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau_m = tau_m
        self.mFlowValve_nominal = mFlowValve_nominal
        # self.flowCoefficient = flowCoefficient
        self.mFlowPump_nominal = mFlowPump_nominal
        self.KvCheckValve = KvCheckValve
        self.dp1_nominal = dp1_nominal
        self.dpPump = dpPump
        self.dpValve_nominal = dpValve_nominal
        self.dpSystem = dpSystem
        self.dpFixedSystem = dpFixedSystem
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
                                "flowCoefficient.hasValue": "Kv",
                                "mFlowPump_nominal": "mFlowPump_nominal",
                                "KvCheckValve": "KvCheckValve",
                                "dp1_nominal": "dp1_nominal",
                                "dpPump": "dpPump",
                                "dpSystem": "dpSystem",
                                "dpFixedSystem": "dpFixedSystem",
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


        