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
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, regularize
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil import Coil
import twin4build.base as base
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.SAREF.Meter)
    node1 = Node(cls=base.SAREF.Flow)
    node2 = Node(cls=base.S4BLDG.Coil) #waterside
    node3 = Node(cls=base.S4BLDG.Coil) #airside
    node4 = Node(cls=base.S4BLDG.Coil) #supersystem
    node5 = Node(cls=base.S4BLDG.Pump) #before waterside
    # node8 = Node(cls=base.System) #after airside
    node10 = Node(cls=base.S4BLDG.Valve)
    node11 = Node(cls=base.S4BLDG.Valve)
    node12 = Node(cls=base.SAREF.OpeningPosition)
    node13 = Node(cls=base.S4BLDG.Controller)
    node14 = Node(cls=base.SAREF.Sensor)
    # node15 = Node(cls=(base.Fan, base.AirToAirHeatRecovery, base.Coil, base.Sensor))
    node15 = Node(cls=(base.S4BLDG.Fan, base.S4BLDG.Coil, base.SAREF.Sensor))

    node16 = Node(cls=base.SAREF.PropertyValue)
    node17 = Node(cls=base.XSD.float)
    node18 = Node(cls=base.S4BLDG.NominalUa)
    node19 = Node(cls=base.SAREF.PropertyValue)
    node20 = Node(cls=base.XSD.float)
    node21 = Node(cls=base.S4BLDG.FlowCoefficient)
    node22 = Node(cls=base.SAREF.PropertyValue)
    node23 = Node(cls=base.XSD.float)
    node24 = Node(cls=base.S4BLDG.FlowCoefficient)

    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="SensorSystem", priority=10)
    sp.add_triple(Exact(subject=node0, object=node1, predicate=base.SAREF.observes))
    sp.add_triple(SinglePath(subject=node5, object=node2, predicate=base.FSO.feedsFluidTo))
    sp.add_triple(SinglePath(subject=node2, object=node10, predicate=base.FSO.returnsFluidTo))
    sp.add_triple(SinglePath(subject=node2, object=node11, predicate=base.FSO.returnsFluidTo))
    # sp.add_triple(Exact(subject=node3, object=node8, predicate="suppliesFluidTo"))
    sp.add_triple(Exact(subject=node2, object=node4, predicate=base.S4SYST.subSystemOf))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=base.S4SYST.subSystemOf))
    sp.add_triple(SinglePath(subject=node0, object=node3, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node14, object=node5, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node15, object=node3, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node10, object=node5, predicate=base.FSO.suppliesFluidTo))
    
    sp.add_triple(Exact(subject=node11, object=node12, predicate=base.SAREF.hasProperty))
    sp.add_triple(Exact(subject=node13, object=node12, predicate=base.SAREF.controls))

    sp.add_triple(Optional_(subject=node16, object=node17, predicate=base.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node16, object=node18, predicate=base.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node4, object=node16, predicate=base.SAREF.hasPropertyValue))
    sp.add_triple(Optional_(subject=node19, object=node20, predicate=base.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node19, object=node21, predicate=base.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node10, object=node19, predicate=base.SAREF.hasPropertyValue))
    sp.add_triple(Optional_(subject=node22, object=node23, predicate=base.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node22, object=node24, predicate=base.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node11, object=node22, predicate=base.SAREF.hasPropertyValue))
    sp.add_input("airFlowRate", node0)
    sp.add_input("inletAirTemperature", node15, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_input("supplyWaterTemperature", node14, "supplyWaterTemperature")
    sp.add_input("valvePosition", node13, "inputSignal")

    sp.add_parameter("nominalUa.hasValue", node17)
    sp.add_parameter("flowCoefficient.hasValue", node23)



    # sp.add_modeled_node(node2)
    # sp.add_modeled_node(node3)
    sp.add_modeled_node(node4)
    sp.add_modeled_node(node5)
    sp.add_modeled_node(node10)
    sp.add_modeled_node(node11)
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

        self.input = {"valvePosition": tps.Scalar(),
                      "airFlowRate": tps.Scalar(),
                      "supplyWaterTemperature": tps.Scalar(),
                      "inletAirTemperature": tps.Scalar()}
        
        self.output = {"outletWaterTemperature": tps.Scalar(), 
                       "outletAirTemperature": tps.Scalar(),
                       "inletWaterTemperature": tps.Scalar(),
                       "valvePosition": tps.Scalar()}
        
        
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
        
        self.input_conversion = {"valvePosition": do_nothing,
                                      "airFlowRate": regularize(0.01),
                                      "supplyWaterTemperature": to_degK_from_degC,
                                      "inletAirTemperature": to_degK_from_degC}
        
        self.output_conversion = {"outletWaterTemperature": to_degC_from_degK,
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
                    stepSize=None,
                    model=None):
        '''
            This function initializes the FMU component by setting the start_time and fmu_filename attributes, 
            and then sets the parameters for the FMU model.
        '''
        if self.INITIALIZED:
            self.reset()
        else:
            self.initialize_fmu()
            self.INITIALIZED = True


        