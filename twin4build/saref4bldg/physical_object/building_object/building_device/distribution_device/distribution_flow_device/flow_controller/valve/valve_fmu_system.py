from .valve import Valve
from typing import Union
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import os
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes, Optional
import twin4build.base as base
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing

def get_signature_pattern():
    node0 = Node(cls=base.Valve, id="<n<SUB>1</SUB>(Valve)>") #supply valve
    node1 = Node(cls=base.Controller, id="<n<SUB>2</SUB>(Controller)>")
    node2 = Node(cls=base.OpeningPosition, id="<n<SUB>3</SUB>(Property)>")
    sp = SignaturePattern(ownedBy="ValveFMUSystem")

    sp.add_edge(Exact(object=node1, subject=node2, predicate="controls"))
    sp.add_edge(Exact(object=node2, subject=node0, predicate="isPropertyOf"))

    sp.add_input("valvePosition", node1, "inputSignal")
    sp.add_modeled_node(node0)

    return sp



class ValveFMUSystem(FMUComponent, Valve):
    sp = [get_signature_pattern()]
    def __init__(self,
                 m_flow_nominal=None,
                 dpFixed_nominal=None,
                **kwargs):
        Valve.__init__(self, **kwargs)
        self.start_time = 0
        fmu_filename = "Valve_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)
        self.m_flow_nominal = m_flow_nominal
        self.dpFixed_nominal = dpFixed_nominal

        self.input = {"valvePosition": None}
        self.output = {"waterFlowRate": None,
                       "valvePosition": None}
        
        #Used in finite difference jacobian approximation for uncertainty analysis.
        self.inputLowerBounds = {"valvePosition": 0}
        self.inputUpperBounds = {"valvePosition": 1}

        self.FMUinputMap = {"valvePosition": "u"}
        self.FMUoutputMap = {"waterFlowRate": "m_flow"}
        self.FMUparameterMap = {"m_flow_nominal": "m_flow_nominal",
                                "flowCoefficient.hasValue": "Kv",
                                "dpFixed_nominal": "dpFixed_nominal"}
        
        self.parameter = {"m_flow_nominal": {"lb": 0.0001, "ub": 5},
                            "flowCoefficient.hasValue": {"lb": 0.1, "ub": 100},
                            "dpFixed_nominal": {"lb": 0, "ub": 10000}
                          
        }
        
        self.input_conversion = {"valvePosition": do_nothing}
        self.output_conversion = {"waterFlowRate": do_nothing,
                                       "valvePosition": do_nothing}

        self.INITIALIZED = False
        self._config = {"parameters": list(self.parameter.keys())}

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
