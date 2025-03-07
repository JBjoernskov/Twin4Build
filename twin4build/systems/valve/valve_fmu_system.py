import twin4build.systems.utils.fmu.fmu_component as fmu_component
from twin4build.utils.uppath import uppath
import os
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
import twin4build.core as core
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.Valve) #supply valve
    node1 = Node(cls=core.S4BLDG.Controller)
    node2 = Node(cls=core.SAREF.OpeningPosition)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="ValveFMUSystem")

    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.SAREF.controls))
    sp.add_triple(Exact(subject=node2, object=node0, predicate=core.SAREF.isPropertyOf))

    sp.add_input("valvePosition", node1, "inputSignal")
    sp.add_modeled_node(node0)

    return sp

class ValveFMUSystem(fmu_component.FMUComponent):
    sp = [get_signature_pattern()]
    def __init__(self,
                 m_flow_nominal=None,
                 dpFixed_nominal=None,
                **kwargs):
        super().__init__(**kwargs)
        
        self.start_time = 0
        fmu_filename = "Valve_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = fmu_component.unzip_fmu(self.fmu_path)
        self.m_flow_nominal = m_flow_nominal
        self.dpFixed_nominal = dpFixed_nominal

        self.input = {"valvePosition": tps.Scalar()}
        self.output = {"waterFlowRate": tps.Scalar(),
                       "valvePosition": tps.Scalar()}
        
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
