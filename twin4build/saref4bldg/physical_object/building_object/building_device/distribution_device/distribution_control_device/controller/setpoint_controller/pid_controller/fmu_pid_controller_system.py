import twin4build.base as base
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import numpy as np
import os
from twin4build.utils.fmu.unit_converters.functions import do_nothing
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact

def get_signature_pattern():
    node0 = Node(cls=(base.SetpointController,))
    node1 = Node(cls=(base.Sensor,))
    node2 = Node(cls=(base.Property,))
    node3 = Node(cls=(base.Property,))
    node4 = Node(cls=(base.Schedule,))
    sp = SignaturePattern(ownedBy="FMUPIDControllerSystem")
    sp.add_edge(Exact(object=node0, subject=node2, predicate="actuatesProperty"))
    sp.add_edge(Exact(object=node0, subject=node3, predicate="controlsProperty"))
    sp.add_edge(Exact(object=node1, subject=node3, predicate="measuresProperty"))
    sp.add_edge(Exact(object=node0, subject=node4, predicate="hasSetpointSchedule"))
    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node4)
    sp.add_modeled_node(node0)
    return sp


class FMUPIDControllerSystem(FMUComponent, base.SetpointController):
    sp = [get_signature_pattern()]
    def __init__(self,
                 kp=None,
                 Ti=None,
                 Td=None,
                **kwargs):
        base.SetpointController.__init__(self, **kwargs)
        self.start_time = 0
        fmu_filename = "Controller_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)
        self.kp = kp
        self.Ti = Ti
        self.Td = Td

        self.input = {"actualValue": None,
                        "setpointValue": None}
        self.output = {"inputSignal": None}

        #Used in finite difference jacobian approximation for uncertainty analysis.
        self.inputLowerBounds = {"actualValue": -np.inf,
                        "setpointValue": -np.inf}
        self.inputUpperBounds = {"actualValue": np.inf,
                        "setpointValue": np.inf}
        
        self.FMUinputMap = {"actualValue": "u_m",
                            "setpointValue": "u_s"}
        self.FMUoutputMap = {"inputSignal": "y"}
        self.FMUparameterMap = {"kp": "k",
                                "Ti": "Ti",
                                "Td": "Td"}
        
        self.input_unit_conversion = {"actualValue": do_nothing,
                                      "setpointValue": do_nothing}
        self.output_unit_conversion = {"inputSignal": do_nothing}

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


        