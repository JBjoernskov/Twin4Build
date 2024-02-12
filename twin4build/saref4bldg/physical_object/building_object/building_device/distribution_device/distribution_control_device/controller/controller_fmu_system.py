from .controller import Controller
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import numpy as np
import os
from twin4build.utils.fmu.unit_converters.functions import do_nothing
import twin4build.base as base
from twin4build.utils.context_signature.context_signature import ContextSignature, Node 

def get_context_signature():
    node0 = Node(cls=(base.Controller,))
    node1 = Node(cls=(base.Sensor,))
    node2 = Node(cls=(base.Property,))
    node3 = Node(cls=(base.Property,))
    cs = ContextSignature()
    cs.add_edge(node0, node2, "actuatesProperty")
    cs.add_edge(node0, node3, "controlsProperty")
    cs.add_edge(node1, node3, "measuresProperty")
    cs.add_input("airFlow", node0)
    return cs

class ControllerFMUSystem(FMUComponent, Controller):
    cs = get_context_signature()
    def __init__(self,
                 kp=None,
                 Ti=None,
                 Td=None,
                **kwargs):
        Controller.__init__(self, **kwargs)
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


        