import twin4build.base as base
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import numpy as np
import os
from twin4build.utils.fmu.unit_converters.functions import do_nothing
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches

def get_signature_pattern():
    node0 = Node(cls=(base.SetpointController,), id="<Controller\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Property,), id="<Property\nn<SUB>3</SUB>>")
    node3 = Node(cls=(base.Schedule,), id="<Schedule\nn<SUB>4</SUB>>")
    sp = SignaturePattern(ownedBy="PIControllerFMUSystem")
    sp.add_edge(Exact(object=node0, subject=node2, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="observes"))
    sp.add_edge(Exact(object=node0, subject=node3, predicate="hasProfile"))
    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_modeled_node(node0)
    return sp


class PIControllerFMUSystem(FMUComponent, base.SetpointController):
    sp = [get_signature_pattern()]
    def __init__(self,
                 kp=None,
                 Ti=None,
                **kwargs):
        # base.SetpointController.__init__(self, **kwargs)
        super().__init__(**kwargs)
        self.start_time = 0
        assert isinstance(self.isReverse, bool), "Attribute \"isReverse\" is of type \"" + str(type(self.isReverse)) + "\" but must be of type \"" + str(bool) + "\""
        if self.isReverse:
            fmu_filename = "Controller_0reverse_0FMU.fmu"
        else:
            fmu_filename = "Controller_0direct_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)
        self.kp = kp
        self.Ti = Ti

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
                                "Ti": "Ti"}
        
        self.input_conversion = {"actualValue": do_nothing,
                                      "setpointValue": do_nothing}
        self.output_conversion = {"inputSignal": do_nothing}

        self.INITIALIZED = False

        self._config = {"parameters": ["kp", "Ti", "isReverse"],}

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


        