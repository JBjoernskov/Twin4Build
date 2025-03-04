import twin4build.base as base
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import numpy as np
import os
from twin4build.utils.unit_converters.functions import do_nothing
import twin4build.base as base
from twin4build.translator.translator import SignaturePattern, Node, Exact, MultiPath
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.S4BLDG.SetpointController)
    node1 = Node(cls=base.SAREF.Sensor)
    node2 = Node(cls=base.SAREF.Property)
    node3 = Node(cls=base.S4BLDG.Schedule)
    node4 = Node(cls=base.XSD.boolean)
    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="PIControllerFMUSystem")
    sp.add_triple(Exact(subject=node0, object=node2, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node0, object=node3, predicate=base.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node0, object=node4, predicate=base.S4BLDG.isReverse))


    # sp.add_triple(Exact(subject=node4, object=node5, predicate=base.SAREF.hasValue))
    # sp.add_triple(Exact(subject=node4, object=node6, predicate=base.SAREF.isValueOfProperty))
    # sp.add_triple(Optional_(subject=node0, object=node4, predicate=base.SAREF.hasPropertyValue))
    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_parameter("isReverse", node4)
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

        self.input = {"actualValue": tps.Scalar(),
                        "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}

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


        