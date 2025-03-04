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
    node0 = Node(cls=(base.S4BLDG.RulebasedController))
    node1 = Node(cls=(base.SAREF.Sensor))
    node2 = Node(cls=(base.SAREF.Property))
    node3 = Node(cls=(base.S4BLDG.Schedule))
    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="OnOffControllerSystem")
    sp.add_triple(Exact(subject=node0, object=node2, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node0, object=node3, predicate=base.SAREF.hasProfile))
    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_modeled_node(node0)
    return sp


class OnOffControllerSystem(base.RulebasedController):
    sp = [get_signature_pattern()]
    def __init__(self,
                 offValue=0,
                 onValue=1,
                 isReverse=False,
                **kwargs):
        # base.SetpointController.__init__(self, **kwargs)
        super().__init__(**kwargs)
        self.offValue = offValue
        self.onValue = onValue
        self.isReverse = isReverse

        self.input = {"actualValue": tps.Scalar(),
                        "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {"parameters": ["offValue", "onValue", "isReverse"],}

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
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
            This function calls the do_step method of the FMU component, and then sets the output of the FMU model.
        '''
        if self.isReverse:
            if self.input["actualValue"] < self.input["setpointValue"]:
                self.output["inputSignal"].set(self.onValue)
            else:
                self.output["inputSignal"].set(self.offValue)
        else:
            if self.input["actualValue"] > self.input["setpointValue"]:
                self.output["inputSignal"].set(self.onValue)
            else:
                self.output["inputSignal"].set(self.offValue)


        