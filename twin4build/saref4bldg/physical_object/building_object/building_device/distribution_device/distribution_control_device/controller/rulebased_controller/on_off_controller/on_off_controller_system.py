import twin4build.base as base
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import numpy as np
import os
from twin4build.utils.fmu.unit_converters.functions import do_nothing
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches

def get_signature_pattern():
    node0 = Node(cls=(base.RulebasedController,), id="<Controller\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Property,), id="<Property\nn<SUB>3</SUB>>")
    node3 = Node(cls=(base.Schedule,), id="<Schedule\nn<SUB>4</SUB>>")
    sp = SignaturePattern(ownedBy="OnOffControllerSystem")
    sp.add_edge(Exact(object=node0, subject=node2, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="observes"))
    sp.add_edge(Exact(object=node0, subject=node3, predicate="hasProfile"))
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

        self.input = {"actualValue": None,
                        "setpointValue": None}
        self.output = {"inputSignal": None}
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
                    stepSize=None):
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
                self.output["inputSignal"] = self.onValue
            else:
                self.output["inputSignal"] = self.offValue
        else:
            if self.input["actualValue"] > self.input["setpointValue"]:
                self.output["inputSignal"] = self.onValue
            else:
                self.output["inputSignal"] = self.offValue


        