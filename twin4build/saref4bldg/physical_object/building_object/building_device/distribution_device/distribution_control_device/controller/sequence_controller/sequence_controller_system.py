import twin4build.base as base
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import twin4build.systems as systems
import numpy as np
import os
from twin4build.utils.unit_converters.functions import do_nothing
import twin4build.base as base
from twin4build.translator.translator import SignaturePattern, Node, Exact, MultiPath
from twin4build.utils.get_object_properties import get_object_properties
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rgetattr import rgetattr
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.S4BLDG.SetpointController)
    node1 = Node(cls=base.S4BLDG.RulebasedController)
    node2 = Node(cls=base.SAREF.Property)
    node3 = Node(cls=base.SAREF.Property)
    node4 = Node(cls=base.SAREF.Property)

    node5 = Node(cls=base.S4BLDG.Schedule)
    node6 = Node(cls=base.S4BLDG.Schedule)
    node7 = Node(cls=base.SAREF.Sensor)
    node8 = Node(cls=base.SAREF.Sensor)
    node9 = Node(cls=base.SAREF.Property)
    
    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="SequenceControllerSystem")
    sp.add_triple(Exact(subject=node0, object=node2, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node1, object=node4, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node0, object=node3, predicate=base.SAREF.controls))
    sp.add_triple(Exact(subject=node1, object=node3, predicate=base.SAREF.controls))
    sp.add_triple(Exact(subject=node0, object=node5, predicate=base.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node1, object=node6, predicate=base.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node7, object=node2, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node8, object=node4, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node0, object=node9, predicate=base.SAREF.controls))
    sp.add_triple(Exact(subject=node1, object=node9, predicate=base.SAREF.controls))

    sp.add_input("actualValueSetpointController", node7, "measuredValue")
    sp.add_input("actualValueRulebasedController", node8, "measuredValue")
    sp.add_input("setpointValueSetpointController", node5, "scheduleValue")
    sp.add_input("setpointValueRulebasedController", node6, "scheduleValue")
    sp.add_modeled_node(node0)
    sp.add_modeled_node(node1)
    return sp


class SequenceControllerSystem(base.Controller):
    sp = [get_signature_pattern()]
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.base_components = kwargs["base_components"]
        base_setpoint_controller = [component for component in self.base_components if isinstance(component, base.SetpointController)][0]
        base_rulebased_controller = [component for component in self.base_components if isinstance(component, base.RulebasedController)][0]
        self.setpoint_controller = systems.PIControllerFMUSystem(**get_object_properties(base_setpoint_controller))
        self.rulebased_controller = systems.OnOffControllerSystem(**get_object_properties(base_rulebased_controller))

#         id=f"setpoint_controller - {self.id}", 
# id=f"rulebased_controller - {self.id}", 

        self.input = {"actualValueSetpointController": tps.Scalar(),
                        "actualValueRulebasedController": tps.Scalar(),
                        "setpointValueSetpointController": tps.Scalar(),
                        "setpointValueRulebasedController": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {"parameters": []}

        for attr in self.setpoint_controller.config["parameters"]:
            new_attr = f"{attr}__{self.setpoint_controller.id}"
            rsetattr(self, new_attr, rgetattr(self.setpoint_controller, attr))
            self._config["parameters"].append(new_attr)

        for attr in self.rulebased_controller.config["parameters"]:
            new_attr = f"{attr}__{self.rulebased_controller.id}"
            rsetattr(self, new_attr, rgetattr(self.rulebased_controller, attr))
            self._config["parameters"].append(new_attr)
        

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

        for attr in self.setpoint_controller.config["parameters"]:
            new_attr = f"{attr}__{self.setpoint_controller.id}"
            rsetattr(self.setpoint_controller, attr, rgetattr(self, new_attr))

        for attr in self.rulebased_controller.config["parameters"]:
            new_attr = f"{attr}__{self.rulebased_controller.id}"
            rsetattr(self.rulebased_controller, attr, rgetattr(self, new_attr))

        self.setpoint_controller.input["actualValue"] = self.input["actualValueSetpointController"]
        self.setpoint_controller.input["setpointValue"] = self.input["setpointValueSetpointController"]
        self.rulebased_controller.input["actualValue"] = self.input["actualValueRulebasedController"]
        self.rulebased_controller.input["setpointValue"] = self.input["setpointValueRulebasedController"]

        self.setpoint_controller.output = self.output.copy()
        self.setpoint_controller.initialize(startTime,
                                        endTime,
                                        stepSize)
        self.rulebased_controller.output = self.output.copy()
        self.rulebased_controller.initialize(startTime,
                                        endTime,
                                        stepSize)


    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.setpoint_controller.input["actualValue"].set(self.input["actualValueSetpointController"])
        self.setpoint_controller.input["setpointValue"].set(self.input["setpointValueSetpointController"])
        self.setpoint_controller.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)

        self.rulebased_controller.input["actualValue"].set(self.input["actualValueRulebasedController"])
        self.rulebased_controller.input["setpointValue"].set(self.input["setpointValueRulebasedController"])
        self.rulebased_controller.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)

        self.output["inputSignal"].set(max(next(iter(self.setpoint_controller.output.values())), next(iter(self.rulebased_controller.output.values()))))

        