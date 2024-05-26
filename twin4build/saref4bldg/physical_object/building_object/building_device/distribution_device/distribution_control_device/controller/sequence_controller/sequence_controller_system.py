import twin4build.base as base
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
import twin4build.components as components
import numpy as np
import os
from twin4build.utils.fmu.unit_converters.functions import do_nothing
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches
from twin4build.utils.get_object_properties import get_object_properties
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rgetattr import rgetattr

def get_signature_pattern1():
    node0 = Node(cls=(base.SetpointController,), id="<Controller\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.RulebasedController,), id="<Controller\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Property,), id="<Property\nn<SUB>3</SUB>>")
    node3 = Node(cls=(base.Property,), id="<Property\nn<SUB>4</SUB>>")
    node4 = Node(cls=(base.Property,), id="<Property\nn<SUB>5</SUB>>")


    node5 = Node(cls=(base.Schedule,), id="<Schedule\nn<SUB>6</SUB>>")
    node6 = Node(cls=(base.Schedule,), id="<Schedule\nn<SUB>7</SUB>>")
    node7 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>8</SUB>>")
    node8 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>9</SUB>>")
    
    sp = SignaturePattern(ownedBy="SequenceControllerSystem", priority=20)
    sp.add_edge(Exact(object=node0, subject=node2, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node4, predicate="observes"))
    sp.add_edge(Exact(object=node0, subject=node3, predicate="controls"))
    sp.add_edge(Exact(object=node1, subject=node3, predicate="controls"))
    sp.add_edge(Exact(object=node0, subject=node5, predicate="hasProfile"))
    sp.add_edge(Exact(object=node1, subject=node6, predicate="hasProfile"))
    sp.add_edge(Exact(object=node7, subject=node2, predicate="observes"))
    sp.add_edge(Exact(object=node8, subject=node4, predicate="observes"))
    sp.add_input("actualValueSetpointController", node7, "measuredValue")
    sp.add_input("actualValueRulebasedController", node8, "measuredValue")
    sp.add_modeled_node(node0)
    sp.add_modeled_node(node1)
    return sp

def get_signature_pattern2():
    node0 = Node(cls=(base.SetpointController,), id="<Controller\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.RulebasedController,), id="<Controller\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Property,), id="<Property\nn<SUB>3</SUB>>")
    node3 = Node(cls=(base.Property,), id="<Property\nn<SUB>4</SUB>>")
    node4 = Node(cls=(base.Property,), id="<Property\nn<SUB>5</SUB>>")
    


    node5 = Node(cls=(base.Schedule,), id="<Schedule\nn<SUB>6</SUB>>")
    node6 = Node(cls=(base.Schedule,), id="<Schedule\nn<SUB>7</SUB>>")
    node7 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>8</SUB>>")
    node8 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>9</SUB>>")
    node9 = Node(cls=(base.Property,), id="<Property\nn<SUB>10</SUB>>")
    
    sp = SignaturePattern(ownedBy="SequenceControllerSystem", priority=20)
    sp.add_edge(Exact(object=node0, subject=node2, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node4, predicate="observes"))
    sp.add_edge(Exact(object=node0, subject=node3, predicate="controls"))
    sp.add_edge(Exact(object=node1, subject=node3, predicate="controls"))
    sp.add_edge(Exact(object=node0, subject=node5, predicate="hasProfile"))
    sp.add_edge(Exact(object=node1, subject=node6, predicate="hasProfile"))
    sp.add_edge(Exact(object=node7, subject=node2, predicate="observes"))
    sp.add_edge(Exact(object=node8, subject=node4, predicate="observes"))
    sp.add_edge(Exact(object=node0, subject=node9, predicate="controls"))
    sp.add_edge(Exact(object=node1, subject=node9, predicate="controls"))

    sp.add_input("actualValueSetpointController", node7, "measuredValue")
    sp.add_input("actualValueRulebasedController", node8, "measuredValue")
    sp.add_input("setpointValueSetpointController", node5, "scheduleValue")
    sp.add_input("setpointValueRulebasedController", node6, "scheduleValue")
    sp.add_modeled_node(node0)
    sp.add_modeled_node(node1)
    return sp


class SequenceControllerSystem(base.Controller):
    sp = [get_signature_pattern1(), get_signature_pattern2()]
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.base_components = kwargs["base_components"]
        base_setpoint_controller = [component for component in self.base_components if isinstance(component, base.SetpointController)][0]
        base_rulebased_controller = [component for component in self.base_components if isinstance(component, base.RulebasedController)][0]
        self.setpoint_controller = components.PIControllerFMUSystem(**get_object_properties(base_setpoint_controller))
        self.rulebased_controller = components.OnOffControllerSystem(**get_object_properties(base_rulebased_controller))

#         id=f"setpoint_controller - {self.id}", 
# id=f"rulebased_controller - {self.id}", 

        self.input = {"actualValueSetpointController": None,
                        "actualValueRulebasedController": None,
                        "setpointValueSetpointController": None,
                        "setpointValueRulebasedController": None}
        self.output = {"inputSignal": None}
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
                    stepSize=None):
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

        self.setpoint_controller.output = self.output
        self.setpoint_controller.initialize(startTime,
                                        endTime,
                                        stepSize)
        self.rulebased_controller.output = self.output
        self.rulebased_controller.initialize(startTime,
                                        endTime,
                                        stepSize)


    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.setpoint_controller.input["actualValue"] = self.input["actualValueSetpointController"]
        self.setpoint_controller.input["setpointValue"] = self.input["setpointValueSetpointController"]
        self.setpoint_controller.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        self.rulebased_controller.input["actualValue"] = self.input["actualValueRulebasedController"]
        self.rulebased_controller.input["setpointValue"] = self.input["setpointValueRulebasedController"]
        self.rulebased_controller.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        # if self.addUncertainty:
        #     for key in self.do_step_instance.output:
        #         self.output[key] = self.do_step_instance.output[key] + np.random.normal(0, self.standardDeviation)
        # else:
        self.output["inputSignal"] = max(next(iter(self.setpoint_controller.output.values())), next(iter(self.rulebased_controller.output.values())))