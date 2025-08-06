# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.translator.translator import Exact, MultiPath, Node, SignaturePattern
from twin4build.utils.get_object_properties import get_object_properties
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr


def get_signature_pattern():
    node0 = Node(cls=core.namespace.S4BLDG.SetpointController)
    node1 = Node(cls=core.namespace.S4BLDG.RulebasedController)
    node2 = Node(cls=core.namespace.SAREF.Property)
    node3 = Node(cls=core.namespace.SAREF.Property)
    node4 = Node(cls=core.namespace.SAREF.Property)

    node5 = Node(cls=core.namespace.S4BLDG.Schedule)
    node6 = Node(cls=core.namespace.S4BLDG.Schedule)
    node7 = Node(cls=core.namespace.SAREF.Sensor)
    node8 = Node(cls=core.namespace.SAREF.Sensor)
    node9 = Node(cls=core.namespace.SAREF.Property)

    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="SequenceControllerSystem"
    )
    sp.add_triple(
        Exact(subject=node0, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node1, object=node4, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node0, object=node3, predicate=core.namespace.SAREF.controls)
    )
    sp.add_triple(
        Exact(subject=node1, object=node3, predicate=core.namespace.SAREF.controls)
    )
    sp.add_triple(
        Exact(subject=node0, object=node5, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_triple(
        Exact(subject=node1, object=node6, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_triple(
        Exact(subject=node7, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node8, object=node4, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node0, object=node9, predicate=core.namespace.SAREF.controls)
    )
    sp.add_triple(
        Exact(subject=node1, object=node9, predicate=core.namespace.SAREF.controls)
    )

    sp.add_input("actualValueSetpointController", node7, "measuredValue")
    sp.add_input("actualValueRulebasedController", node8, "measuredValue")
    sp.add_input("setpointValueSetpointController", node5, "scheduleValue")
    sp.add_input("setpointValueRulebasedController", node6, "scheduleValue")
    sp.add_modeled_node(node0)
    sp.add_modeled_node(node1)
    return sp


class SequenceControllerSystem:
    sp = [get_signature_pattern()]

    def __init__(self, **kwargs):
        self.base_components = kwargs["base_components"]
        base_setpoint_controller = [
            component
            for component in self.base_components
            if isinstance(component, core.SetpointController)
        ][0]
        base_rulebased_controller = [
            component
            for component in self.base_components
            if isinstance(component, core.RulebasedController)
        ][0]
        self.setpoint_controller = systems.PIControllerFMUSystem(
            **get_object_properties(base_setpoint_controller)
        )
        self.rulebased_controller = systems.OnOffControllerSystem(
            **get_object_properties(base_rulebased_controller)
        )

        #         id=f"setpoint_controller - {self.id}",
        # id=f"rulebased_controller - {self.id}",

        self.input = {
            "actualValueSetpointController": tps.Scalar(),
            "actualValueRulebasedController": tps.Scalar(),
            "setpointValueSetpointController": tps.Scalar(),
            "setpointValueRulebasedController": tps.Scalar(),
        }
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

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """
        This function initializes the FMU component by setting the start_time and fmu_filename attributes,
        and then sets the parameters for the FMU model.
        """

        for attr in self.setpoint_controller.config["parameters"]:
            new_attr = f"{attr}__{self.setpoint_controller.id}"
            rsetattr(self.setpoint_controller, attr, rgetattr(self, new_attr))

        for attr in self.rulebased_controller.config["parameters"]:
            new_attr = f"{attr}__{self.rulebased_controller.id}"
            rsetattr(self.rulebased_controller, attr, rgetattr(self, new_attr))

        self.setpoint_controller.input["actualValue"] = self.input[
            "actualValueSetpointController"
        ]
        self.setpoint_controller.input["setpointValue"] = self.input[
            "setpointValueSetpointController"
        ]
        self.rulebased_controller.input["actualValue"] = self.input[
            "actualValueRulebasedController"
        ]
        self.rulebased_controller.input["setpointValue"] = self.input[
            "setpointValueRulebasedController"
        ]

        self.setpoint_controller.output = self.output.copy()
        self.setpoint_controller.initialize(startTime, endTime, stepSize, simulator)
        self.rulebased_controller.output = self.output.copy()
        self.rulebased_controller.initialize(startTime, endTime, stepSize, simulator)

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        self.setpoint_controller.input["actualValue"].set(
            self.input["actualValueSetpointController"], stepIndex
        )
        self.setpoint_controller.input["setpointValue"].set(
            self.input["setpointValueSetpointController"], stepIndex
        )
        self.setpoint_controller.do_step(
            secondTime=secondTime,
            dateTime=dateTime,
            stepSize=stepSize,
            stepIndex=stepIndex,
        )

        self.rulebased_controller.input["actualValue"].set(
            self.input["actualValueRulebasedController"], stepIndex
        )
        self.rulebased_controller.input["setpointValue"].set(
            self.input["setpointValueRulebasedController"], stepIndex
        )
        self.rulebased_controller.do_step(
            secondTime=secondTime,
            dateTime=dateTime,
            stepSize=stepSize,
            stepIndex=stepIndex,
        )

        self.output["inputSignal"].set(
            max(
                next(iter(self.setpoint_controller.output.values())),
                next(iter(self.rulebased_controller.output.values())),
            ),
            stepIndex,
        )
