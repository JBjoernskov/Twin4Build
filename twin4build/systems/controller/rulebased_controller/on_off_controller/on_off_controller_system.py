# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import Exact, MultiPath, Node, SignaturePattern


def get_signature_pattern():
    node0 = Node(cls=(core.namespace.S4BLDG.RulebasedController))
    node1 = Node(cls=(core.namespace.SAREF.Sensor))
    node2 = Node(cls=(core.namespace.SAREF.Property))
    node3 = Node(cls=(core.namespace.S4BLDG.Schedule))
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="OnOffControllerSystem"
    )
    sp.add_triple(
        Exact(subject=node0, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node1, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node0, object=node3, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_modeled_node(node0)
    return sp


class OnOffControllerSystem(core.System):
    sp = [get_signature_pattern()]

    def __init__(self, offValue=0, onValue=1, isReverse=False, **kwargs):
        super().__init__(**kwargs)
        self.offValue = offValue
        self.onValue = onValue
        self.isReverse = isReverse

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {
            "parameters": ["offValue", "onValue", "isReverse"],
        }

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
        pass

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """
        This function calls the do_step method of the FMU component, and then sets the output of the FMU model.
        """
        if self.isReverse:
            if self.input["actualValue"] < self.input["setpointValue"]:
                self.output["inputSignal"].set(self.onValue, stepIndex)
            else:
                self.output["inputSignal"].set(self.offValue, stepIndex)
        else:
            if self.input["actualValue"] > self.input["setpointValue"]:
                self.output["inputSignal"].set(self.onValue, stepIndex)
            else:
                self.output["inputSignal"].set(self.offValue, stepIndex)
