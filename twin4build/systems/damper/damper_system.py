"""
Damper System Module

This module defines a DamperSystem class which models the behavior of a damper in an air flow system.

Classes:
    DamperSystem: Represents a damper system with configurable parameters and air flow calculations.

Functions:
    get_signature_pattern_1: Returns a SignaturePattern object for the DamperSystem.
"""

import math
import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, MultiPath, Optional_
import twin4build.utils.input_output_types as tps
import datetime
from typing import Optional

def get_signature_pattern():
    """
    Creates and returns a SignaturePattern for the DamperSystem.

    Returns:
        SignaturePattern: A configured SignaturePattern object for the DamperSystem.
    """
    node0 = Node(cls=core.S4BLDG.Damper)
    node1 = Node(cls=core.S4BLDG.Controller)
    node2 = Node(cls=core.SAREF.OpeningPosition)
    node3 = Node(cls=core.SAREF.Property)
    node4 = Node(cls=core.SAREF.PropertyValue)
    node5 = Node(cls=core.XSD.float)
    node6 = Node(cls=core.S4BLDG.NominalAirFlowRate)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="DamperSystem", priority=0)

    # Add edges to the signature pattern
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.SAREF.controls))
    sp.add_triple(Exact(subject=node2, object=node0, predicate=core.SAREF.isPropertyOf))
    sp.add_triple(Exact(subject=node1, object=node3, predicate=core.SAREF.observes))
    sp.add_triple(Optional_(subject=node4, object=node5, predicate=core.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node4, object=node6, predicate=core.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node0, object=node4, predicate=core.SAREF.hasPropertyValue))

    # Configure inputs, parameters, and modeled nodes
    sp.add_input("damperPosition", node1, "inputSignal")
    sp.add_parameter("nominalAirFlowRate", node5)
    sp.add_modeled_node(node0)

    return sp

class DamperSystem(core.System):
    sp = [get_signature_pattern()]

    def __init__(self,
                a=None,
                nominalAirFlowRate=None,
                **kwargs):
        """
        Initialize the DamperSystem.

        Args:
            a (float, optional): Shape parameter for the air flow curve. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent Damper class.
        """
        super().__init__(**kwargs)
        self.a = a
        self.b = None
        self.c = None
        self.nominalAirFlowRate = nominalAirFlowRate

        self.input = {"damperPosition": tps.Scalar()}
        self.output = {"airFlowRate": tps.Scalar(),
                       "damperPosition": tps.Scalar()}
        self.parameter = {
            "a": {"lb": 0.0001, "ub": 5},
            "nominalAirFlowRate": {"lb": 0.0001, "ub": 5}
        }
        self._config = {"parameters": list(self.parameter.keys())}

    @property
    def config(self):
        """
        Get the configuration of the DamperSystem.

        Returns:
            dict: The configuration dictionary.
        """
        return self._config

    def cache(self, startTime=None, endTime=None, stepSize=None):
        """
        Cache method (placeholder).

        Args:
            startTime: The start time for caching.
            endTime: The end time for caching.
            stepSize: The step size for caching.
        """
        pass

    def initialize(self, startTime=None, endTime=None, stepSize=None, simulator=None):
        """
        Initialize the DamperSystem by calculating parameters b and c.

        Args:
            startTime: The start time for initialization.
            endTime: The end time for initialization.
            stepSize: The step size for initialization.
            model: The model object, if any.
        """
        self.c = -self.a  # Ensures that m=0 at u=0
        self.b = math.log((self.nominalAirFlowRate-self.c)/self.a)  # Ensures that m=nominalAirFlowRate at u=1

    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        """
        Perform a single step of the simulation.

        This method calculates the air flow rate based on the current damper position.

        Args:
            secondTime: The current time in seconds.
            dateTime: The current date and time.
            stepSize: The size of the time step.
        """
        m_a = self.a * math.exp(self.b * self.input["damperPosition"]) + self.c
        self.output["damperPosition"].set(self.input["damperPosition"], stepIndex)
        self.output["airFlowRate"].set(m_a, stepIndex)