"""
Damper System Module

This module defines a DamperSystem class which models the behavior of a damper in an air flow system.

Classes:
    DamperSystem: Represents a damper system with configurable parameters and air flow calculations.

Functions:
    get_signature_pattern_1: Returns a SignaturePattern object for the DamperSystem.
"""

import math
from .damper import Damper
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches, Optional
import twin4build.base as base

def get_signature_pattern_1():
    """
    Creates and returns a SignaturePattern for the DamperSystem.

    Returns:
        SignaturePattern: A configured SignaturePattern object for the DamperSystem.
    """
    node0 = Node(cls=base.Damper, id="<n<SUB>1</SUB>(Damper)>") #supply valve
    node1 = Node(cls=base.Controller, id="<n<SUB>2</SUB>(Controller)>")
    node2 = Node(cls=base.OpeningPosition, id="<n<SUB>3</SUB>(OpeningPosition)>")
    node3 = Node(cls=base.Property, id="<n<SUB>4</SUB>(Property)>")
    node4 = Node(cls=base.PropertyValue, id="<n<SUB>5</SUB>(PropertyValue)>")
    node5 = Node(cls=(float, int), id="<n<SUB>6</SUB>(Float)>")
    node6 = Node(cls=base.NominalAirFlowRate, id="<n<SUB>7</SUB>(nominalAirFlowRate)>")
    sp = SignaturePattern(ownedBy="DamperSystem", priority=0)

    # Add edges to the signature pattern
    sp.add_edge(Exact(object=node1, subject=node2, predicate="controls"))
    sp.add_edge(Exact(object=node2, subject=node0, predicate="isPropertyOf"))
    sp.add_edge(Exact(object=node1, subject=node3, predicate="observes"))
    sp.add_edge(Exact(object=node4, subject=node5, predicate="hasValue"))
    sp.add_edge(Exact(object=node4, subject=node6, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node0, subject=node4, predicate="hasPropertyValue"))

    # Configure inputs, parameters, and modeled nodes
    sp.add_input("damperPosition", node1, "inputSignal")
    sp.add_parameter("nominalAirFlowRate.hasValue", node5)
    sp.add_modeled_node(node0)

    return sp

class DamperSystem(Damper):
    """
    A class representing a damper system in an air flow setup.

    This class models the behavior of a damper, calculating the air flow rate based on 
    the damper position and other parameters.

    Attributes:
        a (float): A parameter that determines the shape of the air flow curve.
        b (float): Calculated parameter for the air flow equation.
        c (float): Calculated parameter for the air flow equation.
        input (dict): Dictionary to store input values.
        output (dict): Dictionary to store output values.
        parameter (dict): Dictionary of parameters with their lower and upper bounds.
        _config (dict): Configuration dictionary for the system.

    Args:
        a (float, optional): Shape parameter for the air flow curve. Defaults to 5.
        **kwargs: Additional keyword arguments passed to the parent Damper class.

    Note:
        The air flow rate is calculated using the equation: m = a*exp(b*u) + c,
        where m is the air flow rate, u is the damper position, and a, b, c are parameters.
    """

    sp = [get_signature_pattern_1()]

    def __init__(self, a=5, **kwargs):
        """
        Initialize the DamperSystem.

        Args:
            a (float, optional): Shape parameter for the air flow curve. Defaults to 5.
            **kwargs: Additional keyword arguments passed to the parent Damper class.
        """
        super().__init__(**kwargs)
        self.a = a
        self.b = None
        self.c = None

        self.input = {"damperPosition": None}
        self.output = {"airFlowRate": None}
        self.parameter = {
            "a": {"lb": 0.0001, "ub": 5},
            "nominalAirFlowRate.hasValue": {"lb": 0.0001, "ub": 5}
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

    def initialize(self, startTime=None, endTime=None, stepSize=None, model=None):
        """
        Initialize the DamperSystem by calculating parameters b and c.

        Args:
            startTime: The start time for initialization.
            endTime: The end time for initialization.
            stepSize: The step size for initialization.
            model: The model object, if any.
        """
        self.c = -self.a  # Ensures that m=0 at u=0
        self.b = math.log((self.nominalAirFlowRate.hasValue-self.c)/self.a)  # Ensures that m=nominalAirFlowRate at u=1

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        """
        Perform a single step of the simulation.

        This method calculates the air flow rate based on the current damper position.

        Args:
            secondTime: The current time in seconds.
            dateTime: The current date and time.
            stepSize: The size of the time step.
        """
        m_a = self.a * math.exp(self.b * self.input["damperPosition"]) + self.c
        self.output["damperPosition"] = self.input["damperPosition"]
        self.output["airFlowRate"] = m_a