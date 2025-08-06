# Standard library imports
import datetime
from typing import Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    Exact,
    MultiPath,
    Node,
    Optional_,
    SignaturePattern,
    SinglePath,
)


def get_signature_pattern():
    """
    Creates and returns a SignaturePattern for the DamperSystem.

    Returns:
        SignaturePattern: A configured SignaturePattern object for the DamperSystem.
    """
    node0 = Node(cls=core.namespace.S4BLDG.Damper)
    node1 = Node(cls=core.namespace.S4BLDG.Controller)
    node2 = Node(cls=core.namespace.SAREF.OpeningPosition)
    node3 = Node(cls=core.namespace.SAREF.Property)
    node4 = Node(cls=core.namespace.SAREF.PropertyValue)
    node5 = Node(cls=core.namespace.XSD.float)
    node6 = Node(cls=core.namespace.S4BLDG.NominalAirFlowRate)
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="DamperSystem", priority=0
    )

    # Add edges to the signature pattern
    sp.add_triple(
        Exact(subject=node1, object=node2, predicate=core.namespace.SAREF.controls)
    )
    sp.add_triple(
        Exact(subject=node2, object=node0, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_triple(
        Exact(subject=node1, object=node3, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Optional_(subject=node4, object=node5, predicate=core.namespace.SAREF.hasValue)
    )
    sp.add_triple(
        Optional_(
            subject=node4,
            object=node6,
            predicate=core.namespace.SAREF.isValueOfProperty,
        )
    )
    sp.add_triple(
        Optional_(
            subject=node0, object=node4, predicate=core.namespace.SAREF.hasPropertyValue
        )
    )

    # Configure inputs, parameters, and modeled nodes
    sp.add_input("damperPosition", node1, "inputSignal")
    sp.add_parameter("nominalAirFlowRate", node5)
    sp.add_modeled_node(node0)

    return sp


def get_signature_pattern_brick():
    """
    Creates and returns a BRICK-only SignaturePattern for the DamperSystem.

    Returns:
        SignaturePattern: A configured BRICK-only SignaturePattern object for the DamperSystem.
    """
    node0 = Node(cls=core.namespace.BRICK.Damper)
    node1 = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    node2 = Node(cls=core.namespace.BRICK.Damper_Position_Sensor)
    node3 = Node(cls=core.namespace.BRICK.Air_Flow_Sensor)
    node4 = Node(cls=core.namespace.BRICK.Air_Flow_Setpoint)
    node5 = Node(cls=core.namespace.XSD.float)
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="DamperSystemBrick", priority=1
    )

    # Add edges to the signature pattern
    sp.add_triple(
        Exact(subject=node1, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_triple(
        Exact(subject=node2, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_triple(
        Exact(subject=node3, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_triple(
        Exact(subject=node4, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_triple(
        Optional_(subject=node4, object=node5, predicate=core.namespace.BRICK.hasValue)
    )

    # Configure inputs, parameters, and modeled nodes
    sp.add_input("damperPosition", node1, "setpoint")
    sp.add_parameter("nominalAirFlowRate", node5)
    sp.add_modeled_node(node0)

    return sp


class DamperTorchSystem(core.System, nn.Module):
    r"""
    A damper system model implemented with PyTorch for gradient-based optimization.

    This model represents a damper that controls air flow rate based on damper position,
    using an exponential equation for accurate flow control representation.

    Mathematical Formulation
    -----------------------

    The damper characteristic is calculated using an exponential equation:

        .. math::

            \dot{m} = a \cdot e^{b \cdot u} + c

    where:
       - :math:`\dot{m}` is the air flow rate [m³/s]
       - :math:`a` is the shape parameter
       - :math:`b` is calculated to ensure :math:`\dot{m} = \dot{m}_{nom}` at :math:`u = 1`
       - :math:`c` is calculated to ensure :math:`\dot{m} = 0` at :math:`u = 0`
       - :math:`u` is the damper position (0-1)
       - :math:`\dot{m}_{nom}` is the nominal air flow rate [m³/s]

    The parameters :math:`b` and :math:`c` are calculated during initialization:

        .. math::

            c = -a

        .. math::

            b = \ln(\frac{\dot{m}_{nom} - c}{a})

    where:
       - :math:`c = -a` ensures zero flow at closed position
       - :math:`b` is calculated to ensure nominal flow at fully open position

    Parameters
    ----------
    a : float
        Shape parameter for the air flow curve. Controls the non-linearity
        of the damper characteristic. Higher values result in more non-linear behavior.
    nominalAirFlowRate : float
        Nominal air flow rate [m³/s] at fully open position

    Attributes
    ----------
    input : Dict[str, Scalar]
        Dictionary containing input ports:
        - "damperPosition": Damper position (0-1)
    output : Dict[str, Scalar]
        Dictionary containing output ports:
        - "damperPosition": Damper position (0-1)
        - "airFlowRate": Air flow rate [m³/s]
    parameter : Dict[str, Dict[str, float]]
        Dictionary containing parameter bounds for calibration:
        - "a": {"lb": 0.0001, "ub": 5}
        - "nominalAirFlowRate": {"lb": 0.0001, "ub": 5}
    a : torch.tps.Parameter
        Shape parameter, stored as a PyTorch parameter
    nominalAirFlowRate : torch.tps.Parameter
        Nominal air flow rate [m³/s], stored as a PyTorch parameter
    b : torch.Tensor
        Exponential coefficient calculated during initialization
    c : torch.Tensor
        Offset coefficient calculated during initialization

    Notes
    -----
    Damper Characteristics:
       - The exponential characteristic provides a more realistic representation
         of damper behavior compared to a linear relationship
       - The shape parameter 'a' controls the non-linearity of the flow curve
       - Higher values of 'a' result in more non-linear behavior
       - The model ensures zero flow at closed position and nominal flow at
         fully open position

    Implementation Details:
       - The model uses PyTorch tensors for gradient-based optimization
       - Parameters 'a' and 'nominalAirFlowRate' are stored as non-trainable
         PyTorch parameters
       - Parameters 'b' and 'c' are calculated during initialization
       - The model assumes ideal damper behavior (no hysteresis or deadband)
    """

    sp = [get_signature_pattern(), get_signature_pattern_brick()]

    def __init__(
        self,
        a: float = 1,
        nominalAirFlowRate: float = 100
        * 1.225
        / 3600,  # 1 air-change per hour for 100 m³ space
        **kwargs,
    ):
        """
        Initialize the damper system model.

        Args:
            a: Shape parameter for the air flow curve
            nominalAirFlowRate: Nominal air flow rate [m³/s]
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store parameters as tps.Parameters for gradient tracking
        self.a = tps.Parameter(
            torch.tensor(a, dtype=torch.float64), requires_grad=False
        )
        self.nominalAirFlowRate = tps.Parameter(
            torch.tensor(nominalAirFlowRate, dtype=torch.float64), requires_grad=False
        )

        # Define inputs and outputs as private variables
        self._input = {"damperPosition": tps.Scalar()}
        self._output = {"damperPosition": tps.Scalar(0), "airFlowRate": tps.Scalar(0)}

        # Define parameters for calibration
        self.parameter = {
            "a": {"lb": 0.0001, "ub": 5},
            "nominalAirFlowRate": {"lb": 0.0001, "ub": 5},
        }

        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    @property
    def config(self):
        """Get the configuration of the damper system."""
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the damper system.

        Returns:
            dict: Dictionary containing input ports:
                - "damperPosition": Damper position (0-1)
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the damper system.

        Returns:
            dict: Dictionary containing output ports:
                - "damperPosition": Damper position (0-1)
                - "airFlowRate": Air flow rate [m³/s]
        """
        return self._output

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """Initialize the damper system."""
        # Initialize I/O
        for input in self.input.values():
            input.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )
        for output in self.output.values():
            output.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )

        # Calculate b and c parameters
        self.c = -self.a.get()  # Ensures that m=0 at u=0
        self.b = torch.log(
            (self.nominalAirFlowRate.get() - self.c) / self.a.get()
        )  # Ensures that m=nominalAirFlowRate at u=1

        self.INITIALIZED = True

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """
        Perform one step of the damper system simulation.

        The damper characteristic is calculated using an exponential equation:
        m = a * exp(b * u) + c
        where:
        - m is the air flow rate
        - a is the shape parameter
        - b is calculated to ensure m=nominalAirFlowRate at u=1
        - c is calculated to ensure m=0 at u=0
        - u is the damper position (0-1)
        """
        # Get input damper position (assumed to be a tensor)
        damper_position = self.input["damperPosition"].get()

        # Calculate air flow rate using exponential equation
        air_flow_rate = self.a.get() * torch.exp(self.b * damper_position) + self.c

        # Update outputs
        self.output["damperPosition"].set(damper_position, stepIndex)
        self.output["airFlowRate"].set(air_flow_rate, stepIndex)
