# Standard library imports
import datetime
from typing import List

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    Exact,
    Node,
    Optional_,
    SignaturePattern,
)


class DamperTorchSystem(core.System, nn.Module):
    r"""
    A damper system model implemented with PyTorch for gradient-based optimization.

    This model represents a damper that controls air flow rate based on damper position,
    using an exponential equation for accurate flow control representation. Supports
    vectorized operation across multiple parallel branches via the n_c dimension.

    Args:
        a : Shape parameter for the air flow curve. Controls the non-linearity
            of the damper characteristic. Higher values result in more non-linear behavior.
        nominalAirFlowRate : Nominal air flow rate [kg/s] at fully open position.

    Mathematical Formulation
    ========================

    The damper characteristic is calculated using an exponential equation:

        .. math::

            \dot{m} = a \cdot e^{b \cdot u} + c

    where:
       - :math:`\dot{m}` is the air flow rate [kg/s]
       - :math:`a` is the shape parameter
       - :math:`b` is calculated to ensure :math:`\dot{m} = \dot{m}_{nom}` at :math:`u = 1`
       - :math:`c` is calculated to ensure :math:`\dot{m} = 0` at :math:`u = 0`
       - :math:`u` is the damper position (0-1)
       - :math:`\dot{m}_{nom}` is the nominal air flow rate [kg/s]

    The parameters :math:`b` and :math:`c` are calculated during initialization:

        .. math::

            c = -a

        .. math::

            b = \ln(\frac{\dot{m}_{nom} - c}{a})

    where:
       - :math:`c = -a` ensures zero flow at closed position
       - :math:`b` is calculated to ensure nominal flow at fully open position

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
       - Parameters 'a' and 'nominalAirFlowRate' are stored as tps.Parameter and
         expanded to n_c dimension during initialize() for parallel branches
       - Parameters 'b' and 'c' are calculated during initialization
       - The model assumes ideal damper behavior (no hysteresis or deadband)
       - Uses tps.Scalar for ports (not tps.Vector) - multiple parallel instances
         are handled via the n_c dimension, not the n_v dimension
       - n_c (parallel components) is set before initialize() and used for vectorization
    """

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
            a: Shape parameter for the air flow curve.
            nominalAirFlowRate: Nominal air flow rate [kg/s].
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Create parameters as scalars - expanded to n_c in initialize()
        self.a = tps.Parameter(
            torch.tensor(a, dtype=torch.float64), requires_grad=False
        )
        self.nominalAirFlowRate = tps.Parameter(
            torch.tensor(nominalAirFlowRate, dtype=torch.float64), requires_grad=False
        )

        # Define inputs and outputs using Scalar (n_c handles vectorization)
        self._input = {"damperPosition": tps.Scalar()}
        self._output = {
            "damperPosition": tps.Scalar(),
            "airFlowRate": tps.Scalar(),
        }

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
                - "damperPosition": Damper position (0-1). Shape: (n_s, n_c).
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the damper system.

        Returns:
            dict: Dictionary containing output ports:
                - "damperPosition": Damper position (0-1). Shape: (n_s, n_c).
                - "airFlowRate": Air flow rate [kg/s]. Shape: (n_s, n_c).
        """
        return self._output

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: int,
    ) -> None:
        """Initialize the damper system."""
        # Initialize I/O
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        
        # Initialize input/output ports (Scalar with n_c handled by data shape)
        for port in self.input.values():
            port.initialize(n_t=max_timesteps, n_s=batch_size)
                
        for port in self.output.values():
            port.initialize(n_t=max_timesteps, n_s=batch_size)

        # Expand parameters to n_c dimension for vectorization
        self.a = self.a.expand_to_n_c(self.n_c)
        self.nominalAirFlowRate = self.nominalAirFlowRate.expand_to_n_c(self.n_c)

        # Calculate b and c parameters (vectorized for n_c)
        self.c = -self.a.get()  # Ensures that m=0 at u=0
        self.b = torch.log(
            (self.nominalAirFlowRate.get() - self.c) / self.a.get()
        )  # Ensures that m=nominalAirFlowRate at u=1

        self.INITIALIZED = True

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one step of the damper system simulation.

        The damper characteristic is calculated using an exponential equation:
        m = a * exp(b * u) + c
        where:
        - m is the air flow rate [kg/s]
        - a is the shape parameter (shape: (n_c,))
        - b is calculated to ensure m=nominalAirFlowRate at u=1
        - c is calculated to ensure m=0 at u=0
        - u is the damper position (0-1)
        
        All calculations are vectorized via n_c dimension.
        """
        # Get input damper position - shape: (n_s, n_c)
        damper_position = self.input["damperPosition"].get()

        # Calculate air flow rate using exponential equation
        # Broadcasting: (n_s, n_c) * (n_c,) -> (n_s, n_c)
        air_flow_rate = self.a.get() * torch.exp(self.b * damper_position) + self.c

        # Update outputs - shape: (n_s, n_c)
        self.output["damperPosition"]._set(damper_position, i_t=step_index)
        self.output["airFlowRate"]._set(air_flow_rate, i_t=step_index)


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the damper component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the damper component.
    """
    node0 = Node(cls=core.namespace.S4BLDG.Damper)
    node1 = Node(cls=core.namespace.S4BLDG.Controller)
    node2 = Node(cls=core.namespace.SAREF.OpeningPosition)
    node3 = Node(cls=core.namespace.SAREF.Property)
    node4 = Node(cls=core.namespace.SAREF.PropertyValue)
    node5 = Node(cls=core.namespace.XSD.float)
    node6 = Node(cls=core.namespace.S4BLDG.NominalAirFlowRate)
    sp = SignaturePattern(id="damper_signature_pattern")

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


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the damper component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the damper component.
    """
    node0 = Node(cls=core.namespace.BRICK.Damper)
    node1 = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    node2 = Node(cls=core.namespace.BRICK.Damper_Position_Sensor)
    node3 = Node(cls=core.namespace.BRICK.Air_Flow_Sensor)
    node4 = Node(cls=core.namespace.BRICK.Air_Flow_Setpoint)
    node5 = Node(cls=core.namespace.XSD.float)
    sp = SignaturePattern(id="damper_signature_pattern_brick")

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


DamperTorchSystem.add_signature_pattern(brick_signature_pattern())
DamperTorchSystem.add_signature_pattern(saref_signature_pattern())
