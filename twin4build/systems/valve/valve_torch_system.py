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
    StepRule,
    AnyPathRule,
    Node,
    OptionalRule,
    SignaturePattern,
    PathRule,
)


class ValveTorchSystem(core.System, nn.Module):
    r"""
    A valve system model implemented with PyTorch for gradient-based optimization.

    This model represents a valve that controls water flow rate based on valve position.
    The valve characteristic is modeled using the valve authority equation, which provides
    a more accurate representation of the valve's behavior compared to a simple linear
    relationship.

    Args:
        waterFlowRateMax: Maximum water flow rate [kg/s]
        valveAuthority: Valve authority (0-1)
        **kwargs: Additional keyword arguments

    Mathematical Formulation
    -----------------------

    The valve characteristic is calculated using the valve authority equation:

        .. math::

            u_{norm} = \frac{u}{\sqrt{u^2 (1-a) + a}}

    where:
       - :math:`u` is the valve position (0-1)
       - :math:`a` is the valve authority (0-1)
       - :math:`u_{norm}` is the normalized valve position

    The water flow rate is then calculated as:

        .. math::

            \dot{m}_w = u_{norm} \cdot \dot{m}_{w,max}

    where:
       - :math:`\dot{m}_w` is the water flow rate [kg/s]
       - :math:`\dot{m}_{w,max}` is the maximum water flow rate [kg/s]

    Notes
    -----
    Valve Authority Characteristics:
       - Linear (a = 0): Flow rate is directly proportional to valve position
       - Equal Percentage (a = 1): Flow rate changes exponentially with valve position
       - Mixed (0 < a < 1): Combination of linear and equal percentage characteristics

    Implementation Details:
       - The model uses PyTorch tensors for gradient-based optimization
       - All parameters are stored as non-trainable PyTorch parameters
       - The valve authority equation provides better control at low flow rates
       - The model assumes ideal valve behavior (no hysteresis or deadband)
    """

    def __init__(
        self,
        waterFlowRateMax: Optional[float] = 1000
        / (
            (60 - 45) * 4180
        ),  # Provide 1000 W of heating power when cooling from 60 to 45 degrees
        valveAuthority: Optional[float] = 1,  # Linear relation by default
        **kwargs,
    ):
        """
        Initialize the valve system model.

        Args:
            waterFlowRateMax: Maximum water flow rate [kg/s]
            valveAuthority: Valve authority (0-1)
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store parameters as tps.Parameters for gradient tracking
        self.waterFlowRateMax = tps.Parameter(
            torch.tensor(waterFlowRateMax, dtype=torch.float64),
            requires_grad=False,
            scaling="log",
        )
        self.valveAuthority = tps.Parameter(
            torch.tensor(valveAuthority, dtype=torch.float64), requires_grad=False
        )

        # Define inputs and outputs as private variables
        self._input = {"valvePosition": tps.Scalar()}
        self._output = {"valvePosition": tps.Scalar(), "waterFlowRate": tps.Scalar(0)}

        # Define parameters for calibration.  Physically realistic
        # ranges for an FCU / VAV reheat valve -- wider bounds give the
        # auto-estimator useless feasible space and let it pin
        # ``valveAuthority`` to 0 (valve has *no* effect on flow) or
        # ``waterFlowRateMax`` to ridiculous 10 kg/s.  ``Valve`` is
        # always wired downstream of a single coil in the supported
        # topology, so loop-level sizing fits inside the bounds below.
        #
        # ``waterFlowRateMax`` is log-scaled (see ``tps.Parameter``
        # constructor above), so its lower bound MUST be > 0 -- the
        # normalisation otherwise hits a ``log(0)`` assertion inside
        # :class:`tps.TensorParameter`.
        self.parameter = {
            # FCU / VAV reheat valve: typical 0.01 - 0.5 kg/s.  Lower
            # cap at 5e-3 still allows a very small zone; below that
            # the valve cannot deliver enough water to register a
            # measurable supply-air rise.
            "waterFlowRateMax": {"lb": 5e-3, "ub": 1.0},
            # Authority < 0.3 in this characteristic (see class
            # docstring: ``u_norm = u / sqrt(u^2*(1-a) + a)``) makes
            # the valve barely affect flow over much of its travel and
            # the resulting hydraulic loop is uncontrollable in
            # practice, so estimates that low are not physical.  Upper
            # bound 1.0 is the ideal linear-flow limit.
            "valveAuthority": {"lb": 0.3, "ub": 1.0},
        }

        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    @property
    def config(self):
        """Get the configuration of the valve system."""
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the valve system.

        Returns:
            dict: Dictionary containing input ports:
                - "valvePosition": Valve position (0-1)
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the valve system.

        Returns:
            dict: Dictionary containing output ports:
                - "valvePosition": Valve position (0-1)
                - "waterFlowRate": Water flow rate [kg/s]
        """
        return self._output

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the valve system."""
        # Initialize I/O
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        if hasattr(self, "_n_c_compiled") and self._n_c_compiled > 1:
            self.n_c = self._n_c_compiled
        else:
            self.n_c = 1

        for input in self.input.values():
            input.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )
        for output in self.output.values():
            output.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )

        # Expand parameters to n_c dimension for vectorization
        self.waterFlowRateMax = self.waterFlowRateMax.expand_to_n_c(self.n_c)
        self.valveAuthority = self.valveAuthority.expand_to_n_c(self.n_c)

        self.INITIALIZED = True

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one step of the valve system simulation.

        The valve characteristic is calculated using the valve authority equation:
        u_norm = u / sqrt(u^2 * (1-a) + a)
        where:
        - u is the valve position (0-1)
        - a is the valve authority (0-1)
        - u_norm is the normalized valve position

        The water flow rate is then calculated as:
        m_w = u_norm * waterFlowRateMax

        Thin port-I/O wrapper around :meth:`forward` (the single source of
        truth for the math).
        """
        # Clone to detach from the input _tensor's storage, preventing
        # version-counter conflicts with jacrev when _tensor is overwritten
        # at the next timestep by _assign_component_inputs.
        inputs = {"valvePosition": self.input["valvePosition"].get().clone()}
        _, outs = self.forward(None, inputs, self._forward_params(), step_size)
        self.output["valvePosition"]._set(outs["valvePosition"], i_t=step_index)
        self.output["waterFlowRate"]._set(outs["waterFlowRate"], i_t=step_index)

    #: Physical parameters, in a fixed order (the ``forward`` theta contract).
    PARAM_NAMES = ("waterFlowRateMax", "valveAuthority")

    def forward(self, x, inputs, params, sample_time):
        """Pure algebraic map ``(inputs, params) -> outputs`` (stateless).

        Functorch-compatible re-expression of :meth:`do_step`.  ``inputs`` provides
        ``valvePosition``; ``params`` a dict for :attr:`PARAM_NAMES`.  ``x`` (an
        empty state) is passed through.  Returns
        ``(x, {"valvePosition", "waterFlowRate"})``.
        """
        vp = inputs["valvePosition"]
        a = params["valveAuthority"]
        u_norm = vp / torch.sqrt(vp**2 * (1 - a) + a)
        m_w = u_norm * params["waterFlowRateMax"]
        return x, {"valvePosition": vp, "waterFlowRate": m_w}


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the valve component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the valve component.
    """
    node0 = Node(cls=core.namespace.S4BLDG.Valve)  # supply valve
    node1 = Node(cls=core.namespace.S4BLDG.Controller)
    node2 = Node(cls=core.namespace.SAREF.OpeningPosition)
    sp = SignaturePattern()

    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.controls)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.SAREF.isPropertyOf)
    )

    sp.add_input("valvePosition", node1, "inputSignal")
    sp.add_modeled_node(node0)

    return sp


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the valve component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the valve component.
    """
    node0 = Node(cls=core.namespace.BRICK.Valve)
    node1 = Node(cls=core.namespace.BRICK.Valve_Position_Setpoint)
    node2 = Node(cls=core.namespace.BRICK.Water_Flow_Sensor)

    sp = SignaturePattern(id="valve_signature_pattern_brick")

    sp.add_rule(
        StepRule(subject=node1, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )

    sp.add_input("valvePosition", node1, "setpoint")
    sp.add_modeled_node(node0)

    return sp


ValveTorchSystem.add_signature_pattern(brick_signature_pattern())
ValveTorchSystem.add_signature_pattern(saref_signature_pattern())
