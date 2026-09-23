# Standard library imports
import datetime
from typing import List, Optional

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    StepRule,
    Node,
    OptionalRule,
    SignaturePattern,
)


class DamperSystem(core.System, nn.Module):
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
    ------------------------

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
        c: Optional[float] = None,
        c_tied: Optional[bool] = None,
        exhaustFlowRatio: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the damper system model.
        Args:
            a: Shape parameter for the air flow curve.
            nominalAirFlowRate: Nominal air flow rate [kg/s].
            exhaustFlowRatio: The terminal's exhaust flow as a ratio of its
                supply flow [-] (``exhaustAirFlowRate`` output); estimable.

        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Create parameters as scalars - expanded to n_c in initialize()
        self.a = tps.Parameter(
            torch.tensor(a, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.nominalAirFlowRate = tps.Parameter(
            torch.tensor(nominalAirFlowRate, dtype=tps.float_dtype()),
            requires_grad=False,
        )
        # Offset ``c`` of ``m = a exp(b u) + c``.  Tied (the default): ``c = -a``
        # follows ``a`` so the closed damper passes nothing.  Given (or set
        # later, see :meth:`set_c`): a free parameter, and ``a + c`` is the
        # flow through the closed damper -- the minimum / leakage flow a
        # pressure-independent VAV keeps while its fan runs.
        self._c_tied = (c is None) if c_tied is None else bool(c_tied)
        self.c = tps.Parameter(
            torch.tensor(-a if c is None else c, dtype=tps.float_dtype()),
            requires_grad=False,
        )

        # A terminal's exhaust: its supply flow times a ratio (the exhaust
        # side of a VAV box is rarely metered per branch).
        self.exhaustFlowRatio = tps.Parameter(
            torch.tensor(exhaustFlowRatio, dtype=tps.float_dtype()),
            requires_grad=False,
        )
        # Define inputs and outputs using Scalar (n_c handles vectorization).
        # ``fanSpeed`` gates the flow: a pressure-controlled branch passes
        # nothing while the unit's fan is stopped (1.0 when unwired).
        self._input = {
            "damperPosition": tps.Scalar(),
            "fanSpeed": tps.Scalar(1.0, optional=True),
        }
        self._output = {
            "damperPosition": tps.Scalar(),
            "airFlowRate": tps.Scalar(),
            "exhaustAirFlowRate": tps.Scalar(),
        }


        # Define parameters for calibration.  Tightened to the
        # physically-realistic VAV-branch / AHU-damper range so the
        # auto-estimator can't pin a damper at 1e-4 kg/s (effectively
        # zero flow, makes the coil's energy balance singular) or run
        # the shape coefficient ``a`` into a region where the
        # exponential characteristic ``m = a*exp(b*u) + c`` is monotone
        # but numerically ill-conditioned.
        self.parameter = {
            # log-scaled (lb > 0 mandatory).  ``a`` is a unit-less
            # shape coefficient; values much above 5 give very steep
            # rise near ``u=0`` and saturate immediately, values below
            # 0.1 give nearly linear damper response (lose the physics
            # of the equal-percentage characteristic).
            "a": {"lb": 0.1, "ub": 5.0},
            # Branch / AHU damper kg/s.  Range covers a 100 m³ VAV
            # zone at 1 ach (~ 0.03 kg/s) up to a large primary AHU
            # branch (~ 5 kg/s).  Below 0.01 kg/s the coil's
            # energy balance becomes singular.
            "nominalAirFlowRate": {"lb": 0.001, "ub": 5.0},
            # Offset of the characteristic (kg/s); estimable once untied.
            # ``a + c`` is the closed-damper flow: ``c < -a`` gives a dead
            # band (the flow is clamped at zero), ``c > -a`` a leakage.
            "c": {"lb": -5.0, "ub": 1.0},
            # Exhaust-to-supply ratio of the terminal.
            "exhaustFlowRatio": {"lb": 0.3, "ub": 1.5},
        }

        self._config = {
            "parameters": ["a", "nominalAirFlowRate", "c", "c_tied", "exhaustFlowRatio"]
        }

        self.INITIALIZED = False

    @property
    def c_tied(self) -> bool:
        """``True`` while ``c`` follows ``-a`` (zero flow when closed)."""
        return self._c_tied

    @c_tied.setter
    def c_tied(self, value) -> None:
        self._c_tied = bool(value)

    def set_c(self, value) -> None:
        """Untie ``c`` and set it: from now on ``c`` is its own (estimable)
        parameter and ``a + c`` is the closed-damper flow."""
        self.c = tps.Parameter(
            torch.as_tensor(value, dtype=tps.float_dtype()).clone(), requires_grad=False
        )
        self._c_tied = False

    def get_estimable_parameters(self):
        """Own estimable parameters; ``c`` only once untied (tied, it is
        derived from ``a`` and estimating it would change nothing)."""
        return [e for e in super().get_estimable_parameters() if not (self._c_tied and e[1] == "c")]

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

        # Determine n_c.  Order of preference:
        #   1. ``_n_c_batched`` set by the batcher (overrides everything).
        #   2. An ``n_c`` already assigned by an outer wrapper (e.g. the
        #      vectorized :class:`AirHandlingUnitSystem` flattens
        #      its (n_s, n_c, n_v) Vector inputs into a per-branch damper
        #      ``n_c = n_c_ahu * n_v`` *before* calling ``initialize``).
        #   3. Default to 1 when neither caller set anything > 1.
        if hasattr(self, "_n_c_batched") and getattr(self, "_n_c_batched") > 1:
            self.n_c = self._n_c_batched
        elif self.n_c <= 1:
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
        self.a = self.a.expand_to_n_c(self.n_c)
        self.nominalAirFlowRate = self.nominalAirFlowRate.expand_to_n_c(self.n_c)
        if self._c_tied:
            # Re-tie: ``c`` mirrors the (possibly re-estimated) ``a``.
            self.c = tps.Parameter(
                (-self.a.get()).detach().clone(), requires_grad=False
            )
        self.c = self.c.expand_to_n_c(self.n_c)
        self.exhaustFlowRatio = self.exhaustFlowRatio.expand_to_n_c(self.n_c)

        # ``b`` ensures m = nominalAirFlowRate at u = 1 (vectorized for n_c)

        self.b = torch.log(
            (self.nominalAirFlowRate.get() - self.c.get()) / self.a.get()
        )

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
        b and c are recomputed from the current a and nominalAirFlowRate
        so that gradients flow correctly during estimation.

        Thin port-I/O wrapper around :meth:`forward` (the single source of
        truth for the math).
        """
        inputs = {
            "damperPosition": self.input["damperPosition"].get(),
            "fanSpeed": self.input["fanSpeed"].get(),
        }
        _, outs = self.forward(None, inputs, self._forward_params(), step_size)
        self.output["damperPosition"]._set(
            outs["damperPosition"], i_t=step_index, ic=self.n_c
        )
        self.output["airFlowRate"]._set(
            outs["airFlowRate"], i_t=step_index, ic=self.n_c
        )
        self.output["exhaustAirFlowRate"]._set(
            outs["exhaustAirFlowRate"], i_t=step_index, ic=self.n_c
        )


    #: Physical parameters, in a fixed order (the ``forward`` theta contract).
    PARAM_NAMES = ("nominalAirFlowRate", "a", "c", "exhaustFlowRatio")

    #: Fan speed (0-1) above which the fan is fully "on" for the branch flow.
    FAN_ON_SPEED = 0.1

    @classmethod
    def fan_gate(cls, speed):
        """0 with the fan stopped, 1 once it runs (linear in between): the
        branch is pressure-controlled, so the damper sets the flow while the
        fan runs, and nothing moves when it does not."""
        return torch.clamp(speed / cls.FAN_ON_SPEED, 0.0, 1.0)

    def forward(self, x, inputs, params, sample_time):
        """Pure algebraic map ``(inputs, params) -> outputs`` (stateless).

        Functorch-compatible re-expression of :meth:`do_step`.  ``inputs`` provides
        ``damperPosition`` and, optionally, ``fanSpeed`` (no gating without
        it); ``params`` a dict for :attr:`PARAM_NAMES`.  ``x`` (an empty
        state) is passed through.  Returns
        ``(x, {"damperPosition", "airFlowRate", "exhaustAirFlowRate"})``.
        """
        dp = inputs["damperPosition"]
        a = params["a"]
        c = -a if self._c_tied else params["c"]
        air_flow_rate = self.characteristic(a, params["nominalAirFlowRate"], dp, c)
        fan_speed = inputs.get("fanSpeed")
        if fan_speed is not None:
            air_flow_rate = air_flow_rate * self.fan_gate(fan_speed)
        ratio = params.get("exhaustFlowRatio")
        if ratio is None:
            ratio = self.exhaustFlowRatio.get()
        return x, {
            "damperPosition": dp,
            "airFlowRate": air_flow_rate,
            "exhaustAirFlowRate": air_flow_rate * ratio,
        }


    @staticmethod
    def characteristic(a, nominal, position, c=None):
        """``m(u) = a exp(b u) + c`` with ``b`` such that ``m(1) = nominal``;
        ``c = -a`` (zero flow when closed) unless given.  The flow is clamped
        at zero (``c < -a`` is a dead band) and ``nominal > c`` is enforced
        so ``b`` stays finite.  Shared with ``OccupancySystem``."""
        c = -a if c is None else c
        b = torch.log(torch.clamp(nominal - c, min=1e-9) / a)
        return torch.clamp(a * torch.exp(b * position) + c, min=0.0)


# Deprecated aliases (removed in twin4build 2.1)
DamperTorchSystem = DamperSystem
