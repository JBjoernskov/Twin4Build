# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter


class DamperSystem(core.System):
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

    The offset ``c`` is optional.  Without it the damper is tied: ``c = -a``,
    so the closed damper passes nothing, and estimating ``a`` moves ``c`` with
    it (:meth:`derive`).  Given (``DamperSystem(c=-1.9)``, or ``damper.c =
    -1.9`` later), ``c`` is its own parameter and ``a + c`` is the flow through
    the closed damper: a leakage or the minimum flow a VAV box keeps when
    positive, a dead band when negative.

    Implementation Details:
       - The model assumes ideal damper behavior (no hysteresis or deadband)
       - Parallel dampers are copies along the first dimension, not slots of a
         vector port
    """

    REMOVED_ATTRIBUTES = {
        "set_c": "give the offset: damper.c = value (or DamperSystem(c=value))",
        **{name: "c is optional: absent, it is tied (c = -a); damper.c = value unties it"
           for name in ("c_tied", "_c_tied")},
    }

    damperPosition = Input(doc="Damper position (0-1)")
    fanSpeed = Input(default=1.0, doc="Fan speed (0-1): nothing flows while the fan is stopped")
    damperPosition = Output(doc="Damper position (0-1)")
    airFlowRate = Output(doc="Air flow rate [kg/s]")
    exhaustAirFlowRate = Output(doc="The terminal's exhaust: the supply flow times exhaustFlowRatio [kg/s]")
    # Ranges: a VAV zone of 100 m3 at 1 ach (~0.03 kg/s) to a large primary
    # AHU branch (~5 kg/s); below 0.01 kg/s the coil's energy balance turns
    # singular.  ``a`` much above 5 saturates at once, below 0.1 is linear.
    nominalAirFlowRate = Parameter(
        100 * 1.225 / 3600, lb=0.001, ub=5.0, relative=(0.25, 4.0), doc="Nominal air flow rate [kg/s] (fully open)"
    )
    a = Parameter(1.0, lb=0.1, ub=5.0, scaling="log", doc="Shape of the characteristic [-]")
    c = Parameter(None, lb=-5.0, ub=1.0, doc="Offset of the characteristic [kg/s]; absent: tied to -a")
    exhaustFlowRatio = Parameter(1.0, lb=0.3, ub=1.5, doc="The terminal's exhaust over its supply flow [-]")

    #: What each output depends on (:mod:`twin4build.model.signal_dependencies`); ``c`` counts once given, the
    #: union of the two configurations.  Stateless.
    OUTPUT_DEPENDENCIES = {
        "damperPosition": {"inputs": ("damperPosition",)},
        "airFlowRate": {"inputs": ("damperPosition", "fanSpeed"), "parameters": ("nominalAirFlowRate", "a", "c")},
        "exhaustAirFlowRate": {"inputs": ("damperPosition", "fanSpeed"),
                               "parameters": ("nominalAirFlowRate", "a", "c", "exhaustFlowRatio")},
    }

    #: Fan speed (0-1) above which the fan is fully "on" for the branch flow.
    FAN_ON_SPEED = 0.1

    @classmethod
    def fan_gate(cls, speed):
        """0 with the fan stopped, 1 once it runs (linear in between): the
        branch is pressure-controlled, so the damper sets the flow while the
        fan runs, and nothing moves when it does not."""
        return torch.clamp(speed / cls.FAN_ON_SPEED, 0.0, 1.0)

    def derive(self, p, step_size):
        """Without ``c`` the damper is tied: ``c = -a`` (a closed damper
        passes nothing)."""
        return {} if "c" in p else {"c": -p["a"]}

    def forward(self, x, inputs, p, step_size):
        """Stateless: ``m(u) = a exp(b u) + c`` gated by the fan; the exhaust is
        the supply flow times ``exhaustFlowRatio``."""
        dp = inputs["damperPosition"]
        air_flow_rate = self.characteristic(p["a"], p["nominalAirFlowRate"], dp, p["c"])
        fan_speed = inputs.get("fanSpeed")
        if fan_speed is not None:
            air_flow_rate = air_flow_rate * self.fan_gate(fan_speed)
        return x, {
            "damperPosition": dp,
            "airFlowRate": air_flow_rate,
            "exhaustAirFlowRate": air_flow_rate * p["exhaustFlowRatio"],
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
