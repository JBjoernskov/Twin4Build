# Local application imports
import twin4build.core as core
from twin4build.systems.controller.setpoint_controller.cascade_controller.cascade_controller_system import (
    CascadeControllerSystem,
)
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter
from twin4build.systems.utils.smooth_saturation import clamp


class SATLinearRuleSystem(core.System):
    r"""
    SAT Linear Rule -- standalone rule-based controller.

    Maps AHU supply air temperature to a minimum airflow setpoint using a
    simple linear compensation:

    .. math::

        u = \text{clamp}(u_{base} + K \cdot (T_{SAT} - T_{design}),\; u_{min},\; u_{max})

    Physical Rationale
    ------------------

    The cooling delivered by a VAV box is:

    .. math::

        \dot{Q} = \dot{m} \cdot c_p \cdot (T_{zone} - T_{SAT})

    When :math:`T_{SAT}` rises, less cooling per unit airflow, so a higher
    minimum airflow setpoint is needed. When :math:`T_{SAT}` drops, less
    airflow suffices.

    I/O Ports
    ---------

    Inputs:
        - ``supplyAirTemp``: AHU supply air temperature (°C)

    Outputs:
        - ``inputSignal``: Minimum airflow setpoint fraction (0-1)

    Args:
        base_position: Output at design SAT (default: 0.3)
        sat_design: Design supply air temperature in °C (default: 13.0, ~55°F)
        gain: Compensation gain -- output increase per °C above design
            (default: 0.05, i.e. +5% per degree)
        output_min: Minimum output (default: 0.0)
        output_max: Maximum output (default: 1.0)
        **kwargs: Additional keyword arguments passed to parent classes

    Example:
        >>> rule = SATLinearRuleSystem(
        ...     base_position=0.3, sat_design=13.0, gain=0.05,
        ...     id="sat_rule"
        ... )
        >>> # At SAT=13°C: output = 0.3 (base)
        >>> # At SAT=18°C: output = 0.3 + 0.05*5 = 0.55
        >>> # At SAT=23°C: output = 0.3 + 0.05*10 = 0.80
    """

    supplyAirTemp = Input(doc="Supply air temperature [degC]")
    inputSignal = Output(init=0.0, doc="The minimum airflow command (0-1)")
    base_position = Parameter(0.3, lb=0.0, ub=1.0, doc="Command at the design supply temperature")
    sat_design = Parameter(13.0, lb=0.0, ub=30.0, doc="Design supply air temperature [degC]")
    gain = Parameter(0.05, lb=-0.5, ub=0.5, doc="Command per degree off the design temperature [1/K]")
    output_min = Parameter(0.0, lb=0.0, ub=1.0, doc="Lower limit of the command")
    output_max = Parameter(1.0, lb=0.0, ub=1.0, doc="Upper limit of the command")

    def forward(self, x, inputs, params, step_size):
        """Stateless: the linear SAT compensation.

        ``output = clamp(base_position + gain * (SAT - sat_design),
        output_min, output_max)``
        """
        deviation = inputs["supplyAirTemp"] - params["sat_design"]
        raw_output = params["base_position"] + params["gain"] * deviation
        output = clamp(
            raw_output, lower=params["output_min"], upper=params["output_max"]
        )
        return x, {"inputSignal": output}

    def reset_state(self) -> None:
        """Reset controller state (stateless controller, no-op)."""
        pass


class SATCompensatedControllerSystem(CascadeControllerSystem):
    r"""
    SAT-Compensated Cascade Damper Controller.

    Two-stage cascade that models the real VAV control sequence where the
    AHU supply air temperature drives the minimum airflow setpoint, and a
    PI flow controller modulates the damper to track that setpoint.

    .. code-block:: text

        supplyAirTemp ──> SATLinearRule ──[min_flow_sp]──> PID ──> damper_position
                                                           │
         actualValue_b (airflow) ──────────────────────────┘

    **Stage A** (outer, rule-based): ``SATLinearRuleSystem``
        Maps AHU supply air temperature to a minimum airflow setpoint fraction:
        ``min_flow = clamp(base + gain * (SAT - design), min_a, max_a)``

    **Stage B** (inner, PID): ``PIDControllerSystem``
        Tracks the flow setpoint by modulating the damper:
        ``damper_pos = PID(min_flow - actual_flow)``

    External I/O (inherited from CascadeControllerSystem)
    -----------------------------------------------------

    Inputs:
        - ``setpointValue_a``: Unused (SAT rule ignores setpoints)
        - ``actualValue_a``: AHU supply air temperature (routed to SAT rule)
        - ``actualValue_b``: Actual airflow measurement (routed to PID)

    Outputs:
        - ``inputSignal``: Damper position command (0-1)

    Args:
        base_position: Output at design SAT for SAT rule (default: 0.3)
        sat_design: Design supply air temperature °C (default: 13.0)
        gain: SAT compensation gain (default: 0.05)
        output_min_a: Minimum flow setpoint from SAT rule (default: 0.0)
        output_max_a: Maximum flow setpoint from SAT rule (default: 1.0)
        kp_b: Proportional gain for inner PID (default: 0.5)
        Ti_b: Integral time constant for inner PID (default: 5.0)
        Td_b: Derivative time constant for inner PID (default: 0.0)
        output_min_b: Minimum damper position (default: 0.0)
        output_max_b: Maximum damper position (default: 1.0)
        isReverse_b: Error direction for inner PID (default: True;
            excess flow → close damper)
        **kwargs: Additional keyword arguments (must include ``id``)

    Example:
        >>> controller = SATCompensatedControllerSystem(
        ...     base_position=0.3, sat_design=13.0, gain=0.05,
        ...     kp_b=0.5, Ti_b=5.0, isReverse_b=True,
        ...     id="sat_cascade_ctrl"
        ... )
    """

    def __init__(
        self,
        # SAT rule parameters (ctrl_a)
        base_position: float = 0.3,
        sat_design: float = 13.0,
        gain: float = 0.05,
        output_min_a: float = 0.0,
        output_max_a: float = 1.0,
        # Inner PID parameters (ctrl_b)
        kp_b: float = 0.5,
        Ti_b: float = 5.0,
        Td_b: float = 0.0,
        output_min_b: float = 0.0,
        output_max_b: float = 1.0,
        isReverse_b: bool = True,
        **kwargs,
    ):
        super().__init__(
            controller_a=SATLinearRuleSystem,
            controller_a_kwargs={
                "base_position": base_position,
                "sat_design": sat_design,
                "gain": gain,
                "output_min": output_min_a,
                "output_max": output_max_a,
            },
            controller_b=PIDControllerSystem,
            controller_b_kwargs={
                "kp": kp_b,
                "Ti": Ti_b,
                "Td": Td_b,
                "output_min": output_min_b,
                "output_max": output_max_b,
                "is_reverse": isReverse_b,
            },
            **kwargs,
        )


# Deprecated aliases (removed in twin4build 2.1)
SATCompensatedControllerTorchSystem = SATCompensatedControllerSystem
