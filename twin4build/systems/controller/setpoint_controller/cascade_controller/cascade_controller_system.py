# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.saref4syst.declarations import Child, Input, Output


class CascadeControllerSystem(core.System):
    r"""
    Cascade controller: two sub-controllers, the outer one's output the inner one's setpoint.

    Controller A's clamped output becomes the setpoint for Controller B:

    .. code-block:: text

        setpointValue_a ──┐
                          ├── ctrl_a ──[output]── ctrl_b ── inputSignal
        actualValue_a ────┘                        │
        actualValue_b ────────────────────────────┘

    **ctrl_a** (outer loop) receives the external setpoint/feedback and produces
    an intermediate signal.  For setpoint-based controllers (PID), it receives
    ``setpointValue_a`` and ``actualValue_a``.  For direct-signal controllers
    (e.g. SATCompensated), it receives ``actualValue_a`` as its driving signal.

    **ctrl_b** (inner loop) receives ctrl_a's output as its setpoint and
    ``actualValue_b`` as feedback, and produces the final output.

    Both are PIDs (the children ``ctrl_a`` and ``ctrl_b``); a subclass
    declares other children (:class:`SATCompensatedControllerSystem`: a SAT
    rule outside, a PID inside).  Each child's inputs are routed by its ports:
    a PID gets the setpoint and the feedback, a SAT rule its driving signal
    on ``supplyAirTemp``.

    Example use-cases:
        - **VAV damper (PID+PID)**: A = temperature loop, B = airflow loop
        - **SAT-compensated VAV (SAT+PID)**: A = SAT→flow setpoint, B = flow→damper

    Args:
        ctrl_a: The outer PID's values, a dict (``ctrl_a=dict(kp=0.1, Ti=10.0)``).
        ctrl_b: The inner PID's values, a dict.

    Example:
        >>> cascade = CascadeControllerSystem(
        ...     ctrl_a=dict(kp=0.1, Ti=10.0, is_reverse=True),
        ...     ctrl_b=dict(kp=0.5, Ti=5.0, is_reverse=True),
        ...     id="cascade_pid",
        ... )
    """

    #: The old flat names of the children's values, and the dicts they came in
    #: (removed in 2.1).
    DEPRECATED_KEYWORDS = {
        **{f"{old}_{side}": f"ctrl_{side}.{new}" for side in ("a", "b") for old, new in (('kp', 'kp'), ('Ti', 'Ti'), ('Td', 'Td'), ('output_min', 'output_min'), ('output_max', 'output_max'), ('isReverse', 'is_reverse'))},
        "controller_a_kwargs": "ctrl_a",
        "controller_b_kwargs": "ctrl_b",
    }
    REMOVED_ATTRIBUTES = {
        name: "a cascade's children are declared: a subclass declares other ones (ctrl_a = Child(...))"
        for name in ("controller_a", "controller_b", "controller_a_class", "controller_b_class")
    }

    setpointValue_a = Input(doc="The outer loop's setpoint (not read by a SAT rule)")
    actualValue_a = Input(doc="The outer loop's feedback, or a SAT rule's driving signal")
    actualValue_b = Input(doc="The inner loop's feedback")
    inputSignal = Output(init=0.0, doc="The inner loop's command")
    ctrl_a = Child(PIDControllerSystem, kp=0.1, Ti=10.0, Td=0.0, is_reverse=True)
    ctrl_b = Child(PIDControllerSystem, kp=0.5, Ti=5.0, Td=0.0, is_reverse=True)

    @staticmethod
    def _resolve_sub_params(sub, prefix, params):
        """Full physical-parameter dict for a sub-controller: estimated values
        from ``params`` (keyed ``"<prefix>.<name>"``), the rest from the
        sub-controller's own values."""
        out = {}
        for name in sub.PARAM_NAMES:
            key = f"{prefix}.{name}"
            out[name] = params[key] if key in params else getattr(sub, name).get()
        return out

    @staticmethod
    def _route_forward_inputs(ctrl, setpoint, feedback):
        """The inputs a sub-controller's ``forward`` expects, from the cascade's
        signals."""
        if "setpointValue" in ctrl.input and "actualValue" in ctrl.input:
            return {"setpointValue": setpoint, "actualValue": feedback}
        if "supplyAirTemp" in ctrl.input:
            return {"supplyAirTemp": feedback}
        raise ValueError(
            f"Cannot route signals to {ctrl.__class__.__name__}: unrecognized input ports {list(ctrl.input.keys())}"
        )

    def derive(self, p, step_size):
        """Each sub-controller's parameters (estimated values from ``p``, the
        rest the sub-controller's own) and what it derives from them."""
        p_a = self._resolve_sub_params(self.ctrl_a, "ctrl_a", p)
        p_b = self._resolve_sub_params(self.ctrl_b, "ctrl_b", p)
        return {
            "ctrl_a_params": self.ctrl_a.full_params(p_a, step_size),
            "ctrl_b_params": self.ctrl_b.full_params(p_b, step_size),
        }

    def forward(self, x, inputs, params, step_size):
        """One step of the cascade: ctrl_a -> ctrl_b.

        State is ``[ctrl_a_state | ctrl_b_state]`` (the order
        :meth:`System.get_state` produces; a stateless sub-controller such as
        ``SATLinearRuleSystem`` contributes zero width).  ``params`` is keyed
        by the composite attr path (``"ctrl_a.kp"``, ``"ctrl_b.Ti"``, ...).
        Controller A's output feeds controller B within the same step.
        """
        p_a, p_b = params["ctrl_a_params"], params["ctrl_b_params"]

        n_a = self.ctrl_a.state_size()
        x_a, x_b = x[..., :n_a], x[..., n_a:]

        in_a = self._route_forward_inputs(
            self.ctrl_a, inputs["setpointValue_a"], inputs["actualValue_a"]
        )
        x_a_n, out_a = self.ctrl_a.forward(x_a, in_a, p_a, step_size)

        in_b = self._route_forward_inputs(
            self.ctrl_b, out_a["inputSignal"], inputs["actualValue_b"]
        )
        x_b_n, out_b = self.ctrl_b.forward(x_b, in_b, p_b, step_size)

        return torch.cat([x_a_n, x_b_n], dim=-1), {"inputSignal": out_b["inputSignal"]}
