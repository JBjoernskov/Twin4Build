# Standard library imports
import datetime
from typing import List, Optional, Type

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)


class CascadeControllerSystem(core.System, nn.Module):
    r"""
    Generic Cascade Controller composed of two arbitrary sub-controllers.

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

    The sub-controller types are passed as class + kwargs, similar to how
    ``ControllerIdentificationTorchSystem`` accepts candidate controllers.

    Signal Routing
    --------------

    Each sub-controller's inputs are detected automatically:

    - If ctrl has ``setpointValue``/``actualValue`` ports (PID-like):
      routes setpoint and feedback normally.
    - If ctrl has ``supplyAirTemp`` port (SAT-compensated):
      routes the feedback/signal to ``supplyAirTemp``.
    - Other controller types: extend ``_route_inputs()`` as needed.

    Example use-cases:
        - **VAV damper (PID+PID)**: A = temperature loop, B = airflow loop
        - **SAT-compensated VAV (SAT+PID)**: A = SAT→flow setpoint, B = flow→damper

    Args:
        controller_a: Class for outer-loop controller (default: PIDControllerSystem)
        controller_a_kwargs: Keyword arguments for controller A (default: PID defaults)
        controller_b: Class for inner-loop controller (default: PIDControllerSystem)
        controller_b_kwargs: Keyword arguments for controller B (default: PID defaults)

        Legacy PID-specific kwargs (used only when controller_a/b are None):
            kp_a, Ti_a, Td_a, output_min_a, output_max_a, isReverse_a:
                PID parameters for controller A
            kp_b, Ti_b, Td_b, output_min_b, output_max_b, isReverse_b:
                PID parameters for controller B

    Example:
        >>> # Classic cascade PID (backward compatible)
        >>> cascade = CascadeControllerSystem(
        ...     kp_a=0.1, Ti_a=10.0, isReverse_a=True,
        ...     kp_b=0.5, Ti_b=5.0, isReverse_b=True,
        ...     id="cascade_pid"
        ... )
        >>>
        >>> # SAT-compensated + PID cascade
        >>> from twin4build.systems.controller.rulebased_controller \
        ...     .sat_compensated_controller.sat_compensated_controller_torch_system \
        ...     import SATCompensatedControllerTorchSystem
        >>> cascade = CascadeControllerSystem(
        ...     controller_a=SATCompensatedControllerTorchSystem,
        ...     controller_a_kwargs={"base_position": 0.3, "sat_design": 13.0, "gain": 0.05},
        ...     controller_b=PIDControllerSystem,
        ...     controller_b_kwargs={"kp": 0.5, "Ti": 5.0, "isReverse": True},
        ...     id="sat_cascade"
        ... )
    """

    def __init__(
        self,
        # Generic interface (takes priority)
        controller_a: Optional[Type[core.System]] = None,
        controller_a_kwargs: Optional[dict] = None,
        controller_b: Optional[Type[core.System]] = None,
        controller_b_kwargs: Optional[dict] = None,
        # Legacy PID-specific kwargs (backward compat, used when controller_a/b are None)
        kp_a=0.1,
        Ti_a=10.0,
        Td_a=0.0,
        kp_b=0.5,
        Ti_b=5.0,
        Td_b=0.0,
        output_min_a=0.0,
        output_max_a=1.0,
        output_min_b=0.0,
        output_max_b=1.0,
        isReverse_a=True,
        isReverse_b=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        base_id = kwargs.get("id", "cascade")

        # --- Build controller A ---
        if controller_a is not None:
            # Generic: use provided class + kwargs
            a_kwargs = controller_a_kwargs or {}
            a_kwargs.setdefault("id", f"{base_id}_ctrl_a")
            self.ctrl_a = controller_a(**a_kwargs)
        else:
            # Legacy: build PID from individual kwargs
            self.ctrl_a = PIDControllerSystem(
                kp=kp_a,
                Ti=Ti_a,
                Td=Td_a,
                output_min=output_min_a,
                output_max=output_max_a,
                isReverse=isReverse_a,
                id=f"{base_id}_ctrl_a",
            )

        # --- Build controller B ---
        if controller_b is not None:
            # Generic: use provided class + kwargs
            b_kwargs = controller_b_kwargs or {}
            b_kwargs.setdefault("id", f"{base_id}_ctrl_b")
            self.ctrl_b = controller_b(**b_kwargs)
        else:
            # Legacy: build PID from individual kwargs
            self.ctrl_b = PIDControllerSystem(
                kp=kp_b,
                Ti=Ti_b,
                Td=Td_b,
                output_min=output_min_b,
                output_max=output_max_b,
                isReverse=isReverse_b,
                id=f"{base_id}_ctrl_b",
            )

        # Store controller classes for introspection
        self.controller_a_class = type(self.ctrl_a)
        self.controller_b_class = type(self.ctrl_b)

        # --- External I/O ---
        self.input = {
            "setpointValue_a": tps.Scalar(),  # setpoint for controller A (unused by direct-signal controllers)
            "actualValue_a": tps.Scalar(),  # feedback for A / direct signal for A
            "actualValue_b": tps.Scalar(),  # feedback for controller B
        }
        self.output = {"inputSignal": tps.Scalar(0)}

        # Config: enumerate sub-controller parameters with prefixes
        self._config = {"parameters": []}
        for prefix, ctrl in [("ctrl_a", self.ctrl_a), ("ctrl_b", self.ctrl_b)]:
            if hasattr(ctrl, "_config") and "parameters" in ctrl._config:
                for param in ctrl._config["parameters"]:
                    self._config["parameters"].append(f"{prefix}.{param}")

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: int,
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        # Initialize external inputs
        self.input["setpointValue_a"].initialize(n_t=max_timesteps, n_s=batch_size)
        self.input["actualValue_a"].initialize(n_t=max_timesteps, n_s=batch_size)
        self.input["actualValue_b"].initialize(n_t=max_timesteps, n_s=batch_size)

        # Initialize external output
        self.output["inputSignal"].initialize(n_t=max_timesteps, n_s=batch_size)

        # Propagate n_c to sub-controllers before their initialization
        self.ctrl_a.n_c = self.n_c
        self.ctrl_b.n_c = self.n_c

        # Initialize both sub-controllers
        self.ctrl_a.initialize(start_time, end_time, step_size)
        self.ctrl_b.initialize(start_time, end_time, step_size)

    @staticmethod
    def _route_inputs(ctrl, setpoint, feedback, step_index):
        """Route signals to a sub-controller based on its input port types.

        Args:
            ctrl: The sub-controller instance
            setpoint: The setpoint signal (may be unused by direct-signal controllers)
            feedback: The feedback / driving signal
            step_index: Current simulation step
        """
        if "setpointValue" in ctrl.input and "actualValue" in ctrl.input:
            # PID-like controller: uses setpoint and feedback
            ctrl.input["setpointValue"].set(setpoint, step_index)
            ctrl.input["actualValue"].set(feedback, step_index)
        elif "supplyAirTemp" in ctrl.input:
            # SAT-compensated controller: feedback signal is the supply air temp
            ctrl.input["supplyAirTemp"].set(feedback, step_index)
        else:
            raise ValueError(
                f"Cannot route signals to {ctrl.__class__.__name__}: "
                f"unrecognized input ports {list(ctrl.input.keys())}"
            )

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        # --- Controller A (outer loop) ---
        self._route_inputs(
            self.ctrl_a,
            setpoint=self.input["setpointValue_a"].get(),
            feedback=self.input["actualValue_a"].get(),
            step_index=step_index,
        )
        self.ctrl_a.do_step(second_time, date_time, step_size, step_index)

        # --- Controller B (inner loop): setpoint = A's output ---
        self._route_inputs(
            self.ctrl_b,
            setpoint=self.ctrl_a.output["inputSignal"].get(),
            feedback=self.input["actualValue_b"].get(),
            step_index=step_index,
        )
        self.ctrl_b.do_step(second_time, date_time, step_size, step_index)

        # --- Cascade output = B's output ---
        self.output["inputSignal"]._set(
            self.ctrl_b.output["inputSignal"].get(), i_t=step_index
        )

    def reset_state(self) -> None:
        """Reset the state of both sub-controllers."""
        for ctrl in (self.ctrl_a, self.ctrl_b):
            if hasattr(ctrl, "reset_state"):
                ctrl.reset_state()

    # -- composed-map support (mirrors BuildingSpaceTorchSystem) -------------

    @staticmethod
    def _resolve_sub_params(sub, prefix, params):
        """Full physical-parameter dict for a sub-controller: estimated values
        from ``params`` (keyed ``"<prefix>.<name>"``), the rest from the
        sub-controller's own ``tps.Parameter`` defaults."""
        out = {}
        for name in sub.PARAM_NAMES:
            key = f"{prefix}.{name}"
            out[name] = params[key] if key in params else getattr(sub, name).get()
        return out

    @staticmethod
    def _route_forward_inputs(ctrl, setpoint, feedback):
        """Pure-input analogue of :meth:`_route_inputs`: the inputs dict a
        sub-controller's ``forward`` expects, given the cascade signals."""
        if "setpointValue" in ctrl.input and "actualValue" in ctrl.input:
            # PID-like controller: uses setpoint and feedback
            return {"setpointValue": setpoint, "actualValue": feedback}
        if "supplyAirTemp" in ctrl.input:
            # SAT-compensated controller: feedback signal is the supply air temp
            return {"supplyAirTemp": feedback}
        raise ValueError(
            f"Cannot route signals to {ctrl.__class__.__name__}: "
            f"unrecognized input ports {list(ctrl.input.keys())}"
        )

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step of the composite cascade = ctrl_a -> ctrl_b.

        State is ``[ctrl_a_state | ctrl_b_state]`` (the order
        :meth:`System.get_state` produces; a stateless sub-controller such as
        ``SATLinearRuleSystem`` contributes zero width).  ``params`` is keyed
        by the composite attr path (``"ctrl_a.kp"``, ``"ctrl_b.Ti"``, ...).
        Controller A's output feeds controller B *within the same step*,
        exactly like :meth:`do_step`'s sequential sub-stepping.
        """
        # Identity-keyed cache: a sequential rollout re-calls forward with the
        # SAME params dict every step (see OneStepComposer._params_for).
        cache = getattr(self, "_fwd_param_cache", None)
        if cache is None or cache[0] is not params:
            cache = (
                params,
                self._resolve_sub_params(self.ctrl_a, "ctrl_a", params),
                self._resolve_sub_params(self.ctrl_b, "ctrl_b", params),
            )
            self._fwd_param_cache = cache
        _, p_a, p_b = cache

        n_a = self.ctrl_a.state_size()
        x_a, x_b = x[..., :n_a], x[..., n_a:]

        in_a = self._route_forward_inputs(
            self.ctrl_a, inputs["setpointValue_a"], inputs["actualValue_a"]
        )
        x_a_n, out_a = self.ctrl_a.forward(x_a, in_a, p_a, sample_time)

        in_b = self._route_forward_inputs(
            self.ctrl_b, out_a["inputSignal"], inputs["actualValue_b"]
        )
        x_b_n, out_b = self.ctrl_b.forward(x_b, in_b, p_b, sample_time)

        return torch.cat([x_a_n, x_b_n], dim=-1), {
            "inputSignal": out_b["inputSignal"]
        }


# Backward-compatible alias
CascadePIDControllerSystem = CascadeControllerSystem
