# Standard library imports
import datetime
from twin4build.systems.saref4syst.system import System
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.controller.rulebased_controller.sat_compensated_controller.sat_compensated_controller_system import (
    SATCompensatedControllerSystem,
)
from twin4build.systems.controller.setpoint_controller.cascade_controller.cascade_controller_system import (
    CascadeControllerSystem,
)
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.controller.rulebased_controller.on_off_controller.smooth_on_off_controller_system import (
    SmoothOnOffControllerSystem,
)
from twin4build.systems.utils.sigmoid_gate import BandGate, SigmoidGate
from twin4build.translator.translator import (
    StepRule,
    SetStepRule,
    ModeledNode,
    AnyPathRule,
    Node,
    Predicate,
    SignaturePattern,
)  # noqa: F401 (StepRule/SetStepRule/AnyPathRule used in patterns below)


class ControllerIdentificationSystem(core.System, nn.Module):
    r"""
    Controller Identification System using Continuous Relaxation.

    This class implements a differentiable controller identification framework
    that can estimate control logic from observed actuator data. It composes
    candidate controller Systems and uses continuous relaxation to identify:

    1. Which sensor(s) provide feedback (β weights)
    2. Which setpoint signal(s) are being tracked (γ weights)
    3. Which candidate controller(s) are active (α weights)
    4. The parameters of each candidate controller

    The system supports multiple actuators, each with its own set of candidate
    controllers. This is designed to work with the twin4build Estimator class.

    Mathematical Formulation
    ------------------------

    **Control Error (per actuator):**

    .. math::

        e_t = \sum_j \gamma_j \cdot sp_{jt} - \sum_i \beta_i \cdot y_{it}

    **Predicted Actuator Output (per actuator a):**

    .. math::

        \hat{u}_{a,t} = \sum_k \alpha_{a,k} \cdot C_k(e_t; \theta_k)

    where :math:`C_k` is the k-th candidate controller.

    **Binarization Penalty:**

    .. math::

        P(x) = x(1 - x)

    Args:
        n_sensors: Number of candidate feedback sensors (default: 1)
        n_setpoints: Number of candidate setpoint signals (default: 1)
        n_actuators: Number of actuator outputs (default: 1)
        candidate_controllers: List of controller System classes to use as candidates.
            If None, uses default PIDControllerSystem with different configurations.
        candidate_controller_kwargs: List of kwargs dicts for each candidate controller.
            If None, uses default configurations.
        **kwargs: Additional keyword arguments passed to parent classes

    Example:
        >>> # Default usage with built-in candidates
        >>> controller = ControllerIdentificationSystem(
        ...     n_sensors=1,
        ...     n_setpoints=1,
        ...     n_actuators=1,
        ...     id="my_controller"
        ... )
        >>>
        >>> # Custom candidates
        >>> controller = ControllerIdentificationSystem(
        ...     candidate_controllers=[PIDControllerSystem, PIDControllerSystem],
        ...     candidate_controller_kwargs=[
        ...         {"kp": 0.1, "Ti": 100, "Td": 0},  # PI
        ...         {"kp": 0.1, "Ti": 100, "Td": 10}, # PID
        ...     ],
        ...     id="custom_controller"
        ... )

    Note:
        This component deliberately has **no composer-style** ``forward``:
        its estimable parameters (``alpha_{a}``, ``beta_{a}``, ``gamma_{a}``,
        ``gamma_gate_{a}``, ...) are multi-element (``n_c > 1``)
        ``tps.Parameter`` vectors, which the composed fast paths reject
        upstream (``Estimator._composer_theta_spec`` raises on any
        multi-branch parameter) -- so a pure ``forward`` here could never be
        exercised by those paths.  Estimation with theta on this component
        always uses the exact object-graph objective;
        ``OneStepComposer._validate_theta_influence`` guarantees the
        fallback instead of silently freezing theta-dependent signals.
    """

    # Controller type constants -- each defines a signal routing strategy
    CTRL_SETPOINT = "setpoint"
    CTRL_CASCADE = "cascade"

    # Gate class used for the per-actuator setpoint-based gate
    # ("gate_{a}") constructed in :meth:`_build_components`.  Subclasses
    # can override to swap the gate type without re-implementing the
    # whole component-building routine.  The base class ships with
    # ``BandGate`` (smooth band-pass) which is the right default for any
    # zone-temperature loop with a deadband; subclasses targeting other
    # control regimes can set this to e.g. ``SigmoidGate`` for a single-
    # threshold gate.
    _gate_class: Type[SigmoidGate] = BandGate

    # ------------------------------------------------------------------
    # Estimable-parameter bounds (used by :meth:`get_estimable_parameters`).
    # All bounds live as class constants so :class:`Estimator` callers can
    # rely on a single source of truth and subclasses can tighten/loosen a
    # specific knob without re-implementing the whole contract.
    # ``x0`` is read from the component's *current* state (which the
    # rewire pipeline writes data-driven seeds into); these constants only
    # supply the ``(lower, upper)`` half of each parameter tuple.
    # ------------------------------------------------------------------
    _KP_BOUNDS: Tuple[float, float] = (0.1, 10.0)
    _TI_BOUNDS: Tuple[float, float] = (1.0, 1800.0)
    _OUTPUT_MIN_BOUNDS: Tuple[float, float] = (0.0, 1.0)
    _DEFAULT_OUTPUT_BOUNDS: Tuple[float, float] = (0.0, 1.0)
    _GATE_THRESHOLD_BOUNDS: Tuple[float, float] = (-0.5, 1.5)
    _GATE_BAND_BOUNDS: Tuple[float, float] = (0.05, 5.0)
    _GAMMA_GATE_BOUNDS: Tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        n_sensors: Optional[int] = None,
        n_setpoints: Optional[int] = None,
        n_on_off_signals: Optional[int] = None,
        n_actuators: int = 1,
        # --- Type-based candidate specification ---
        # Each type defines its signal routing.  Provide one or more lists;
        # entries within a list are individual candidate instances of that type.
        setpoint_controllers: Optional[List[Type[core.System]]] = None,
        setpoint_controller_kwargs: Optional[List[dict]] = None,
        cascade_controllers: Optional[List[Type[core.System]]] = None,
        cascade_controller_kwargs: Optional[List[dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.n_sensors = n_sensors
        self.n_setpoints = n_setpoints
        # ``n_on_off_signals`` indexes a third input port (``onOffSignal``)
        # whose sole consumer is the ``gamma_gate``-weighted gate input.
        # It is wired *independently* from the PI-error setpoint pool so
        # the rewire pipeline (which prunes feedback-sensor / setpoint
        # connections it has classified) cannot mutate the gate's input
        # space.  Translator signature patterns auto-mirror every Brick
        # ``Zone_Air_Temperature_Setpoint`` into this port, so by default
        # the gate has the schedule setpoints available; downstream code
        # can additionally wire schedule / occupancy entities here without
        # affecting the PI loop identification.
        self.n_on_off_signals = n_on_off_signals
        self.n_actuators = n_actuators

        # Build the ordered candidate list from per-type arguments.
        # If nothing is provided, fall back to defaults.
        any_type_given = (
            setpoint_controllers is not None
            or cascade_controllers is not None
        )

        if not any_type_given:
            setpoint_controllers = [PIDControllerSystem, PIDControllerSystem]
            setpoint_controller_kwargs = [
                {"kp": 0.3, "Ti": 5.0, "Td": 0.0, "is_reverse": True},
                {"kp": 0.3, "Ti": 5.0, "Td": 0.0, "is_reverse": False},
            ]
            cascade_controllers = [CascadeControllerSystem]
            cascade_controller_kwargs = [
                {
                    "kp_a": 0.1, "Ti_a": 10.0, "Td_a": 0.0,
                    "kp_b": 0.5, "Ti_b": 5.0, "Td_b": 0.0,
                    "isReverse_a": True, "isReverse_b": True,
                },
            ]

        # Normalize Nones to empty lists
        setpoint_controllers = setpoint_controllers or []
        setpoint_controller_kwargs = setpoint_controller_kwargs or [{} for _ in setpoint_controllers]
        cascade_controllers = cascade_controllers or []
        cascade_controller_kwargs = cascade_controller_kwargs or [{} for _ in cascade_controllers]

        # Validate lengths
        for label, classes, kws in [
            ("setpoint", setpoint_controllers, setpoint_controller_kwargs),
            ("cascade", cascade_controllers, cascade_controller_kwargs),
        ]:
            assert len(classes) == len(kws), (
                f"{label}_controllers ({len(classes)}) and "
                f"{label}_controller_kwargs ({len(kws)}) must have the same length"
            )

        # Flatten into ordered candidate list: [(class, kwargs, type_tag), ...]
        self._candidate_entries: List[Tuple[Type[core.System], dict, str]] = []
        for cls, kw in zip(setpoint_controllers, setpoint_controller_kwargs):
            self._candidate_entries.append((cls, kw, self.CTRL_SETPOINT))
        for cls, kw in zip(cascade_controllers, cascade_controller_kwargs):
            self._candidate_entries.append((cls, kw, self.CTRL_CASCADE))

        assert len(self._candidate_entries) > 0, (
            "At least one candidate controller must be provided"
        )

        self.n_candidates = len(self._candidate_entries)
        self._candidate_types = [e[2] for e in self._candidate_entries]
        self.candidate_controller_classes = [e[0] for e in self._candidate_entries]
        self._candidate_controller_kwargs = [e[1] for e in self._candidate_entries]
        self._has_cascade = self.CTRL_CASCADE in self._candidate_types

        # Build input dictionary
        self._input = {
            "sensorValue": tps.Vector(),  # Feedback sensor values (n_sensors)
            "setpointValue": tps.Vector(),  # PI-error setpoint signals (n_setpoints)
            # Gate input bus -- indexed by ``gamma_gate``.  Decoupled from
            # ``setpointValue`` so the rewire pipeline can prune the PI
            # setpoint pool without mutating the gate's input space.  The
            # translator's CITS signature patterns auto-mirror every
            # Brick ``Zone_Air_Temperature_Setpoint`` into this port, so
            # the schedule (with setback transitions) stays available
            # to the gate even when the rewire winner picks a different
            # setpoint for the PI error term.  Sized by
            # ``n_on_off_signals``.
            "onOffSignal": tps.Vector(),
        }

        # Output: one signal per actuator
        self._output = {"inputSignal": tps.Vector()}

        self._built = False
        self._config = {
            "parameters": [
                "n_sensors",
                "n_setpoints",
                "n_on_off_signals",
                "n_actuators",
            ]
        }
        self.INITIALIZED = False

        # Build immediately if all sizes are explicitly provided.
        if (
            n_sensors is not None
            and n_setpoints is not None
            and n_on_off_signals is not None
        ):
            self._build_components()

    def _get_n_actuators_from_connections(self) -> int:
        """Detect n_actuators by counting unique output slots used on inputSignal.

        Mirrors :meth:`get_n_v_from_connections` but operates on outgoing
        connections from the ``inputSignal`` output port instead of incoming
        connections to an input port.

        Returns:
            int: The number of actuator slots (max output_port_index + 1), or
                 None if no outgoing connections to inputSignal are found.
        """
        max_index = 0
        found = False
        for conn in self.connected_through:
            if conn.output_port == "inputSignal":
                for cp in conn.connects_system_at:
                    idx = cp.output_port_index.get(conn)
                    if idx is not None:
                        found = True
                        if isinstance(idx, int):
                            max_index = max(max_index, idx)
                        elif hasattr(idx, "max"):
                            max_index = max(max_index, int(idx.max().item()))
        return max_index + 1 if found else None

    def _build_components(self) -> None:
        """Create candidate controllers and selection-weight parameters.

        Called automatically from ``__init__`` when ``n_sensors`` and
        ``n_setpoints`` are both provided, or lazily from ``initialize()``
        after the translator has wired up connections (so the actual sizes
        can be read from :meth:`get_n_v_from_connections`).
        """
        n_sensors = self.n_sensors
        n_setpoints = self.n_setpoints
        n_on_off_signals = self.n_on_off_signals
        n_actuators = self.n_actuators

        alpha_init = 0.5
        beta_init = 0.5
        gamma_init = 0.5

        # Create candidate controller instances for each actuator
        for a in range(n_actuators):
            for c, (CtrlClass, ctrl_kwargs, _ctype) in enumerate(self._candidate_entries):
                ctrl_id = f"{self.id}_a{a}_c{c}"
                ctrl = CtrlClass(id=ctrl_id, **ctrl_kwargs)
                setattr(self, f"candidate_{a}_{c}", ctrl)

        # Selection weights - all per-actuator:
        # - alpha_a: selects among candidates for actuator a (n_c = n_candidates)
        # - beta_a: selects feedback sensors for actuator a (n_c = n_sensors)
        # - gamma_a: selects setpoint signals for actuator a (n_c = n_setpoints)
        for a in range(n_actuators):
            # Alpha: candidate controller selection
            setattr(
                self,
                f"alpha_{a}",
                tps.Parameter(
                    torch.full((self.n_candidates,), alpha_init, dtype=tps.float_dtype()),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                    n_c=self.n_candidates,
                ),
            )

            # Beta: feedback sensor selection (per-actuator)
            setattr(
                self,
                f"beta_{a}",
                tps.Parameter(
                    torch.full((n_sensors,), beta_init, dtype=tps.float_dtype()),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                    n_c=n_sensors,
                ),
            )

            # Gamma: setpoint signal selection (per-actuator)
            setattr(
                self,
                f"gamma_{a}",
                tps.Parameter(
                    torch.full((n_setpoints,), gamma_init, dtype=tps.float_dtype()),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                    n_c=n_setpoints,
                ),
            )

            if self._has_cascade:
                setattr(
                    self,
                    f"beta_b_{a}",
                    tps.Parameter(
                        torch.full((n_sensors,), beta_init, dtype=tps.float_dtype()),
                        min_value=0.0,
                        max_value=1.0,
                        requires_grad=False,
                        n_c=n_sensors,
                    ),
                )

            # Setpoint-based gate sub-system per actuator.  Type is
            # configurable via the ``_gate_class`` class attribute
            # (defaults to ``BandGate``).  ``BandGate`` is parameterized
            # by ``threshold`` (lower edge ``T_lo``) and ``band`` (width
            # ``w = T_hi - T_lo``, clamped ``>= 0``) so the upper edge
            # cannot invert below the lower one; the default band
            # ``[18, 24]`` °C matches typical zone deadbands.  When
            # subclasses override ``_gate_class`` to a different gate
            # type (e.g. ``SigmoidGate``) the ``band`` kwarg is
            # forwarded too, so a custom gate that does not accept it
            # should silently ignore the extra keyword.
            setattr(
                self,
                f"gate_{a}",
                self._gate_class(
                    threshold=18.0,
                    band=6.0,
                    steepness=100.0,
                    id=f"{self.id}_gate_{a}",
                ),
            )

            # gamma_gate: selects which onOffSignal slot drives the gate.
            # Sized by ``n_on_off_signals`` (NOT ``n_setpoints``) -- the
            # gate input bus is structurally distinct from the PI-error
            # setpoint bus, so the rewire pipeline can prune the latter
            # without mutating the former.
            setattr(
                self,
                f"gamma_gate_{a}",
                tps.Parameter(
                    torch.full((n_on_off_signals,), gamma_init, dtype=tps.float_dtype()),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                    n_c=n_on_off_signals,
                ),
            )

            # alpha_gate: 0 = pass-through (no gating), 1 = full gating
            setattr(
                self,
                f"alpha_gate_{a}",
                tps.Parameter(
                    torch.tensor(0.5, dtype=tps.float_dtype()),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                ),
            )

            # default_output: actuator output when gate is inactive (0-1)
            setattr(
                self,
                f"default_output_{a}",
                tps.Parameter(
                    torch.tensor(0.0, dtype=tps.float_dtype()),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                ),
            )

        # Per-onOffSignal normalization buffers (shared across actuators).
        # The gate input is computed as the gamma_gate-weighted sum of
        # the onOffSignal slots.  Without normalization the gate
        # threshold/band x0 must be in physical units (e.g., degC) and
        # therefore tied to a specific signal type.  Mapping each signal
        # to ``[0, 1]`` via these buffers makes the gate seeds
        # room/unit-agnostic; populate them via
        # :func:`pi_loop_rewire._populate_on_off_signal_norm_bounds`
        # after the upstream sensors have been initialised.  Defaults
        # are identity transform (min=0, max=1) so existing models
        # behave unchanged until the bounds are explicitly written.
        #
        # Plain ``torch.Tensor`` (not ``tps.Parameter``) -- they are
        # constants, never estimated, never gradient-tracked.
        self.on_off_signal_norm_min = torch.zeros(
            n_on_off_signals, dtype=tps.float_dtype()
        )
        self.on_off_signal_norm_max = torch.ones(
            n_on_off_signals, dtype=tps.float_dtype()
        )

        # Build config - include parameters from all candidate controllers
        config_params = [
            "n_sensors",
            "n_setpoints",
            "n_actuators",
        ]
        for a in range(n_actuators):
            for c in range(self.n_candidates):
                ctrl = getattr(self, f"candidate_{a}_{c}")
                if hasattr(ctrl, "_config") and "parameters" in ctrl._config:
                    for param in ctrl._config["parameters"]:
                        config_params.append(f"candidate_{a}_{c}.{param}")
        self._config = {"parameters": config_params}

        self._built = True

    @property
    def config(self):
        return self._config

    @property
    def input(self) -> dict:
        return self._input

    @property
    def output(self) -> dict:
        return self._output

    def _get_candidate(self, actuator: int, candidate: int) -> core.System:
        """Get candidate controller for actuator a, candidate c."""
        return getattr(self, f"candidate_{actuator}_{candidate}")

    def _get_alpha(self, actuator: int, candidate: int) -> torch.Tensor:
        """Get alpha parameter for candidate c of actuator a."""
        return getattr(self, f"alpha_{actuator}").get()[candidate]

    def _get_beta(self, actuator: int, sensor: int) -> torch.Tensor:
        """Get beta parameter for sensor i of actuator a."""
        return getattr(self, f"beta_{actuator}").get()[sensor]

    def _get_gamma(self, actuator: int, setpoint: int) -> torch.Tensor:
        """Get gamma parameter for setpoint j of actuator a."""
        return getattr(self, f"gamma_{actuator}").get()[setpoint]

    def _get_alpha_vector(self, actuator: int) -> torch.Tensor:
        """Get full alpha vector for actuator a."""
        return getattr(self, f"alpha_{actuator}").get()

    def _get_beta_vector(self, actuator: int) -> torch.Tensor:
        """Get full beta vector for actuator a."""
        return getattr(self, f"beta_{actuator}").get()

    def _get_gamma_vector(self, actuator: int) -> torch.Tensor:
        """Get full gamma vector for actuator a."""
        return getattr(self, f"gamma_{actuator}").get()

    def _get_beta_b_vector(self, actuator: int) -> torch.Tensor:
        """Get full beta_b vector (cascade B-loop sensor selection) for actuator a."""
        return getattr(self, f"beta_b_{actuator}").get()

    def _get_gate(self, actuator: int) -> SigmoidGate:
        """Get the setpoint-based gate sub-system for actuator ``a``.

        Returns a :class:`BandGate` by default (constructed in
        ``__init__``); the annotation stays :class:`SigmoidGate` because
        ``BandGate`` is a subclass and callers only use the shared
        interface (``compute_gate``, ``threshold``, ``steepness``).
        """
        return getattr(self, f"gate_{actuator}")

    def _get_gamma_gate_vector(self, actuator: int) -> torch.Tensor:
        """Get gamma_gate vector (gate setpoint selection) for actuator a."""
        return getattr(self, f"gamma_gate_{actuator}").get()

    def _get_alpha_gate(self, actuator: int) -> torch.Tensor:
        """Get alpha_gate scalar (gate activation) for actuator a."""
        return getattr(self, f"alpha_gate_{actuator}").get()

    def _get_default_output(self, actuator: int) -> torch.Tensor:
        """Get default_output scalar (output when gate is inactive) for actuator a."""
        return getattr(self, f"default_output_{actuator}").get()

    def get_estimable_parameters(
        self,
    ) -> List[Tuple[Any, str, Any, float, float]]:
        """Return the standard CITS parameter tuples for the auto-estimator.

        Implements the ``parameters="auto"`` contract consumed by
        :meth:`twin4build.Estimator.estimate`: each tuple is
        ``(component, attribute_path, x0, lower_bound, upper_bound)``
        suitable for direct inclusion in the ``parameters`` list.

        The set per actuator ``a`` and candidate ``c`` is:

          * ``candidate_{a}_{c}.kp``        -- PI proportional gain
          * ``candidate_{a}_{c}.Ti``        -- PI integral time
          * ``candidate_{a}_{c}.output_min``-- saturation lower bound
          * ``default_output_{a}``          -- fallback output when gate inactive
          * ``gate_{a}.threshold``          -- gate lower edge (T_lo)
          * ``gate_{a}.band``               -- gate width (BandGate only)
          * ``gamma_gate_{a}``              -- onOffSignal-slot selector

        ``x0`` is read straight from the current state of each parameter,
        which the rewire pipeline (:meth:`SimulationModel.rewire`) writes
        data-driven seeds into.  Bounds come from the class-level
        ``_*_BOUNDS`` constants; override on a subclass to tighten.

        Returns:
            A list of parameter tuples, possibly empty when the CITS has
            not been built yet (``self._built is False``).
        """
        if not getattr(self, "_built", False):
            return []

        params: List[Tuple[Any, str, Any, float, float]] = []

        def _scalar(p) -> float:
            v = p.get()
            try:
                return float(v.item())
            except AttributeError:
                return float(v)

        for a in range(self.n_actuators):
            # Per-candidate PID-like knobs (kp / Ti / output_min).  Skip
            # attributes that do not exist on the candidate class (e.g.
            # output_min is absent on some controllers).
            for c in range(self.n_candidates):
                cand = getattr(self, f"candidate_{a}_{c}", None)
                if cand is None:
                    continue
                for attr, bounds in (
                    ("kp", self._KP_BOUNDS),
                    ("Ti", self._TI_BOUNDS),
                    ("output_min", self._OUTPUT_MIN_BOUNDS),
                ):
                    p = getattr(cand, attr, None)
                    if p is None or not hasattr(p, "get"):
                        continue
                    if not getattr(p, "requires_grad", True):
                        # Skip parameters that the candidate has frozen
                        # (e.g. ``Td`` on the PI subclass).
                        continue
                    params.append(
                        (
                            self,
                            f"candidate_{a}_{c}.{attr}",
                            _scalar(p),
                            bounds[0],
                            bounds[1],
                        )
                    )

            # default_output_{a}
            default_out = getattr(self, f"default_output_{a}", None)
            if default_out is not None and hasattr(default_out, "get"):
                params.append(
                    (
                        self,
                        f"default_output_{a}",
                        _scalar(default_out),
                        self._DEFAULT_OUTPUT_BOUNDS[0],
                        self._DEFAULT_OUTPUT_BOUNDS[1],
                    )
                )

            # gate_{a}.threshold and gate_{a}.band (BandGate-specific).
            gate = getattr(self, f"gate_{a}", None)
            if gate is not None:
                t_param = getattr(gate, "threshold", None)
                if t_param is not None and hasattr(t_param, "get"):
                    params.append(
                        (
                            self,
                            f"gate_{a}.threshold",
                            _scalar(t_param),
                            self._GATE_THRESHOLD_BOUNDS[0],
                            self._GATE_THRESHOLD_BOUNDS[1],
                        )
                    )
                b_param = getattr(gate, "band", None)
                if b_param is not None and hasattr(b_param, "get"):
                    params.append(
                        (
                            self,
                            f"gate_{a}.band",
                            _scalar(b_param),
                            self._GATE_BAND_BOUNDS[0],
                            self._GATE_BAND_BOUNDS[1],
                        )
                    )

            # gamma_gate_{a} (vector parameter -- per-slot weight).
            gg = getattr(self, f"gamma_gate_{a}", None)
            if gg is not None and hasattr(gg, "get"):
                val = gg.get()
                if hasattr(val, "detach"):
                    x0_vec = val.detach().cpu().numpy().flatten().tolist()
                else:
                    try:
                        x0_vec = list(val)
                    except TypeError:
                        x0_vec = float(val)
                params.append(
                    (
                        self,
                        f"gamma_gate_{a}",
                        x0_vec,
                        self._GAMMA_GATE_BOUNDS[0],
                        self._GAMMA_GATE_BOUNDS[1],
                    )
                )

        return params

    def _append_pid_params(
        self, params: list, ctrl: core.System, prefix: str
    ) -> None:
        """Append estimable PID-like parameters from *ctrl* to *params*.

        Works for PIDControllerSystem, On-Off controllers, and any other
        controller that exposes the standard attrs.
        """
        target = ctrl
        target_prefix = prefix

        if hasattr(target, "kp"):
            params.append((self, f"{target_prefix}.kp", 0.01, 0.001, 10, "private"))
        if hasattr(target, "Ti"):
            params.append((self, f"{target_prefix}.Ti", 10, 1.0, 10000.0, "private"))
        if hasattr(target, "Td"):
            params.append((self, f"{target_prefix}.Td", 0.0, 0.0, 0.0001, "private"))
        if hasattr(target, "output_min"):
            params.append((self, f"{target_prefix}.output_min", 0.5, 0.0, 1.0, "private"))
        if hasattr(target, "off_value"):
            params.append((self, f"{target_prefix}.offValue", 0.0, 0, 1.0, "private"))
        if hasattr(target, "on_value"):
            params.append((self, f"{target_prefix}.onValue", 1.0, 0.0, 1.0, "private"))
        # ``steepness`` is intentionally NOT estimated.  It is a numerical
        # hyperparameter that controls the smoothness of the soft on/off
        # transition; the underlying physical behavior is binary.  The
        # value is set at construction time and (optionally) annealed
        # globally via the saturation_mode / steepness_override helpers
        # in ``smooth_saturation.py``.

    @staticmethod
    def _summary_pid_params(lines: list, ctrl: core.System, indent: int = 6) -> None:
        """Append human-readable PID parameter lines for *ctrl*."""
        pad = " " * indent
        for attr in ("kp", "Ti", "Td", "output_min", "output_max"):
            if hasattr(ctrl, attr):
                lines.append(f"{pad}{attr}: {getattr(ctrl, attr).get().item():.6f}")
        if hasattr(ctrl, "is_reverse"):
            lines.append(f"{pad}isReverse: {ctrl.isReverse}")
        for attr in ("off_value", "on_value", "steepness"):
            if hasattr(ctrl, attr):
                lines.append(f"{pad}{attr}: {getattr(ctrl, attr).get().item():.6f}")

    @staticmethod
    def _append_pid_scales(scales: list, ctrl: core.System) -> None:
        """Append 1.0 gradient scales for each PID-like parameter on *ctrl*."""
        target = ctrl.pid if hasattr(ctrl, "pid") else ctrl
        # Keep this list in sync with ``_append_pid_params`` above.
        # ``steepness`` is intentionally excluded (hyperparameter, not estimated).
        for attr in ("kp", "Ti", "Td", "output_min", "off_value", "on_value"):
            if hasattr(target, attr):
                scales.append(1.0)

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: int,
    ) -> None:
        """Initialize the controller system and all candidate controllers.

        If ``n_sensors`` or ``n_setpoints`` were not provided at construction
        time (None), they are inferred here from the wired connections via
        :meth:`get_n_v_from_connections`.  This allows the translator to
        instantiate the system without knowing the signal counts upfront.
        """
        # Lazy build: detect n_sensors / n_setpoints / n_on_off_signals /
        # n_actuators from connections.
        if not self._built:
            n_s = self.get_n_v_from_connections("sensorValue")
            n_sp = self.get_n_v_from_connections("setpointValue")
            n_oo = self.get_n_v_from_connections("onOffSignal")
            n_act = self._get_n_actuators_from_connections()
            if n_s is not None:
                self.n_sensors = n_s
            if n_sp is not None:
                self.n_setpoints = n_sp
            if n_oo is not None:
                self.n_on_off_signals = n_oo
            if n_act is not None:
                self.n_actuators = n_act
            assert (
                self.n_sensors is not None
                and self.n_setpoints is not None
                and self.n_on_off_signals is not None
            ), (
                f"ControllerIdentificationSystem '{self.id}': "
                "n_sensors, n_setpoints, and n_on_off_signals must be "
                "provided at construction or inferred from translator "
                "connections (signature patterns must wire ``sensorValue``, "
                "``setpointValue`` and ``onOffSignal``)."
            )
            self._build_components()

        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        # Initialize Vector inputs with their sizes
        self.input["sensorValue"].initialize(
            n_t=max_timesteps, n_s=batch_size, n_v=self.n_sensors
        )
        self.input["setpointValue"].initialize(
            n_t=max_timesteps, n_s=batch_size, n_v=self.n_setpoints
        )
        self.input["onOffSignal"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
            size=self.n_on_off_signals,
        )

        # Initialize output
        self.output["inputSignal"].initialize(
            n_t=max_timesteps, n_s=batch_size, n_v=self.n_actuators
        )

        # Initialize all candidate controllers
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                self._get_candidate(a, c).initialize(start_time, end_time, step_size)

        # Initialize gate sub-systems
        for a in range(self.n_actuators):
            self._get_gate(a).initialize(start_time, end_time, step_size)

        self.INITIALIZED = True

    def _compute_weighted_signals(self, actuator: int) -> Tuple[torch.Tensor, ...]:
        """Compute weighted sensor and setpoint signals for a specific actuator.

        Uses ratio normalization (w_i / sum(w)) for beta, gamma, and beta_b weights.

        Args:
            actuator: The actuator index to compute signals for.

        Returns:
            Tuple of (weighted_setpoint, weighted_feedback[, weighted_feedback_b])
            tensors, each shape (n_s, n_c).  weighted_feedback_b is only included
            when cascade candidates exist.
        """
        # Tensor shapes: (n_s, n_c, n_v) where n_v is n_sensors or n_setpoints
        sensor_values = self.input["sensorValue"].get()  # (n_s, n_c, n_sensors)
        setpoint_values = self.input["setpointValue"].get()  # (n_s, n_c, n_setpoints)

        # Ratio-normalised weights
        gamma = self._get_gamma_vector(actuator)  # (n_setpoints,)
        gamma_norm = gamma / (torch.sum(gamma) + 1e-8)

        beta = self._get_beta_vector(actuator)  # (n_sensors,)
        beta_norm = beta / (torch.sum(beta) + 1e-8)

        # Weighted setpoint: sum_j gamma_norm_j * sp_jt
        weighted_setpoint = torch.sum(
            gamma_norm * setpoint_values, dim=-1
        )  # (n_s, n_c)

        # Weighted sensor feedback: sum_i beta_norm_i * y_it
        weighted_feedback = torch.sum(beta_norm * sensor_values, dim=-1)  # (n_s, n_c)

        if self._has_cascade:
            beta_b = self._get_beta_b_vector(actuator)  # (n_sensors,)
            beta_b_norm = beta_b / (torch.sum(beta_b) + 1e-8)
            weighted_feedback_b = torch.sum(
                beta_b_norm * sensor_values, dim=-1
            )  # (n_s, n_c)
            return weighted_setpoint, weighted_feedback, weighted_feedback_b

        return weighted_setpoint, weighted_feedback

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Perform one simulation step.

        For each actuator:
        1. Compute per-actuator weighted setpoint and feedback signals
        2. Run all candidate controllers with actuator-specific error signal
        3. Combine outputs using normalized alpha weights
        """
        n_s = self.input["sensorValue"].n_s
        n_c = self.input["sensorValue"].n_c

        # Process each actuator
        actuator_outputs = torch.zeros(n_s, n_c, self.n_actuators, dtype=tps.float_dtype())

        for a in range(self.n_actuators):
            # Compute per-actuator weighted signals (each actuator has own beta/gamma/beta_b)
            signals = self._compute_weighted_signals(a)
            weighted_setpoint = signals[0]
            weighted_feedback = signals[1]
            weighted_feedback_b = signals[2] if len(signals) > 2 else None

            # Run all candidate controllers and collect outputs
            candidate_outputs = []
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                ctype = self._candidate_types[c]

                if ctype == self.CTRL_CASCADE:
                    ctrl.input["setpointValue_a"].set(weighted_setpoint, step_index)
                    ctrl.input["actualValue_a"].set(weighted_feedback, step_index)
                    ctrl.input["actualValue_b"].set(weighted_feedback_b, step_index)
                elif ctype == self.CTRL_SETPOINT:
                    ctrl.input["setpointValue"].set(weighted_setpoint, step_index)
                    ctrl.input["actualValue"].set(weighted_feedback, step_index)

                ctrl.do_step(second_time, date_time, step_size, step_index)
                candidate_outputs.append(ctrl.output["inputSignal"].get())

            # Stack outputs: (n_candidates, n_s, n_c)
            candidate_outputs = torch.stack(candidate_outputs, dim=0)

            # Flatten to (n_candidates, n_s * n_c) for einsum
            orig_shape = candidate_outputs.shape[1:]  # (n_s, n_c) or similar
            candidate_outputs_flat = candidate_outputs.reshape(self.n_candidates, -1)

            # Ratio-normalised weighted sum of candidate outputs
            alpha = self._get_alpha_vector(a)
            alpha_norm = alpha / (torch.sum(alpha) + 1e-8)
            combined_output = torch.einsum(
                "c,cb->b", alpha_norm, candidate_outputs_flat
            )

            # Reshape back
            combined_output = combined_output.reshape(orig_shape)

            # On/off-signal-based gating with default output.  The gate
            # input bus (``onOffSignal``) is structurally distinct from
            # the PI-error setpoint bus (``setpointValue``), so the
            # rewire pipeline can prune the latter without mutating the
            # former.  Each onOffSignal slot is first mapped to a
            # unit-free ``[0, 1]`` range using per-slot min/max bounds
            # (populated from data by
            # ``_populate_on_off_signal_norm_bounds`` -- defaults to
            # identity).  This decouples the gate threshold/band x0
            # from physical units so the same seed (e.g.,
            # ``threshold=0.1, band=0.8``) can be used across rooms /
            # signal types without manual tuning.
            on_off_signal_values = self.input["onOffSignal"].get()  # (n_s, n_c, n_on_off_signals)
            oo_range = (
                self.on_off_signal_norm_max - self.on_off_signal_norm_min
            ).clamp(min=1e-6)
            on_off_signal_values_norm = (
                on_off_signal_values - self.on_off_signal_norm_min
            ) / oo_range
            gamma_gate = self._get_gamma_gate_vector(a)
            gamma_gate_norm = gamma_gate / (torch.sum(gamma_gate) + 1e-8)
            gate_input = torch.sum(
                gamma_gate_norm * on_off_signal_values_norm, dim=-1
            )

            gate_signal = self._get_gate(a).compute_gate(gate_input)

            alpha_g = self._get_alpha_gate(a)
            gate = (1 - alpha_g) + alpha_g * gate_signal

            default_out = self._get_default_output(a)
            actuator_outputs[..., a] = gate * combined_output + (1 - gate) * default_out

        # Set final output - shape (n_s, n_c, n_actuators)
        self.output["inputSignal"].set(actuator_outputs, step_index)

    def compute_binarization_penalty(self) -> torch.Tensor:
        """Compute binarization penalty P(x) = x(1-x) for all selection weights.

        This penalty encourages weights toward 0 or 1, promoting crisp selection.

        Returns:
            torch.Tensor: Sum of x*(1-x) over all alpha, beta, gamma weights.
        """
        penalty = torch.tensor(0.0, dtype=tps.float_dtype())
        for a in range(self.n_actuators):
            alpha = self._get_alpha_vector(a)
            penalty = penalty + torch.sum(alpha * (1 - alpha))
            beta = self._get_beta_vector(a)
            penalty = penalty + torch.sum(beta * (1 - beta))
            gamma = self._get_gamma_vector(a)
            penalty = penalty + torch.sum(gamma * (1 - gamma))
            if self._has_cascade:
                beta_b = self._get_beta_b_vector(a)
                penalty = penalty + torch.sum(beta_b * (1 - beta_b))
            # Gate selection weights
            gamma_gate = self._get_gamma_gate_vector(a)
            penalty = penalty + torch.sum(gamma_gate * (1 - gamma_gate))
            alpha_gate = self._get_alpha_gate(a)
            penalty = penalty + torch.sum(alpha_gate * (1 - alpha_gate))
            # Polarity: push toward -1 or +1 via (1 - p^2)
            polarity = self._get_gate(a).polarity.get()
            penalty = penalty + torch.sum(1 - polarity ** 2)
        return penalty

    def compute_regularization_penalty(self) -> torch.Tensor:
        """Standard interface for Estimator to compute regularization penalty.

        Returns the binarization penalty P(x) = x(1-x) over all selection weights.
        """
        return self.compute_binarization_penalty()

    def get_selection_weights(self) -> Dict[str, torch.Tensor]:
        """Get all selection weights as a dictionary.

        All weights are per-actuator:
        - alpha_{a}_{c}: Individual alpha weights per actuator/candidate
        - alpha_{a}: Full alpha vector for actuator a
        - beta_{a}_{i}: Individual beta weights per actuator/sensor
        - beta_{a}: Full beta vector for actuator a
        - gamma_{a}_{j}: Individual gamma weights per actuator/setpoint
        - gamma_{a}: Full gamma vector for actuator a
        """
        weights = {}

        for a in range(self.n_actuators):
            # Alpha weights
            alpha_vec = self._get_alpha_vector(a)
            weights[f"alpha_{a}"] = alpha_vec.detach().clone()
            for c in range(self.n_candidates):
                weights[f"alpha_{a}_{c}"] = alpha_vec[c].detach().clone()

            # Beta weights
            beta_vec = self._get_beta_vector(a)
            weights[f"beta_{a}"] = beta_vec.detach().clone()
            for i in range(self.n_sensors):
                weights[f"beta_{a}_{i}"] = beta_vec[i].detach().clone()

            # Gamma weights
            gamma_vec = self._get_gamma_vector(a)
            weights[f"gamma_{a}"] = gamma_vec.detach().clone()
            for j in range(self.n_setpoints):
                weights[f"gamma_{a}_{j}"] = gamma_vec[j].detach().clone()

            # Beta_b weights (cascade B-loop sensor selection)
            if self._has_cascade:
                beta_b_vec = self._get_beta_b_vector(a)
                weights[f"beta_b_{a}"] = beta_b_vec.detach().clone()
                for i in range(self.n_sensors):
                    weights[f"beta_b_{a}_{i}"] = beta_b_vec[i].detach().clone()

            # Gate weights
            gamma_gate_vec = self._get_gamma_gate_vector(a)
            weights[f"gamma_gate_{a}"] = gamma_gate_vec.detach().clone()
            for j in range(self.n_setpoints):
                weights[f"gamma_gate_{a}_{j}"] = gamma_gate_vec[j].detach().clone()
            weights[f"alpha_gate_{a}"] = self._get_alpha_gate(a).detach().clone()
            gate = self._get_gate(a)
            weights[f"gate_{a}_threshold"] = gate.threshold.get().detach().clone()
            weights[f"gate_{a}_steepness"] = gate.steepness.get().detach().clone()
            weights[f"gate_{a}_polarity"] = gate.polarity.get().detach().clone()
            weights[f"default_output_{a}"] = self._get_default_output(a).detach().clone()

        return weights

    def get_candidate_controller(self, actuator: int, candidate: int) -> core.System:
        """Get a specific candidate controller instance."""
        return self._get_candidate(actuator, candidate)

    def get_identified_structure(
        self, threshold: float = 0.5
    ) -> Dict[str, Union[List, Dict]]:
        """Get the identified controller structure after thresholding.

        Args:
            threshold: Threshold for binary decision

        Returns:
            Dict describing the identified structure per actuator
        """
        weights = self.get_selection_weights()

        structure = {
            "actuators": {},
            "sensors": [],
            "setpoints": [],
        }

        # Identify active candidates per actuator
        for a in range(self.n_actuators):
            active_candidates = []
            for c in range(self.n_candidates):
                if weights.get(f"alpha_{a}_{c}", torch.tensor(0)).item() > threshold:
                    ctrl = self._get_candidate(a, c)
                    active_candidates.append(
                        {
                            "index": c,
                            "class": ctrl.__class__.__name__,
                            "id": ctrl.id,
                        }
                    )
            structure["actuators"][a] = active_candidates

        # Identify active sensors per actuator
        structure["sensors"] = {}
        for a in range(self.n_actuators):
            active_sensors = []
            for i in range(self.n_sensors):
                if weights.get(f"beta_{a}_{i}", torch.tensor(0)).item() > threshold:
                    active_sensors.append(i)
            structure["sensors"][a] = active_sensors

        # Identify active setpoints per actuator
        structure["setpoints"] = {}
        for a in range(self.n_actuators):
            active_setpoints = []
            for j in range(self.n_setpoints):
                if weights.get(f"gamma_{a}_{j}", torch.tensor(0)).item() > threshold:
                    active_setpoints.append(j)
            structure["setpoints"][a] = active_setpoints

        # Identify active B-loop sensors per actuator (cascade controllers)
        if self._has_cascade:
            structure["sensors_b"] = {}
            for a in range(self.n_actuators):
                active_sensors_b = []
                for i in range(self.n_sensors):
                    if (
                        weights.get(f"beta_b_{a}_{i}", torch.tensor(0)).item()
                        > threshold
                    ):
                        active_sensors_b.append(i)
                structure["sensors_b"][a] = active_sensors_b

        return structure

    def reset_state(self) -> None:
        """Reset the state of all candidate controllers."""
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                # Cascade controllers handle their own sub-PID reset
                if hasattr(ctrl, "reset_state"):
                    ctrl.reset_state()
                # Reset PID controller state (direct PID candidates)
                if hasattr(ctrl, "err_prev"):
                    ctrl.err_prev = torch.zeros_like(ctrl.err_prev)
                if hasattr(ctrl, "err_prev_m1"):
                    ctrl.err_prev_m1 = torch.zeros_like(ctrl.err_prev_m1)
                if hasattr(ctrl, "u_prev"):
                    ctrl.u_prev = torch.zeros_like(ctrl.u_prev)

    def summary(self) -> str:
        """Get a summary of the identified controller."""
        lines = ["=" * 60]
        lines.append("Controller Identification Summary")
        lines.append("=" * 60)
        lines.append(f"\nConfiguration:")
        lines.append(f"  Actuators: {self.n_actuators}")
        lines.append(f"  Sensors: {self.n_sensors}")
        lines.append(f"  Setpoints: {self.n_setpoints}")
        lines.append(f"  Candidates: {self.n_candidates}")
        from collections import Counter
        type_counts = Counter(self._candidate_types)
        for ctype, count in type_counts.items():
            lines.append(f"    {ctype}: {count}")

        # Selection weights
        lines.append("\nSelection Weights:")
        weights = self.get_selection_weights()

        lines.append("  Alpha (candidate selection):")
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                ctype = self._candidate_types[c]
                val = weights[f"alpha_{a}_{c}"].item()
                lines.append(f"    α_{a},{c} [{ctype}] ({ctrl.__class__.__name__}): {val:.4f}")

        lines.append("  Beta (sensor selection per actuator):")
        for a in range(self.n_actuators):
            lines.append(f"    Actuator {a}:")
            for i in range(self.n_sensors):
                val = weights[f"beta_{a}_{i}"].item()
                lines.append(f"      β_{a},{i}: {val:.4f}")

        lines.append("  Gamma (setpoint selection per actuator):")
        for a in range(self.n_actuators):
            lines.append(f"    Actuator {a}:")
            for j in range(self.n_setpoints):
                val = weights[f"gamma_{a}_{j}"].item()
                lines.append(f"      γ_{a},{j}: {val:.4f}")

        if self._has_cascade:
            lines.append("  Beta_b (cascade B-loop sensor selection per actuator):")
            for a in range(self.n_actuators):
                lines.append(f"    Actuator {a}:")
                for i in range(self.n_sensors):
                    val = weights[f"beta_b_{a}_{i}"].item()
                    lines.append(f"      β_b_{a},{i}: {val:.4f}")

        lines.append("  Gate (setpoint-based gating per actuator):")
        for a in range(self.n_actuators):
            lines.append(f"    Actuator {a}:")
            alpha_g = weights[f"alpha_gate_{a}"].item()
            lines.append(f"      α_gate: {alpha_g:.4f}")
            thresh = weights[f"gate_{a}_threshold"].item()
            steep = weights[f"gate_{a}_steepness"].item()
            pol = weights[f"gate_{a}_polarity"].item()
            lines.append(f"      threshold: {thresh:.4f}")
            # BandGate exposes a ``band`` width parameter and a derived
            # ``threshold_high`` property.  Print both edges and the
            # width so the identified band ``[T_lo, T_hi]`` is visible.
            gate_a = getattr(self, f"gate_{a}", None)
            if gate_a is not None and hasattr(gate_a, "band"):
                width_raw = gate_a.band.get()
                width = width_raw.item() if hasattr(width_raw, "item") else float(width_raw)
                thresh_hi = thresh + width
                lines.append(f"      threshold_high: {thresh_hi:.4f}")
                lines.append(
                    f"      band: [{thresh:.4f}, {thresh_hi:.4f}] (width={width:.4f})"
                )
            lines.append(f"      steepness: {steep:.4f}")
            pol_label = "active above" if pol > 0 else "active below"
            lines.append(f"      polarity: {pol:.4f} ({pol_label})")
            default_out = weights[f"default_output_{a}"].item()
            lines.append(f"      default_output: {default_out:.4f}")
            for j in range(self.n_on_off_signals):
                val = weights[f"gamma_gate_{a}_{j}"].item()
                lines.append(f"      γ_gate_{a},{j}: {val:.4f}")

        # Candidate controller parameters
        lines.append("\nCandidate Controller Parameters:")
        for a in range(self.n_actuators):
            lines.append(f"  Actuator {a}:")
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                ctype = self._candidate_types[c]
                lines.append(f"    Candidate {c} [{ctype}] ({ctrl.__class__.__name__}):")

                if ctype == self.CTRL_CASCADE:
                    for sub_name in ("ctrl_a", "ctrl_b"):
                        sub = getattr(ctrl, sub_name)
                        lines.append(f"      {sub_name} ({sub.__class__.__name__}):")
                        self._summary_pid_params(lines, sub, indent=8)
                else:
                    self._summary_pid_params(lines, ctrl, indent=6)

        # Identified structure
        structure = self.get_identified_structure()
        lines.append("\nIdentified Structure (threshold=0.5):")
        for a in range(self.n_actuators):
            lines.append(f"  Actuator {a}:")
            lines.append(f"    Active sensors: {structure['sensors'].get(a, [])}")
            lines.append(f"    Active setpoints: {structure['setpoints'].get(a, [])}")
            if self._has_cascade:
                lines.append(
                    f"    Active B-loop sensors: {structure.get('sensors_b', {}).get(a, [])}"
                )
            candidates = structure["actuators"].get(a, [])
            lines.append(f"    Active controllers: {[c['class'] for c in candidates]}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_estimator_parameters(self) -> List[tuple]:
        """Get parameter specifications for use with twin4build Estimator.

        Returns a list of parameter tuples for all selection weights and
        candidate controller parameters. Uses vector Parameters with n_c
        dimension for efficient handling of multiple weights.

        Returns:
            List of tuples: [(component, attr, x0, lb, ub, "private"), ...]
                - x0 can be scalar (broadcast to n_c) or array of shape (n_c,)
                - lb, ub can be scalar (broadcast) or array of shape (n_c,)
        """
        # Build candidate controllers if not yet built (e.g. when called before
        # initialize() after auto-translation, where n_sensors/n_setpoints are
        # inferred from wired connections rather than constructor arguments).
        if not self._built:
            n_s = self.get_n_v_from_connections("sensorValue")
            n_sp = self.get_n_v_from_connections("setpointValue")
            n_oo = self.get_n_v_from_connections("onOffSignal")
            n_act = self._get_n_actuators_from_connections()
            if n_s is not None:
                self.n_sensors = n_s
            if n_sp is not None:
                self.n_setpoints = n_sp
            if n_oo is not None:
                self.n_on_off_signals = n_oo
            if n_act is not None:
                self.n_actuators = n_act
            if (
                self.n_sensors is not None
                and self.n_setpoints is not None
                and self.n_on_off_signals is not None
            ):
                self._build_components()

        params = []

        weight_lb = 0.0
        weight_ub = 1.0

        # Initial values: uniform distribution (1/n)
        alpha_x0 = 0.5
        beta_x0 = 0.5
        gamma_x0 = 0.5

        # All selection weights are per-actuator:
        # - alpha_{a}: candidate selection (n_c = n_candidates)
        # - beta_{a}: sensor selection (n_c = n_sensors)
        # - gamma_{a}: setpoint selection (n_c = n_setpoints)
        for a in range(self.n_actuators):
            params.append(
                (self, f"alpha_{a}", alpha_x0, weight_lb, weight_ub, "private")
            )
            params.append((self, f"beta_{a}", beta_x0, weight_lb, weight_ub, "private"))
            params.append(
                (self, f"gamma_{a}", gamma_x0, weight_lb, weight_ub, "private")
            )
            if self._has_cascade:
                params.append(
                    (self, f"beta_b_{a}", beta_x0, weight_lb, weight_ub, "private")
                )

            # Gate parameters (steepness is a hyperparameter, not estimated)
            params.append(
                (self, f"gamma_gate_{a}", gamma_x0, weight_lb, weight_ub, "private")
            )
            params.append(
                (self, f"gate_{a}.threshold", 18.0, 10.0, 30.0, "private")
            )
            # Only BandGate (and subclasses) expose a ``band`` width.
            # Advertise it to the estimator so users who swap in a
            # BandGate can learn the zone-specific deadband width;
            # callers using a plain SigmoidGate simply won't see this
            # parameter.  The upper edge ``threshold_high`` is derived
            # from ``threshold + band`` and not estimated directly,
            # which structurally prevents ``T_hi < T_lo``.
            gate_a = getattr(self, f"gate_{a}", None)
            if gate_a is not None and hasattr(gate_a, "band"):
                params.append(
                    (self, f"gate_{a}.band", 4.0, 0.0, 20.0, "private")
                )
            params.append(
                (self, f"gate_{a}.polarity", 1.0, -1.0, 1.0, "private")
            )
            params.append(
                (self, f"alpha_gate_{a}", 0.5, weight_lb, weight_ub, "private")
            )
            params.append(
                (self, f"default_output_{a}", 0.0, 0.0, 1.0, "private")
            )

        # Candidate controller parameters - accessed through parent using dot notation
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                ctype = self._candidate_types[c]
                prefix = f"candidate_{a}_{c}"

                if ctype == self.CTRL_CASCADE:
                    # Cascade: expose both sub-controller params (ctrl_a, ctrl_b)
                    # ctrl_a saturation = intermediate clamp (estimated)
                    # ctrl_b saturation = final output (fixed at 0/1)
                    for sub_name in ("ctrl_a", "ctrl_b"):
                        sub = getattr(ctrl, sub_name)
                        sub_prefix = f"{prefix}.{sub_name}"
                        if hasattr(sub, "kp"):
                            params.append(
                                (self, f"{sub_prefix}.kp", 0.1, 0.001, 10, "private")
                            )
                        if hasattr(sub, "Ti"):
                            params.append(
                                (self, f"{sub_prefix}.Ti", 10, 1.0, 10000.0, "private")
                            )
                        if hasattr(sub, "Td"):
                            params.append(
                                (self, f"{sub_prefix}.Td", 0.0, 0.0, 0.0001, "private")
                            )
                        if sub_name == "ctrl_a":
                            if hasattr(sub, "output_min"):
                                params.append(
                                    (self, f"{sub_prefix}.output_min", 0.0, 0.0, 1.0, "private")
                                )
                            if hasattr(sub, "output_max"):
                                params.append(
                                    (self, f"{sub_prefix}.output_max", 1.0, 0.0, 1.0, "private")
                                )
                else:
                    # Standard setpoint controllers (PID, on-off, etc.)
                    self._append_pid_params(params, ctrl, prefix)

        return params

    def get_gradient_scales(self, weight_scale: float = 0.01) -> List[float]:
        """Get recommended gradient scaling factors for each parameter.

        Selection weights (alpha, beta, gamma) typically have gradients ~100x
        larger than controller parameters due to directly multiplying signals.
        This method returns scaling factors that balance the gradients.

        Note: With vector Parameters, the scales are per-parameter (not per-element).
        The Estimator handles expanding scales to match n_c dimensions internally.

        Args:
            weight_scale: Scaling factor for selection weight gradients.
                Default 0.01 reduces weight gradients by 100x.

        Returns:
            List of gradient scales matching the order from get_estimator_parameters()

        Example:
            >>> params = controller.get_estimator_parameters()
            >>> scales = controller.get_gradient_scales()
            >>> result = estimator.estimate(
            ...     parameters=params,
            ...     gradient_scales=scales,
            ...     ...
            ... )
        """
        scales = []

        for a in range(self.n_actuators):
            scales.append(weight_scale)  # alpha
            scales.append(weight_scale)  # beta
            scales.append(weight_scale)  # gamma
            if self._has_cascade:
                scales.append(weight_scale)  # beta_b
            scales.append(weight_scale)  # gamma_gate
            scales.append(1.0)  # gate.threshold
            scales.append(1.0)  # gate.polarity
            scales.append(weight_scale)  # alpha_gate
            scales.append(1.0)  # default_output

        # Controller parameters - no scaling (1.0)
        # Must match the order and count of get_estimator_parameters()
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                ctype = self._candidate_types[c]

                if ctype == self.CTRL_CASCADE:
                    for sub_name in ("ctrl_a", "ctrl_b"):
                        sub = getattr(ctrl, sub_name)
                        for attr_name in ("kp", "Ti", "Td", "output_min", "output_max"):
                            if hasattr(sub, attr_name):
                                scales.append(1.0)
                else:
                    self._append_pid_scales(scales, ctrl)

        return scales



def brick_signature_pattern_vav():
    """
    BRICK signature pattern for VAV zone controller identification.

    A single pattern matching a VAV box with all directly connected signals:
    feedback sensors, setpoints, and actuator commands (with timeseries IDs).

    Groups are formed as the cross-product of (sensor × setpoint × actuator)
    per VAV.  resolve_port_indices uses unique-value ordinals (not raw group
    indices) so each signal type is correctly indexed independent of the
    cross-product size:

      sensors   → sensorValue[0..n_sensors-1]
      setpoints → setpointValue[0..n_setpoints-1]
      actuators → CITS groups (for SensorSystem command pattern index resolution)

    Only commands with ref:hasTimeseriesId are included (purely logical flags
    like Heating_Mode which lack timeseries data are excluded).

    Topology::

        VAV  hasPoint  <Zone_Air_Temperature_Sensor>    → sensorValue[0]
        VAV  hasPoint  <Supply_Air_Temperature_Sensor>  → sensorValue[1]
        VAV  hasPoint  <Air_Flow_Sensor>                → sensorValue[2]
        VAV  hasPoint  <Supply_Air_Flow_Sensor>         → sensorValue[3]
        VAV  hasPoint  <Zone_Air_Temperature_Setpoint>  → setpointValue[0]
        VAV  hasPoint  <Zone_Air_Temperature_Setpoint>  → setpointValue[1]
        VAV  hasPoint  <Command (with ts-id)>           → (actuator slot for index resolution)
    """
    vav = Node(cls=core.namespace.BRICK.VAV)
    sensors = Node(
        cls=(
            core.namespace.BRICK.Zone_Air_Temperature_Sensor,
            core.namespace.BRICK.Supply_Air_Temperature_Sensor,
            core.namespace.BRICK.Air_Flow_Sensor,
            core.namespace.BRICK.Supply_Air_Flow_Sensor,
        )
    )
    setpoints = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Setpoint)
    actuators = Node(cls=core.namespace.BRICK.Command)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="controller_identification_vav_brick")
    # The three VAV ``hasPoint`` rules use ``SetStepRule`` to collapse
    # the per-point cross-product into one group per VAV: ``sensors``,
    # ``setpoints`` and ``actuators`` each bind to the *tuple of all*
    # matching points. Downstream ``StepRule`` hops (the
    # actuator → externalref → timeseries_id chain) are auto-broadcast
    # per element by the matcher so they remain scalar rules here.
    sp.add_rule(SetStepRule(subject=vav, object=sensors, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=vav, object=setpoints, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=vav, object=actuators, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(StepRule(subject=actuators, object=externalref, predicate=core.namespace.BRICKREF.hasExternalReference))
    sp.add_rule(StepRule(subject=externalref, object=timeseries_id, predicate=core.namespace.BRICKREF.hasTimeseriesId))

    sp.add_connection(sensors, "measuredValue", "sensorValue", input_port_index=sensors)
    sp.add_connection(setpoints, "measuredValue", "setpointValue", input_port_index=setpoints)
    # Auto-mirror every setpoint into the gate-input bus.  See the
    # corresponding pattern on ``ControllerIdentificationPISystem``
    # for the full rationale -- in short, the ``onOffSignal`` port is
    # never pruned by the rewire and gives the gate access to the
    # schedule even when the rewire winner picks a different setpoint
    # for the PI error term.
    sp.add_connection(setpoints, "measuredValue", "onOffSignal", input_port_index=setpoints)
    # The VAV controller entity is not a first-class node in BRICK; it is
    # identified jointly by the VAV, its sensor/setpoint points and its
    # command actuators. Expressing this as a ``ModeledNode`` group makes
    # the composite identity explicit and leaves the member SM nodes
    # (notably the ``BRICK.Command`` actuators) available for other
    # systems (e.g. ``SensorSystem``) to model on their own via the
    # non-exclusive mutex semantics.
    ModeledNode([vav, sensors, setpoints, actuators])
    return sp


# NOTE: BRICK pattern registration disabled for the generic CITS class.  The
# matching patterns are now registered on
# :class:`ControllerIdentificationPISystem` so the translator builds
# PI-only CITS instances (with the joint regression loop classifier as a
# pre-step).  The generic class still works programmatically; it just no
# longer matches BRICK topologies during translation.  Re-enable this line
# (and the damper variant below) if you want both classes to compete.
# ControllerIdentificationSystem.add_signature_pattern(brick_signature_pattern_vav())


def brick_signature_pattern_vav_damper():
    """
    BRICK signature pattern for VAV damper controller identification.

    Damper commands are modeled indirectly in BRICK via a Damper equipment
    entity rather than as a direct hasPoint of the VAV:

        Damper  isPartOf   VAV
        Damper  hasPoint   <Damper_Position_Setpoint (with ts-id)>

    Sensors and setpoints are still direct hasPoint of the VAV.
    Each matched damper command becomes its own CITS instance
    (one CITS per actuator, n_actuators=1).
    """
    vav = Node(cls=core.namespace.BRICK.VAV)
    sensors = Node(
        cls=(
            core.namespace.BRICK.Zone_Air_Temperature_Sensor,
            core.namespace.BRICK.Supply_Air_Temperature_Sensor,
            core.namespace.BRICK.Air_Flow_Sensor,
            core.namespace.BRICK.Supply_Air_Flow_Sensor,
        )
    )
    setpoints = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Setpoint)
    damper_equip = Node(cls=core.namespace.BRICK.Damper)
    damper_cmd = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="controller_identification_vav_damper_brick")
    # ``sensors`` and ``setpoints`` collect all points per VAV as sets;
    # ``damper_cmd`` collects all damper commands per damper equipment.
    # The scalar edges (isPartOf, the actuator → externalref →
    # timeseries_id chain) stay as plain ``StepRule`` and are
    # auto-broadcast over the set-bound endpoints.
    sp.add_rule(SetStepRule(subject=vav, object=sensors, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=vav, object=setpoints, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(StepRule(subject=damper_equip, object=vav, predicate=core.namespace.BRICK.isPartOf))
    sp.add_rule(SetStepRule(subject=damper_equip, object=damper_cmd, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(StepRule(subject=damper_cmd, object=externalref, predicate=core.namespace.BRICKREF.hasExternalReference))
    sp.add_rule(StepRule(subject=externalref, object=timeseries_id, predicate=core.namespace.BRICKREF.hasTimeseriesId))

    sp.add_connection(sensors, "measuredValue", "sensorValue", input_port_index=sensors)
    sp.add_connection(setpoints, "measuredValue", "setpointValue", input_port_index=setpoints)
    # Auto-mirror setpoints into the gate-input bus (see sibling
    # pattern for rationale).
    sp.add_connection(setpoints, "measuredValue", "onOffSignal", input_port_index=setpoints)
    # See ``brick_signature_pattern_vav``: the damper controller is an
    # implicit entity, identified by the (VAV, sensors, setpoints,
    # damper equipment, damper command) tuple.
    ModeledNode([vav, sensors, setpoints, damper_equip, damper_cmd])
    return sp


# NOTE: see comment above the previous registration; re-enable this if both
# CITS variants should compete during translation.
# ControllerIdentificationSystem.add_signature_pattern(brick_signature_pattern_vav_damper())

# Deprecated aliases (removed in twin4build 2.1)
ControllerIdentificationTorchSystem = ControllerIdentificationSystem
