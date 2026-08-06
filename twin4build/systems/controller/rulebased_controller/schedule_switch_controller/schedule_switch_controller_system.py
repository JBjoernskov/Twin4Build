# Standard library imports
import datetime
from typing import List

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class ScheduleSwitchControllerSystem(core.System, nn.Module):
    r"""
    Differentiable Schedule Switch Controller that blends between an upstream
    input signal and a learnable override value based on a weekly schedule.

    All parameters are tunable via gradient-based optimization.

    Two schedule parameterizations are available, controlled by ``factored``:

    **Factored mode** (``factored=True``, 31 parameters, recommended):

    The schedule is the outer product of a 24-element hour profile
    :math:`h_w` and a 7-element day profile :math:`d_w`:

    .. math::

        s(h, d) = h_w[h] \cdot d_w[d]

    where :math:`h_w[h], d_w[d] \in [0, 1]`.  This gives **31** parameters
    and each parameter is constrained by data from all hours (for day weights)
    or all days (for hour weights), greatly reducing overfitting risk.

    **Independent mode** (``factored=False``, 168 parameters):

    Each (hour, day) pair has its own independent weight
    :math:`s_{h,d} \in [0, 1]`.  This is more flexible but prone to
    overfitting when training data covers only a few weeks.

    **Combined output (blend between input and override):**

    .. math::

        u = s \cdot x + (1 - s) \cdot v_{\text{override}}

    where :math:`x` is the upstream input signal and :math:`v_{\text{override}}`
    is a learnable override value in [0, 1].

    - When :math:`s \approx 1` (schedule active): :math:`u \approx x` — the
      input signal passes through.
    - When :math:`s \approx 0` (schedule inactive): :math:`u \approx v_{\text{override}}`
      — the output is the learned override value.

    Args:
        hour_weights: Iterable of 24 floats in [0, 1].
        day_weights: Iterable of 7 floats in [0, 1].
        schedule_weights: Optional 2-D structure of shape (24, 7) with
            values in [0, 1].  Only used when ``factored=False``.
        override_value: Float in [0, 1] for the output when the schedule is
            inactive. Default 0.5 (undecided, will be learned).
        factored: If True (default), keep the bilinear hour*day
            parameterization during optimization (31 params).  If False,
            use 168 independent (hour, day) weights.
        **kwargs: Additional keyword arguments passed to parent classes.

    Example:
        >>> gate = ScheduleSwitchControllerSystem(
        ...     hour_weights=[0.0]*7 + [1.0]*10 + [0.0]*7,
        ...     day_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ...     override_value=1.0,
        ...     factored=True,
        ...     id="office_schedule",
        ... )
    """

    N_HOURS = 24
    N_DAYS = 7
    N_SCHEDULE_WEIGHTS = N_HOURS * N_DAYS  # 168

    def __init__(
        self,
        schedule_weights: List[List[float]] = None,
        hour_weights: List[float] = None,
        day_weights: List[float] = None,
        override_value: float = 0.5,
        factored: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self._factored = factored

        # --- Resolve hour_weights and day_weights ---
        if hour_weights is None:
            hour_weights = [0.5] * self.N_HOURS
        if day_weights is None:
            day_weights = [0.5] * self.N_DAYS
        if len(hour_weights) != self.N_HOURS:
            raise ValueError(
                f"hour_weights must have {self.N_HOURS} values, got {len(hour_weights)}"
            )
        if len(day_weights) != self.N_DAYS:
            raise ValueError(
                f"day_weights must have {self.N_DAYS} values, got {len(day_weights)}"
            )

        if self._factored:
            # --- Factored mode: store 24 + 7 = 31 parameters ---
            for h in range(self.N_HOURS):
                setattr(
                    self,
                    f"hour_weight_{h}",
                    tps.Parameter(
                        torch.tensor(float(hour_weights[h]), dtype=torch.float64),
                        min_value=0.0,
                        max_value=1.0,
                        requires_grad=False,
                    ),
                )
            for d in range(self.N_DAYS):
                setattr(
                    self,
                    f"day_weight_{d}",
                    tps.Parameter(
                        torch.tensor(float(day_weights[d]), dtype=torch.float64),
                        min_value=0.0,
                        max_value=1.0,
                        requires_grad=False,
                    ),
                )

            self._config = {
                "parameters": (
                    [f"hour_weight_{h}" for h in range(self.N_HOURS)]
                    + [f"day_weight_{d}" for d in range(self.N_DAYS)]
                    + ["override_value"]
                ),
            }
        else:
            # --- Independent mode: store 168 parameters ---
            if schedule_weights is not None:
                sw = schedule_weights
                if len(sw) != self.N_HOURS or any(
                    len(row) != self.N_DAYS for row in sw
                ):
                    raise ValueError(
                        f"schedule_weights must be shape ({self.N_HOURS}, {self.N_DAYS}), "
                        f"got {len(sw)} rows"
                    )
            else:
                sw = [
                    [
                        float(hour_weights[h]) * float(day_weights[d])
                        for d in range(self.N_DAYS)
                    ]
                    for h in range(self.N_HOURS)
                ]

            for h in range(self.N_HOURS):
                for d in range(self.N_DAYS):
                    setattr(
                        self,
                        f"schedule_h{h}_d{d}",
                        tps.Parameter(
                            torch.tensor(float(sw[h][d]), dtype=torch.float64),
                            min_value=0.0,
                            max_value=1.0,
                            requires_grad=False,
                        ),
                    )

            self._config = {
                "parameters": (
                    [
                        f"schedule_h{h}_d{d}"
                        for h in range(self.N_HOURS)
                        for d in range(self.N_DAYS)
                    ]
                    + ["override_value"]
                ),
            }

        # --- Override value: output when schedule is inactive ---
        self.override_value = tps.Parameter(
            torch.tensor(float(override_value), dtype=torch.float64),
            min_value=0.0,
            max_value=1.0,
            requires_grad=False,
        )

        # --- I/O ---
        # hourOfDay / dayOfWeek are measured-data ports: NOT connected to any
        # producer.  ``do_step`` publishes the wall-clock features here each
        # step so the composed fast paths (Simulator.compose) can capture
        # them per step like any exogenous signal -- they are pure time
        # features, independent of any estimated parameter.
        self.input = {
            "inputSignal": tps.Scalar(),
            "hourOfDay": tps.Scalar(),
            "dayOfWeek": tps.Scalar(),
        }
        self.output = {"inputSignal": tps.Scalar()}

        # All schedule weights + override are estimable -> route through the
        # ``forward`` params dict (PARAM_NAMES is per-instance because the
        # weight set depends on ``factored``).
        self.PARAM_NAMES = tuple(self._config["parameters"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def factored(self) -> bool:
        return self._factored

    @property
    def config(self):
        return self._config

    def _get_schedule_weight(self, hour: int, day: int) -> tps.Parameter:
        """Get a single schedule weight parameter (independent mode only)."""
        if self._factored:
            raise RuntimeError(
                "_get_schedule_weight is not available in factored mode. "
                "Use get_schedule_matrix() instead."
            )
        return getattr(self, f"schedule_h{hour}_d{day}")

    def _schedule_weight_params(self) -> List[tps.Parameter]:
        """Return all schedule weight parameters."""
        if self._factored:
            return [
                getattr(self, f"hour_weight_{h}") for h in range(self.N_HOURS)
            ] + [getattr(self, f"day_weight_{d}") for d in range(self.N_DAYS)]
        else:
            return [
                getattr(self, f"schedule_h{h}_d{d}")
                for h in range(self.N_HOURS)
                for d in range(self.N_DAYS)
            ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: int,
    ) -> None:
        """Initialize the controller for simulation."""
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        for inp in self.input.values():
            inp.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
            )
        self.output["inputSignal"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
        )

        if self._factored:
            for h in range(self.N_HOURS):
                attr = f"hour_weight_{h}"
                setattr(self, attr, getattr(self, attr).expand_to_n_c(self.n_c))
            for d in range(self.N_DAYS):
                attr = f"day_weight_{d}"
                setattr(self, attr, getattr(self, attr).expand_to_n_c(self.n_c))
        else:
            for h in range(self.N_HOURS):
                for d in range(self.N_DAYS):
                    attr = f"schedule_h{h}_d{d}"
                    setattr(
                        self, attr, getattr(self, attr).expand_to_n_c(self.n_c)
                    )

        self.override_value = self.override_value.expand_to_n_c(self.n_c)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step schedule blend (functorch-safe, stateless).

        Computes ``output = s * input + (1 - s) * override_value`` where *s*
        is the schedule weight for the current (hour, day) pair.  The time
        features enter through the ``hourOfDay`` / ``dayOfWeek`` data ports
        (captured per step by the composed fast paths), so all estimable
        weights keep their gradient paths through ``params``.
        """
        input_signal = inputs["inputSignal"]
        hour_idx = inputs["hourOfDay"].reshape(-1).long()
        day_idx = inputs["dayOfWeek"].reshape(-1).long()

        if self._factored:
            # Bilinear: schedule_weight = hour_w[h] * day_w[d]
            hw = torch.stack(
                [params[f"hour_weight_{h}"] for h in range(self.N_HOURS)], dim=0
            )  # (24, n_c)
            dw = torch.stack(
                [params[f"day_weight_{d}"] for d in range(self.N_DAYS)], dim=0
            )  # (7, n_c)
            schedule_signal = hw[hour_idx] * dw[day_idx]  # (n_s, n_c)
        else:
            # Independent: look up (hour, day) weight directly
            all_weights = torch.stack(
                [
                    params[f"schedule_h{h}_d{d}"]
                    for h in range(self.N_HOURS)
                    for d in range(self.N_DAYS)
                ],
                dim=0,
            ).reshape(self.N_HOURS, self.N_DAYS, -1)
            schedule_signal = all_weights[hour_idx, day_idx]  # (n_s, n_c)

        # --- Blend: active -> input, inactive -> override_value ---
        override = params["override_value"]
        output_signal = (
            schedule_signal * input_signal + (1 - schedule_signal) * override
        )
        return x, {"inputSignal": output_signal}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one simulation step: blend between input signal and override value.

        Thin port-I/O wrapper: extracts the wall-clock features, publishes
        them on the (unconnected) data ports and delegates the math to
        :meth:`forward`.
        """
        # --- Extract time features from date_time ---
        if isinstance(date_time, np.ndarray):
            dt_list = [pd.Timestamp(dt) for dt in date_time.flat]
        elif not isinstance(date_time, list):
            dt_list = [pd.Timestamp(date_time)]
        else:
            dt_list = [pd.Timestamp(dt) for dt in date_time]

        hours = []
        weekdays = []
        for dt in dt_list:
            if pd.isna(dt):
                hours.append(0)
                weekdays.append(0)
            else:
                hours.append(dt.hour)
                weekdays.append(dt.weekday())

        hour_t = torch.tensor(hours, dtype=torch.float64).unsqueeze(-1)  # (n_s, 1)
        day_t = torch.tensor(weekdays, dtype=torch.float64).unsqueeze(-1)

        # Publish the time features on the (unconnected) data input ports so
        # the composed fast paths can capture them per step.
        self.input["hourOfDay"]._set(hour_t, i_t=step_index)
        self.input["dayOfWeek"]._set(day_t, i_t=step_index)

        inputs = {
            "inputSignal": self.input["inputSignal"].get(),
            "hourOfDay": hour_t,
            "dayOfWeek": day_t,
        }
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["inputSignal"].set(outs["inputSignal"], step_index)

    # ------------------------------------------------------------------
    # Parameter estimation support
    # ------------------------------------------------------------------

    def get_estimator_parameters(self) -> List[tuple]:
        """
        Return parameter specifications for gradient-based estimation.

        Returns:
            List of tuples
            ``(component, attr_name, x0, lb, ub, parameter_type)``.
        """
        params = []
        if self._factored:
            for h in range(self.N_HOURS):
                params.append(
                    (self, f"hour_weight_{h}", 0.5, 0.0, 1.0, "private")
                )
            for d in range(self.N_DAYS):
                params.append(
                    (self, f"day_weight_{d}", 0.5, 0.0, 1.0, "private")
                )
        else:
            for h in range(self.N_HOURS):
                for d in range(self.N_DAYS):
                    params.append(
                        (self, f"schedule_h{h}_d{d}", 0.5, 0.0, 1.0, "private")
                    )

        params.append((self, "override_value", 1, 0.0, 1.0, "private"))
        return params

    def compute_binarization_penalty(self) -> torch.Tensor:
        """Compute the binarization penalty P(x) = x(1-x) for all schedule weights.

        The penalty is zero when weights are at 0 or 1 (fully decided) and maximal
        at 0.5 (undecided). This encourages the optimizer to push all schedule
        weights towards crisp on/off decisions.

        Returns:
            torch.Tensor: Total binarization penalty summed over all schedule weights.
        """
        penalty = torch.tensor(0.0, dtype=torch.float64)
        for p in self._schedule_weight_params():
            w = p.get()
            penalty = penalty + torch.sum(w * (1 - w))
        return penalty

    def compute_regularization_penalty(self) -> torch.Tensor:
        """Standard interface for Estimator to compute regularization penalty.

        This method is automatically called by the Estimator when
        regularization_lambda > 0 is specified.

        Returns:
            torch.Tensor: Regularization penalty (binarization penalty for this component).
        """
        return self.compute_binarization_penalty()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_schedule_signal(
        self,
        date_time: datetime.datetime,
    ) -> torch.Tensor:
        """
        Compute the schedule signal for a given datetime (useful for debugging).

        Args:
            date_time: The datetime to evaluate.

        Returns:
            Schedule signal between 0 (OFF) and 1 (ON), shape ``(n_c,)``.
        """
        hour = date_time.hour
        weekday = date_time.weekday()
        if self._factored:
            hw = getattr(self, f"hour_weight_{hour}").get()
            dw = getattr(self, f"day_weight_{weekday}").get()
            return hw * dw
        else:
            return getattr(self, f"schedule_h{hour}_d{weekday}").get()

    def get_schedule_matrix(self) -> torch.Tensor:
        """
        Return the full 24x7 schedule matrix (useful for visualization).

        Returns:
            Tensor of shape ``(24, 7)`` or ``(24, 7, n_c)`` with schedule weights.
        """
        if self._factored:
            hw = torch.stack(
                [getattr(self, f"hour_weight_{h}").get() for h in range(self.N_HOURS)],
                dim=0,
            )  # (24,) or (24, n_c)
            dw = torch.stack(
                [getattr(self, f"day_weight_{d}").get() for d in range(self.N_DAYS)],
                dim=0,
            )  # (7,) or (7, n_c)
            if hw.dim() == 1:
                return torch.outer(hw, dw)  # (24, 7)
            else:
                return hw.unsqueeze(1) * dw.unsqueeze(0)  # (24, 7, n_c)
        else:
            rows = []
            for h in range(self.N_HOURS):
                row = []
                for d in range(self.N_DAYS):
                    row.append(getattr(self, f"schedule_h{h}_d{d}").get())
                rows.append(torch.stack(row, dim=0))
            return torch.stack(rows, dim=0)

    def reset_state(self) -> None:
        """Reset controller state (no-op for schedule switch controller)."""
        pass

# Deprecated aliases (removed in twin4build 2.1)
ScheduleSwitchControllerTorchSystem = ScheduleSwitchControllerSystem
