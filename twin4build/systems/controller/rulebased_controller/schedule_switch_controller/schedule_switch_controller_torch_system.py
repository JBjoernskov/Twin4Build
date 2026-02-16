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


class ScheduleSwitchControllerTorchSystem(core.System, nn.Module):
    r"""
    Differentiable Schedule Switch Controller that blends between an upstream
    input signal and a learnable override value based on a weekly schedule.

    All parameters are tunable via gradient-based optimization.

    Mathematical Formulation
    ------------------------

    **Independent hour-day schedule weights (168 parameters):**

    Each (hour, day) pair has its own independent weight
    :math:`s_{h,d} \in [0, 1]` for :math:`h = 0, \ldots, 23` and
    :math:`d = 0, \ldots, 6` (Mon-Sun).

    .. math::

        s = s_{\lfloor t \rfloor, \text{weekday}}

    This eliminates the bilinear product :math:`h_t \cdot w_d` from the old
    factored formulation, removing saddle-point non-convexity and making each
    weight independently optimisable.

    **Combined output (blend between input and override):**

    .. math::

        u = s \cdot x + (1 - s) \cdot v_{\text{override}}

    where :math:`x` is the upstream input signal and :math:`v_{\text{override}}`
    is a learnable override value in [0, 1].

    - When :math:`s \approx 1` (schedule active): :math:`u \approx x` — the
      input signal passes through.
    - When :math:`s \approx 0` (schedule inactive): :math:`u \approx v_{\text{override}}`
      — the output is the learned override value (e.g. 1.0 for a damper that
      springs fully open at night).

    Args:
        schedule_weights: Optional dict or 2-D structure of shape (24, 7) with
            values in [0, 1].  If *None*, all 168 weights default to 0.5
            (undecided, will be learned).  Can also be constructed from the
            legacy ``hour_weights`` / ``day_weights`` interface (see below).
        hour_weights: (Legacy) Iterable of 24 floats in [0, 1].  If provided
            together with ``day_weights``, the 168 schedule weights are
            initialised as the outer product ``h_i * w_d``.
        day_weights: (Legacy) Iterable of 7 floats in [0, 1].
        override_value: Float in [0, 1] for the output when the schedule is
            inactive. Default 0.5 (undecided, will be learned).
        **kwargs: Additional keyword arguments passed to parent classes.

    Example:
        >>> # Weekday office-hours gate with night override at 100%
        >>> gate = ScheduleSwitchControllerTorchSystem(
        ...     hour_weights=[0.0]*7 + [1.0]*10 + [0.0]*7,
        ...     day_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ...     override_value=1.0,
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # --- Build the 168 schedule weights ---
        if schedule_weights is not None:
            # Direct specification: list of 24 lists of 7 values (hour × day)
            sw = schedule_weights
            if len(sw) != self.N_HOURS or any(len(row) != self.N_DAYS for row in sw):
                raise ValueError(
                    f"schedule_weights must be shape ({self.N_HOURS}, {self.N_DAYS}), "
                    f"got {len(sw)} rows"
                )
        elif hour_weights is not None or day_weights is not None:
            # Legacy interface: build from outer product h_i * w_d
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
            sw = [
                [float(hour_weights[h]) * float(day_weights[d]) for d in range(self.N_DAYS)]
                for h in range(self.N_HOURS)
            ]
        else:
            # Default: all 0.5 (undecided)
            sw = [[0.5] * self.N_DAYS for _ in range(self.N_HOURS)]

        # Create 168 independent parameters: schedule_h{hour}_d{day}
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

        # --- Override value: output when schedule is inactive ---
        self.override_value = tps.Parameter(
            torch.tensor(float(override_value), dtype=torch.float64),
            min_value=0.0,
            max_value=1.0,
            requires_grad=False,
        )

        # --- I/O ---
        self.input = {"inputSignal": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}

        self._config = {
            "parameters": (
                [f"schedule_h{h}_d{d}" for h in range(self.N_HOURS) for d in range(self.N_DAYS)]
                + ["override_value"]
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self._config

    def _schedule_weight_params(self) -> List[tps.Parameter]:
        """Return all 168 schedule weight parameters in (hour, day) order."""
        return [
            getattr(self, f"schedule_h{h}_d{d}")
            for h in range(self.N_HOURS)
            for d in range(self.N_DAYS)
        ]

    def _get_schedule_weight(self, hour: int, day: int) -> tps.Parameter:
        """Get a single schedule weight parameter."""
        return getattr(self, f"schedule_h{hour}_d{day}")

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

        self.input["inputSignal"].initialize(
            n_timesteps=max_timesteps,
            batch_size=batch_size,
        )
        self.output["inputSignal"].initialize(
            n_timesteps=max_timesteps,
            batch_size=batch_size,
        )

        # Expand all schedule parameters to n_c dimension for vectorization
        for h in range(self.N_HOURS):
            for d in range(self.N_DAYS):
                attr = f"schedule_h{h}_d{d}"
                setattr(self, attr, getattr(self, attr).expand_to_n_c(self.n_c))

        self.override_value = self.override_value.expand_to_n_c(self.n_c)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one simulation step: blend between input signal and override value.

        Computes ``output = s * input + (1 - s) * override_value`` where *s* is
        the independent schedule weight for the current (hour, day) pair.
        """
        input_signal = self.input["inputSignal"].get()

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

        # --- Look up schedule weight for each (hour, day) pair ---
        # Build full (24, 7, n_c) tensor and index into it
        all_weights = torch.stack(
            [self._get_schedule_weight(h, d).get()
             for h in range(self.N_HOURS)
             for d in range(self.N_DAYS)],
            dim=0,
        ).reshape(self.N_HOURS, self.N_DAYS, -1)  # (24, 7, n_c)

        hour_idx = torch.tensor(hours, dtype=torch.long)      # (n_s,)
        day_idx = torch.tensor(weekdays, dtype=torch.long)     # (n_s,)
        schedule_signal = all_weights[hour_idx, day_idx]       # (n_s, n_c)

        # --- Blend: active -> input, inactive -> override_value ---
        override = self.override_value.get()  # (n_c,)
        output_signal = schedule_signal * input_signal + (1 - schedule_signal) * override

        self.output["inputSignal"].set(output_signal, step_index)

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
        for h in range(self.N_HOURS):
            for d in range(self.N_DAYS):
                params.append((self, f"schedule_h{h}_d{d}", 0.5, 0.0, 1.0, "private"))

        params.append((self, "override_value", 1, 0.0, 1.0, "private"))
        return params

    def compute_binarization_penalty(self) -> torch.Tensor:
        """Compute the binarization penalty P(x) = x(1-x) for all schedule weights.

        The penalty is zero when weights are at 0 or 1 (fully decided) and maximal
        at 0.5 (undecided). This encourages the optimizer to push all schedule
        weights towards crisp on/off decisions.

        Returns:
            torch.Tensor: Total binarization penalty summed over all 168 schedule weights.
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
        return self._get_schedule_weight(hour, weekday).get()

    def get_schedule_matrix(self) -> torch.Tensor:
        """
        Return the full 24×7 schedule matrix (useful for visualization).

        Returns:
            Tensor of shape ``(24, 7)`` or ``(24, 7, n_c)`` with schedule weights.
        """
        rows = []
        for h in range(self.N_HOURS):
            row = []
            for d in range(self.N_DAYS):
                row.append(self._get_schedule_weight(h, d).get())
            rows.append(torch.stack(row, dim=0))
        return torch.stack(rows, dim=0)  # (24, 7, ...) or (24, 7)

    def reset_state(self) -> None:
        """Reset controller state (no-op for schedule switch controller)."""
        pass
