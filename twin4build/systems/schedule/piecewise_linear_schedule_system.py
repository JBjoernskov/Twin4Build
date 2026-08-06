# Standard library imports
import copy
import datetime
from typing import Dict, List, Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem

_EMPTY_RULESET = {
    "ruleset_default_value": 0,
    "ruleset_start_minute": [],
    "ruleset_end_minute": [],
    "ruleset_start_hour": [],
    "ruleset_end_hour": [],
    "ruleset_value": [],
}


class PiecewiseLinearScheduleSystem(PiecewiseLinearSystem, ScheduleSystem):
    """A schedule system using piecewise linear interpolation.

    Combines PiecewiseLinearSystem and ScheduleSystem to create a scheduling
    system that interpolates between schedule points using piecewise linear
    functions.

    The simplest usage provides ``defaultX`` / ``defaultY`` — plain lists that
    the JSON config system can serialise reliably::

        PiecewiseLinearScheduleSystem(
            defaultX=[-12, 5, 20],
            defaultY=[60, 50, 20],
            id="supply_water_schedule",
        )

    A ``weekday_ruleset`` is built automatically.  Time-varying curves are
    still possible by providing the standard ruleset dicts with
    ``{"X": [...], "Y": [...]}`` values.

    Args:
        defaultX: Default X breakpoints (list of floats).
        defaultY: Default Y breakpoints (list of floats).
        **kwargs: Keyword arguments passed to parent classes including
            weekday_ruleset, weekend_ruleset, per-day rulesets,
            and add_noise.
    """

    # Override inherited ScheduleSystem.sp so the translator does NOT
    # auto-match this class from semantic models.
    sp = None

    # NOT composable: unlike the parent ``PiecewiseLinearSystem`` (fixed
    # interpolation table -> pure ``forward``), this schedule re-resolves its
    # (X, Y) table from the wall clock every step (``_resolve_xy(date_time)``)
    # -- a time source.  Overriding ``forward`` back to ``None`` makes
    # ``_has_real_forward`` treat it as exogenous, so the composed fast paths
    # capture its output per step (theta-independent by construction).
    forward = None

    def __init__(
        self,
        default_x: Optional[List[float]] = None,
        default_y: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        from twin4build.utils.deprecation import deprecate_args

        legacy = deprecate_args(
            ["defaultX", "defaultY"],
            ["default_x", "default_y"],
            [None, None],
            kwargs,
        )
        default_x = legacy.get("default_x", default_x)
        default_y = legacy.get("default_y", default_y)

        # Auto-create a scalar weekday_ruleset when only default_x/y given.
        if default_x is not None and default_y is not None:
            if kwargs.get("weekday_ruleset") is None and kwargs.get(
                "weekday_ruleset"
            ) is None:
                kwargs["weekday_ruleset"] = copy.deepcopy(_EMPTY_RULESET)

        super().__init__(**kwargs)

        self.default_x = default_x
        self.default_y = default_y

        self._input = {"x": tps.Scalar()}
        self._output = {"scheduleValue": tps.Scalar()}
        self._config = {
            "parameters": [
                "default_x",
                "default_y",
                "weekday_ruleset",
                "weekend_ruleset",
                "monday_ruleset",
                "tuesday_ruleset",
                "wednesday_ruleset",
                "thursday_ruleset",
                "friday_ruleset",
                "saturday_ruleset",
                "sunday_ruleset",
                "add_noise",
            ]
        }

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
    ) -> None:
        assert (
            (self.default_x is not None and self.default_y is not None)
            or self.weekday_ruleset is not None
            or self.weekend_ruleset is not None
        ), (
            f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: "
            "Provide defaultX/defaultY or a weekday_ruleset/weekend_ruleset."
        )

        # Inherit missing day rulesets from weekday / weekend defaults
        if self.monday_ruleset is None:
            self.monday_ruleset = self.weekday_ruleset
        if self.tuesday_ruleset is None:
            self.tuesday_ruleset = self.weekday_ruleset
        if self.wednesday_ruleset is None:
            self.wednesday_ruleset = self.weekday_ruleset
        if self.thursday_ruleset is None:
            self.thursday_ruleset = self.weekday_ruleset
        if self.friday_ruleset is None:
            self.friday_ruleset = self.weekday_ruleset
        if self.saturday_ruleset is None:
            self.saturday_ruleset = (
                self.weekend_ruleset
                if self.weekend_ruleset is not None
                else self.weekday_ruleset
            )
        if self.sunday_ruleset is None:
            self.sunday_ruleset = (
                self.weekend_ruleset
                if self.weekend_ruleset is not None
                else self.weekday_ruleset
            )

        # Ensure standard ruleset keys exist so get_schedule_value works
        required_keys = [
            "ruleset_start_minute",
            "ruleset_end_minute",
            "ruleset_start_hour",
            "ruleset_end_hour",
            "ruleset_value",
        ]
        for ruleset_dict in [
            self.monday_ruleset,
            self.tuesday_ruleset,
            self.wednesday_ruleset,
            self.thursday_ruleset,
            self.friday_ruleset,
            self.saturday_ruleset,
            self.sunday_ruleset,
        ]:
            if ruleset_dict is not None:
                for key in required_keys:
                    if key not in ruleset_dict:
                        ruleset_dict[key] = []

        # Initialize I/O buffers (skip ScheduleSystem.initialize which
        # assumes scalar schedule values)
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for inp in self.input.values():
            inp.initialize(n_t=max_timesteps, n_s=batch_size)
        for out in self.output.values():
            out.initialize(n_t=max_timesteps, n_s=batch_size)

    def _resolve_xy(self, date_time: datetime.datetime, device=None):
        """Return (X_points, Y_points) tensors for the given datetime.

        Resolution order:
        1. ``get_schedule_value`` dict with ``"X"`` / ``"Y"`` keys
           (time-varying piecewise-linear curves via the ruleset mechanism).
        2. ``self.default_x`` / ``self.default_y`` (constant curve stored as
           plain lists — always survives JSON config round-trips).

        ``device`` places the per-step table where the input port lives (this
        runs in the step loop, outside initialize()'s device context).
        """
        schedule_value = self.get_schedule_value(date_time)

        if (
            isinstance(schedule_value, dict)
            and "X" in schedule_value
            and "Y" in schedule_value
        ):
            return (
                torch.tensor(
                    schedule_value["X"], dtype=tps.float_dtype(), device=device
                ),
                torch.tensor(
                    schedule_value["Y"], dtype=tps.float_dtype(), device=device
                ),
            )

        if self.default_x is not None and self.default_y is not None:
            return (
                torch.tensor(
                    self.default_x, dtype=tps.float_dtype(), device=device
                ),
                torch.tensor(
                    self.default_y, dtype=tps.float_dtype(), device=device
                ),
            )

        raise TypeError(
            f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: "
            f"get_schedule_value returned {type(schedule_value).__name__} "
            f"({schedule_value!r}) instead of a dict with 'X'/'Y' keys, and "
            f"no defaultX/defaultY fallback is set."
        )

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        dt = date_time[0] if hasattr(date_time, "__len__") else date_time
        x_port = self.input["x"].get()
        X_points, Y_points = self._resolve_xy(dt, device=x_port.device)

        self._XY = torch.stack([X_points, Y_points]).T
        sorted_indices = torch.argsort(self._XY[:, 0])
        self._XY = self._XY[sorted_indices]
        self._X = self._XY[:, 0]
        self._Y = self._XY[:, 1]
        self._get_a_b_vectors()

        x_input = self.input["x"].get()
        original_shape = x_input.shape
        y_output = self._get_Y(x_input.reshape(-1))
        self.output["scheduleValue"]._set(
            y_output.reshape(original_shape), i_t=step_index
        )
