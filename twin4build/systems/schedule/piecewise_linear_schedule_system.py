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

    A ``weekDayRulesetDict`` is built automatically.  Time-varying curves are
    still possible by providing the standard ruleset dicts with
    ``{"X": [...], "Y": [...]}`` values.

    Args:
        defaultX: Default X breakpoints (list of floats).
        defaultY: Default Y breakpoints (list of floats).
        **kwargs: Keyword arguments passed to parent classes including
            weekDayRulesetDict, weekendRulesetDict, per-day rulesets,
            and add_noise.
    """

    # Override inherited ScheduleSystem.sp so the translator does NOT
    # auto-match this class from semantic models.
    sp = None

    def __init__(
        self,
        defaultX: Optional[List[float]] = None,
        defaultY: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        # Auto-create a scalar weekDayRulesetDict when only defaultX/Y given.
        if defaultX is not None and defaultY is not None:
            if "weekDayRulesetDict" not in kwargs or kwargs["weekDayRulesetDict"] is None:
                kwargs["weekDayRulesetDict"] = copy.deepcopy(_EMPTY_RULESET)

        super().__init__(**kwargs)

        self.defaultX = defaultX
        self.defaultY = defaultY

        self._input = {"x": tps.Scalar()}
        self._output = {"scheduleValue": tps.Scalar()}
        self._config = {
            "parameters": [
                "defaultX",
                "defaultY",
                "weekDayRulesetDict",
                "weekendRulesetDict",
                "mondayRulesetDict",
                "tuesdayRulesetDict",
                "wednesdayRulesetDict",
                "thursdayRulesetDict",
                "fridayRulesetDict",
                "saturdayRulesetDict",
                "sundayRulesetDict",
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
            (self.defaultX is not None and self.defaultY is not None)
            or self.weekDayRulesetDict is not None
            or self.weekendRulesetDict is not None
        ), (
            f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: "
            "Provide defaultX/defaultY or a weekDayRulesetDict/weekendRulesetDict."
        )

        # Inherit missing day rulesets from weekday / weekend defaults
        if self.mondayRulesetDict is None:
            self.mondayRulesetDict = self.weekDayRulesetDict
        if self.tuesdayRulesetDict is None:
            self.tuesdayRulesetDict = self.weekDayRulesetDict
        if self.wednesdayRulesetDict is None:
            self.wednesdayRulesetDict = self.weekDayRulesetDict
        if self.thursdayRulesetDict is None:
            self.thursdayRulesetDict = self.weekDayRulesetDict
        if self.fridayRulesetDict is None:
            self.fridayRulesetDict = self.weekDayRulesetDict
        if self.saturdayRulesetDict is None:
            self.saturdayRulesetDict = (
                self.weekendRulesetDict
                if self.weekendRulesetDict is not None
                else self.weekDayRulesetDict
            )
        if self.sundayRulesetDict is None:
            self.sundayRulesetDict = (
                self.weekendRulesetDict
                if self.weekendRulesetDict is not None
                else self.weekDayRulesetDict
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
            self.mondayRulesetDict,
            self.tuesdayRulesetDict,
            self.wednesdayRulesetDict,
            self.thursdayRulesetDict,
            self.fridayRulesetDict,
            self.saturdayRulesetDict,
            self.sundayRulesetDict,
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

    def _resolve_xy(self, date_time: datetime.datetime):
        """Return (X_points, Y_points) tensors for the given datetime.

        Resolution order:
        1. ``get_schedule_value`` dict with ``"X"`` / ``"Y"`` keys
           (time-varying piecewise-linear curves via the ruleset mechanism).
        2. ``self.defaultX`` / ``self.defaultY`` (constant curve stored as
           plain lists — always survives JSON config round-trips).
        """
        schedule_value = self.get_schedule_value(date_time)

        if isinstance(schedule_value, dict) and "X" in schedule_value and "Y" in schedule_value:
            return (
                torch.tensor(schedule_value["X"], dtype=torch.float64),
                torch.tensor(schedule_value["Y"], dtype=torch.float64),
            )

        if self.defaultX is not None and self.defaultY is not None:
            return (
                torch.tensor(self.defaultX, dtype=torch.float64),
                torch.tensor(self.defaultY, dtype=torch.float64),
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
        X_points, Y_points = self._resolve_xy(dt)

        self._XY = torch.stack([X_points, Y_points]).T
        sorted_indices = torch.argsort(self._XY[:, 0])
        self._XY = self._XY[sorted_indices]
        self._X = self._XY[:, 0]
        self._Y = self._XY[:, 1]
        self._get_a_b_vectors()

        x_input = self.input["x"].get()
        original_shape = x_input.shape
        y_output = self._get_Y(x_input.reshape(-1))
        self.output["scheduleValue"]._set(y_output.reshape(original_shape), i_t=step_index)
