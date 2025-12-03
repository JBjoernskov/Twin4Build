# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class OnOffSystem(core.System):
    r"""
    On-Off System.

    If value>=threshold set to on_value else set to off_value

    Args:
        threshold: Threshold value
        is_on_value: Value to set when value>=threshold
        is_off_value: Value to set when value<threshold
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        threshold: float = None,
        is_on_value: float = None,
        is_off_value: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.is_off_value = is_off_value

        self.input = {"value": tps.Scalar(), "criteriaValue": tps.Scalar()}
        self.output = {"value": tps.Scalar()}

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
        simulator: core.Simulator,
    ) -> None:
        pass

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        if self.input["criteriaValue"] >= self.threshold:
            self.output["value"].set(self.input["value"], step_index)
        else:
            self.output["value"].set(self.is_off_value, step_index)
