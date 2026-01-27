# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch

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
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for input in self.input.values():
            input.initialize(n_t=max_timesteps, n_s=batch_size)
        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=batch_size)

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        criteria_value = self.input["criteriaValue"].get()
        input_value = self.input["value"].get()

        # Vectorized conditional: where criteria >= threshold, use input_value, else use is_off_value
        output_value = torch.where(
            criteria_value >= self.threshold,
            input_value,
            torch.full_like(input_value, self.is_off_value),
        )
        self.output["value"]._set(output_value, i_t=step_index)
