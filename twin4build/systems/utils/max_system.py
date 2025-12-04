# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class MaxSystem(core.System):
    r"""
    Max System.

    This class implements a max system for a given system.

    Args:
        **kwargs: Additional keyword arguments
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input = {"inputs": tps.Vector()}
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
            input.initialize(n_timesteps=max_timesteps, batch_size=batch_size)
        for output in self.output.values():
            output.initialize(n_timesteps=max_timesteps, batch_size=batch_size)

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        self.output["value"].set(torch.max(self.input["inputs"].get(), dim=-1).values, step_index)
