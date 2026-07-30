# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class PassInputToOutput(core.System):
    r"""
    Pass Input to Output System.

    This component simply passes inputs to outputs during simulation.

    Args:
        **kwargs: Additional keyword arguments
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"value": tps.Scalar()}
        self.output = {"value": tps.Scalar()}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

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

    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step pass-through (functorch-safe, stateless)."""
        return x, {"value": inputs["value"]}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        inputs = {"value": self.input["value"].get()}
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["value"]._set(outs["value"], i_t=step_index)
