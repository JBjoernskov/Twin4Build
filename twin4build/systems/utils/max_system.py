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

        n_v = self.get_n_v_from_connections("inputs")
        self.input["inputs"].initialize(n_t=max_timesteps, n_s=batch_size, n_v=n_v)

        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=batch_size)

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        inputs = self.input["inputs"].get()
        k = 50.0
        smooth_max = torch.logsumexp(k * inputs, dim=-1) / k
        self.output["value"]._set(smooth_max, step_index)

    #: No physical parameters (the ``forward`` theta contract).
    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):
        """Pure algebraic map ``(inputs,) -> outputs`` (stateless).

        Functorch-compatible re-expression of :meth:`do_step`.  ``inputs``
        provides ``inputs`` with the vector of values along the last dim.
        Returns ``(x, {"value"})``.
        """
        vals = inputs["inputs"]
        k = 50.0
        smooth_max = torch.logsumexp(k * vals, dim=-1) / k
        return x, {"value": smooth_max.reshape(1)}
