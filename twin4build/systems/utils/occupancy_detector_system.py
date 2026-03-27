# Standard library imports
import datetime
from typing import List

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.smooth_saturation import smooth_saturation


class OccupancyDetectorSystem(core.System, nn.Module):
    r"""Smooth binary occupancy detector (mimics a PIR sensor).

    Converts a continuous occupancy estimate :math:`N_{occ}` into a smooth
    0--1 signal using a sigmoid:

    .. math::

        \sigma = \frac{1}{1 + e^{-k\,(N_{occ} - T)}}

    where :math:`T` is the ``threshold`` and :math:`k` is the ``steepness``.

    Args:
        threshold: Occupancy level that triggers "occupied" (estimable).
        steepness: Controls sigmoid sharpness (higher = sharper).
        **kwargs: Forwarded to ``core.System`` (must include ``id``).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        steepness: float = 100.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.threshold = tps.Parameter(
            torch.tensor(threshold, dtype=torch.float64), requires_grad=False
        )
        self.steepness = tps.Parameter(
            torch.tensor(steepness, dtype=torch.float64), requires_grad=False
        )

        self._input = {"occupancy": tps.Scalar()}
        self._output = {"occupancySignal": tps.Scalar()}
        self._config = {"parameters": ["threshold", "steepness"]}
        self.INITIALIZED = False

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for inp in self.input.values():
            inp.initialize(n_t=max_timesteps, n_s=batch_size)
        for out in self.output.values():
            out.initialize(n_t=max_timesteps, n_s=batch_size)

        self.threshold = self.threshold.expand_to_n_c(self.n_c)
        self.steepness = self.steepness.expand_to_n_c(self.n_c)
        self.INITIALIZED = True

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        occupancy = self.input["occupancy"].get()
        error = occupancy - self.threshold.get()
        u = 0.5 + error * self.steepness.get()
        signal = smooth_saturation(u, lower=0.0, upper=1.0, curve_start=0.1)
        self.output["occupancySignal"]._set(signal, i_t=step_index)
