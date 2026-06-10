# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class ScalarProductSystem(core.System):
    r"""Multiplies two scalar inputs element-wise with an optional scaling factor.

    Computes ``output = input_1 * input_2 * scale_factor`` at each time step.
    Useful for combining a physical quantity (e.g. power in W) with a weighting
    signal (e.g. electricity price in DKK/kWh) and a unit-conversion constant
    (e.g. ``step_size / 3600 / 1000`` to go from W to kWh).

    Args:
        scale_factor: Constant multiplier applied after the element-wise
            product of the two inputs. Defaults to 1.0.
        **kwargs: Additional keyword arguments passed to the parent
            :class:`core.System`.

    Inputs:
        input_1 (Scalar): First operand.
        input_2 (Scalar): Second operand.

    Outputs:
        output (Scalar): Result of ``input_1 * input_2 * scale_factor``.
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.input = {
            "input_1": tps.Scalar(),
            "input_2": tps.Scalar(),
        }
        self.output = {
            "output": tps.Scalar(),
        }

        self._scale_factor = scale_factor

        self._config = {"parameters": ["scale_factor"]}

    @property
    def config(self):
        return self._config

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float) -> None:
        self._scale_factor = value

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
        for inp in self.input.values():
            inp.initialize(n_t=max_timesteps, n_s=batch_size)
        for out in self.output.values():
            out.initialize(n_t=max_timesteps, n_s=batch_size)

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        v1 = self.input["input_1"].get()
        v2 = self.input["input_2"].get()
        self.output["output"]._set(v1 * v2 * self._scale_factor, i_t=step_index)
