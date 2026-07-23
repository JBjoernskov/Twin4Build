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

    Gates a signal based on a criteria input: if the "criteriaValue" input is
    greater than or equal to ``threshold``, the "value" input is passed through
    to the output; otherwise the output is set to ``is_off_value``.

    Args:
        threshold: Threshold that the "criteriaValue" input is compared against.
        is_on_value: Currently unused. The argument is accepted but never
            stored; when the criteria is met, the "value" input is passed
            through instead.
        is_off_value: Output value when criteriaValue < threshold.
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

    PARAM_NAMES = ()  # threshold / is_off_value are plain numbers (structural)

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step gate (functorch-safe, stateless).

        ``threshold`` and ``is_off_value`` are plain (non-estimable) numbers,
        read from ``self`` as structural constants.
        """
        criteria_value = inputs["criteriaValue"]
        input_value = inputs["value"]

        # Vectorized conditional: where criteria >= threshold, use input_value, else use is_off_value
        output_value = torch.where(
            criteria_value >= self.threshold,
            input_value,
            torch.full_like(input_value, self.is_off_value),
        )
        return x, {"value": output_value}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        inputs = {
            "value": self.input["value"].get(),
            "criteriaValue": self.input["criteriaValue"].get(),
        }
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["value"]._set(outs["value"], i_t=step_index)
