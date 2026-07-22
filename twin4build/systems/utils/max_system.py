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

    Reduces a vector input to a single scalar output by taking a smooth
    (differentiable) maximum over its elements, computed as
    :math:`\mathrm{logsumexp}(k \cdot x) / k` with sharpness :math:`k = 50`.
    Useful e.g. for combining several control signals into one, while keeping
    the operation differentiable for gradient-based optimization.

    Inputs:
        - "inputs": Vector of values to take the maximum over.

    Outputs:
        - "value": Smooth maximum of the input values.

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
        """Thin port-I/O wrapper around :meth:`forward` (the single source of
        truth for the math)."""
        _, outs = self.forward(
            None, {"inputs": self.input["inputs"].get()}, {}, step_size
        )
        self.output["value"]._set(outs["value"], step_index)

    #: No physical parameters (the ``forward`` theta contract).
    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):
        """Pure algebraic map ``(inputs,) -> outputs`` (stateless).

        ``inputs`` provides ``inputs`` with the vector of values along the
        last dim; the smooth max reduces that dim, so the result keeps
        whatever batch dims the input carried (``(1,)`` under the composer,
        ``(n_s, n_c)`` under ``do_step``).  Returns ``(x, {"value"})``.
        """
        vals = inputs["inputs"]
        k = 50.0
        smooth_max = torch.logsumexp(k * vals, dim=-1) / k
        return x, {"value": smooth_max}
