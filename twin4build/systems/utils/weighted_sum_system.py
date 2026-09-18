# Standard library imports
import datetime
from typing import List, Optional, Sequence, Union

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class WeightedSumSystem(core.System):
    r"""
    Weighted Sum System.

    Reduces a vector input to one scalar output, the weighted sum of its
    elements:

    .. math::

        y = \sum_i w_i \, x_i

    Without weights every element counts once (a plain sum).  Useful for
    totals over many components (the heating power of every radiator in a
    building) and for weighted means (an area-weighted overheating over
    every zone, with the weights the zones' floor areas over the total).
    Differentiable, so a weighted sum can be an objective of the Optimizer.

    Inputs:
        - "inputs": Vector of values to sum, one slot per connection.

    Outputs:
        - "value": The weighted sum of the input values.

    Args:
        weights: One weight per slot of ``inputs``, in slot order.  ``None``
            (the default) sums the elements unweighted.
        **kwargs: Additional keyword arguments
    """

    def __init__(self, weights: Optional[Union[Sequence[float], float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.input = {"inputs": tps.Vector()}
        self.output = {"value": tps.Scalar()}
        self.weights = weights
        self._config = {"parameters": ["weights"]}

    @property
    def config(self):
        return self._config

    @property
    def weights(self) -> Optional[List[float]]:
        """The weights in slot order, or ``None`` for a plain sum."""
        return self._weights

    @weights.setter
    def weights(self, value: Optional[Union[Sequence[float], float]]) -> None:
        if value is None:
            self._weights = None
        else:
            # A one-slot weight list comes back from a saved model as a
            # scalar literal.
            value = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
            self._weights = [float(v) for v in value]
        self._w = None

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
        self.n_c = int(getattr(self, "_n_c_batched", 1))
        indices = [
            int(cp.input_port_index[conn])
            for cp in self.connects_at
            if cp.input_port == "inputs"
            for conn in cp.connects_system_through
        ]
        n_v = max(indices, default=-1) + 1
        if n_v == 0 and self.input["inputs"].n_v:
            # Preserve an explicitly configured standalone vector. Compiled
            # graph instances infer the width from their connections above.
            n_v = self.input["inputs"].n_v
        if self._weights is not None and len(self._weights) != n_v:
            raise ValueError(
                f"{self.id}: {len(self._weights)} weights for {n_v} input slots"
            )
        self._w = (
            None
            if self._weights is None
            else torch.tensor(self._weights, dtype=tps.float_dtype())
        )
        self.input["inputs"].initialize(
            n_t=max_timesteps, n_s=batch_size, n_c=self.n_c, n_v=n_v
        )
        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=batch_size, n_c=self.n_c)

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

    #: No physical parameters (the ``forward`` theta contract); the weights
    #: are fixed constants read from the instance.
    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):
        """Pure algebraic map ``(inputs,) -> outputs`` (stateless).

        ``inputs`` provides ``inputs`` with the vector of values along the
        last dim; the weighted sum reduces that dim, so the result keeps
        whatever batch dims the input carried (``(1,)`` under the composer,
        ``(n_s, n_c)`` under ``do_step``).  Returns ``(x, {"value"})``.
        """
        vals = inputs["inputs"]
        if self._w is None:
            return x, {"value": vals.sum(dim=-1)}
        w = self._w.to(device=vals.device, dtype=vals.dtype)
        return x, {"value": (vals * w).sum(dim=-1)}
