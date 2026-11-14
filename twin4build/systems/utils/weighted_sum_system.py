# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Setting


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

    weights = Setting(None, doc="One weight per slot of inputs, in slot order; None sums unweighted")
    inputs = Input(size="wired", doc="Values to sum, one slot per connection")
    value = Output(doc="The weighted sum of the input values")

    def forward(self, x, inputs, p, step_size):
        """Stateless: the weighted sum over the slots of ``inputs``."""
        values = inputs["inputs"]
        if self.weights is None:
            return x, {"value": values.sum(dim=-1)}
        # One tensor, made where the values are (a one-slot weight list comes
        # back from a saved model as a scalar: reshaped).  Through Python
        # floats, element by element, a compiled step lifts a host constant
        # and copies it to the device at every step, which a step captured as
        # a CUDA graph cannot do.
        w = torch.as_tensor(self.weights, dtype=values.dtype, device=values.device).reshape(-1)
        if w.shape[0] != values.shape[-1]:
            raise ValueError(f"{self.id}: {w.shape[0]} weights for {values.shape[-1]} input slots")
        return x, {"value": (values * w).sum(dim=-1)}
