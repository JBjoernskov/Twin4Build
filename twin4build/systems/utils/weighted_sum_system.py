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

    def _weight_list(self):
        # A one-slot weight list comes back from a saved model as a scalar.
        return [float(w) for w in torch.as_tensor(self.weights, dtype=torch.float64).reshape(-1)]

    def forward(self, x, inputs, p, step_size):
        """Stateless: the weighted sum over the slots of ``inputs``."""
        values = inputs["inputs"]
        if self.weights is None:
            return x, {"value": values.sum(dim=-1)}
        weights = self._weight_list()
        if len(weights) != values.shape[-1]:
            raise ValueError(f"{self.id}: {len(weights)} weights for {values.shape[-1]} input slots")
        w = torch.tensor(weights, dtype=values.dtype, device=values.device)
        return x, {"value": (values * w).sum(dim=-1)}
