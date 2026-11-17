# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output


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

    inputs = Input(size="wired", doc="Values to take the maximum over, one slot per connection")
    value = Output(doc="Smooth maximum of the input values")

    def forward(self, x, inputs, p, step_size):
        """Stateless: ``logsumexp(k x) / k`` over the slots, ``k = 50``."""
        k = 50.0
        return x, {"value": torch.logsumexp(k * inputs["inputs"], dim=-1) / k}

