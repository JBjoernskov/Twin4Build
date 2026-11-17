# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output


class PassInputToOutput(core.System):
    r"""
    Pass Input to Output System.

    This component simply passes inputs to outputs during simulation.

    Args:
        **kwargs: Additional keyword arguments
    """

    value = Input(doc="The value")
    value = Output(doc="The same value")

    def forward(self, x, inputs, p, step_size):
        """Stateless: the input as it is."""
        return x, {"value": inputs["value"]}

