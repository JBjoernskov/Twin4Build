# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output


class ToDegCFromDegK(core.System):
    """
    System for converting temperature from Kelvin to Celsius.

    Mathematical Formulation:

    .. math::

       T_{C} = T_{K} - 273.15
    """

    K = Input(doc="Temperature [K]")
    C = Output(doc="Temperature [°C]")

    def forward(self, x, inputs, p, step_size):
        return x, {"C": inputs["K"] - 273.15}
