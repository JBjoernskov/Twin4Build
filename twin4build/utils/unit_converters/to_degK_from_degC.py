# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output


class ToDegKFromDegC(core.System):
    """
    System for converting temperature from Celsius to Kelvin.

    Mathematical Formulation:

    .. math::

       T_{K} = T_{C} + 273.15
    """

    C = Input(doc="Temperature [°C]")
    K = Output(doc="Temperature [K]")

    def forward(self, x, inputs, p, step_size):
        return x, {"K": inputs["C"] + 273.15}
