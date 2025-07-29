# Local application imports
from twin4build.systems.saref4syst.system import System


class ToDegCFromDegK(System):
    """
    System for converting temperature from Kelvin to Celsius.

    Mathematical Formulation:

    .. math::

       T_{C} = T_{K} - 273.15

    where:
    - :math:`T_{C}` is temperature in Celsius
    - :math:`T_{K}` is temperature in Kelvin
    """

    def __init__(self):
        super().__init__()
        self.input = {"K": None}
        self.output = {"C": None}

    def initialize(self, startTime=None, endTime=None, stepSize=None, model=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["C"].set(self.input["K"] - 273.15)
