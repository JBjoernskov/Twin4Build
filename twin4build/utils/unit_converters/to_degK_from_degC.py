# Local application imports
from twin4build.systems.saref4syst.system import System


class ToDegKFromDegC(System):
    """
    System for converting temperature from Celsius to Kelvin.

    Mathematical Formulation:

    .. math::

       T_{K} = T_{C} + 273.15

    where:
    - :math:`T_{K}` is temperature in Kelvin
    - :math:`T_{C}` is temperature in Celsius
    """

    def __init__(self):
        super().__init__()
        self.input = {"C": None}
        self.output = {"K": None}

    def initialize(self, start_time=None, end_time=None, step_size=None, model=None):
        pass

    def do_step(self, second_time=None, date_time=None, step_size=None):
        self.output["K"].set(self.input["C"] + 273.15)
