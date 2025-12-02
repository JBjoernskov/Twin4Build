# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps 
import datetime


class ToDegKFromDegC(core.System):
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
        self.input = {"C": tps.Scalar()}
        self.output = {"K": tps.Scalar()}

    def initialize(self, start_time: datetime.datetime, end_time: datetime.datetime, step_size: int):
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        self.input["C"].initialize(n_timesteps=max_timesteps, batch_size=batch_size)
        self.output["K"].initialize(n_timesteps=max_timesteps, batch_size=batch_size)

    def do_step(self, second_time: float, date_time: datetime.datetime, step_size: int, step_index: int):
        self.output["K"].set(self.input["C"].get() + 273.15, step_index=step_index)
