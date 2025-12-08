# Standard library imports
import datetime

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class ToDegCFromDegK(core.System):
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
        self.input = {"K": tps.Scalar()}
        self.output = {"C": tps.Scalar()}

    def initialize(
        self, start_time: datetime.datetime, end_time: datetime.datetime, step_size: int
    ):
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        self.input["K"].initialize(n_timesteps=max_timesteps, batch_size=batch_size)
        self.output["C"].initialize(n_timesteps=max_timesteps, batch_size=batch_size)

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ):
        self.output["C"].set(self.input["K"].get() - 273.15, step_index=step_index)
