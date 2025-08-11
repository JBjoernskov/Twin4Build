# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class PassInputToOutput(core.System):
    r"""
    Pass Input to Output System.

    This component simply passes inputs to outputs during simulation.

    Args:
        **kwargs: Additional keyword arguments
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"value": tps.Scalar()}
        self.output = {"value": tps.Scalar()}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
        simulator: core.Simulator,
    ) -> None:
        pass

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        step_size: int,
        stepIndex: int,
    ) -> None:
        self.output["value"].set(self.input["value"], stepIndex)
