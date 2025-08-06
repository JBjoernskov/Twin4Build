# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class PassInputToOutput(core.System):
    """
    This component simply passes inputs to outputs during simulation.
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
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        pass

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        self.output["value"].set(self.input["value"], stepIndex)
