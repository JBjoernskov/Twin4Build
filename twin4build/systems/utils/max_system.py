# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class MaxSystem(core.System):
    """
    If value>=threshold set to on_value else set to off_value
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input = {"inputs": tps.Vector()}
        self.output = {"value": tps.Scalar()}

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
        self.output["value"].set(torch.max(self.input["inputs"].get()), stepIndex)
