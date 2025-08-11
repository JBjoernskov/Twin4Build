# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class MaxSystem(core.System):
    r"""
    Max System.

    This class implements a max system for a given system.

    Args:
        **kwargs: Additional keyword arguments
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input = {"inputs": tps.Vector()}
        self.output = {"value": tps.Scalar()}

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
        self.output["value"].set(torch.max(self.input["inputs"].get()), stepIndex)
