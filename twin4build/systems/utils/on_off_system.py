# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class OnOffSystem(core.System):
    """
    If value>=threshold set to on_value else set to off_value
    """

    def __init__(self, threshold=None, is_on_value=None, is_off_value=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.is_off_value = is_off_value

        self.input = {"value": tps.Scalar(), "criteriaValue": tps.Scalar()}
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
        if self.input["criteriaValue"] >= self.threshold:
            self.output["value"].set(self.input["value"], stepIndex)
        else:
            self.output["value"].set(self.is_off_value, stepIndex)
