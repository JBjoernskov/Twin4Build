# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class RulebasedSetpointInputControllerSystem(core.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self.interval = 99
        self._config = {"parameters": ["interval"]}

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
        self.hold_high_signal = False
        self.hold_mid_signal = False
        self.hold_low_signal = False

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:

        setpoint = self.input["setpointValue"]
        high_threshold = setpoint + 400
        mid_threshold = setpoint + 200
        low_threshold = setpoint

        if self.input["actualValue"] > high_threshold or self.hold_high_signal:
            self.output["inputSignal"].set(1, stepIndex)
            if self.input["actualValue"] > high_threshold - self.interval:
                self.hold_high_signal = True
            else:
                self.hold_high_signal = False

        elif self.input["actualValue"] > mid_threshold or self.hold_mid_signal:
            self.output["inputSignal"].set(0.7, stepIndex)
            if self.input["actualValue"] > mid_threshold - self.interval:
                self.hold_mid_signal = True
            else:
                self.hold_mid_signal = False

        elif self.input["actualValue"] > low_threshold or self.hold_low_signal:
            self.output["inputSignal"].set(0.45, stepIndex)
            if self.input["actualValue"] > low_threshold - self.interval:
                self.hold_low_signal = True
            else:
                self.hold_low_signal = False

        else:
            self.output["inputSignal"].set(0, stepIndex)
