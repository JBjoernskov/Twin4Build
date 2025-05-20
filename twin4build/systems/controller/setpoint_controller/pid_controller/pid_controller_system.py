from scipy.optimize import least_squares
import numpy as np
import twin4build.utils.input_output_types as tps
import twin4build.core as core
import datetime
from typing import Optional

class PIDControllerSystem(core.System):
    def __init__(self, 
                # isTemperatureController=None,
                # isCo2Controller=None,
                kp=None,
                ki=None,
                kd=None,
                **kwargs):
        super().__init__(**kwargs)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.input = {"actualValue": tps.Scalar(),
                    "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {"parameters": ["kp",
                                       "ki",
                                       "kd"]}

    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    simulator=None):
        self.acc_err = 0
        self.prev_err = 0

    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        err = self.input["setpointValue"]-self.input["actualValue"]
        p = err*self.kp
        i = self.acc_err*self.ki
        d = (err-self.prev_err)*self.kd
        signal_value = p + i + d
        if signal_value>1:
            signal_value = 1
            self.acc_err = 1/self.ki
            self.prev_err = 0
        elif signal_value<0:
            signal_value = 0
            self.acc_err = 0
            self.prev_err = 0
        else:
            self.acc_err += err
            self.prev_err = err

        self.output["inputSignal"].set(signal_value, stepIndex)
