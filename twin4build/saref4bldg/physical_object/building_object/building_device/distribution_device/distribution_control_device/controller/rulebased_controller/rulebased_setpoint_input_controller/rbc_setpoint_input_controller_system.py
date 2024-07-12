from twin4build.base import RulebasedController
import sys
import os

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 9)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")
logger.info("Controller Model Rule Based File")

class RulebasedSetpointInputControllerSystem(RulebasedController):
    def __init__(self, 
                **kwargs):
        super().__init__(**kwargs)
        logger.info("[Controller Model Rule Based] : Entered in Initialise Funtion")
        self.input = {"actualValue": None,
                    "setpointValue": None}
        self.output = {"inputSignal": None}
        self.interval = 99
        self._config = {"parameters": ["interval"]}

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
                    model=None):
        self.hold_high_signal = False
        self.hold_mid_signal = False
        self.hold_low_signal = False

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):

        setpoint = self.input["setpointValue"]
        high_threshold = setpoint + 400
        mid_threshold = setpoint + 200
        low_threshold = setpoint

        if self.input["actualValue"] > high_threshold or self.hold_high_signal:
            self.output["inputSignal"] = 1
            if self.input["actualValue"] > high_threshold - self.interval:
                self.hold_high_signal = True
            else:
                self.hold_high_signal = False
        
        elif self.input["actualValue"] > mid_threshold or self.hold_mid_signal:
            self.output["inputSignal"] = 0.7
            if self.input["actualValue"] > mid_threshold - self.interval:
                self.hold_mid_signal = True
            else:
                self.hold_mid_signal = False

        elif self.input["actualValue"] > low_threshold or self.hold_low_signal:
            self.output["inputSignal"] = 0.45
            if self.input["actualValue"] > low_threshold - self.interval:
                self.hold_low_signal = True
            else:
                self.hold_low_signal = False

        else:
            self.output["inputSignal"] = 0



