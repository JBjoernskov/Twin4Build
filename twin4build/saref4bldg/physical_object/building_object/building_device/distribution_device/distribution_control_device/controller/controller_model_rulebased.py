from .controller import Controller

import sys
import os
import numpy as np
uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 9)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

logger.info("Controller Model Rule Based File")

class ControllerModelRuleBased(Controller):
    def __init__(self, 
                **kwargs):
        super().__init__(**kwargs)
        logger.info("[Controller Model Rule Based] : Entered in Initialise Funtion")
        self.input = {"actualValue": None}
        self.output = {"inputSignal": None}
        self.interval = 100
        

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        self.hold_900_signal = False
        self.hold_750_signal = False
        self.hold_600_signal = False

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        if self.input["actualValue"]>900 or self.hold_900_signal:
            self.output["inputSignal"] = 1
            if self.input["actualValue"]>900-self.interval:
                self.hold_900_signal = True
            else:
                self.hold_900_signal = False
        
        elif self.input["actualValue"]>750 or self.hold_750_signal:
            self.output["inputSignal"] = 0.7
            if self.input["actualValue"]>750-self.interval:
                self.hold_750_signal = True
            else:
                self.hold_750_signal = False

        elif self.input["actualValue"]>600 or self.hold_600_signal:
            self.output["inputSignal"] = 0.45
            if self.input["actualValue"]>600-self.interval:
                self.hold_600_signal = True
            else:
                self.hold_600_signal = False

        else:
            self.holdUntilValue = np.inf
            self.output["inputSignal"] = 0

        logger.info("[Controller Model Rule Based] : Exited from Update Funtion")


