from .controller import Controller

import sys
import os

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 9)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

logger.info("Controller Model Rule Based File")

class ControllerModelRulebased(Controller):
    def __init__(self, 
                **kwargs):
        super().__init__(**kwargs)

        logger.info("[Controller Model Rule Based] : Entered in Initialise Funtion")

    def update_output(self):
        if self.input["actualValue"]>900:
            self.output["inputSignal"] = 1
        
        if self.input["actualValue"]>866:
            self.output["inputSignal"] = 0.9

        if self.input["actualValue"]>833:
            self.output["inputSignal"] = 0.8

        if self.input["actualValue"]>800:
            self.output["inputSignal"] = 0.7

        if self.input["actualValue"]>766:
            self.output["inputSignal"] = 0.6

        if self.input["actualValue"]>733:
            self.output["inputSignal"] = 0.5

        if self.input["actualValue"]>700:
            self.output["inputSignal"] = 0.4

        elif self.input["actualValue"]>666:
            self.output["inputSignal"] = 0.3

        elif self.input["actualValue"]>633:
            self.output["inputSignal"] = 0.2

        elif self.input["actualValue"]>600:
            self.output["inputSignal"] = 0.1
        else:
            self.output["inputSignal"] = 0

        logger.info("[Controller Model Rule Based] : Eexited from Update Funtion")


