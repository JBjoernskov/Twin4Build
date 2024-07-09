from twin4build.saref4syst.system import System
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

class PassInputToOutput(System):
    """
    This component simply passes inputs to outputs during simulation.
    """
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        logger.info("[Pass Input To Output] : Entered in Initialise Function")
        self._config = {"parameters": []}

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
        pass
        
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output = self.input
        # self.outputUncertainty = self.inputUncertainty
        