from twin4build.saref4syst.system import System
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

class PassInputToOutput(System):
    """
    This component models a generic dynamic input based on prescribed time series data. 
    It extracts and samples the second column of a csv file given by "filename".
    """
    def __init__(self,
                filename=None,
                **kwargs):
        super().__init__(**kwargs)
        logger.info("[Pass Input To Output] : Entered in Initialise Function")

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass
        
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output = self.input
        # self.outputUncertainty = self.inputUncertainty
        