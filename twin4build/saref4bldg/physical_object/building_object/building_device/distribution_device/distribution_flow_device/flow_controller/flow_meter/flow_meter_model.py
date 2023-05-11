from .flow_meter import FlowMeter

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")


logger.info("[Flow Meter Model]")

class FlowMeterModel(FlowMeter):
    """
    Not in use at the moment
    """
    def __init__(self,
                isSupplyFlowMeter=None,
                isReturnFlowMeter=None,
                **kwargs):
        super().__init__(**kwargs)
        self.isSupplyFlowMeter = isSupplyFlowMeter
        self.isReturnFlowMeter = isReturnFlowMeter

        logger.info("Entered in Initialise Function")

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["airFlowRate"] = sum(self.input.values())
