from .flow_meter import FlowMeter
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

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["airFlowRate"] = sum(self.input.values())
