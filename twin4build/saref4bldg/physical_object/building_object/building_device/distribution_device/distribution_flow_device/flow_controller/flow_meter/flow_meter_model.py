from .flow_meter import FlowMeter
class FlowMeterModel(FlowMeter):
    def __init__(self,
                isSupplyFlowMeter = None,
                isReturnFlowMeter = None,
                **kwargs):
        super().__init__(**kwargs)
        self.isSupplyFlowMeter = isSupplyFlowMeter
        self.isReturnFlowMeter = isReturnFlowMeter

    def update_output(self):
        self.output["airFlowRate"] = sum(self.input.values())
