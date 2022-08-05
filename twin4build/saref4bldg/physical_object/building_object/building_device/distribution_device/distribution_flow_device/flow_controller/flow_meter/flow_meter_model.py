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
        if self.isSupplyFlowMeter:
            self.output["supplyAirFlowRate"] = sum(self.input.values())
        elif self.isReturnFlowMeter:
            self.output["returnAirFlowRate"] = sum(self.input.values())
        else:
            raise Exception("FlowMeter is neither defined as supply or return. Set either \"isSupplyFlowMeter\" or \"isReturnFlowMeter\" to True")