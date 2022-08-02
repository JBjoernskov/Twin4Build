from saref4syst.system import System
class FlowMeterModel(System):
    def __init__(self,
                isSupplyFlowMeter = None,
                isReturnFlowMeter = None):
        self.isSupplyFlowMeter = isSupplyFlowMeter
        self.isReturnFlowMeter = isReturnFlowMeter

    def update_output(self):
        if self.isSupplyFlowMeter:
            self.output["supplyAirFlowRate"] = sum(self.input.values())
        elif self.isReturnFlowMeter:
            self.output["returnAirFlowRate"] = sum(self.input.values())
        else:
            raise Exception("FlowMeter is neither defined as supply or return. Set either \"isSupplyFlowMeter\" or \"isReturnFlowMeter\" to True")