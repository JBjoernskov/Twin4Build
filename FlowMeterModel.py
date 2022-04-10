
class FlowMeterModel():
    def __init__(self):
        pass

    def update_output(self):
        if self.isSupplyFlowMeter:
            self.output["supplyAirFlowRate"] = sum(self.input.values())
        elif self.isReturnFlowMeter:
            self.output["returnAirFlowRate"] = sum(self.input.values())
        else:
            raise Exception("FlowMeter is neither defined as supply or return. Set either \"isSupplyFlowMeter\" or \"isReturnFlowMeter\" to True")