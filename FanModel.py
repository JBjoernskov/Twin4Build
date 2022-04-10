
class FanModel():
    def __init__(self):
        pass

    def update_output(self):
        if self.isSupplyFan:
            f_flow = self.input["supplyAirFlowRate"]/self.nominalAirFlowRate
            f_pl = self.c1 + self.c2*f_flow + self.c3*f_flow**2 + self.c4*f_flow**3
            W_fan = f_pl*self.nominalPowerRate
            self.output["Power"] = W_fan
        elif self.isReturnFan:
            f_flow = self.input["returnAirFlowRate"]/self.nominalAirFlowRate
            f_pl = self.c1 + self.c2*f_flow + self.c3*f_flow**2 + self.c4*f_flow**3
            W_fan = f_pl*self.nominalPowerRate

            if self.save:
                self.savedOutput["Power"].append(W_fan)
            self.output["Power"] = W_fan
            
        else:
            raise Exception("Fan is neither defined as supply or return. Set either \"isSupplyFan\" or \"isReturnFan\" to True")