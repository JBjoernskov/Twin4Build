
import math

class DamperModel():
    def __init__(self,
                isSupplyDamper = None,
                isReturnDamper = None):
        
        self.isSupplyDamper = isSupplyDamper
        self.isReturnDamper = isReturnDamper

        if self.isSupplyDamper:
            self.DamperSignalName = "supplyDamperSignal"
            self.AirFlowRateName = "supplyAirFlowRate" + str(self.systemId)
        elif self.isReturnDamper:
            self.DamperSignalName = "returnDamperSignal"
            self.AirFlowRateName = "returnAirFlowRate" + str(self.systemId)
        else:
            raise Exception("Damper is neither defined as supply or return. Set either \"isSupplyDamper\" or \"isReturnDamper\" to True")

    def update_output(self):
        m_a = self.a*math.exp(self.b*self.input[self.DamperSignalName]) + self.c
        self.output[self.AirFlowRateName] = m_a


    