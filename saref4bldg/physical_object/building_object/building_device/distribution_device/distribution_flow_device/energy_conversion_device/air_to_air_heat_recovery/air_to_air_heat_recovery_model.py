from numpy import NaN
from air_to_air_heat_recovery import AirToAirHeatRecovery
class AirToAirHeatRecoveryModel(AirToAirHeatRecovery):
    def __init__(self,
                specificHeatCapacityAir = None,
                **kwargs):
        super().__init__(**kwargs)
        self.specificHeatCapacityAir = specificHeatCapacityAir 
        self.eps_75_h = 0.8
        self.eps_75_c = 0.8 
        self.eps_100_h = 0.8 
        self.eps_100_c = 0.8
        

    def update_output(self):
        m_a_max = max(self.primaryAirFlowRateMax, self.secondaryAirFlowRateMax)
        if self.input["outdoorTemperature"] < self.input["indoorTemperature"]:
            eps_75 = self.eps_75_h
            eps_100 = self.eps_100_h
        else:
            eps_75 = self.eps_75_c
            eps_100 = self.eps_100_c
        f_flow = 0.5*(self.input["supplyAirFlowRate"] + self.input["returnAirFlowRate"])/m_a_max
        eps_op = eps_75 + (eps_100-eps_75)*(f_flow-0.75)/(1-0.75)
        C_sup = self.input["supplyAirFlowRate"]*self.specificHeatCapacityAir
        C_exh = self.input["returnAirFlowRate"]*self.specificHeatCapacityAir
        C_min = min(C_sup, C_exh)
        if C_sup == 0:
            T_a_sup_out = NaN
        else:
            T_a_sup_out = self.input["outdoorTemperature"] + eps_op*(self.input["indoorTemperature"] - self.input["outdoorTemperature"])*(C_min/C_sup)
        self.output["supplyAirTemperature"] = T_a_sup_out
