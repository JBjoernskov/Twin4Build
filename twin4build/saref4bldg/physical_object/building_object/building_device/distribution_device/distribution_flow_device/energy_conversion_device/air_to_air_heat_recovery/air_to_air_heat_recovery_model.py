from numpy import NaN
from typing import Union
from .air_to_air_heat_recovery import AirToAirHeatRecovery
import twin4build.saref.measurement.measurement as measurement

class AirToAirHeatRecoveryModel(AirToAirHeatRecovery):
    def __init__(self,
                specificHeatCapacityAir: Union[measurement.Measurement, None] = None,
                eps_75_h: Union[measurement.Measurement, None] = None,
                eps_75_c: Union[measurement.Measurement, None] = None,
                eps_100_h: Union[measurement.Measurement, None] = None,
                eps_100_c: Union[measurement.Measurement, None] = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(specificHeatCapacityAir, measurement.Measurement) or specificHeatCapacityAir is None, "Attribute \"specificHeatCapacityAir\" is of type \"" + str(type(specificHeatCapacityAir))+ "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(eps_75_h, measurement.Measurement) or eps_75_h is None, "Attribute \"eps_75_h\" is of type \"" + str(type(eps_75_h)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(eps_75_c, measurement.Measurement) or eps_75_c is None, "Attribute \"eps_75_c\" is of type \"" + str(type(eps_75_c)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(eps_100_h, measurement.Measurement) or eps_100_h is None, "Attribute \"eps_100_h\" is of type \"" + str(type(eps_100_h)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(eps_100_c, measurement.Measurement) or eps_100_c is None, "Attribute \"eps_100_c\" is of type \"" + str(type(eps_100_c)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.specificHeatCapacityAir = specificHeatCapacityAir
        self.eps_75_h = eps_75_h
        self.eps_75_c = eps_75_c
        self.eps_100_h = eps_100_h
        self.eps_100_c = eps_100_c
        

    def do_step(self):
        m_a_max = max(self.primaryAirFlowRateMax.hasValue, self.secondaryAirFlowRateMax.hasValue)
        if self.input["primaryTemperatureIn"] < self.input["secondaryTemperatureIn"]:
            eps_75 = self.eps_75_h.hasValue
            eps_100 = self.eps_100_h.hasValue
        else:
            eps_75 = self.eps_75_c.hasValue
            eps_100 = self.eps_100_c.hasValue
        f_flow = 0.5*(self.input["primaryAirFlowRate"] + self.input["secondaryAirFlowRate"])/m_a_max
        eps_op = eps_75 + (eps_100-eps_75)*(f_flow-0.75)/(1-0.75)
        C_sup = self.input["primaryAirFlowRate"]*self.specificHeatCapacityAir.hasValue
        C_exh = self.input["secondaryAirFlowRate"]*self.specificHeatCapacityAir.hasValue
        C_min = min(C_sup, C_exh)
        if C_sup < 1e-5:
            T_p_out = NaN
        else:
            T_p_out = self.input["primaryTemperatureIn"] + eps_op*(self.input["secondaryTemperatureIn"] - self.input["primaryTemperatureIn"])*(C_min/C_sup)
        self.output["primaryTemperatureOut"] = T_p_out
