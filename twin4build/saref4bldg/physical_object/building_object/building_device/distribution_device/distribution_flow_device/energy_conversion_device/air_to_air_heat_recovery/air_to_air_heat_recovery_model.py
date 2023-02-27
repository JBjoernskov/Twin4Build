from numpy import NaN
from typing import Union
from scipy.optimize import least_squares
import numpy as np
from .air_to_air_heat_recovery import AirToAirHeatRecovery
import twin4build.saref.measurement.measurement as measurement

class AirToAirHeatRecoveryModel(AirToAirHeatRecovery):
    def __init__(self,
                specificHeatCapacityAir: Union[measurement.Measurement, None] = None,
                eps_75_h: Union[float, None] = None,
                eps_75_c: Union[float, None] = None,
                eps_100_h: Union[float, None] = None,
                eps_100_c: Union[float, None] = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(specificHeatCapacityAir, measurement.Measurement) or specificHeatCapacityAir is None, "Attribute \"specificHeatCapacityAir\" is of type \"" + str(type(specificHeatCapacityAir))+ "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(eps_75_h, float) or eps_75_h is None, "Attribute \"eps_75_h\" is of type \"" + str(type(eps_75_h)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(eps_75_c, float) or eps_75_c is None, "Attribute \"eps_75_c\" is of type \"" + str(type(eps_75_c)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(eps_100_h, float) or eps_100_h is None, "Attribute \"eps_100_h\" is of type \"" + str(type(eps_100_h)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(eps_100_c, float) or eps_100_c is None, "Attribute \"eps_100_c\" is of type \"" + str(type(eps_100_c)) + "\" but must be of type \"" + str(float) + "\""
        self.specificHeatCapacityAir = specificHeatCapacityAir
        self.eps_75_h = eps_75_h
        self.eps_75_c = eps_75_c
        self.eps_100_h = eps_100_h
        self.eps_100_c = eps_100_c

    def initialize(self):
        pass        

    def do_step(self, time=None, stepSize=None):
        self.output.update(self.input)
        m_a_max = max(self.primaryAirFlowRateMax.hasValue, self.secondaryAirFlowRateMax.hasValue)
        if self.input["primaryTemperatureIn"] < self.input["secondaryTemperatureIn"]:
            eps_75 = self.eps_75_h
            eps_100 = self.eps_100_h
            feasibleMode = "Heating"
        else:
            eps_75 = self.eps_75_c
            eps_100 = self.eps_100_c
            feasibleMode = "Cooling"

        operationMode = "Heating" if self.input["primaryTemperatureIn"]<self.input["primaryTemperatureOutSetpoint"] else "Cooling"

        if feasibleMode==operationMode:
            f_flow = 0.5*(self.input["primaryAirFlowRate"] + self.input["secondaryAirFlowRate"])/m_a_max
            eps_op = eps_75 + (eps_100-eps_75)*(f_flow-0.75)/(1-0.75)
            C_sup = self.input["primaryAirFlowRate"]*self.specificHeatCapacityAir.hasValue
            C_exh = self.input["secondaryAirFlowRate"]*self.specificHeatCapacityAir.hasValue
            C_min = min(C_sup, C_exh)
            if C_sup < 1e-5:
                self.output["primaryTemperatureOut"] = NaN
            else:
                self.output["primaryTemperatureOut"] = self.input["primaryTemperatureIn"] + eps_op*(self.input["secondaryTemperatureIn"] - self.input["primaryTemperatureIn"])*(C_min/C_sup)

            if operationMode=="Heating" and self.output["primaryTemperatureOut"]>self.input["primaryTemperatureOutSetpoint"]:
                self.output["primaryTemperatureOut"] = self.input["primaryTemperatureOutSetpoint"]
            elif operationMode=="Cooling" and self.output["primaryTemperatureOut"]<self.input["primaryTemperatureOutSetpoint"]:
                self.output["primaryTemperatureOut"] = self.input["primaryTemperatureOutSetpoint"]
        else:
            self.output["primaryTemperatureOut"] = self.input["primaryTemperatureIn"]


    def do_period(self, input):
        self.clear_report()
        for index, row in input.iterrows():
            for key in input:
                self.input[key] = row[key]
            self.do_step()
            self.update_report()
        output_predicted = np.array(self.savedOutput["primaryTemperatureOut"])
        return output_predicted

    def obj_fun(self, x, input, output):
        self.eps_75_h = x[0]
        self.eps_75_c = x[1]
        self.eps_100_h = x[2]
        self.eps_100_c = x[3]
        output_predicted = self.do_period(input)
        res = output_predicted-output #residual of predicted vs measured
        print(f"Loss: {np.sum(res**2)}")
        return res

    def calibrate(self, input=None, output=None):
        x0 = np.array([self.eps_75_h, self.eps_75_c, self.eps_100_h, self.eps_100_c])
        lb = [0, 0, 0, 0]
        ub = [1, 1, 1, 1]
        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output))
        self.eps_75_h, self.eps_75_c, self.eps_100_h, self.eps_100_c = sol.x
        print(sol)