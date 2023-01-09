from .fan import Fan
from typing import Union
import twin4build.saref.measurement.measurement as measurement
class FanModel(Fan):
    def __init__(self,
                c1: Union[measurement.Measurement, None] = None,
                c2: Union[measurement.Measurement, None] = None,
                c3: Union[measurement.Measurement, None] = None,
                c4: Union[measurement.Measurement, None] = None,
                timeStep = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(c1, measurement.Measurement) or c1 is None, "Attribute \"capacityControlType\" is of type \"" + str(type(c1)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(c2, measurement.Measurement) or c2 is None, "Attribute \"capacityControlType\" is of type \"" + str(type(c2)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(c3, measurement.Measurement) or c3 is None, "Attribute \"capacityControlType\" is of type \"" + str(type(c3)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(c4, measurement.Measurement) or c4 is None, "Attribute \"capacityControlType\" is of type \"" + str(type(c4)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.timeStep = timeStep
        
    def do_step(self):
        if self.input["airFlowRate"] < 1e-5:
            self.output["Power"] = 0
        else:
            f_flow = self.input["airFlowRate"]/self.nominalAirFlowRate.hasValue
            f_pl = self.c1.hasValue + self.c2.hasValue*f_flow + self.c3.hasValue*f_flow**2 + self.c4.hasValue*f_flow**3
            W_fan = f_pl*self.nominalPowerRate.hasValue
            self.output["Power"] = W_fan
            self.output["Energy"] =  self.output["Energy"] + W_fan*self.timeStep/3600/1000

        