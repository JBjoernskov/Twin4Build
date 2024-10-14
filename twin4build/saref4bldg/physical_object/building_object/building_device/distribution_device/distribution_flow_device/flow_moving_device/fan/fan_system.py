from .fan import Fan
from typing import Union
import twin4build.utils.input_output_types as tps


class FanSystem(Fan):
    def __init__(self,
                c1: Union[float, None]=None,
                c2: Union[float, None]=None,
                c3: Union[float, None]=None,
                c4: Union[float, None]=None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(c1, float) or c1 is None, "Attribute \"c1\" is of type \"" + str(type(c1)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(c2, float) or c2 is None, "Attribute \"c2\" is of type \"" + str(type(c2)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(c3, float) or c3 is None, "Attribute \"c3\" is of type \"" + str(type(c3)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(c4, float) or c4 is None, "Attribute \"c4\" is of type \"" + str(type(c4)) + "\" but must be of type \"" + str(float) + "\""
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.input = {"airFlowRate": tps.Scalar()}
        self.output = {"Power": tps.Scalar(),
                       "Energy": 0}
        self._config = {"parameters": ["c1", "c2", "c3", "c4", "nominalAirFlowRate.hasValue", "nominalPowerRate.hasValue"]}

    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        pass
        
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        f_flow = self.input["airFlowRate"]/self.nominalAirFlowRate.hasValue
        f_pl = self.c1 + self.c2*f_flow + self.c3*f_flow**2 + self.c4*f_flow**3
        W_fan = f_pl*self.nominalPowerRate.hasValue
        self.output["Power"].set(W_fan)
        self.output["Energy"].set(self.output["Energy"] + W_fan*stepSize/3600/1000)
