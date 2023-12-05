import math
from .damper import Damper
class DamperSystem(Damper):
    def __init__(self,
                a=None,
                b=None,
                c=None,
                **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c

        if self.c is None:
            self.c = -self.a
        
        if self.b is None:
            self.b = math.log((self.nominalAirFlowRate.hasValue-self.c)/self.a)

        self.input = {"damperPosition": None}
        self.output = {"airFlowRate": None}

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        m_a = self.a*math.exp(self.b*self.input["damperPosition"]) + self.c
        self.output["damperPosition"] = self.input["damperPosition"]
        self.output["airFlowRate"] = m_a


    