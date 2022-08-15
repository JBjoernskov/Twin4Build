import math
from .damper import Damper
class DamperModel(Damper):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.a = 5
        self.c = -self.a
        self.b = math.log((self.nominalAirFlowRate.hasValue-self.c)/self.a)

    def update_output(self):
        m_a = self.a*math.exp(self.b*self.input["damperPosition"]) + self.c
        self.output["airFlowRate"] = m_a


    