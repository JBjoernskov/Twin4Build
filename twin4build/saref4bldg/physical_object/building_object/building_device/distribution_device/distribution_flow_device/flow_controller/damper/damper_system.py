import math
from .damper import Damper
class DamperSystem(Damper):
    """
    Parameters:
        - nominalAirFlowRate: The nominal air flow rate through the damper when it is fully open.
        - a: A parameter that determines the shape of the curve defined by the equation: 
        m = a*exp(b*u) + c, 
        where m is the air flow rate, u is the damper position, and a is a parameter that determines
        the shape of the curve. The parameters b, and c are calculated to ensure that m=0 when the damper is fully closed (u=0) and m=nominalAirFlowRate when the damper is fully open (u=1).
        


    Inputs: 
        - damperPosition: The position of the damper as a value between 0 and 1. 
        0 means the damper is closed and 1 means the damper is fully open.

    Outputs:
        - airFlowRate: The air flow rate through the damper. The air flow rate is calculated using the equation: 
        

    """
    def __init__(self,
                a=5,
                **kwargs):
        
        super().__init__(**kwargs)
        self.a = a
        # self.b = b
        # self.c = c

        # if self.c is None:
        self.c = -self.a # Ensures that m=0 at u=0
        
        # if self.b is None:
        self.b = math.log((self.nominalAirFlowRate.hasValue-self.c)/self.a) #Ensures that m=nominalAirFlowRate at u=1

        self.input = {"damperPosition": None}
        self.output = {"airFlowRate": None}
        self._config = {"parameters": ["a",
                                       "b",
                                       "c",
                                       "nominalAirFlowRate.hasValue"]}

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
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        m_a = self.a*math.exp(self.b*self.input["damperPosition"]) + self.c
        self.output["damperPosition"] = self.input["damperPosition"]
        self.output["airFlowRate"] = m_a


