import math
from .damper import Damper
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches
import twin4build.base as base

def get_signature_pattern():
    node0 = Node(cls=base.Damper, id="<n<SUB>1</SUB>(Damper)>") #supply valve
    node1 = Node(cls=base.Controller, id="<n<SUB>2</SUB>(Controller)>")
    node2 = Node(cls=base.OpeningPosition, id="<n<SUB>3</SUB>(Property)>")
    sp = SignaturePattern(ownedBy="DamperSystem")

    sp.add_edge(Exact(object=node1, subject=node2, predicate="controls"))
    sp.add_edge(Exact(object=node2, subject=node0, predicate="isPropertyOf"))

    sp.add_input("damperPosition", node1, "inputSignal")
    sp.add_modeled_node(node0)

    return sp


class DamperSystem(Damper):
    sp = [get_signature_pattern()]
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
        self.b = None
        self.c = None

        # if self.c is None:
        
        
        # if self.b is None:
        

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
        self.c = -self.a # Ensures that m=0 at u=0
        self.b = math.log((self.nominalAirFlowRate.hasValue-self.c)/self.a) #Ensures that m=nominalAirFlowRate at u=1
        

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        m_a = self.a*math.exp(self.b*self.input["damperPosition"]) + self.c
        self.output["damperPosition"] = self.input["damperPosition"]
        self.output["airFlowRate"] = m_a


