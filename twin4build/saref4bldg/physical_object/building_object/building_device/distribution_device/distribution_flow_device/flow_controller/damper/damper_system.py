import math
from .damper import Damper
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches, Optional
import twin4build.base as base

def get_signature_pattern_1():
    node0 = Node(cls=base.Damper, id="<n<SUB>1</SUB>(Damper)>") #supply valve
    node1 = Node(cls=base.Controller, id="<n<SUB>2</SUB>(Controller)>")
    node2 = Node(cls=base.OpeningPosition, id="<n<SUB>3</SUB>(Property)>")
    node3 = Node(cls=base.PropertyValue, id="<n<SUB>4</SUB>(PropertyValue)>")
    node4 = Node(cls=(float, int), id="<n<SUB>5</SUB>(Float)>")
    node5 = Node(cls=base.NominalAirFlowRate, id="<n<SUB>6</SUB>(nominalAirFlowRate)>")
    sp = SignaturePattern(ownedBy="DamperSystem")

    sp.add_edge(Exact(object=node1, subject=node2, predicate="controls"))
    sp.add_edge(Exact(object=node2, subject=node0, predicate="isPropertyOf"))
    sp.add_edge(Exact(object=node3, subject=node4, predicate="hasValue"))
    sp.add_edge(Exact(object=node3, subject=node5, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node0, subject=node3, predicate="hasPropertyValue"))

    sp.add_input("damperPosition", node1, "inputSignal")
    sp.add_parameter("nominalAirFlowRate.hasValue", node4)
    sp.add_modeled_node(node0)

    return sp

def get_signature_pattern_2():
    node0 = Node(cls=base.Damper, id="<n<SUB>1</SUB>(Damper)>") #supply valve
    node1 = Node(cls=base.Controller, id="<n<SUB>2</SUB>(Controller)>")
    node2 = Node(cls=base.OpeningPosition, id="<n<SUB>3</SUB>(OpeningPosition)>")
    node3 = Node(cls=base.Property, id="<n<SUB>4</SUB>(Property)>")
    node4 = Node(cls=base.PropertyValue, id="<n<SUB>5</SUB>(PropertyValue)>")
    node5 = Node(cls=(float, int), id="<n<SUB>6</SUB>(Float)>")
    node6 = Node(cls=base.NominalAirFlowRate, id="<n<SUB>7</SUB>(nominalAirFlowRate)>")
    sp = SignaturePattern(ownedBy="DamperSystem", priority=10)

    sp.add_edge(Exact(object=node1, subject=node2, predicate="controls"))
    sp.add_edge(Exact(object=node2, subject=node0, predicate="isPropertyOf"))
    sp.add_edge(Exact(object=node1, subject=node3, predicate="observes"))
    sp.add_edge(Exact(object=node4, subject=node5, predicate="hasValue"))
    sp.add_edge(Exact(object=node4, subject=node6, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node0, subject=node4, predicate="hasPropertyValue"))

    sp.add_input("damperPosition", node1, "inputSignal")
    sp.add_parameter("nominalAirFlowRate.hasValue", node5)
    # sp.add_input("damperPosition", node3, "inputSignal")
    sp.add_modeled_node(node0)

    return sp

class DamperSystem(Damper):
    sp = [get_signature_pattern_2()]
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
        self.parameter = {"a": {"lb": 0.0001, "ub": 5},
                            "nominalAirFlowRate.hasValue": {"lb": 0.0001, "ub": 5}}
        self._config = {"parameters": list(self.parameter.keys())}

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


