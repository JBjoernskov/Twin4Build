from .coil import Coil
from typing import Union
from twin4build.utils.constants import Constants
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches, Optional, IgnoreIntermediateNodes

def get_signature_pattern_1():
    node0 = Node(cls=base.Coil, id="<n<SUB>0</SUB>(CoolingCoil)>")
    node1 = Node(cls=base.Temperature, id="<n<SUB>1</SUB>(InletTemperature)>")
    node2 = Node(cls=base.Temperature, id="<n<SUB>2</SUB>(OutletTemperature)>")
    node3 = Node(cls=base.Sensor, id="<n<SUB>3</SUB>(InletSensor)>")
    node4 = Node(cls=base.Sensor, id="<n<SUB>4</SUB>(OutletSensor)>")
    node5 = Node(cls=base.System, id="<n<SUB>5</SUB>(VentilationSystem)>")
    node6 = Node(cls=base.PropertyValue, id="<n<SUB>6</SUB>(InletPropertyValue)>")
    node7 = Node(cls=base.PropertyValue, id="<n<SUB>7</SUB>(OutletPropertyValue)>")
    node8 = Node(cls=base.FlowJunction, id="<n<SUB>8</SUB>(FlowJunction)>")
    node10 = Node(cls=base.NominalAirFlowRate, id="<n<SUB>10</SUB>(AirFlowRate)>")
    node11 = Node(cls=base.PropertyValue, id="<n<SUB>11</SUB>(PropertyValue)>")

    # airflowrate max (config)
    node12 = Node(cls=base.PropertyValue, id="<n<SUB>12</SUB>(PropertyValue)>")
    node13 = Node(cls=(float, int), id="<n<SUB>13</SUB>(Float)>")
    node14 = Node(cls=base.AirFlowRateMax, id="<n<SUB>14</SUB>(AirFlowRateMax)>")

    sp = SignaturePattern(ownedBy="CoolingCoilSystem", priority=0)

    # Inlet Temperature
    sp.add_edge(Exact(object=node0, subject=node6, predicate="hasProperty"))
    sp.add_edge(Exact(object=node6, subject=node3, predicate="hasValue"))
    sp.add_edge(Exact(object=node3, subject=node1, predicate="observes"))

    # Outlet Temperature
    sp.add_edge(Exact(object=node0, subject=node7, predicate="hasProperty"))
    sp.add_edge(Exact(object=node7, subject=node4, predicate="hasValue"))
    sp.add_edge(Exact(object=node4, subject=node2, predicate="observes"))

    # Air Flowrate
    sp.add_edge(Exact(object=node10, subject=node0, predicate="isPropertyOf"))
    sp.add_edge(Exact(object=node11, subject=node10, predicate="isValueOfProperty"))
    sp.add_edge(Exact(object=node11, subject=node8, predicate="hasValue"))

    # Air Flow Rate Max
    sp.add_edge(Exact(object=node12, subject=node14, predicate="isValueOfProperty"))
    sp.add_edge(Exact(object=node14, subject=node0, predicate="isPropertyOf"))
    sp.add_edge(Exact(object=node12, subject=node13, predicate="hasValue"))


    sp.add_modeled_node(node0)
    sp.add_parameter("airFlowRateMax.hasValue", node13)
    sp.add_input("AirFlowRate", node8, "totalAirFlowRate")

    return sp


class CoilCoolingSystem(Coil):
    sp = [get_signature_pattern_1()]
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.specificHeatCapacityAir = Constants.specificHeatCapacity["air"]

        self.input = {"inletAirTemperature": None,
                      "outletAirTemperatureSetpoint": None,
                      "airFlowRate": None}
        self.output = {"Power": None, 
                       "outletAirTemperature": None}
        self._config = {"parameters": ["airFlowRateMax.hasValue",
                                       "nominalSensibleCapacity.hasValue"]}

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
        '''
            simulate the cooling behavior of a coil. It calculates the heat transfer rate based on the 
            input temperature difference and air flow rate, and sets the output air temperature to the desired setpoint.
        '''
        
        self.output.update(self.input)
        if self.input["inletAirTemperature"] > self.input["outletAirTemperatureSetpoint"]:
            Q = self.input["airFlowRate"]*self.specificHeatCapacityAir*(self.input["inletAirTemperature"] - self.input["outletAirTemperatureSetpoint"])
        else:
            Q = 0
        self.output["Power"] = Q
        self.output["outletAirTemperature"] = self.input["outletAirTemperatureSetpoint"]


        