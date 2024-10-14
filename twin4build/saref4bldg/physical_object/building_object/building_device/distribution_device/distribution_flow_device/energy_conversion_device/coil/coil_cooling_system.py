from .coil import Coil
from typing import Union
from twin4build.utils.constants import Constants
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches, Optional, IgnoreIntermediateNodes
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.Coil, id="<n<SUB>1</SUB>(Coil)>")
    node1 = Node(cls=base.FlowJunction, id="<n<SUB>2</SUB>(FlowJunction)>")
    node2 = Node(cls=(base.Fan, base.AirToAirHeatRecovery, base.Coil), id="<Fan, AirToAirHeatRecovery, Coil\nn<SUB>3</SUB>>")

    node = Node(cls=base.PropertyValue, id="<PropertyValue\nn<SUB>23</SUB>>")
    node = Node(cls=(float, int), id="<Float, Int\nn<SUB>24</SUB>>")
    node = Node(cls=base.FlowCoefficient, id="<FlowCoefficient\nn<SUB>25</SUB>>")

    sp = SignaturePattern(ownedBy="CoolingCoilSystem", priority=0)

    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node1, predicate="suppliesFluidTo"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="hasFluidSuppliedBy"))

    sp.add_modeled_node(node0)
    # sp.add_parameter("airFlowRateMax.hasValue", node13)
    sp.add_input("airFlowRate", node1, "airFlowRateIn")
    sp.add_input("inletAirTemperature", node2, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))

    return sp


class CoilCoolingSystem(Coil):
    # sp = [get_signature_pattern()]
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.specificHeatCapacityAir = Constants.specificHeatCapacity["air"]

        self.input = {"inletAirTemperature": tps.Scalar(),
                      "outletAirTemperatureSetpoint": tps.Scalar(),
                      "airFlowRate": tps.Scalar()}
        self.output = {"power": tps.Scalar(), 
                       "outletAirTemperature": tps.Scalar()}
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
        self.output["Power"].set(Q)
        self.output["outletAirTemperature"].set(self.input["outletAirTemperatureSetpoint"])


        