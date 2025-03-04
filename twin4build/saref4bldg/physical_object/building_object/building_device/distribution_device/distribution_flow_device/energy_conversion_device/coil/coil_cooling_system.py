from .coil import Coil
from typing import Union
from twin4build.utils.constants import Constants
import twin4build.base as base
from twin4build.translator.translator import SignaturePattern, Node, Exact, MultiPath, Optional_, SinglePath
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.S4BLDG.Coil)
    node1 = Node(cls=base.SAREF.FlowJunction)
    node2 = Node(cls=(base.S4BLDG.Fan, base.S4BLDG.AirToAirHeatRecovery, base.S4BLDG.Coil))

    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="CoilCoolingSystem")

    sp.add_triple(SinglePath(subject=node0, object=node1, predicate="suppliesFluidTo"))
    sp.add_triple(SinglePath(subject=node0, object=node2, predicate="hasFluidSuppliedBy"))

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


        