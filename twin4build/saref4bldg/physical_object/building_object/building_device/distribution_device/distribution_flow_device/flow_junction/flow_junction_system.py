import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_junction.flow_junction as flow_junction
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes, Optional, MultipleMatches
import twin4build.base as base
def get_signature_pattern():
    node0 = Node(cls=base.FlowJunction, id="<FlowJunction\nn<SUB>1</SUB>>") #flow junction
    node1 = Node(cls=base.Damper, id="<Damper\nn<SUB>2</SUB>>") #damper
    node2 = Node(cls=(base.Coil, base.AirToAirHeatRecovery, base.Fan), id="<Coil, AirToAirHeatRecovery, Fan\nn<SUB>3</SUB>>") #building space
    sp = SignaturePattern(ownedBy="FlowJunctionSystem", priority=160)
    sp.add_edge(MultipleMatches(object=node0, subject=node1, predicate="suppliesFluidTo"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="hasFluidSuppliedBy"))

    sp.add_input("airFlowRate", node1)
    sp.add_input("supplyAirTemperature", node2, "flowTemperatureIn")
    sp.add_modeled_node(node0)
    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")

    return sp

class SupplyFlowJunctionSystem(flow_junction.FlowJunction):
    sp = [get_signature_pattern()]
    def __init__(self,
                airFlowRateBias = None,
                **kwargs):
        super().__init__(**kwargs)
        if airFlowRateBias is not None:
            self.airFlowRateBias = airFlowRateBias
        else:
            self.airFlowRateBias = 0

        self.input = {"airFlowRateOut": None,
                      }
        self.output = {"totalAirFlowRate": None,
                       "flowTemperatureOut": None,
                       "flowRate": None}
        self._config = {"parameters": ["airFlowRateBias"]}


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
        self.output["totalAirFlowRate"] = sum(v for k, v in self.input.items() if "airFlowRate" in k) + self.airFlowRateBias