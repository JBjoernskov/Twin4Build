import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_junction.flow_junction as flow_junction
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes, Optional, MultipleMatches
import twin4build.base as base
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.FlowJunction, id="<FlowJunction\nn<SUB>1</SUB>>") #flow junction
    node1 = Node(cls=base.Damper, id="<Damper\nn<SUB>2</SUB>>") #damper
    node2 = Node(cls=(base.Coil, base.AirToAirHeatRecovery, base.Fan), id="<Coil, AirToAirHeatRecovery, Fan\nn<SUB>3</SUB>>") #building space
    sp = SignaturePattern(ownedBy="SupplyFlowJunctionSystem", priority=160)
    sp.add_edge(MultipleMatches(object=node0, subject=node1, predicate="suppliesFluidTo"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="hasFluidSuppliedBy"))
    sp.add_input("airFlowRateOut", node1, "airFlowRate")
    sp.add_modeled_node(node0)
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

        self.input = {"airFlowRateOut": tps.Vector()}
        self.output = {"airFlowRateIn": tps.Scalar()}
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
        # print("=========")
        # print("inputs")
        # for i in self.input:
        #     print(i, self.input[i].get())
        self.output["airFlowRateIn"].set((self.input["airFlowRateOut"].get().sum()) + self.airFlowRateBias)
        # print("outputs")
        # for i in self.output:
        #     print(i, self.output[i].get())
