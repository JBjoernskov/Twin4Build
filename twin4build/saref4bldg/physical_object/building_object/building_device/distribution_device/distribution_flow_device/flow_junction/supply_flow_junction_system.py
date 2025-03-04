import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_junction.flow_junction as flow_junction
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_, MultiPath
import twin4build.base as base
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.SAREF.FlowJunction) #flow junction
    node1 = Node(cls=base.S4BLDG.Damper) #damper
    node2 = Node(cls=(base.S4BLDG.Coil, base.S4BLDG.AirToAirHeatRecovery, base.S4BLDG.Fan)) #building space
    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="SupplyFlowJunctionSystem", priority=160)
    sp.add_triple(MultiPath(subject=node0, object=node1, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node0, object=node2, predicate=base.FSO.hasFluidSuppliedBy))
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
        self.output["airFlowRateIn"].set((self.input["airFlowRateOut"].get().sum()) + self.airFlowRateBias)

