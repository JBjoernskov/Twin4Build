import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_, MultiPath
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.FlowJunction) #flow junction
    node1 = Node(cls=core.S4BLDG.Damper) #damper
    node2 = Node(cls=(core.S4BLDG.Coil, core.S4BLDG.AirToAirHeatRecovery, core.S4BLDG.Fan)) #building space
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SupplyFlowJunctionSystem", priority=160)
    sp.add_triple(MultiPath(subject=node0, object=node1, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node0, object=node2, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_input("airFlowRateOut", node1, "airFlowRate")
    sp.add_modeled_node(node0)
    return sp

class SupplyFlowJunctionSystem(core.System):
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

