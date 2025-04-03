import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_, MultiPath
import twin4build.utils.input_output_types as tps
import numpy as np

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.FlowJunction) #flow junction
    node1 = Node(cls=core.S4BLDG.Damper) #damper
    node2 = Node(cls=core.S4BLDG.BuildingSpace) #building space
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="ReturnFlowJunctionSystem", priority=160)
    sp.add_triple(MultiPath(subject=node0, object=node1, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.FSO.hasFluidReturnedBy))

    sp.add_input("airFlowRateIn", node1, "airFlowRate")
    sp.add_input("airTemperatureIn", node2, "indoorTemperature")
    # sp.add_input("inletAirTemperature", node15, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_modeled_node(node0)
    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")
    return sp

class ReturnFlowJunctionSystem(core.System):
    sp = [get_signature_pattern()]
    def __init__(self,
                airFlowRateBias = None,
                **kwargs):
        super().__init__(**kwargs)
        if airFlowRateBias is not None:
            self.airFlowRateBias = airFlowRateBias
        else:
            self.airFlowRateBias = 0

        self.input = {"airFlowRateIn": tps.Vector(),
                      "airTemperatureIn": tps.Vector(),}
        self.output = {"airFlowRateOut": tps.Scalar(),
                       "airTemperatureOut": tps.Scalar()}
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
        with np.errstate(invalid='raise'):
            m_dot_in = self.input["airFlowRateIn"].get().sum()
            Q_dot_in = self.input["airTemperatureIn"].get()*self.input["airFlowRateIn"].get()
            tol = 1e-5
            if m_dot_in > tol:
                self.output["airFlowRateOut"].set(m_dot_in + self.airFlowRateBias)
                self.output["airTemperatureOut"].set(Q_dot_in.sum()/self.output["airFlowRateOut"].get())
            else:
                self.output["airFlowRateOut"].set(0)
                self.output["airTemperatureOut"].set(20)
