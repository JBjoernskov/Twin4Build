from twin4build.utils.constants import Constants
import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, MultiPath, Optional_, SinglePath
import twin4build.utils.input_output_types as tps
import datetime
from typing import Optional

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.AirToAirHeatRecovery)
    node1 = Node(cls=core.S4BLDG.OutdoorEnvironment)
    node2 = Node(cls=core.S4BLDG.FlowJunction)
    node3 = Node(cls=core.S4BLDG.FlowJunction)
    node4 = Node(cls=core.S4BLDG.PrimaryAirFlowRateMax)
    node5 = Node(cls=core.SAREF.PropertyValue)
    node6 = Node(cls=core.XSD.float)
    node7 = Node(cls=core.S4BLDG.SecondaryAirFlowRateMax)
    node8 = Node(cls=core.SAREF.PropertyValue)
    node9 = Node(cls=core.XSD.float)
    node10 = Node(cls=core.S4BLDG.AirToAirHeatRecovery) #primary
    node11 = Node(cls=core.S4BLDG.AirToAirHeatRecovery) #secondary
    node12 = Node(cls=core.S4BLDG.Controller)
    node13 = Node(cls=core.SAREF.Motion)
    node14 = Node(cls=core.S4BLDG.Schedule)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="AirToAirHeatRecoverySystem")

    # buildingTemperature (SecondaryTemperatureIn)
    sp.add_triple(SinglePath(subject=node10, object=node1, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(SinglePath(subject=node10, object=node2, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node11, object=node3, predicate=core.FSO.hasFluidReturnedBy))

    sp.add_triple(Optional_(subject=node5, object=node6, predicate=core.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node5, object=node4, predicate=core.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node0, object=node5, predicate=core.SAREF.hasPropertyValue))

    # airFlowRateMax
    sp.add_triple(Optional_(subject=node8, object=node9, predicate=core.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node8, object=node7, predicate=core.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node0, object=node8, predicate=core.SAREF.hasPropertyValue))

    sp.add_triple(Exact(subject=node10, object=node0, predicate=core.S4SYST.subSystemOf))
    sp.add_triple(Exact(subject=node11, object=node0, predicate=core.S4SYST.subSystemOf))

    sp.add_triple(Exact(subject=node12, object=node13, predicate=core.SAREF.controls))
    sp.add_triple(Exact(subject=node13, object=node0, predicate=core.SAREF.isPropertyOf))
    sp.add_triple(Exact(subject=node12, object=node14, predicate=core.SAREF.hasProfile))

    sp.add_parameter("primaryAirFlowRateMax", node6)
    sp.add_parameter("secondaryAirFlowRateMax", node9)

    sp.add_input("primaryTemperatureIn", node1, "outdoorTemperature")
    sp.add_input("secondaryTemperatureIn", node3, "airTemperatureOut")
    sp.add_input("primaryAirFlowRate", node2, "airFlowRateIn")
    sp.add_input("secondaryAirFlowRate", node3, "airFlowRateOut")
    sp.add_input("primaryTemperatureOutSetpoint", node14, "scheduleValue")

    sp.add_modeled_node(node0)
    sp.add_modeled_node(node10)
    sp.add_modeled_node(node11)

    return sp

class AirToAirHeatRecoverySystem(core.System):
    sp = [get_signature_pattern()]
    def __init__(self,
                eps_75_h=None,
                eps_100_h=None,
                eps_75_c=None,
                eps_100_c=None,
                primaryAirFlowRateMax=None,
                secondaryAirFlowRateMax=None,
                **kwargs):
        super().__init__(**kwargs)
        self.eps_75_h = eps_75_h
        self.eps_100_h = eps_100_h
        self.eps_75_c = eps_75_c
        self.eps_100_c = eps_100_c
        self.primaryAirFlowRateMax = primaryAirFlowRateMax
        self.secondaryAirFlowRateMax = secondaryAirFlowRateMax

        self.input = {"primaryAirFlowRate": tps.Scalar(),
                      "secondaryAirFlowRate": tps.Scalar(),
                      "primaryTemperatureIn": tps.Scalar(),
                      "secondaryTemperatureIn": tps.Scalar(),
                      "primaryTemperatureOutSetpoint": tps.Scalar()}
        self.output = {"primaryTemperatureOut": tps.Scalar(),
                       "secondaryTemperatureOut": tps.Scalar()}
        self._config = {"parameters": ["eps_75_h",
                                       "eps_100_h",
                                       "eps_75_c",
                                       "eps_100_c",
                                       "primaryAirFlowRateMax",
                                       "secondaryAirFlowRateMax"]}

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

    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        '''
            Performs one simulation step based on the inputs and attributes of the object.
        '''
        self.output.update(self.input)
        tol = 1e-5
        if self.input["primaryAirFlowRate"]>tol and self.input["secondaryAirFlowRate"]>tol:
            m_a_max = max(self.primaryAirFlowRateMax, self.secondaryAirFlowRateMax)
            if self.input["primaryTemperatureIn"] < self.input["secondaryTemperatureIn"]:
                eps_75 = self.eps_75_h
                eps_100 = self.eps_100_h
                feasibleMode = "Heating"
            else:
                eps_75 = self.eps_75_c
                eps_100 = self.eps_100_c
                feasibleMode = "Cooling"

            operationMode = "Heating" if self.input["primaryTemperatureIn"]<self.input["primaryTemperatureOutSetpoint"] else "Cooling"

            if feasibleMode==operationMode:
                f_flow = 0.5*(self.input["primaryAirFlowRate"] + self.input["secondaryAirFlowRate"])/m_a_max
                eps_op = eps_75 + (eps_100-eps_75)*(f_flow-0.75)/(1-0.75)
                C_sup = self.input["primaryAirFlowRate"]*Constants.specificHeatCapacity["air"]
                C_exh = self.input["secondaryAirFlowRate"]*Constants.specificHeatCapacity["air"]
                C_min = min(C_sup, C_exh)
                # if C_sup < 1e-5:
                #     self.output["primaryTemperatureOut"] = NaN
                # else:
                self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"] + eps_op*(self.input["secondaryTemperatureIn"] - self.input["primaryTemperatureIn"])*(C_min/C_sup), stepIndex)

                if operationMode=="Heating" and self.output["primaryTemperatureOut"]>self.input["primaryTemperatureOutSetpoint"]:
                    self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureOutSetpoint"], stepIndex)
                elif operationMode=="Cooling" and self.output["primaryTemperatureOut"]<self.input["primaryTemperatureOutSetpoint"]:
                    self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureOutSetpoint"], stepIndex)
                
                 # Calculate secondaryTemperatureOut using energy conservation
                primary_delta_T = self.output["primaryTemperatureOut"].get() - self.input["primaryTemperatureIn"].get()
                secondary_delta_T = primary_delta_T * (C_sup/C_exh)
                self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"].get() - secondary_delta_T, stepIndex)
                    
            else:
                self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"], stepIndex)
                self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"], stepIndex)
        else:
            self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"], stepIndex) #np.nan
            self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"], stepIndex)

