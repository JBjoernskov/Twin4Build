from twin4build.utils.constants import Constants
import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, MultiPath, Optional_, SinglePath
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.Coil) # air heating
    node1 = Node(cls=core.S4BLDG.FlowJunction)
    node2 = Node(cls=(core.S4BLDG.Fan, core.S4BLDG.AirToAirHeatRecovery, core.S4BLDG.Coil))
    node3 = Node(cls=core.S4BLDG.Coil) # water heating
    node4 = Node(cls=core.S4BLDG.Coil) #supersystem heating
    node5 = Node(cls=core.S4BLDG.Controller)
    node6 = Node(cls=core.SAREF.OpeningPosition)
    node7 = Node(cls=core.S4BLDG.Schedule)
    node8 = Node(cls=core.S4BLDG.Coil) # air cooling
    node9 = Node(cls=core.S4BLDG.Coil) # water cooling
    node10 = Node(cls=core.S4BLDG.Coil) #supersystem cooling




    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="CoilHeatingCoolingSystem", priority=0)

    sp.add_triple(SinglePath(subject=node0, object=node1, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node8, object=node2, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(SinglePath(subject=node0, object=node8, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node0, object=node4, predicate=core.S4SYST.subSystemOf))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=core.S4SYST.subSystemOf))
    sp.add_triple(Exact(subject=node8, object=node10, predicate=core.S4SYST.subSystemOf))
    sp.add_triple(Exact(subject=node9, object=node10, predicate=core.S4SYST.subSystemOf))
    sp.add_triple(Exact(subject=node5, object=node6, predicate=core.SAREF.controls))
    sp.add_triple(Exact(subject=node6, object=node4, predicate=core.SAREF.isPropertyOf)) #We just need to know that the OpeningPosition is a property of the supersystem
    sp.add_triple(Exact(subject=node5, object=node7, predicate=core.SAREF.hasProfile))

    sp.add_modeled_node(node0)
    sp.add_modeled_node(node3)
    sp.add_modeled_node(node4)
    sp.add_modeled_node(node5)
    sp.add_input("airFlowRate", node1, "airFlowRateIn")
    sp.add_input("inletAirTemperature", node2, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_input("outletAirTemperatureSetpoint", node7, "scheduleValue")

    return sp

class CoilHeatingCoolingSystem(core.System):
    sp = [get_signature_pattern()]
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.specificHeatCapacityAir = Constants.specificHeatCapacity["air"]

        self.input = {"inletAirTemperature": tps.Scalar(),
                      "outletAirTemperatureSetpoint": tps.Scalar(),
                      "airFlowRate": tps.Scalar()}
        self.output = {"heatingPower": tps.Scalar(),
                       "coolingPower": tps.Scalar(),
                       "outletAirTemperature": tps.Scalar()}
        self._config = {"parameters": []}

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
         updates the input and output variables of the coil and calculates the power output and air temperature based on the input air temperature, air flow rate, and air temperature setpoint. 
         If the air flow rate is zero, the output power and air temperature are set to NaN
        '''
        self.output.update(self.input)
        tol = 1e-5
        if self.input["airFlowRate"]>tol:
            if self.input["inletAirTemperature"] < self.input["outletAirTemperatureSetpoint"]:
                heatingPower = self.input["airFlowRate"]*self.specificHeatCapacityAir*(self.input["outletAirTemperatureSetpoint"] - self.input["inletAirTemperature"])
                coolingPower = 0

            else:
                heatingPower = 0
                coolingPower = self.input["airFlowRate"]*self.specificHeatCapacityAir*(self.input["inletAirTemperature"] - self.input["outletAirTemperatureSetpoint"])
            self.output["heatingPower"].set(heatingPower)
            self.output["coolingPower"].set(coolingPower)
            
        else:
            self.output["heatingPower"].set(0)
            self.output["coolingPower"].set(0)
        self.output["outletAirTemperature"].set(self.input["outletAirTemperatureSetpoint"])
            

        


