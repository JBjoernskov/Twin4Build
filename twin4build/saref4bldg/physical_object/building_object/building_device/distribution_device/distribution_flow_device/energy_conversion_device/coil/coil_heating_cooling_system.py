from .coil import Coil
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil as coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants
import numpy as np
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches, Optional, IgnoreIntermediateNodes
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.Coil, id="<n<SUB>1</SUB>(Coil)>")
    node1 = Node(cls=base.FlowJunction, id="<n<SUB>2</SUB>(FlowJunction)>")
    node2 = Node(cls=(base.Fan, base.AirToAirHeatRecovery, base.Coil), id="<Fan, AirToAirHeatRecovery, Coil\nn<SUB>3</SUB>>")
    node3 = Node(cls=base.Coil, id="<n<SUB>4</SUB>(Coil)>")
    node4 = Node(cls=base.Coil, id="<n<SUB>5</SUB>(Coil)>") #supersystem
    node5 = Node(cls=base.Coil, id="<n<SUB>6</SUB>(Coil)>") #supersystem
    node6 = Node(cls=base.Controller, id="<n<SUB>7</SUB>(Controller)>")
    node7 = Node(cls=base.OpeningPosition, id="<n<SUB>8</SUB>(OpeningPosition)>")
    node8 = Node(cls=base.Schedule, id="<n<SUB>9</SUB>(Schedule)>")




    sp = SignaturePattern(ownedBy="CoilHeatingCoolingSystem", priority=0)
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node1, predicate="suppliesFluidTo"))
    sp.add_edge(IgnoreIntermediateNodes(object=node3, subject=node2, predicate="hasFluidSuppliedBy"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node3, predicate="hasFluidSuppliedBy"))
    sp.add_edge(Exact(object=node0, subject=node4, predicate="subSystemOf"))
    sp.add_edge(Exact(object=node3, subject=node5, predicate="subSystemOf"))
    sp.add_edge(Exact(object=node6, subject=node7, predicate="controls"))
    sp.add_edge(Exact(object=node7, subject=node5, predicate="isPropertyOf")) #We just need to know that the OpeningPosition is a property of the supersystem
    sp.add_edge(Exact(object=node6, subject=node8, predicate="hasProfile"))

    sp.add_modeled_node(node0)
    sp.add_modeled_node(node3)
    sp.add_modeled_node(node4)
    sp.add_modeled_node(node5)
    # sp.add_parameter("airFlowRateMax.hasValue", node13)
    sp.add_input("airFlowRate", node1, "airFlowRateIn")
    sp.add_input("inletAirTemperature", node2, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_input("outletAirTemperatureSetpoint", node8, "scheduleValue")

    return sp

class CoilHeatingCoolingSystem(coil.Coil):
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
        print("=========")
        for i in self.input:
            print(i, self.input[i].get())
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
            

        


