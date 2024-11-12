import numpy as np
from typing import Union
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery as air_to_air_heat_recovery
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants
import sys
import os
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, MultipleMatches, Optional, IgnoreIntermediateNodes
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.AirToAirHeatRecovery, id="<n<SUB>1</SUB>(AirToAirHeatRecovery)>")
    node1 = Node(cls=base.OutdoorEnvironment, id="<n<SUB>2</SUB>(OutdoorEnvironment)>")
    node2 = Node(cls=base.FlowJunction, id="<n<SUB>3</SUB>(FlowJunction)>")
    node3 = Node(cls=base.FlowJunction, id="<n<SUB>4</SUB>(FlowJunction)>")
    node4 = Node(cls=base.PrimaryAirFlowRateMax, id="<n<SUB>5</SUB>(PrimaryAirFlowRateMax)>")
    node5 = Node(cls=base.PropertyValue, id="<n<SUB>6</SUB>(PropertyValue)>")
    node6 = Node(cls=(float, int), id="<n<SUB>7</SUB>(Float)>")
    node7 = Node(cls=base.SecondaryAirFlowRateMax, id="<n<SUB>8</SUB>(SecondaryAirFlowRateMax)>")
    node8 = Node(cls=base.PropertyValue, id="<n<SUB>9</SUB>(PropertyValue)>")
    node9 = Node(cls=(float, int), id="<n<SUB>10</SUB>(Float)>")
    node10 = Node(cls=base.AirToAirHeatRecovery, id="<n<SUB>11</SUB>(AirToAirHeatRecovery)>") #primary
    node11 = Node(cls=base.AirToAirHeatRecovery, id="<n<SUB>12</SUB>(AirToAirHeatRecovery)>") #secondary
    node12 = Node(cls=base.Controller, id="<n<SUB>13</SUB>(Controller)>")
    node13 = Node(cls=base.Motion, id="<n<SUB>14</SUB>(OpeningPosition)>")
    node14 = Node(cls=base.Schedule, id="<n<SUB>15</SUB>(Schedule)>")
    
    sp = SignaturePattern(ownedBy="AirToAirHeatRecoverySystem", priority=0)

    # buildingTemperature (SecondaryTemperatureIn)
    sp.add_edge(IgnoreIntermediateNodes(object=node10, subject=node1, predicate="hasFluidSuppliedBy"))
    sp.add_edge(IgnoreIntermediateNodes(object=node10, subject=node2, predicate="suppliesFluidTo"))
    sp.add_edge(IgnoreIntermediateNodes(object=node11, subject=node3, predicate="hasFluidReturnedBy"))

    sp.add_edge(Exact(object=node5, subject=node6, predicate="hasValue"))
    sp.add_edge(Exact(object=node5, subject=node4, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node0, subject=node5, predicate="hasPropertyValue"))

    # airFlowRateMax
    sp.add_edge(Exact(object=node8, subject=node9, predicate="hasValue"))
    sp.add_edge(Exact(object=node8, subject=node7, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node0, subject=node8, predicate="hasPropertyValue"))

    sp.add_edge(Exact(object=node10, subject=node0, predicate="subSystemOf"))
    sp.add_edge(Exact(object=node11, subject=node0, predicate="subSystemOf"))

    sp.add_edge(Exact(object=node12, subject=node13, predicate="controls"))
    sp.add_edge(Exact(object=node13, subject=node0, predicate="isPropertyOf"))
    sp.add_edge(Exact(object=node12, subject=node14, predicate="hasProfile"))

    sp.add_parameter("primaryAirFlowRateMax.hasValue", node6)
    sp.add_parameter("secondaryAirFlowRateMax.hasValue", node9)

    sp.add_input("primaryTemperatureIn", node1, "outdoorTemperature")
    sp.add_input("secondaryTemperatureIn", node3, "airTemperatureOut")
    sp.add_input("primaryAirFlowRate", node2, "airFlowRateIn")
    sp.add_input("secondaryAirFlowRate", node3, "airFlowRateOut")
    sp.add_input("primaryTemperatureOutSetpoint", node14, "scheduleValue")

    sp.add_modeled_node(node0)
    sp.add_modeled_node(node10)
    sp.add_modeled_node(node11)

    return sp




class AirToAirHeatRecoverySystem(air_to_air_heat_recovery.AirToAirHeatRecovery):
    sp = [get_signature_pattern()]
    def __init__(self,
                eps_75_h: Union[float, None]=None,
                eps_75_c: Union[float, None]=None,
                eps_100_h: Union[float, None]=None,
                eps_100_c: Union[float, None]=None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(eps_75_h, float) or eps_75_h is None, "Attribute \"eps_75_h\" is of type \"" + str(type(eps_75_h)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(eps_75_c, float) or eps_75_c is None, "Attribute \"eps_75_c\" is of type \"" + str(type(eps_75_c)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(eps_100_h, float) or eps_100_h is None, "Attribute \"eps_100_h\" is of type \"" + str(type(eps_100_h)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(eps_100_c, float) or eps_100_c is None, "Attribute \"eps_100_c\" is of type \"" + str(type(eps_100_c)) + "\" but must be of type \"" + str(float) + "\""
        self.eps_75_h = eps_75_h
        self.eps_75_c = eps_75_c
        self.eps_100_h = eps_100_h
        self.eps_100_c = eps_100_c

        self.input = {"primaryTemperatureIn": tps.Scalar(), 
                    "secondaryTemperatureIn": tps.Scalar(),
                    "primaryAirFlowRate": tps.Scalar(),
                    "secondaryAirFlowRate": tps.Scalar(),
                    "primaryTemperatureOutSetpoint": tps.Scalar()}
        self.output = {"primaryTemperatureOut": tps.Scalar(),
                       "secondaryTemperatureOut": tps.Scalar()}
        self._config = {"parameters": ["eps_75_h",
                                       "eps_75_c",
                                       "eps_100_h",
                                       "eps_100_c",
                                       "primaryAirFlowRateMax.hasValue",
                                       "secondaryAirFlowRateMax.hasValue"]}

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
            Performs one simulation step based on the inputs and attributes of the object.
        '''
        self.output.update(self.input)
        tol = 1e-5
        if self.input["primaryAirFlowRate"]>tol and self.input["secondaryAirFlowRate"]>tol:
            m_a_max = max(self.primaryAirFlowRateMax.hasValue, self.secondaryAirFlowRateMax.hasValue)
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
                self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"] + eps_op*(self.input["secondaryTemperatureIn"] - self.input["primaryTemperatureIn"])*(C_min/C_sup))

                if operationMode=="Heating" and self.output["primaryTemperatureOut"]>self.input["primaryTemperatureOutSetpoint"]:
                    self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureOutSetpoint"])
                elif operationMode=="Cooling" and self.output["primaryTemperatureOut"]<self.input["primaryTemperatureOutSetpoint"]:
                    self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureOutSetpoint"])
                
                 # Calculate secondaryTemperatureOut using energy conservation
                primary_delta_T = self.output["primaryTemperatureOut"].get() - self.input["primaryTemperatureIn"].get()
                secondary_delta_T = primary_delta_T * (C_exh/C_sup)
                self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"].get() - secondary_delta_T)
                    
            else:
                self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"])
                self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"])
        else:
            self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"]) #np.nan
            self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"])

