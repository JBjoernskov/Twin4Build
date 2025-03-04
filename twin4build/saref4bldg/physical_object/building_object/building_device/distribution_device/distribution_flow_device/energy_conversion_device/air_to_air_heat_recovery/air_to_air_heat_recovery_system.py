import numpy as np
from typing import Union
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery as air_to_air_heat_recovery
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants
import sys
import os
import twin4build.base as base
from twin4build.translator.translator import SignaturePattern, Node, Exact, MultiPath, Optional_, SinglePath
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.S4BLDG.AirToAirHeatRecovery)
    node1 = Node(cls=base.S4BLDG.OutdoorEnvironment)
    node2 = Node(cls=base.SAREF.FlowJunction)
    node3 = Node(cls=base.SAREF.FlowJunction)
    node4 = Node(cls=base.S4BLDG.PrimaryAirFlowRateMax)
    node5 = Node(cls=base.SAREF.PropertyValue)
    node6 = Node(cls=base.XSD.float)
    node7 = Node(cls=base.S4BLDG.SecondaryAirFlowRateMax)
    node8 = Node(cls=base.SAREF.PropertyValue)
    node9 = Node(cls=base.XSD.float)
    node10 = Node(cls=base.S4BLDG.AirToAirHeatRecovery) #primary
    node11 = Node(cls=base.S4BLDG.AirToAirHeatRecovery) #secondary
    node12 = Node(cls=base.S4BLDG.Controller)
    node13 = Node(cls=base.SAREF.Motion)
    node14 = Node(cls=base.S4BLDG.Schedule)
    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="AirToAirHeatRecoverySystem")

    # buildingTemperature (SecondaryTemperatureIn)
    sp.add_triple(SinglePath(subject=node10, object=node1, predicate=base.FSO.hasFluidSuppliedBy))
    sp.add_triple(SinglePath(subject=node10, object=node2, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node11, object=node3, predicate=base.FSO.hasFluidReturnedBy))

    sp.add_triple(Exact(subject=node5, object=node6, predicate=base.SAREF.hasValue))
    sp.add_triple(Exact(subject=node5, object=node4, predicate=base.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node0, object=node5, predicate=base.SAREF.hasPropertyValue))

    # airFlowRateMax
    sp.add_triple(Exact(subject=node8, object=node9, predicate=base.SAREF.hasValue))
    sp.add_triple(Exact(subject=node8, object=node7, predicate=base.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node0, object=node8, predicate=base.SAREF.hasPropertyValue))

    sp.add_triple(Exact(subject=node10, object=node0, predicate=base.S4SYST.subSystemOf))
    sp.add_triple(Exact(subject=node11, object=node0, predicate=base.S4SYST.subSystemOf))

    sp.add_triple(Exact(subject=node12, object=node13, predicate=base.SAREF.controls))
    sp.add_triple(Exact(subject=node13, object=node0, predicate=base.SAREF.isPropertyOf))
    sp.add_triple(Exact(subject=node12, object=node14, predicate=base.SAREF.hasProfile))

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
                secondary_delta_T = primary_delta_T * (C_sup/C_exh)
                self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"].get() - secondary_delta_T)
                    
            else:
                self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"])
                self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"])
        else:
            self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"]) #np.nan
            self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"])

