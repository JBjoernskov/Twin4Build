from .valve import Valve
import twin4build.saref.measurement.measurement as measurement
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater as space_heater
from twin4build.utils.constants import Constants
class ValveModel(Valve):
    def __init__(self, 
                waterFlowRateMax = None, 
                valveAuthority = None,
                **kwargs):
        super().__init__(**kwargs)
        # assert isinstance(waterFlowRateMax, measurement.Measurement) or waterFlowRateMax is None, "Attribute \"closeOffRating\" is of type \"" + str(type(waterFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(valveAuthority, measurement.Measurement) or valveAuthority is None, "Attribute \"valveAuthority\" is of type \"" + str(type(valveAuthority)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.valveAuthority = valveAuthority ###

    def initialize(self):
        # self.waterFlowRateMax = self.flowCoefficient.hasValue/(1/self.testPressure.hasValue)**0.5/3600*1000
        space_heater_component = [component for component in self.connectedTo if isinstance(component, space_heater.SpaceHeater)][0]
        self.waterFlowRateMax = abs(space_heater_component.outputCapacity.hasValue/Constants.specificHeatCapacity["Water"]/(space_heater_component.nominalSupplyTemperature-space_heater_component.nominalReturnTemperature))
        

    def do_step(self, time=None, stepSize=None):
        u_norm = self.input["valvePosition"]/(self.input["valvePosition"]**2*(1-self.valveAuthority.hasValue)+self.valveAuthority.hasValue)**(0.5)
        m_w = u_norm*self.waterFlowRateMax
        self.output["valvePosition"] = self.input["valvePosition"]
        self.output["waterFlowRate"] = m_w
    