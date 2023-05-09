from .valve import Valve
import twin4build.saref.measurement.measurement as measurement
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater as space_heater
from twin4build.utils.constants import Constants
class ValveModel(Valve):
    def __init__(self, 
                waterFlowRateMax=None, 
                valveAuthority=None,
                **kwargs):
        super().__init__(**kwargs)
        # assert isinstance(waterFlowRateMax, measurement.Measurement) or waterFlowRateMax is None, "Attribute \"closeOffRating\" is of type \"" + str(type(waterFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(valveAuthority, measurement.Measurement) or valveAuthority is None, "Attribute \"valveAuthority\" is of type \"" + str(type(valveAuthority)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.valveAuthority = valveAuthority

        self.input = {"valvePosition": None}
        self.output = {"waterFlowRate": None}

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        
        '''
        sets the waterFlowRateMax attribute based on 
        the connected SpaceHeater component's output capacity and nominal temperatures.
        '''

        # self.waterFlowRateMax = self.flowCoefficient.hasValue/(1/self.testPressure.hasValue)**0.5/3600*1000
        space_heater_component = [component for component in self.connectedTo if isinstance(component, space_heater.SpaceHeater)][0]
        self.waterFlowRateMax = abs(space_heater_component.outputCapacity.hasValue/Constants.specificHeatCapacity["water"]/(space_heater_component.nominalSupplyTemperature-space_heater_component.nominalReturnTemperature))
        # 0.0224

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):

        '''
            calculates the water flow rate through the valve based on the valvePosition input and the valveAuthority. It then updates the waterFlowRate output accordingly. 
            The valvePosition output is set to the same value as the valvePosition input.
        '''

        u_norm = self.input["valvePosition"]/(self.input["valvePosition"]**2*(1-self.valveAuthority.hasValue)+self.valveAuthority.hasValue)**(0.5)
        m_w = u_norm*self.waterFlowRateMax
        self.output["valvePosition"] = self.input["valvePosition"]
        self.output["waterFlowRate"] = m_w    