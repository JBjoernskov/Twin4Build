from .valve import Valve
import twin4build.saref.measurement.measurement as measurement
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater as space_heater
from twin4build.utils.constants import Constants

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class ValveModel(Valve):
    def __init__(self, 
                waterFlowRateMax=None, 
                valveAuthority=None,
                **kwargs):
        
        logger.info("[Value Model] : Entered in Initialise Function")

        super().__init__(**kwargs)
        # assert isinstance(waterFlowRateMax, measurement.Measurement) or waterFlowRateMax is None, "Attribute \"closeOffRating\" is of type \"" + str(type(waterFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(waterFlowRateMax, float) or waterFlowRateMax is None, "Attribute \"closeOffRating\" is of type \"" + str(type(waterFlowRateMax)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(valveAuthority, float) or valveAuthority is None, "Attribute \"valveAuthority\" is of type \"" + str(type(valveAuthority)) + "\" but must be of type \"" + str(float) + "\""
        self.waterFlowRateMax = waterFlowRateMax
        self.valveAuthority = valveAuthority

        self.input = {"valvePosition": None}
        self.output = {"waterFlowRate": None}



    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass
        # self.waterFlowRateMax = self.flowCoefficient.hasValue/(1/self.testPressure.hasValue)**0.5/3600*1000
        # space_heater_component = [component for component in self.connectedTo if isinstance(component, space_heater.SpaceHeater)][0]
        # self.waterFlowRateMax = abs(space_heater_component.outputCapacity.hasValue/Constants.specificHeatCapacity["water"]/(space_heater_component.nominalSupplyTemperature-space_heater_component.nominalReturnTemperature))
        # self.waterFlowRateMax =
        # 0.0224
        logger.info("[Value Model] : Exited from Initialise Function")

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        u_norm = self.input["valvePosition"]/(self.input["valvePosition"]**2*(1-self.valveAuthority)+self.valveAuthority)**(0.5)
        m_w = u_norm*self.waterFlowRateMax
        self.output["valvePosition"] = self.input["valvePosition"]
        self.output["waterFlowRate"] = m_w
