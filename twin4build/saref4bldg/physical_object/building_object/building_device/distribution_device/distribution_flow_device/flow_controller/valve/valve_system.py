from .valve import Valve
import twin4build.saref.measurement.measurement as measurement
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater as space_heater
from twin4build.utils.constants import Constants
from twin4build.logger.Logging import Logging
import numpy as np
logger = Logging.get_logger("ai_logfile")

class ValveSystem(Valve):
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
        self.outputUncertainty = {"waterFlowRate": 0}

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass
        # self.waterFlowRateMax = self.flowCoefficient.hasValue/(1/self.testPressure.hasValue)**0.5/3600*1000
        # space_heater_component = [component for component in self.connectedTo if isinstance(component, space_heater.SpaceHeater)][0]
        # self.waterFlowRateMax = abs(space_heater_component.outputCapacity.hasValue/Constants.specificHeatCapacity["water"]/(space_heater_component.nominalSupplyTemperature-space_heater_component.nominalReturnTemperature))
        # self.waterFlowRateMax = 0.0202 #############
        # 0.0224
        logger.info("[Value Model] : Exited from Initialise Function")

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        u_norm = self.input["valvePosition"]/(self.input["valvePosition"]**2*(1-self.valveAuthority)+self.valveAuthority)**(0.5)
        m_w = u_norm*self.waterFlowRateMax
        self.output["valvePosition"] = self.input["valvePosition"]
        self.output["waterFlowRate"] = m_w

    def get_subset_gradient(self, x_key, y_keys=None, as_dict=False):
        if x_key=="valvePosition":
            grad = [(self.valveAuthority*self.waterFlowRateMax)/((-self.valveAuthority*self.input["valvePosition"]+self.valveAuthority+self.input["valvePosition"]**2)**(3/2))]
        
        if x_key=="valveAuthority":
            grad = [(0.5*self.waterFlowRateMax*self.input["valvePosition"]*(self.input["valvePosition"]**2-1))/((-self.valveAuthority*self.input["valvePosition"]**2+self.valveAuthority+self.input["valvePosition"]**2)**1.5)]

        if x_key=="waterFlowRateMax":
            grad = [self.output["waterFlowRate"]/self.waterFlowRateMax]
        if as_dict==False:
            grad = np.array([grad])
        else:
            grad = {key: value for key, value in zip(y_keys, grad)}
        return grad