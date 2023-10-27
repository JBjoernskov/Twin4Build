from .coil import Coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class CoilCoolingSystem(Coil):
    def __init__(self,
                **kwargs):
        
        logger.info("[Coil Cooling Model] : Entered in Initialise Function")

        super().__init__(**kwargs)
        self.specificHeatCapacityAir = Constants.specificHeatCapacity["air"]

        self.input = {"airTemperatureIn": None,
                      "airTemperatureOutSetpoint": None,
                      "airFlowRate": None}
        self.output = {"Power": None, 
                       "airTemperatureOut": None}

        
        logger.info("[Coil Cooling Model] : Exited from Initialise Function")

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
            simulate the cooling behavior of a coil. It calculates the heat transfer rate based on the 
            input temperature difference and air flow rate, and sets the output air temperature to the desired setpoint.
        '''
        
        self.output.update(self.input)
        if self.input["airTemperatureIn"] > self.input["airTemperatureOutSetpoint"]:
            Q = self.input["airFlowRate"]*self.specificHeatCapacityAir*(self.input["airTemperatureIn"] - self.input["airTemperatureOutSetpoint"])
        else:
            Q = 0
        self.output["Power"] = Q
        self.output["airTemperatureOut"] = self.input["airTemperatureOutSetpoint"]


        