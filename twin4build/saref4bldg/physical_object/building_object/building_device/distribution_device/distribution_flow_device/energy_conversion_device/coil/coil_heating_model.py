from .coil import Coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants
from numpy import NaN

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class CoilHeatingModel(Coil):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

        logger.info("[Coil Heating Model] : Entered in Initialise Function")

        self.specificHeatCapacityAir = Constants.specificHeatCapacity["air"]

        self.input = {"airTemperatureIn": None,
                      "airTemperatureOutSetpoint": None,
                      "airFlowRate": None}
        self.output = {"Power": None, 
                       "airTemperatureOut": None}

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
         updates the input and output variables of the coil and calculates the power output and air temperature based on the input air temperature, air flow rate, and air temperature setpoint. 
         If the air flow rate is zero, the output power and air temperature are set to NaN
        '''
        self.output.update(self.input)
        tol = 1e-5
        if self.input["airFlowRate"]>tol:
            if self.input["airTemperatureIn"] < self.input["airTemperatureOutSetpoint"]:
                Q = self.input["airFlowRate"]*self.specificHeatCapacityAir*(self.input["airTemperatureOutSetpoint"] - self.input["airTemperatureIn"])
                self.output["airTemperatureOut"] = self.input["airTemperatureOutSetpoint"]
            else:
                Q = 0
            self.output["Power"] = Q
        else:
            self.output["airTemperatureOut"] = NaN
            self.output["Power"] = NaN

        logger.info("[Coil Heating Model] : Exited from Do step Function")

        


        