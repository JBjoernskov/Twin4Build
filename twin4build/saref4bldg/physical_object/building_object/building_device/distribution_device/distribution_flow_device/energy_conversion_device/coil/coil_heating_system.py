from .coil import Coil
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil as coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants
from numpy import NaN
from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class CoilHeatingSystem(coil.Coil):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

        logger.info("[Coil Heating Model] : Entered in Initialise Function")

        self.specificHeatCapacityAir = Constants.specificHeatCapacity["air"]

        self.input = {"inletAirTemperature": None,
                      "outletAirTemperatureSetpoint": None,
                      "airFlowRate": None}
        self.output = {"Power": None, 
                       "outletAirTemperature": None}
    
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

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
         updates the input and output variables of the coil and calculates the power output and air temperature based on the input air temperature, air flow rate, and air temperature setpoint. 
         If the air flow rate is zero, the output power and air temperature are set to NaN
        '''
        self.output.update(self.input)
        tol = 1e-5
        if self.input["airFlowRate"]>tol:
            if self.input["inletAirTemperature"] < self.input["outletAirTemperatureSetpoint"]:
                Q = self.input["airFlowRate"]*self.specificHeatCapacityAir*(self.input["outletAirTemperatureSetpoint"] - self.input["inletAirTemperature"])
                self.output["outletAirTemperature"] = self.input["outletAirTemperatureSetpoint"]
            else:
                Q = 0
            self.output["Power"] = Q
        else:
            # self.output["outletAirTemperature"] = self.input["outletAirTemperatureSetpoint"]
            self.output["outletAirTemperature"] = NaN
            self.output["Power"] = NaN

        logger.info("[Coil Heating Model] : Exited from Do step Function")

        


        