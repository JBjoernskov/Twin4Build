from .coil import Coil
import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil as coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants
import numpy as np

class CoilHeatingSystem(coil.Coil):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.specificHeatCapacityAir = Constants.specificHeatCapacity["air"]

        self.input = {"inletAirTemperature": None,
                      "outletAirTemperatureSetpoint": None,
                      "airFlowRate": None}
        self.output = {"Power": None,
                       "outletAirTemperature": None}
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
            self.output["outletAirTemperature"] = np.nan
            self.output["Power"] = np.nan

        


        