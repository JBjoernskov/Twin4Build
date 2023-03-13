from .coil import Coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants
from numpy import NaN
class CoilHeatingModel(Coil):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
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
        


        