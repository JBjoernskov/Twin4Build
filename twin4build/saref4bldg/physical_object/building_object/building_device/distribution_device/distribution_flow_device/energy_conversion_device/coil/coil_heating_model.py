from .coil import Coil
from typing import Union
import twin4build.saref.measurement.measurement as measurement
class CoilHeatingModel(Coil):
    def __init__(self,
                specificHeatCapacityAir: Union[measurement.Measurement, None] = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(specificHeatCapacityAir, measurement.Measurement) or specificHeatCapacityAir is None, "Attribute \"specificHeatCapacityAir\" is of type \"" + str(type(specificHeatCapacityAir)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.specificHeatCapacityAir = specificHeatCapacityAir ###

    def initialize(self):
        pass

    def do_step(self, time=None, stepSize=None):
        if self.input["supplyAirTemperature"] < self.input["supplyAirTemperatureSetpoint"]:
            Q = self.input["airFlowRate"]*self.specificHeatCapacityAir.hasValue*(self.input["supplyAirTemperatureSetpoint"] - self.input["supplyAirTemperature"])
        else:
            Q = 0
        self.output["Power"] = Q


        