from .space_heater import SpaceHeater
from typing import Union
import twin4build.saref.measurement.measurement as measurement
class SpaceHeaterModel(SpaceHeater):
    def __init__(self, 
                specificHeatCapacityWater: Union[measurement.Measurement, None] = None, 
                timeStep = None, 
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(specificHeatCapacityWater, measurement.Measurement) or specificHeatCapacityWater is None, "Attribute \"specificHeatCapacityWater\" is of type \"" + str(type(specificHeatCapacityWater)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.specificHeatCapacityWater = specificHeatCapacityWater
        self.timeStep = timeStep
        self.heatTransferCoefficient = self.outputCapacity.hasValue/(self.input["supplyWaterTemperature"]*0.8-self.output["radiatorOutletTemperature"]) 
    
    def update_output(self):
        K1 = (self.input["supplyWaterTemperature"]*self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue + self.heatTransferCoefficient*self.input["indoorTemperature"])/self.thermalMassHeatCapacity.hasValue + self.output["radiatorOutletTemperature"]/self.timeStep
        K2 = 1/self.timeStep + (self.heatTransferCoefficient + self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue)/self.thermalMassHeatCapacity.hasValue
        self.output["radiatorOutletTemperature"] = K1/K2
        Q_r = self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue*(self.input["supplyWaterTemperature"]-self.output["radiatorOutletTemperature"])
        self.output["Power"] = Q_r
        self.output["Energy"] = self.output["Energy"] + Q_r*self.timeStep/3600/1000
