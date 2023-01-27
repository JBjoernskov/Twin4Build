from .space_heater import SpaceHeater
from typing import Union
import twin4build.saref.measurement.measurement as measurement
class SpaceHeaterModel(SpaceHeater):
    def __init__(self, 
                specificHeatCapacityWater: Union[measurement.Measurement, None] = None, 
                stepSize = None, 
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(specificHeatCapacityWater, measurement.Measurement) or specificHeatCapacityWater is None, "Attribute \"specificHeatCapacityWater\" is of type \"" + str(type(specificHeatCapacityWater)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.specificHeatCapacityWater = specificHeatCapacityWater
        self.stepSize = stepSize

        self.nominalSupplyTemperature = int(self.temperatureClassification[0:2])
        self.nominalReturnTemperature = int(self.temperatureClassification[3:5])
        self.nominalRoomTemperature = int(self.temperatureClassification[6:])
        self.heatTransferCoefficient = self.outputCapacity.hasValue/(self.nominalReturnTemperature-self.nominalRoomTemperature)

    def initialize(self):
        pass
    
    def do_step(self, time=None, stepSize=None):
        K1 = (self.input["supplyWaterTemperature"]*self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue + self.heatTransferCoefficient*self.input["indoorTemperature"])/self.thermalMassHeatCapacity.hasValue + self.output["outletWaterTemperature"]/self.stepSize
        K2 = 1/self.stepSize + (self.heatTransferCoefficient + self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue)/self.thermalMassHeatCapacity.hasValue
        self.output["outletWaterTemperature"] = K1/K2

        #Two different ways of calculating heat consumption:
        # 1. Heat delivered to room
        Q_r = self.heatTransferCoefficient*(self.output["outletWaterTemperature"]-self.input["indoorTemperature"])

        # 2. Heat delivered to radiator from heating system
        # Q_r = self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue*(self.input["supplyWaterTemperature"]-self.output["radiatorOutletTemperature"])

        self.output["Power"] = Q_r
        self.output["Energy"] = self.output["Energy"] + Q_r*self.stepSize/3600/1000
