import Saref4Syst

class SpaceHeaterModel(Saref4Syst.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def update_output(self):
        K1 = (self.input["supplyWaterTemperature"]*self.input["waterFlowRate"]*self.specificHeatCapacityWater + self.heatTransferCoefficient*self.input["indoorTemperature"])/self.thermalMassHeatCapacity + self.output["radiatorOutletTemperature"]/self.timeStep
        K2 = 1/self.timeStep + (self.heatTransferCoefficient + self.input["waterFlowRate"]*self.specificHeatCapacityWater)/self.thermalMassHeatCapacity
        self.output["radiatorOutletTemperature"] = K1/K2
        Q_r = self.input["waterFlowRate"]*self.specificHeatCapacityWater*(self.input["supplyWaterTemperature"]-self.output["radiatorOutletTemperature"])
        self.output["Power"] = Q_r
