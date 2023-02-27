from .space_heater import SpaceHeater
from typing import Union
import twin4build.saref.measurement.measurement as measurement
import numpy as np
from scipy.optimize import least_squares
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
        self.output["outletWaterTemperature"] = [self.output["outletWaterTemperature"] for i in range(1)]
        self.input["supplyWaterTemperature"] = [self.input["supplyWaterTemperature"] for i in range(1)]
        
    
    def do_step(self, time=None, stepSize=None):
        n = 1
        self.input["supplyWaterTemperature"] = [self.input["supplyWaterTemperature"] for i in range(n)]
        for i in range(n):
            # K1 = (self.input["supplyWaterTemperature"]*self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue + self.heatTransferCoefficient*self.input["indoorTemperature"])/self.thermalMassHeatCapacity.hasValue + self.output["outletWaterTemperature"]/self.stepSize
            # K2 = 1/self.stepSize + (self.heatTransferCoefficient + self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue)/self.thermalMassHeatCapacity.hasValue
            K1 = (self.input["supplyWaterTemperature"][i]*self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue + self.heatTransferCoefficient/n*self.input["indoorTemperature"])/self.thermalMassHeatCapacity.hasValue/n + self.output["outletWaterTemperature"][i]/self.stepSize
            K2 = 1/self.stepSize + (self.heatTransferCoefficient/n + self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue)/self.thermalMassHeatCapacity.hasValue/n
            self.output["outletWaterTemperature"][i] = K1/K2
            if i!=n-1:
                self.input["supplyWaterTemperature"][i+1] = self.output["outletWaterTemperature"][i]
            # print(self.output["outletWaterTemperature"])

        #Two different ways of calculating heat consumption:
        # 1. Heat delivered to room
        # Q_r = self.heatTransferCoefficient*(self.output["outletWaterTemperature"]-self.input["indoorTemperature"])

        # 2. Heat delivered to radiator from heating system
        Q_r = self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue*(self.input["supplyWaterTemperature"][0]-self.output["outletWaterTemperature"][-1])

        self.output["Power"] = Q_r
        self.output["Energy"] = self.output["Energy"] + Q_r*self.stepSize/3600/1000

    def do_period(self, input):
        self.clear_report()
        self.output["Energy"] = 0
        self.output["outletWaterTemperature"] = [input["indoorTemperature"].iloc[0] for i in range(1)]
        
        for index, row in input.iterrows():            
            for key in input:
                self.input[key] = row[key]
            self.do_step()
            self.update_report()
        output_predicted = np.array(self.savedOutput["Energy"])
        return output_predicted

    def obj_fun(self, x, input, output):
        self.heatTransferCoefficient = x[0]
        self.thermalMassHeatCapacity.hasValue = x[1]
        output_predicted = self.do_period(input)
        res = output_predicted-output #residual of predicted vs measured
        print(f"Loss: {np.sum(res**2)}")
        return res

    def calibrate(self, input=None, output=None):
        x0 = np.array([self.heatTransferCoefficient, self.thermalMassHeatCapacity.hasValue])
        lb = [1, 1]
        ub = [1000, 500000]
        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output))
        self.heatTransferCoefficient, self.thermalMassHeatCapacity.hasValue = sol.x
        print(sol)