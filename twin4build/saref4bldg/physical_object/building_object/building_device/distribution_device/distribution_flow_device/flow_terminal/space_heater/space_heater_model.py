from .space_heater import SpaceHeater
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.constants import Constants
import numpy as np
from scipy.optimize import least_squares
class SpaceHeaterModel(SpaceHeater):
    def __init__(self, 
                **kwargs):
        super().__init__(**kwargs)
        self.specificHeatCapacityWater = Constants.specificHeatCapacity["water"]
        self.nominalSupplyTemperature = int(self.temperatureClassification[0:2])
        self.nominalReturnTemperature = int(self.temperatureClassification[3:5])
        self.nominalRoomTemperature = int(self.temperatureClassification[6:])
        self.heatTransferCoefficient = self.outputCapacity.hasValue/(self.nominalReturnTemperature-self.nominalRoomTemperature)

        self.input = {"supplyWaterTemperature": None,
                      "waterFlowRate": None,
                      "indoorTemperature": None}
        self.output = {"outletWaterTemperature": None,
                       "Power": None,
                       "Energy": None}

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        self.output["outletWaterTemperature"] = [self.output["outletWaterTemperature"] for i in range(1)]
        self.output["Energy"] = 0
        # self.input["supplyWaterTemperature"] = [self.input["supplyWaterTemperature"] for i in range(1)]
        
    
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        n = 1
        self.input["supplyWaterTemperature"] = [self.input["supplyWaterTemperature"] for i in range(n)]
        for i in range(n):
            # K1 = (self.input["supplyWaterTemperature"]*self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue + self.heatTransferCoefficient*self.input["indoorTemperature"])/self.thermalMassHeatCapacity.hasValue + self.output["outletWaterTemperature"]/stepSize
            # K2 = 1/stepSize + (self.heatTransferCoefficient + self.input["waterFlowRate"]*self.specificHeatCapacityWater.hasValue)/self.thermalMassHeatCapacity.hasValue
            K1 = (self.input["supplyWaterTemperature"][i]*self.input["waterFlowRate"]*self.specificHeatCapacityWater + self.heatTransferCoefficient/n*self.input["indoorTemperature"])/(self.thermalMassHeatCapacity.hasValue/n) + self.output["outletWaterTemperature"][i]/stepSize
            K2 = 1/stepSize + (self.heatTransferCoefficient/n + self.input["waterFlowRate"]*self.specificHeatCapacityWater)/(self.thermalMassHeatCapacity.hasValue/n)
            self.output["outletWaterTemperature"][i] = K1/K2
            if i!=n-1:
                self.input["supplyWaterTemperature"][i+1] = self.output["outletWaterTemperature"][i]
            # print(self.output["outletWaterTemperature"])

        #Two different ways of calculating heat consumption:
        # 1. Heat delivered to room
        # Q_r = self.heatTransferCoefficient*(self.output["outletWaterTemperature"]-self.input["indoorTemperature"])

        # 2. Heat delivered to radiator from heating system
        Q_r = self.input["waterFlowRate"]*self.specificHeatCapacityWater*(self.input["supplyWaterTemperature"][0]-self.output["outletWaterTemperature"][-1])

        self.output["Power"] = Q_r
        self.output["Energy"] = self.output["Energy"] + Q_r*stepSize/3600/1000

    def do_period(self, input, stepSize=None):
        self.clear_report()
        self.output["outletWaterTemperature"] = input["indoorTemperature"][0]
        self.initialize()
        
        for index, row in input.iterrows():            
            for key in input:
                self.input[key] = row[key]
            self.do_step(stepSize=stepSize)
            self.update_report()
        output_predicted = np.array(self.savedOutput["Energy"])
        return output_predicted

    def obj_fun(self, x, input, output, stepSize):
        self.heatTransferCoefficient = x[0]
        self.thermalMassHeatCapacity.hasValue = x[1]
        output_predicted = self.do_period(input, stepSize)
        res = output_predicted-output #residual of predicted vs measured
        print(f"Loss: {np.sum(res**2)}")
        return res

    def calibrate(self, input=None, output=None, stepSize=None):
        assert input is not None
        assert output is not None
        assert stepSize is not None
        x0 = np.array([self.heatTransferCoefficient, self.thermalMassHeatCapacity.hasValue])
        lb = [1, 1]
        ub = [1000, 500000]
        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output, stepSize))
        self.heatTransferCoefficient, self.thermalMassHeatCapacity.hasValue = sol.x
        print(sol)