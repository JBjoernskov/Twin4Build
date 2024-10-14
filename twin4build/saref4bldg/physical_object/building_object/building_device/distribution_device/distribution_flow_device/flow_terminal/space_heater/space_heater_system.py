import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater as space_heater
import twin4build.utils.constants as constants
import numpy as np
from scipy.optimize import least_squares
import twin4build.utils.input_output_types as tps

class SpaceHeaterSystem(space_heater.SpaceHeater):
    def __init__(self,
                 heatTransferCoefficient=None,
                **kwargs):
        super().__init__(**kwargs)
        self.specificHeatCapacityWater = constants.Constants.specificHeatCapacity["water"]
        self.nominalSupplyTemperature = int(self.temperatureClassification.hasValue[0:2])
        self.nominalReturnTemperature = int(self.temperatureClassification.hasValue[3:5])
        self.nominalRoomTemperature = int(self.temperatureClassification.hasValue[6:])
        # self.heatTransferCoefficient = self.outputCapacity.hasValue/(self.nominalReturnTemperature-self.nominalRoomTemperature)
        self.heatTransferCoefficient = heatTransferCoefficient

        self.input = {"supplyWaterTemperature": tps.Scalar(),
                      "waterFlowRate": tps.Scalar(),
                      "indoorTemperature": tps.Scalar()}
        self.output = {"outletWaterTemperature": tps.Scalar(),
                       "Power": tps.Scalar(),
                       "Energy": tps.Scalar()}
        self._config = {"parameters": ["heatTransferCoefficient",
                                       "nominalSupplyTemperature",
                                       "nominalReturnTemperature",
                                       "nominalRoomTemperature",
                                       "thermalMassHeatCapacity.hasValue"]}

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
        self.output["outletWaterTemperature"] = [self.output["outletWaterTemperature"] for i in range(10)]
        self.output["Energy"] = 0
        
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
            Advances the model by one time step and calculates the current outlet water temperature, power, and energy output.
        '''
        n = 10
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
        # Q_r = sum([self.heatTransferCoefficient/n*(self.output["outletWaterTemperature"][i]-self.input["indoorTemperature"]) for i in range(n)])

        # 2. Heat delivered to radiator from heating system
        Q_r = self.input["waterFlowRate"]*self.specificHeatCapacityWater*(self.input["supplyWaterTemperature"][0]-self.output["outletWaterTemperature"][-1])

        self.output["Power"].set(Q_r)
        self.output["Energy"].set(self.output["Energy"] + Q_r*stepSize/3600/1000)
