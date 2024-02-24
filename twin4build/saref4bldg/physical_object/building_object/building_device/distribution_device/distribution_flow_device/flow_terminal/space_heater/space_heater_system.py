import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater as space_heater
from typing import Union
import twin4build.utils.constants as constants
import numpy as np
from scipy.optimize import least_squares
from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

logger.info("Space Heater Model")

class SpaceHeaterSystem(space_heater.SpaceHeater):
    def __init__(self,
                 heatTransferCoefficient=None,
                **kwargs):
        
        logger.info("[space heater model] : Entered in Intialise Function")

        super().__init__(**kwargs)
        self.specificHeatCapacityWater = constants.Constants.specificHeatCapacity["water"]
        self.nominalSupplyTemperature = int(self.temperatureClassification[0:2])
        self.nominalReturnTemperature = int(self.temperatureClassification[3:5])
        self.nominalRoomTemperature = int(self.temperatureClassification[6:])
        # self.heatTransferCoefficient = self.outputCapacity.hasValue/(self.nominalReturnTemperature-self.nominalRoomTemperature)
        self.heatTransferCoefficient = heatTransferCoefficient

        self.input = {"supplyWaterTemperature": None,
                      "waterFlowRate": None,
                      "indoorTemperature": None}
        self.output = {"outletWaterTemperature": None,
                       "Power": None,
                       "Energy": None}
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
                    stepSize=None):
        self.output["outletWaterTemperature"] = [self.output["outletWaterTemperature"] for i in range(10)]
        self.output["Energy"] = 0
        # self.input["supplyWaterTemperature"] = [self.input["supplyWaterTemperature"] for i in range(1)]

        logger.info("[space heater model] : Exited from Intialise Function")

        
    
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
            Advances the model by one time step and calculates the current outlet water temperature, power, and energy output.
        '''
        logger.info("[space heater model] : Entered in DoStep Function")
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

        self.output["Power"] = Q_r
        self.output["Energy"] = self.output["Energy"] + Q_r*stepSize/3600/1000

        logger.info("[space heater model] : Exited from DoStep Function")


    def do_period(self, input, stepSize=None):
        '''
            Runs the simulation for the given input over the entire period and returns the predicted energy output.
        '''

        logger.info("[space heater model] : Entered in DoPeriod Function")

        self.clear_report()
        self.output["outletWaterTemperature"] = input["indoorTemperature"][0]
        self.initialize()
        
        for index, row in input.iterrows():            
            for key in input:
                self.input[key] = row[key]
            self.do_step(stepSize=stepSize)
            self.update_report()
        output_predicted = np.array(self.savedOutput["Energy"])

        logger.info("[space heater model] : Exited from DoPeriod Function")

        return output_predicted

    def obj_fun(self, x, input, output, stepSize):
        '''
            Calculates the residual between the predicted and actual energy output for the given 
            input and output data, given the current model parameters.
        '''

        logger.info("[space heater model] : Entered in Object Function")

        self.heatTransferCoefficient = x[0]
        self.thermalMassHeatCapacity.hasValue = x[1]
        output_predicted = self.do_period(input, stepSize)
        res = output_predicted-output #residual of predicted vs measured
        self.loss = np.sum(res**2)
        print(f"MAE: {np.mean(np.abs(res))}")
        print(f"RMSE: {np.mean(res**2)**(0.5)}")

        logger.info("[space heater model] : Exitedd from Object Function")

        return res

    def calibrate(self, input=None, output=None, stepSize=None):
        '''
            Calibrates the model using the given input and output data, 
            optimizing the model parameters to minimize the residual between predicted and 
            actual energy output. Returns the optimized model parameters.
        '''

        logger.info("[space heater model] : Entered in Callibrate Function")


        assert input is not None
        assert output is not None
        assert stepSize is not None
        x0 = np.array([self.heatTransferCoefficient, self.thermalMassHeatCapacity.hasValue])
        lb = [1, 1]
        ub = [400, 50000000]
        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output, stepSize))
        self.heatTransferCoefficient, self.thermalMassHeatCapacity.hasValue = sol.x
        #print("+++++++++++++",self.heatTransferCoefficient, self.thermalMassHeatCapacity.hasValue)

        logger.info("[space heater model] : Exited from Callibrate Function")

        return (self.heatTransferCoefficient, self.thermalMassHeatCapacity.hasValue)