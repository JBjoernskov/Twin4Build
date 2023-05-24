from .space_heater import SpaceHeater
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.fmu.fmu_component import FMUComponent
import os
from twin4build.utils.uppath import uppath
import numpy as np
from scipy.optimize import least_squares
import pandas as pd
class SpaceHeaterModel(FMUComponent, SpaceHeater):
    def __init__(self, 
                stepSize = None, 
                **kwargs):
        SpaceHeater.__init__(self, **kwargs)
        self.nominalSupplyTemperature = int(self.temperatureClassification[0:2])
        self.nominalReturnTemperature = int(self.temperatureClassification[3:5])
        self.nominalRoomTemperature = int(self.temperatureClassification[6:])
        self.heatTransferCoefficient = self.outputCapacity.hasValue/(self.nominalReturnTemperature-self.nominalRoomTemperature)

        self.start_time = 0
        fmu_filename = "Radiator.fmu"
        self.fmu_filename = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)


        self.input = {"supplyWaterTemperature": None,
                      "waterFlowRate": None,
                      "indoorTemperature": None}
        self.output = {"outletWaterTemperature": None,
                       "PowerToRadiator": None,
                       "EnergyToRadiator": None}

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        '''
            This function initializes the FMU component by setting the start_time and fmu_filename attributes, 
            and then sets the parameters for the FMU model.
        '''
        self.initialParameters = {"Q_flow_nominal": self.outputCapacity.hasValue,
                                  "TAir_nominal": self.nominalRoomTemperature+273.15,
                                    "T_a_nominal": self.nominalSupplyTemperature+273.15,
                                    "T_b_nominal": self.nominalReturnTemperature+273.15,
                                    "T_start": self.output["outletTemperature"]+273.15,
                                    "VWat": 5.8e-6*abs(self.outputCapacity.hasValue)}

        FMUComponent.__init__(self, start_time=self.start_time, fmu_filename=self.fmu_filename)

    def do_period(self, input, stepSize=None):
        '''
            This function performs a simulation period for the FMU model with the given input dataframe and optional stepSize.
            It iterates through each row of the input dataframe and sets the input parameters for the FMU model accordingly. 
            It then runs the simulation with the given stepSize and saves the output to a list.
            Finally, it returns the predicted output of the simulation.
        '''
        self.clear_report()        
        start_time = input.index[0].to_pydatetime()
        # print("start")
        for time, row in input.iterrows():
            time_seconds = (time.to_pydatetime()-start_time).total_seconds()
            # print(time_seconds)

            for key in input:
                self.input[key] = row[key]
            self.do_step(secondTime=time_seconds, stepSize=stepSize)
            self.update_report()

        # output_predicted = np.array(self.savedOutput["Energy"])/3600/1000
        output_predicted = np.array(self.savedOutput["PowerToRadiator"])
        return output_predicted

    def obj_fun(self, x, input, output, stepSize):
        '''
            This function calculates the loss (residual) between the predicted and measured output using 
            the least_squares optimization method. It takes in an array x representing the parameter to be optimized, 
            input and output dataframes representing the input and measured output values, respectively. 
            It uses the do_period function to predict the output values with the given x parameter and calculates the 
            residual between the predicted and measured output. It returns the residual.
        '''
        parameters = {"VWat": x[0],
                      "n": x[1],
                      "Q_flow_nominal": x[2]}
        self.initialParameters.update(parameters)
        self.reset()
        # parameters = {"VWat": x[0],
        #               "mDry": x[1]}
        # self.set_parameters(parameters)

        output_predicted = self.do_period(input, stepSize=stepSize)
        res = output_predicted-output #residual of predicted vs measured
        print(f"Loss: {np.sum(res**2)}")
        return res

    def calibrate(self, input=None, output=None, stepSize=None):
        '''
            This function performs calibration using the obj_fun function and the least_squares 
            optimization method with the given input and output. It initializes an array x0 representing the 
            initial parameter value, sets bounds for the parameter optimization, and then uses least_squares
            to find the optimal value for the Radiator.UAEle parameter. 
            Finally, it sets the optimal Radiator.UAEle parameter based on the calibration results.
        '''
        
        x0 = np.array([0.29, 1.24, 2600])
        lb = [0.0001, 1, 100]
        ub = [1, 2, 2601]

        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output, stepSize))
        self.reset()
        # parameters = {"VWat": sol.x[0],
        #               "mDry": sol.x[1]}
        # self.set_parameters(parameters)
        print(sol)

