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
                specificHeatCapacityWater: Union[measurement.Measurement, None] = None, 
                stepSize = None, 
                **kwargs):
        
        SpaceHeater.__init__(self, **kwargs)
        assert isinstance(specificHeatCapacityWater, measurement.Measurement) or specificHeatCapacityWater is None, "Attribute \"specificHeatCapacityWater\" is of type \"" + str(type(specificHeatCapacityWater)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.specificHeatCapacityWater = specificHeatCapacityWater
        self.stepSize = stepSize

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
                       "Power": None,
                       "Energy": None}

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        self.initialParameters = {"Q_flow_nominal": self.outputCapacity.hasValue,
                                    "T_a_nominal": self.nominalSupplyTemperature+273.15,
                                    "T_b_nominal": self.nominalReturnTemperature+273.15,
                                    "T_start": self.output["outletTemperature"]+273.15,
                                    "VWat": 5.8e-6*abs(self.outputCapacity.hasValue),
                                    "mDry": 0.0263*abs(self.outputCapacity.hasValue)}
        FMUComponent.__init__(self, start_time=self.start_time, fmu_filename=self.fmu_filename)

    def do_period(self, input, stepSize=None):
        self.clear_report()        
        start_time = input.index[0].to_pydatetime()
        # print("start")
        for time, row in input.iterrows():
            time_seconds = (time.to_pydatetime()-start_time).total_seconds()
            # print(time_seconds)
            for key in input:
                self.input[key] = row[key]
            self.do_step(secondTime=time_seconds, stepSize=self.stepSize)
            self.update_report()

        # output_predicted = np.array(self.savedOutput["Energy"])/3600/1000
        output_predicted = np.array(self.savedOutput["Power"])
        return output_predicted

    def obj_fun(self, x, input, output, stepSize):
        parameters = {"VWat": x[0],
                      "mDry": x[1],
                      "n": x[2],
                      "Q_flow_nominal": x[3],
                      "T_b_nominal": x[4],
                      "T_a_nominal": x[5]}
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
        x0 = np.array([0.0029, 130, 1.24, 2000, 30+273.15, 45+273.15])
        lb = [0.001, 30, 1, 1500, 29+273.15, 44+273.15]
        ub = [0.01, 200, 2, 3000, 31+273.15, 46+273.15]
        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output, stepSize))
        self.reset()
        # parameters = {"VWat": sol.x[0],
        #               "mDry": sol.x[1]}
        # self.set_parameters(parameters)
        print(sol)

