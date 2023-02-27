from .space_heater import SpaceHeater
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.utils.fmu.fmu_component import FMUComponent
import os
from twin4build.utils.uppath import uppath
import numpy as np
from scipy.optimize import least_squares
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

    def initialize(self):
        FMUComponent.__init__(self, start_time=self.start_time, fmu_filename=self.fmu_filename)

    def do_period(self, input):
        self.clear_report()

        for time, row in input.iterrows():            
            for key in input:
                self.input[key] = row[key]
            self.do_step(time=time, stepSize=self.stepSize)
            self.update_report()

        output_predicted = np.array(self.savedOutput["Energy"])/3600/1000
        return output_predicted

    def obj_fun(self, x, input, output):
        self.reset()
        parameters = {"Radiator.UAEle": x[0]}
        self.set_parameters(parameters)
        output_predicted = self.do_period(input)
        res = output_predicted-output #residual of predicted vs measured
        print(f"Loss: {np.sum(res**2)}")
        return res

    def calibrate(self, input=None, output=None):
        x0 = np.array([1])
        lb = [0.1]
        ub = [1]
        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output))
        #  = sol.x
        parameters = {"Radiator.UAEle": sol.x[0]}
        self.set_parameters(parameters)
        print(sol)
