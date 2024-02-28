import datetime
from dateutil import tz
import unittest
import pickle
import sys
import os
import numpy as np
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.model.tests.test_LBNL_model import fcn

class TestLSEstimator(unittest.TestCase):

    def setUpModelAndEstimator(self):
        """
        Set up the model, estimator, and common parameters used in both tests.
        """
        self.stepSize = 60
        self.startTime = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        self.endTime = datetime.datetime(year=2022, month=2, day=1, hour=21, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

        self.model = Model(id="test_ls_estimator", saveSimulationResult=True)
        self.model.load_model(infer_connections=False, fcn=fcn)
        self.estimator = Estimator(self.model)

        self.coil = self.model.component_dict["coil"]
        self.valve = self.model.component_dict["valve"]
        self.fan = self.model.component_dict["fan"]
        self.controller = self.model.component_dict["controller"]

        
        self.x0 = {self.coil: [1.5, 10, 15, 15, 15, 1500],
            self.valve: [1.5, 1.5, 10000, 2000, 1e+6, 1e+6, 5],
            self.fan: [0.027828, 0.026583, -0.087069, 1.030920, 0.9],
            self.controller: [50, 50, 50]}
        
        self.lb = {self.coil: [0.5, 3, 1, 1, 1, 500],
            self.valve: [0.5, 0.5, 100, 100, 100, 100, 0.1],
            self.fan: [-0.2, -0.7, -0.7, -0.7, 0.7],
            self.controller: [0.05, 1, 0]}
        
        self.ub = {self.coil: [5, 15, 30, 30, 30, 3000],
            self.valve: [2, 5, 1e+5, 1e+5, 5e+6, 5e+6, 500],
            self.fan: [0.2, 1.4, 1.4, 1.4, 1],
            self.controller: [100, 100, 100]}

        self.targetParameters = {
            self.coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
            self.valve: ["mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dpCoil_nominal", "dpPump", "dpSystem", "riseTime"],
            self.fan: ["c1", "c2", "c3", "c4", "f_total"],
            self.controller: ["kp", "Ti", "Td"]
        }

        percentile = 2
        self.targetMeasuringDevices = {
            self.model.component_dict["coil outlet air temperature sensor"]: {"standardDeviation": 0.5/percentile, "scale_factor": 1},
            self.model.component_dict["coil outlet water temperature sensor"]: {"standardDeviation": 0.5/percentile, "scale_factor": 1},
            self.model.component_dict["fan power meter"]: {"standardDeviation": 80/percentile, "scale_factor": 1000},
            self.model.component_dict["valve position sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1}
        }

    def ls_inference(self, ls_res_savedir):

        self.setUpModelAndEstimator()
        with open(ls_res_savedir, 'rb') as handle:
            result = pickle.load(handle)
            ls_params = result.x
            #print("Estimated parameters: ", ls_params)
        self.estimator.simulator.run_ls_inference(self.model, ls_params, self.targetParameters, self.targetMeasuringDevices, self.startTime, self.endTime, self.stepSize, show=False) # Set show=True to plot
        


    @unittest.skipIf(False, 'Currently not used')
    def test_ls_estimator(self):

        self.setUpModelAndEstimator()
        
        self.estimator.estimate(x0=self.x0,
                            lb=self.lb,
                            ub=self.ub,
                            targetParameters=self.targetParameters,
                            targetMeasuringDevices=self.targetMeasuringDevices,
                            startTime=self.startTime,
                            endTime=self.endTime,
                            stepSize=self.stepSize,
                            algorithm="least_squares",
                            options=None #
                            )

        #########################################
        # POST PROCESSING AND INFERENCE 
        self.ls_inference(self.estimator.ls_res_savedir)


if __name__=="__main__":
    t = TestLSEstimator()
    t.test_ls_estimator()

    # #If the resulting parameters have been cached, we can load them and run the inference only:
    # ls_res_savedir = r"generated_files\model_parameters\least_squares_result\model_20231204_164648_.pickle"
    # t.ls_inference(ls_res_savedir)