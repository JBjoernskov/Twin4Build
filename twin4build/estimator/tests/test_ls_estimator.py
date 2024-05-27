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
import twin4build as tb

def fcn(self):

    s = tb.ScheduleSystem(id="s",
                          weekDayRulesetDict={"ruleset_default_value": 0,
                            "ruleset_start_minute": [0],
                            "ruleset_end_minute": [0],
                            "ruleset_start_hour": [6],
                            "ruleset_end_hour": [12],
                            "ruleset_value": [1]
                        },)
    d = tb.DamperSystem(id="d", a=5, nominalAirFlowRate=0.02)
    sensor = tb.SensorSystem(id="sensor")
    self.add_connection(s, d, "scheduleValue", "damperPosition")

class TestLSEstimator(unittest.TestCase):

    def setUpModelAndEstimator(self):
        """
        Set up the model, estimator, and common parameters used in both tests.
        """
        self.stepSize = 60
        self.startTime = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        self.endTime = datetime.datetime(year=2022, month=2, day=1, hour=21, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

        self.model = tb.Model(id="test_ls_estimator", saveSimulationResult=True)
        self.model.load_model(infer_connections=False, fcn=fcn)

        d = self.model.component_dict["d"]
        


        targetParameters = {"private": {"C_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                        }}


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