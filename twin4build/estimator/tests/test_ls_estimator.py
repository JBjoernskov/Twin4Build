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

        
        # self.x0 = {self.coil: [1.5, 10, 15, 15, 15, 1500],
        #     self.valve: [1.5, 1.5, 10000, 2000, 1e+6, 1e+6, 5],
        #     self.fan: [0.027828, 0.026583, -0.087069, 1.030920, 0.9],
        #     self.controller: [50, 50, 50]}
        
        # self.lb = {self.coil: [0.5, 3, 1, 1, 1, 500],
        #     self.valve: [0.5, 0.5, 100, 100, 100, 100, 0.1],
        #     self.fan: [-0.2, -0.7, -0.7, -0.7, 0.7],
        #     self.controller: [0.05, 1, 0]}
        
        # self.ub = {self.coil: [5, 15, 30, 30, 30, 3000],
        #     self.valve: [2, 5, 1e+5, 1e+5, 5e+6, 5e+6, 500],
        #     self.fan: [0.2, 1.4, 1.4, 1.4, 1],
        #     self.controller: [100, 100, 100]}

        # self.targetParameters = {
        #     self.coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
        #     self.valve: ["mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dpCoil_nominal", "dpPump", "dpSystem", "riseTime"],
        #     self.fan: ["c1", "c2", "c3", "c4", "f_total"],
        #     self.controller: ["kp", "Ti", "Td"]
        # }

        space_029A = model.component_dict["[029A][029A_space_heater]"]
        space_031A = model.component_dict["[031A][031A_space_heater]"]
        space_033A = model.component_dict["[033A][033A_space_heater]"]
        space_035A = model.component_dict["[035A][035A_space_heater]"]
        heating_controller_029A = model.component_dict["029A_temperature_heating_controller"]
        heating_controller_031A = model.component_dict["031A_temperature_heating_controller"]
        heating_controller_033A = model.component_dict["033A_temperature_heating_controller"]
        heating_controller_035A = model.component_dict["035A_temperature_heating_controller"]
        space_heater_valve_029A = model.component_dict["029A_space_heater_valve"]
        space_heater_valve_031A = model.component_dict["031A_space_heater_valve"]
        space_heater_valve_033A = model.component_dict["033A_space_heater_valve"]
        space_heater_valve_035A = model.component_dict["035A_space_heater_valve"]


        targetParameters = {"private": {"C_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                        "C_air": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                        "C_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                        "R_out": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 1e-5, "ub": 0.05},
                                        "R_in": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 1e-5, "ub": 0.05},
                                        "R_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 0.001, "ub": 0.05},
                                        "f_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1, "lb": 0, "ub": 2},
                                        "f_air": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1, "lb": 0, "ub": 2},
                                        "m_flow_nominal_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.0202, "lb": 1e-3, "ub": 0.5},
                                        "Q_flow_nominal_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1000, "lb": 100, "ub": 10000},
                                        "n_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1.24, "lb": 1, "ub": 2},
                                        "Kp": {"components": [heating_controller_029A, heating_controller_031A, heating_controller_033A, heating_controller_035A], "x0": 2e-4, "lb": 1e-5, "ub": 3},
                                        "Ti": {"components": [heating_controller_029A, heating_controller_031A, heating_controller_033A, heating_controller_035A], "x0": 3e-1, "lb": 1e-5, "ub": 3},
                                        "m_flow_nominal": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A], "x0": 0.0202, "lb": 1e-3, "ub": 0.3}, #0.0202
                                        "flowCoefficient.hasValue": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A], "x0": 8.7, "lb": 1, "ub": 100},
                                        "dpFixed_nominal": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A], "x0": 1e-6, "lb": 0, "ub": 10000}
                                        },
                            "shared": {"C_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                        "R_int": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 0.001, "ub": 0.05},}
                            }

        self.targetParameters = {"m1_flow_nominal": {"components": [self.coil], "x0": 1.5, "lb": 0.5, "ub": 5},
                                "m2_flow_nominal": {"components": [self.coil], "x0": 10, "lb": 3, "ub": 15},
                                "tau1": {"components": [self.coil], "x0": 15, "lb": 1, "ub": 30},
                                "tau2": {"components": [self.coil], "x0": 15, "lb": 1, "ub": 30},
                                "tau_m": {"components": [self.coil], "x0": 15, "lb": 1, "ub": 30},
                                "nominalUa.hasValue": {"components": [self.coil], "x0": 1500, "lb": 500, "ub": 3000},
                                "mFlowValve_nominal": {"components": [self.valve], "x0": 1.5, "lb": 0.5, "ub": 2},
                                "mFlowPump_nominal": {"components": [self.valve], "x0": 1.5, "lb": 0.5, "ub": 5},
                                "dpCheckValve_nominal": {"components": [self.valve], "x0": 10000, "lb": 100, "ub": 1e+5},
                                "dpCoil_nominal": {"components": [self.valve], "x0": 2000, "lb": 100, "ub": 1e+5},
                                "dpPump": {"components": [self.valve], "x0": 1e+6, "lb": 100, "ub": 5e+6},
                                "dpSystem": {"components": [self.valve], "x0": 1e+6, "lb": 100, "ub": 5e+6},
                                "riseTime": {"components": [self.valve], "x0": 5, "lb": 0.1, "ub": 500},
                                "c1": {"components": [self.fan], "x0": 0.027828, "lb": -0.2, "ub": 0.2},
                                "c2": {"components": [self.fan], "x0": 0.026583, "lb": -0.7, "ub": 1.4},
                                "c3": {"components": [self.fan], "x0": -0.087069, "lb": -0.7, "ub": 1.4},
                                "c4": {"components": [self.fan], "x0": 1.030920, "lb": -0.7, "ub": 1.4},
                                "f_total": {"components": [self.fan], "x0": 0.9, "lb": 0.7, "ub": 1},
                                "kp": {"components": [self.controller], "x0": 50, "lb": 0.05, "ub": 100},
                                "Ti": {"components": [self.controller], "x0": 50, "lb": 1, "ub": 100},
                                

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