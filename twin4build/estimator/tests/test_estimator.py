import datetime
from dateutil import tz
import unittest
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.model.tests.test_LBNL_model import extend_model

class TestEstimator(unittest.TestCase):
    @unittest.skipIf(False, 'Currently not used')
    def test_estimator(self):
        stepSize = 60
        # startPeriod = datetime.datetime(year=2022, month=2, day=1, hour=1, minute=0, second=0) 
        # endPeriod = datetime.datetime(year=2022, month=2, day=15, hour=0, minute=0, second=0)

        model = Model(id="model", saveSimulationResult=True)
        model.load_model(infer_connections=False, extend_model=extend_model)
        estimator = Estimator(model)

        coil = model.component_dict["coil"]
        valve = model.component_dict["valve"]
        fan = model.component_dict["fan"]
        controller = model.component_dict["controller"]

        startPeriod_train = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endPeriod_train = datetime.datetime(year=2022, month=2, day=1, hour=21, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        startPeriod_test = datetime.datetime(year=2022, month=2, day=2, hour=0, minute=0, second=0)
        endPeriod_test = datetime.datetime(year=2022, month=2, day=15, hour=0, minute=0, second=0)

        x0 = {coil: [1.5, 10, 15, 15, 15, 1500],
            valve: [1.5, 1.5, 10000, 2000, 1e+6, 1e+6, 5],
            fan: [0.027828, 0.026583, -0.087069, 1.030920, 0.9],
            controller: [50, 50, 50]}
        
        lb = {coil: [0.5, 3, 1, 1, 1, 500],
            valve: [0.5, 0.5, 100, 100, 100, 100, 0.1],
            fan: [-0.2, -0.7, -0.7, -0.7, 0.7],
            controller: [0.05, 1, 0]}
        
        ub = {coil: [5, 15, 30, 30, 30, 3000],
            valve: [2, 5, 1e+5, 1e+5, 5e+6, 5e+6, 500],
            fan: [0.2, 1.4, 1.4, 1.4, 1],
            controller: [100, 100, 100]}


        targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
                                valve: ["mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dpCoil_nominal", "dpPump", "dpSystem", "riseTime"],
                                fan: ["c1", "c2", "c3", "c4", "f_total"],
                                controller: ["kp", "Ti", "Td"]}
        #################################################################################################################
        
        percentile = 2
        targetMeasuringDevices = {model.component_dict["coil outlet air temperature sensor"]: {"standardDeviation": 0.5/percentile},
                                    model.component_dict["coil outlet water temperature sensor"]: {"standardDeviation": 0.5/percentile},
                                    model.component_dict["fan power meter"]: {"standardDeviation": 80/percentile},
                                    model.component_dict["valve position sensor"]: {"standardDeviation": 0.01/percentile}}
        

        options = {"n_sample": 1, 
                    "n_temperature": 1, 
                    "fac_walker": 2,
                    "prior": "uniform",
                    "walker_initialization": "uniform"}
        
        estimator.estimate(x0=x0,
                            lb=lb,
                            ub=ub,
                            targetParameters=targetParameters,
                            targetMeasuringDevices=targetMeasuringDevices,
                            startPeriod=startPeriod_train,
                            endPeriod=endPeriod_train,
                            startPeriod_test=startPeriod_test,
                            endPeriod_test=endPeriod_test,
                            stepSize=stepSize,
                            algorithm="MCMC",
                            options=options
                            )
