import datetime
from dateutil import tz
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
from twin4build.model.tests.test_LBNL_bypass_coil_model import fcn


def test_estimator():
    np.random.seed(5)

    stepSize = 60
    startTime1 = datetime.datetime(year=2022, month=2, day=1, hour=10, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime1 = datetime.datetime(year=2022, month=2, day=1, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    startTime2 = datetime.datetime(year=2022, month=2, day=2, hour=10, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime2 = datetime.datetime(year=2022, month=2, day=2, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    startTime = [startTime1]#, startTime2]
    endTime = [endTime1]#, endTime2]
    stepSize = [stepSize]#, stepSize]

    model = Model(id="test_estimator_wbypass", saveSimulationResult=True)
    model.load_model(infer_connections=False, fcn=fcn)
    

    coil = model.component_dict["coil_pump_valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]


    
    # p0 = [p + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]


    x0 = {coil: [1.5, 10, 15, 15, 15, 2000, 1, 1, 40, 500, 2000, 5000, 5000],
            fan: [0.08, -0.05, 1.31, -0.55, 0.89],
            controller: [0.1, 1, 0.001]}
    
    lb = {coil: [0.5, 3, 1, 1, 1, 500, 0.5, 0.5, 5, 0, 500, 500, 0],
        fan: [-0.2, -0.7, -0.7, -0.7, 0.7],
        controller: [0, 0.0001, 0]}
    
    ub = {coil: [3, 15, 50, 50, 20, 3000, 3, 3, 150, 10000, 3000, 10000, 10000],
        fan: [0.2, 1.4, 1.4, 1.4, 1],
        controller: [3, 3, 3]}
    
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240111_164945_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp

    


    targetParameters = {
                    coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "KvCheckValve", "dpFixedSystem", "dp1_nominal", "dpPump", "dpSystem"],
                    # coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dp1_nominal", "dpPump", "dpSystem"],
                    fan: ["c1", "c2", "c3", "c4", "f_total"],
                    controller: ["kp", "Ti", "Td"]}
    #################################################################################################################
    
    percentile = 2
    targetMeasuringDevices = {model.component_dict["coil outlet air temperature sensor"]: {"standardDeviation": 0.5/percentile, "scale_factor": 1},
                                model.component_dict["coil outlet water temperature sensor"]: {"standardDeviation": 0.5/percentile, "scale_factor": 1},
                                model.component_dict["fan power meter"]: {"standardDeviation": 80/percentile, "scale_factor": 1000},
                                model.component_dict["valve position sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["coil inlet water temperature sensor"]: {"standardDeviation": 0.5/percentile, "scale_factor": 1}}
    
    
    


    # Options for the PTEMCEE estimation algorithm. If the options argument is not supplied or None is supplied, default options are applied.  
    options = {"n_sample": 500, #500 #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 20, #20 #Number of parallel chains/temperatures.
                "fac_walker": 4, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                "walker_initialization": "uniform", #Initialization of parameters - "gaussian" is also implemented
                # "n_cores": 1,
                "T_max": 1e+4,
                "add_noise_model": False,
                }
    estimator = Estimator(model)
    estimator.estimate(x0=x0,
                        lb=lb,
                        ub=ub,
                        targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        algorithm="MCMC",
                        options=options #
                        )
    # estimator.chain_savedir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "models", "test_estimator_wbypass", "model_parameters", "estimation_results", "chain_logs", "20240307_130004.pickle")
    model.load_chain_log(estimator.chain_savedir)
    options = {"n_sample": 250, #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 10, #Number of parallel chains/temperatures.
                "fac_walker": 4, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                # "model_prior": "sample_gaussian", #Prior distribution - "gaussian" is also implemented
                "prior": "uniform",
                "model_walker_initialization": "sample", #Prior distribution - "gaussian" is also implemented
                "noise_walker_initialization": "uniform",
                # "n_cores": 4,
                "T_max": np.inf,
                "add_noise_model": True,
                }
    estimator.estimate(x0=x0,
                        lb=lb,
                        ub=ub,
                        targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        algorithm="MCMC",
                        options=options #
                        )

    model.load_chain_log(estimator.chain_savedir)
    options = {"n_sample": 20000, #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 1, #Number of parallel chains/temperatures.
                "fac_walker": 4, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                "prior": "uniform",
                "walker_initialization": "sample_hypercube", #Prior distribution - "gaussian" is also implemented
                #"n_cores": 4,
                "T_max": np.inf,
                "add_noise_model": True,
                }
    estimator.estimate(x0=x0,
                        lb=lb,
                        ub=ub,
                        targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        algorithm="MCMC",
                        options=options #
                        )

    # parameter_chain = model.chain_log["chain.x"]
    # parameter_chain = parameter_chain[:,0,:,:]
    # parameter_chain = parameter_chain.reshape((parameter_chain.shape[0]*parameter_chain.shape[1], parameter_chain.shape[2]))
    # estimator.simulator.run_emcee_inference(model, parameter_chain, targetParameters, targetMeasuringDevices, startTime, endTime, stepSize, show=True) # Set show=True to plot

    #########################################
    # POST PROCESSING AND INFERENCE - MIGHT BE MOVED TO METHOD AT SOME POINT
    # Also see the "test_load_emcee_chain.py" script in this folder - implements plotting of the chain convergence, corner plots, etc. 
    # with open(estimator.chain_savedir, 'rb') as handle:
    #     import pickle
    #     import numpy as np
    #     result = pickle.load(handle)
    #     result["chain.T"] = 1/result["chain.betas"]
    # list_ = ["integratedAutoCorrelatedTime", "chain.jumps_accepted", "chain.jumps_proposed", "chain.swaps_accepted", "chain.swaps_proposed"]
    # for key in list_:
    #     result[key] = np.array(result[key])
    # burnin = 0 #Discard the first 0 samples as burnin - In this example we only have (n_sample*n_walker) samples, so we apply 0 burnin. Normally, the first many samples are discarded.
    # print(result["chain.x"].shape)
    # parameter_chain = result["chain.x"][burnin:,0,:,:]
    # parameter_chain = parameter_chain.reshape((parameter_chain.shape[0]*parameter_chain.shape[1], parameter_chain.shape[2]))
    # estimator.simulator.run_emcee_inference(model, parameter_chain, targetParameters, targetMeasuringDevices, startTime, endTime, stepSize, show=True) # Set show=True to plot
    #######################################################

if __name__=="__main__":
    test_estimator()