import datetime
from dateutil import tz
import sys
import os
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.model.tests.test_LBNL_bypass_coil_model import fcn


def test_estimator():
    stepSize = 60
    startTime = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2022, month=2, day=1, hour=12, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False, fcn=fcn)
    estimator = Estimator(model)

    coil = model.component_dict["coil+pump+valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]


    x0 = {coil: [1.5, 10, 15, 15, 15, 2000, 1, 1, 5000, 2000, 25000, 25000],
            fan: [0.08, -0.05, 1.31, -0.55, 0.89],
            controller: [0.1, 1, 0.001]}
    
    lb = {coil: [0.5, 3, 1, 1, 1, 500, 0.5, 0.5, 500, 500, 500, 500],
        fan: [-0.2, -0.7, -0.7, -0.7, 0.7],
        controller: [0, 0.0001, 0]}
    
    ub = {coil: [5, 15, 50, 50, 50, 3000, 3, 3, 8000, 5000, 50000, 50000],
        fan: [0.2, 1.4, 1.4, 1.4, 1],
        controller: [3, 3, 3]}


    targetParameters = {
                    coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dp1_nominal", "dpPump", "dpSystem"],
                    # coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dp1_nominal", "dpPump", "dpSystem"],
                    fan: ["c1", "c2", "c3", "c4", "f_total"],
                    controller: ["kp", "Ti", "Td"]}
    #################################################################################################################
    
    percentile = 2
    targetMeasuringDevices = {model.component_dict["coil outlet air temperature sensor"]: {"standardDeviation": 0.5/percentile},
                                model.component_dict["coil outlet water temperature sensor"]: {"standardDeviation": 0.5/percentile},
                                model.component_dict["fan power meter"]: {"standardDeviation": 80/percentile},
                                model.component_dict["valve position sensor"]: {"standardDeviation": 0.01/percentile},
                                model.component_dict["coil inlet water temperature sensor"]: {"standardDeviation": 0.5/percentile}}
    
    # Options for the PTEMCEE estimation algorithm. If the options argument is not supplied or None is supplied, default options are applied.  
    options = {"n_sample": 10000, #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 6, #Number of parallel chains/temperatures.
                "fac_walker": 4, #Scaling factor for the number of ensemble walkers per chain. Minimum is 2.
                "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                "walker_initialization": "uniform"} #Initialization of parameters - "gaussian" is also implemented
    
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

    #########################################
    # POST PROCESSING AND INFERENCE - MIGHT BE MOVED TO METHOD AT SOME POINT
    # Also see the "test_load_emcee_chain.py" script in this folder - implements plotting of the chain convergence, corner plots, etc. 
    with open(estimator.chain_savedir, 'rb') as handle:
        import pickle
        import numpy as np
        result = pickle.load(handle)
        result["chain.T"] = 1/result["chain.betas"]
    list_ = ["integratedAutoCorrelatedTime", "chain.jumps_accepted", "chain.jumps_proposed", "chain.swaps_accepted", "chain.swaps_proposed"]
    for key in list_:
        result[key] = np.array(result[key])
    burnin = 0 #Discard the first 0 samples as burnin - In this example we only have (n_sample*n_walker) samples, so we apply 0 burnin. Normally, the first many samples are discarded.
    print(result["chain.x"].shape)
    parameter_chain = result["chain.x"][burnin:,0,:,:]
    parameter_chain = parameter_chain.reshape((parameter_chain.shape[0]*parameter_chain.shape[1], parameter_chain.shape[2]))
    estimator.simulator.run_emcee_inference(model, parameter_chain, targetParameters, targetMeasuringDevices, startTime, endTime, stepSize, show=True) # Set show=True to plot
    #######################################################

if __name__=="__main__":
    test_estimator()