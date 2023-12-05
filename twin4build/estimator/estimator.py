import multiprocessing
import math
import os
from tqdm import tqdm
from twin4build.simulator.simulator import Simulator
from twin4build.logger.Logging import Logging
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.uppath import uppath
import numpy as np
from ptemcee.sampler import Sampler, make_ladder
import datetime
import pickle
from twin4build.utils.mkdir_in_root import mkdir_in_root
from fmpy.fmi2 import FMICallException
from scipy.optimize import least_squares
logger = Logging.get_logger("ai_logfile")

#Multiprocessing is used and messes up the logger due to race conditions and access to write the logger file.
logger.disabled = True

class Estimator():
    def __init__(self,
                model=None):
        self.model = model
        self.simulator = Simulator(model)
        self.tol = 1e-5
        logger.info("[Estimator : Initialise Function]")
    
    def estimate(self,
                x0=None,
                lb=None,
                ub=None,
                y_scale=None,
                trackGradients=False,
                targetParameters=None,
                targetMeasuringDevices=None,
                n_initialization_steps=60,
                startTime=None,
                endTime=None,
                do_test=False,
                startTime_test=None,
                endTime_test=None,
                stepSize=None,
                verbose=False,
                algorithm="MCMC",
                options=None):
        
        allowed_algorithms = ["MCMC","least_squares"]
        assert algorithm in allowed_algorithms, f"The \"prior\" argument must be one of the following: {', '.join(allowed_algorithms)} - \"{algorithm}\" was provided."
        if do_test:
            assert do_test and (isinstance(startTime_test, datetime.datetime)==False or isinstance(endTime_test, datetime.datetime)==False), "Both startTime_test and endTime_test must be supplied if do_test is True"

        self.stepSize = stepSize
        self.verbose = verbose 
        self.simulator.get_simulation_timesteps(startTime, endTime, stepSize)
        self.n_initialization_steps = n_initialization_steps
        self.n_train = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
        self.n_init_train = self.n_train + self.n_initialization_steps
        self.startTime_train = startTime
        self.endTime_train = endTime
        self.startTime_test = startTime_test
        self.endTime_test = endTime_test

        
        self.actual_readings = self.simulator.get_actual_readings(startTime=self.startTime_train, endTime=self.endTime_train, stepSize=stepSize).iloc[self.n_initialization_steps:,:]
        self.min_actual_readings = self.actual_readings.min(axis=0)
        self.max_actual_readings = self.actual_readings.max(axis=0)
        self.x0 = np.array([val for lst in x0.values() for val in lst])
        self.lb = np.array([val for lst in lb.values() for val in lst])
        self.ub = np.array([val for lst in ub.values() for val in lst])

        if y_scale is None:
            self.y_scale = [1]*len(targetMeasuringDevices)
        else:
            self.y_scale = y_scale
        self.standardDeviation = np.array([el["standardDeviation"] for el in targetMeasuringDevices.values()])
        self.flat_component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
        self.flat_attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]
        self.trackGradients = trackGradients
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
        bounds = (self.lb,self.ub)
        self.n_obj_eval = 0
        self.best_loss = math.inf


        if algorithm == "MCMC":
            if options is None:
                options = {}
            self.run_emcee_estimation(**options)
        elif algorithm == "least_squares":
            self.run_least_squares_estimation(self.x0, self.lb, self.ub)

    def run_emcee_estimation(self, 
                             n_sample=10000, 
                             n_temperature=15, 
                             fac_walker=2, 
                             T_max=np.inf, 
                             n_cores=multiprocessing.cpu_count(),
                             prior="uniform",
                             walker_initialization="uniform"):
        assert n_cores>=1, "The argument \"n_cores\" must be larger than or equal to 1"
        assert fac_walker>=2, "The argument \"fac_walker\" must be larger than or equal to 2"
        allowed_priors = ["uniform", "gaussian"]
        allowed_walker_initializations = ["uniform", "gaussian"]
        assert prior in allowed_priors, f"The \"prior\" argument must be one of the following: {', '.join(allowed_priors)} - \"{prior}\" was provided."
        assert walker_initialization in allowed_walker_initializations, f"The \"walker_initialization\" argument must be one of the following: {', '.join(allowed_walker_initializations)} - \"{walker_initialization}\" was provided."
        assert np.all(self.x0>=self.lb), "The provided x0 must be larger than the provided lower bound lb"
        assert np.all(self.x0<=self.ub), "The provided x0 must be smaller than the provided upper bound ub"
        assert np.all(np.abs(self.x0-self.lb)>self.tol), f"The difference between x0 and lb must be larger than {str(self.tol)}"
        assert np.all(np.abs(self.x0-self.ub)>self.tol), f"The difference between x0 and ub must be larger than {str(self.tol)}"
        
        self.model.make_pickable()
        self.model.cache(stepSize=self.stepSize,
                        startTime=self.startTime_train,
                        endTime=self.endTime_train)

        ndim = len(self.flat_attr_list)
        n_walkers = int(ndim*fac_walker) #*4 #Round up to nearest even number and multiply by 2
        datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = str('{}_{}_{}'.format(self.model.id, datestr, '.pickle'))
        self.chain_savedir = mkdir_in_root(folder_list=["generated_files", "model_parameters", "chain_logs"], filename=filename)
        diff_lower = np.abs(self.x0-self.lb)
        diff_upper = np.abs(self.ub-self.x0)
        self.standardDeviation_x0 = np.minimum(diff_lower, diff_upper)/2 #Set the standard deviation such that around 95% of the values are within the bounds

        if prior=="uniform":
            logprior = self.uniform_logprior
        elif prior=="gaussian":
            logprior = self.gaussian_logprior
        loglike = self._loglike_exeption_wrapper

        if walker_initialization=="uniform":
            x0_start = np.random.uniform(low=self.lb, high=self.ub, size=(n_temperature, n_walkers, ndim))
        elif walker_initialization=="gaussian":
            x0_start = np.random.normal(loc=self.x0, scale=self.standardDeviation_x0, size=(n_temperature, n_walkers, ndim))
        
        print(f"Using number of cores: {n_cores}")
        print(f"Number of estimated parameters: {ndim}")
        print(f"Number of temperatures: {n_temperature}")
        print(f"Number of ensemble walkers per chain: {n_walkers}")
        adaptive = False if n_temperature==1 else True
        betas = np.array([1]) if n_temperature==1 else make_ladder(ndim, n_temperature, Tmax=T_max)
        pool = multiprocessing.Pool(n_cores, maxtasksperchild=100) #maxtasksperchild is set because the FMUs are leaking memory
        sampler = Sampler(n_walkers, 
                          ndim,
                          loglike,
                          logprior,
                          adaptive=adaptive,
                          betas=betas,
                          mapper=pool.imap) 
        chain = sampler.chain(x0_start)
        n_save_checkpoint = 50 if n_sample>=50 else 1
        result = {"integratedAutoCorrelatedTime": [],
                    "chain.jumps_accepted": [],
                    "chain.jumps_proposed": [],
                    "chain.swaps_accepted": [],
                    "chain.swaps_proposed": [],
                    "chain.logl": [],
                    "chain.logP": [],
                    "chain.x": [],
                    "chain.betas": [],
                    }

        for i, ensemble in tqdm(enumerate(chain.iterate(n_sample)), total=n_sample):
            result["integratedAutoCorrelatedTime"].append(chain.get_acts())
            result["chain.jumps_accepted"].append(chain.jumps_accepted.copy())
            result["chain.jumps_proposed"].append(chain.jumps_proposed.copy())
            result["chain.swaps_accepted"].append(chain.swaps_accepted.copy())
            result["chain.swaps_proposed"].append(chain.swaps_proposed.copy())
        
            if i % n_save_checkpoint == 0:
                result["chain.logl"] = chain.logl[:i]
                result["chain.logP"] = chain.logP[:i]
                result["chain.x"] = chain.x[:i]
                result["chain.betas"] = chain.betas[:i]
                with open(self.chain_savedir, 'wb') as handle:
                    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pool.close()

    def get_solution(self):
        sol_dict = {}
        sol_dict["MSE"] = self.monitor.get_MSE()
        sol_dict["RMSE"] = self.monitor.get_RMSE()
        sol_dict["n_obj_eval"] = self.n_obj_eval
        for component, attr_list in self.targetParameters.items():
            sol_dict[component.id] = []
            for attr in attr_list:
                sol_dict[component.id].append(rgetattr(component, attr))
        return sol_dict

    def _obj_fun_MCMC_exception_wrapper(self, x, data):
        try:
            loss = self._obj_fun_MCMC(x, data)
        except FMICallException as inst:
            loss = 10e+10*np.ones((len(self.targetMeasuringDevices)))
        return loss
    
    def _sim_func_MCMC(self, x):
        # Set parameters for the model
        self.set_parameters_from_array(x)
        self.simulator.simulate(self.model,
                                stepSize=self.stepSize,
                                startTime=self.startTime_train,
                                endTime=self.endTime_train,
                                trackGradients=self.trackGradients,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)
        y = np.zeros((self.actual_readings.shape[0], len(self.targetMeasuringDevices)))
        for i, measuring_device in enumerate(self.targetMeasuringDevices):
            simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
            y[:,i] = simulation_readings
        return y
        
    def _obj_fun_MCMC(self, x, data):
        '''
            This function calculates the loss (residual) between the predicted and measured output using 
            the least_squares optimization method. It takes in an array x representing the parameters to be optimized, 
            sets these parameter values in the model and simulates the model to obtain the predictions. 
        '''
        # Set parameters for the model
        self.set_parameters_from_array(x)
        self.simulator.simulate(self.model,
                                stepSize=self.stepSize,
                                startTime=self.startTime_train,
                                endTime=self.endTime_train,
                                trackGradients=self.trackGradients,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)


        res = np.zeros((self.actual_readings.iloc[:,0].size, len(self.targetMeasuringDevices)))
        for j, (y_scale, measuring_device) in enumerate(zip(self.y_scale, self.targetMeasuringDevices)):
            
            simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
            actual_readings = self.actual_readings[measuring_device.id].to_numpy()
            res[:,j] = (simulation_readings-actual_readings)/y_scale
        self.n_obj_eval+=1
        self.loss = np.sum(res**2, axis=0)
        return self.loss/self.T

    def _loglike_exeption_wrapper(self, theta):
        try:
            loglike = self._loglike(theta)
        except FMICallException as inst:
            loglike = -1e+10
        return loglike

    def _loglike(self, theta):
        '''
            This function calculates the loss (residual) between the predicted and measured output using 
            the least_squares optimization method. It takes in an array x representing the parameters to be optimized, 
            sets these parameter values in the model and simulates the model to obtain the predictions. 
        '''
        # Set parameters for the model
        # x = theta[:-n_sigma]
        # sigma = theta[-n_sigma:]

        outsideBounds = np.any(theta<self.lb) or np.any(theta>self.ub)
        # if outsideBounds: #####################################################h
        #     return -np.inf
        
        self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
        self.simulator.simulate(self.model,
                                stepSize=self.stepSize,
                                startTime=self.startTime_train,
                                endTime=self.endTime_train,
                                trackGradients=self.trackGradients,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)

        res = np.zeros((self.actual_readings.iloc[:,0].size, len(self.targetMeasuringDevices)))
        for j, (y_scale, measuring_device) in enumerate(zip(self.y_scale, self.targetMeasuringDevices)):
            simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
            actual_readings = self.actual_readings[measuring_device.id].to_numpy()
            res[:,j] = (simulation_readings-actual_readings)/y_scale
        self.n_obj_eval+=1
        ss = np.sum(res**2, axis=0)
        loglike = -0.5*np.sum(ss/(self.standardDeviation**2))

        if self.verbose:
            print("=================")
            with np.printoptions(precision=3, suppress=True):
                print(f"Theta: {theta}")
                print(f"Sum of squares: {ss}")
                print(f"Sigma: {self.standardDeviation}")
                print(f"Loglikelihood: {loglike}")
            print("=================")
            print("")
        
        return loglike
    
    def uniform_logprior(self, theta):
        outsideBounds = np.any(theta<self.lb) or np.any(theta>self.ub)
        p = np.sum(np.log(1/(self.ub-self.lb)))
        return -np.inf if outsideBounds else p
    
    def gaussian_logprior(self, theta):
        const = np.log(1/(self.standardDeviation_x0*np.sqrt(2*np.pi)))
        p = -0.5*((self.x0-theta)/self.standardDeviation_x0)**2
        return np.sum(const+p)

    def run_least_squares_estimation(self, x0,lb,ub):
        
        assert np.all(self.x0>=self.lb), "The provided x0 must be larger than the provided lower bound lb"
        assert np.all(self.x0<=self.ub), "The provided x0 must be smaller than the provided upper bound ub"
        assert np.all(np.abs(self.x0-self.lb)>self.tol), f"The difference between x0 and lb must be larger than {str(self.tol)}"
        assert np.all(np.abs(self.x0-self.ub)>self.tol), f"The difference between x0 and ub must be larger than {str(self.tol)}"
        
        self.model.make_pickable()
        self.model.cache(stepSize=self.stepSize,
                        startTime=self.startTime_train,
                        endTime=self.endTime_train)

        datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = str('{}_{}_{}'.format(self.model.id, datestr, '.pickle'))
        self.ls_res_savedir = mkdir_in_root(folder_list=["generated_files", "model_parameters", "least_squares_result"], filename=filename)

        ls_result = least_squares(self._res_fun_least_squares_exception_wrapper, x0, bounds=(lb, ub), verbose=2) #Change verbose to 2 to see the optimization progress

        with open(self.ls_res_savedir, 'wb') as handle:
            pickle.dump(ls_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return ls_result
    
    def _res_fun_least_squares(self, theta):
        '''
            This function calculates the loss (residual) between the predicted and measured output using 
            the least_squares optimization method. It takes in an array x representing the parameters to be optimized, 
            sets these parameter values in the model and simulates the model to obtain the predictions. 
            return: A one-dimensional array of residuals.

        '''
        self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
        self.simulator.simulate(self.model,
                                stepSize=self.stepSize,
                                startTime=self.startTime_train,
                                endTime=self.endTime_train,
                                trackGradients=self.trackGradients,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)

        res = np.zeros((self.actual_readings.iloc[:,0].size, len(self.targetMeasuringDevices)))
        # Populate the residual matrix
        for j, (y_scale, measuring_device) in enumerate(zip(self.y_scale, self.targetMeasuringDevices)):
            simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
            actual_readings = self.actual_readings[measuring_device.id].to_numpy()
            res[:,j] = (simulation_readings-actual_readings)/y_scale
        
        # Flatten the residual matrix for the least_squares optimization method
        res = res.flatten()
        self.n_obj_eval+=1

        return res
    
    def _res_fun_least_squares_exception_wrapper(self, theta):
        try:
            res = self._res_fun_least_squares(theta)
        except FMICallException as inst:
            res = 10e+10*np.ones((len(self.targetMeasuringDevices)))
        return res