"""
An Estimator class will be implemented here.
The estimator estimates the parameters of the components models 
based on a period with measurements from the actual building. 

Pytorch automatic differentiation could be used.
For some period:

0. Initialize parameters as torch.Tensor with "<parameter>.requires_grad=true". 
    All inputs are provided as torch.Tensor with "<input>.requires_grad=false". 
    Selected model parameters can be "frozen" by setting "<parameter>.requires_grad=false"

Repeat 1-3 until convergence or stop criteria

1. Run simulation with inputs
2. Calculate loss based on predicted and measured values
3. Backpropagate and do step to update parameters




Plot Euclidian Distance from final solution and amount of data and accuracy/error
"""
# from memory_profiler import profile
# from memory_profiler import profile
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
import seaborn as sns
from twin4build.simulator.simulator import Simulator
from twin4build.logger.Logging import Logging
# from twin4build.monitor.monitor import Monitor
# from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import twin4build.utils.plot.plot as plot
# from scipy.optimize import basinhopping
# from scipy.optimize._numdiff import approx_derivative
# from scipy.optimize import approx_fprime
# import pygad
import numpy as np
# import pymcmcstat
import matplotlib.dates as mdates
# from pymcmcstat import mcmcplot as mcp
# from pymcmcstat.MCMC import MCMC
# from pymcmcstat.ParallelMCMC import ParallelMCMC
# from pymcmcstat import propagation as up
# from pymcmcstat.chain import ChainProcessing
# from pymcmcstat.structures.ResultsStructure import ResultsStructure
# from pymcmcstat.ParallelMCMC import load_parallel_simulation_results

# import emcee
from ptemcee.sampler import Sampler, make_ladder

# from bayes_opt import BayesianOptimization

# import pymc as pm
# import pytensor
# import arviz as az
# import pytensor.tensor as pt

import matplotlib.pyplot as plt
from fmpy.fmi2 import FMICallException

import datetime
import pickle

logger = Logging.get_logger("ai_logfile")

#Multiprocessing is used and messes up the logger due to race conditions and access to write the logger file.
logger.disabled = True


# pytensor.config.optimizer="None"
# pytensor.config.exception_verbosity="high"

# class LogLikeWithGrad(pt.Op):

#     itypes = [pt.dvector]  # expects a vector of parameter values when called
#     otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

#     def __init__(self, loglike, lb, ub):
#         """
#         Initialise with various things that the function requires. Below
#         are the things that are needed in this particular example.

#         Parameters
#         ----------
#         loglike:
#             The log-likelihood (or whatever) function we've defined
#         data:
#             The "observed" data that our log-likelihood function takes in
#         x:
#             The dependent variable (aka 'x') that our model requires
#         sigma:
#             The noise standard deviation that out function requires.
#         """

#         # add inputs as class attributes
#         self.loglike = loglike

#         # initialise the gradient Op (below)
#         self.logpgrad = LogLikeGrad(loglike, lb, ub)

#     def perform(self, node, inputs, outputs):
#         (theta,) = inputs  # this will contain my variables
#         logl = self.loglike(theta)
#         outputs[0][0] = np.array(logl)  # output the log-likelihood
#         print("AFTER outputs")

#     def grad(self, inputs, g):
#         # the method that calculates the gradients - it actually returns the
#         # vector-Jacobian product - g[0] is a vector of parameter values
#         (theta,) = inputs  # our parameters
#         a = [g[0] * self.logpgrad(theta)]
#         return a

# def my_loglike(theta):
#     return theta

# class LogLikeGrad(pt.Op):
#     """
#     This Op will be called with a vector of values and also return a vector of
#     values - the gradients in each dimension.
#     """

#     itypes = [pt.dvector]
#     otypes = [pt.dvector]

#     def __init__(self, loglike, lb, ub):
#         """
#         Initialise with various things that the function requires. Below
#         are the things that are needed in this particular example.

#         Parameters
#         ----------
#         data:
#             The "observed" data that our log-likelihood function takes in
#         x:
#             The dependent variable (aka 'x') that our model requires
#         sigma:
#             The noise standard deviation that out function requires.
#         """

#         # add inputs as class attributes
#         self.loglike = loglike
#         self.lb = lb
#         self.ub = ub

#     def perform(self, node, inputs, outputs):
#         (theta,) = inputs
#         # calculate gradients
#         outputs[0][0] = theta#approx_fprime(theta, self.loglike)

class Estimator():
    def __init__(self,
                model=None):
        self.model = model
        self.simulator = Simulator(model)
        logger.info("[Estimator : Initialise Function]")
    
    # def on_generation(self, ga_instance):
    #     print("")
    #     print("================================================")
    #     ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    #     ga_instance.logger.info("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    #     print("================================================")
    #     print("")

    # def get_GAoptions(self):
    #     fitness_function = self.obj_fun_GA_exception_wrapper
    #     num_generations = 20
    #     num_parents_mating = 16

    #     sol_per_pop = 81
    #     num_genes = len(self.x0)

    #     gene_space = [{"low": self.lb[i], "high": self.ub[i]} for i in range(num_genes)]

    #     parent_selection_type = "sss"
    #     # keep_parents = 1
    #     keep_elitism = 1
    #     crossover_type = "scattered"

    #     mutation_type = "random"
    #     mutation_percent_genes = 5
    #     GAoptions = {"num_generations": num_generations,
    #                    "num_parents_mating": num_parents_mating,
    #                    "fitness_func": fitness_function,
    #                    "sol_per_pop": sol_per_pop,
    #                    "num_genes": num_genes,
    #                    "gene_space": gene_space,
    #                    "parent_selection_type": parent_selection_type,
    #                    "keep_elitism": keep_elitism,
    #                    "crossover_type": crossover_type,
    #                    "mutation_type": mutation_type,
    #                    "mutation_percent_genes": mutation_percent_genes,
    #                    "on_generation": self.on_generation,
    #                    "parallel_processing": ["process", 8]}
    #     return GAoptions

    def estimate(self,
                x0=None,
                lb=None,
                ub=None,
                y_scale=None,
                trackGradients=True,
                targetParameters=None,
                targetMeasuringDevices=None,
                initialization_steps=None,
                startPeriod=None,
                endPeriod=None,
                startPeriod_test=None,
                endPeriod_test=None,
                stepSize=None):
        if startPeriod_test is None or endPeriod_test is None:
            test_period_supplied = False
            assert startPeriod_test is None and endPeriod_test is None, "Both startPeriod_test and endPeriod_test must be supplied"
        else:
            test_period_supplied = True
        self.stepSize = stepSize
        self.simulator.get_simulation_timesteps(startPeriod, endPeriod, stepSize)
        self.n_initialization_steps = 60
        if test_period_supplied:
            self.n_train = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            self.n_init_train = self.n_train + self.n_initialization_steps
            self.startPeriod_train = startPeriod
            self.endPeriod_train = endPeriod
            self.startPeriod_test = startPeriod_test
            self.endPeriod_test = endPeriod_test
        else:
            split_train = 0.6
            self.n_estimate = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            self.n_train = math.ceil(self.n_estimate*split_train)
            self.n_init_train = self.n_train + self.n_initialization_steps
            self.n_test = self.n_estimate-self.n_train
            
            self.startPeriod_train = startPeriod
            self.endPeriod_train = self.simulator.dateTimeSteps[self.n_initialization_steps+self.n_train]
            self.startPeriod_test = self.simulator.dateTimeSteps[self.n_initialization_steps+self.n_train+1]
            self.endPeriod_test = endPeriod
        
        self.actual_readings = self.simulator.get_actual_readings(startPeriod=self.startPeriod_train, endPeriod=self.endPeriod_train, stepSize=stepSize).iloc[self.n_initialization_steps:,:]
        self.min_actual_readings = self.actual_readings.min(axis=0)
        self.max_actual_readings = self.actual_readings.max(axis=0)
        # self.min_max_diff = self.max_actual_readings-self.min_actual_readings
        # self.actual_readings_min_max_normalized = (self.actual_readings-self.min_actual_readings)/(self.max_actual_readings-self.min_actual_readings)
        self.x0 = np.array([val for lst in x0.values() for val in lst])
        self.lb = np.array([val for lst in lb.values() for val in lst])
        self.ub = np.array([val for lst in ub.values() for val in lst])
        # x_scale = [val for lst in x_scale.values() for val in lst]
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
        self.run_emcee_estimation(n_sample=1, 
                                    n_temperature=1, 
                                    fac_walker=2, 
                                    n_cores=1,
                                    prior="gaussian",
                                    walker_initialization="gaussian")

    def run_emcee_inference(self, model, parameter_chain, targetParameters, targetMeasuringDevices, startPeriod, endPeriod, stepSize):
        simulator = Simulator(model)
        n_samples_max = 100
        n_samples = parameter_chain.shape[0] if parameter_chain.shape[0]<n_samples_max else n_samples_max #100
        sample_indices = np.random.randint(parameter_chain.shape[0], size=n_samples)
        parameter_chain_sampled = parameter_chain[sample_indices]

        component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
        attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

        simulator.get_simulation_timesteps(startPeriod, endPeriod, stepSize)
        time = simulator.dateTimeSteps
        actual_readings = simulator.get_actual_readings(startPeriod=startPeriod, endPeriod=endPeriod, stepSize=stepSize)

        n_cores = 5#multiprocessing.cpu_count()
        pool = multiprocessing.Pool(n_cores)
        pbar = tqdm(total=len(sample_indices))
        cached_predictions = {}
        def _sim_func(simulator, parameter_set):
            try:
                # Set parameters for the model
                hashed = parameter_set.data.tobytes()
                if hashed not in cached_predictions:
                    simulator.model.set_parameters_from_array(parameter_set, component_list, attr_list)
                    simulator.simulate(model,
                                            stepSize=stepSize,
                                            startPeriod=startPeriod,
                                            endPeriod=endPeriod,
                                            trackGradients=False,
                                            targetParameters=targetParameters,
                                            targetMeasuringDevices=targetMeasuringDevices,
                                            show_progress_bar=False)
                    y = np.zeros((len(time), len(targetMeasuringDevices)))
                    for i, measuring_device in enumerate(targetMeasuringDevices):
                        simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))
                        y[:,i] = simulation_readings
                    cached_predictions[hashed] = y
                else:
                    y = cached_predictions[hashed]
                
                pbar.update(1)

            except FMICallException as inst:
                y = None
            return y
        
        y_list = [_sim_func(simulator, parameter_set) for parameter_set in parameter_chain_sampled]
        y_list = [el for el in y_list if el is not None]
        predictions = [[] for i in range(len(targetMeasuringDevices))]
        predictions_w_obs_error = [[] for i in range(len(targetMeasuringDevices))]

        for y in y_list:
            standardDeviation = np.array([el["standardDeviation"] for el in targetMeasuringDevices.values()])
            y_w_obs_error = y + np.random.normal(0, standardDeviation, size=y.shape)
            for col in range(len(targetMeasuringDevices)):
                predictions[col].append(y[:,col])
                predictions_w_obs_error[col].append(y_w_obs_error[:,col])



        intervals = []
        for col in range(len(targetMeasuringDevices)):
            intervals.append({"credible": np.array(predictions[col]),
                            "prediction": np.array(predictions_w_obs_error[col])})

        
        ydata = []
        for measuring_device, value in targetMeasuringDevices.items():
            ydata.append(actual_readings[measuring_device.id].to_numpy())
        ydata = np.array(ydata).transpose()
        self.plot_emcee_inference(intervals, time, ydata)

    def plot_emcee_inference(self, intervals, time, ydata):
        colors = sns.color_palette("deep")
        blue = colors[0]
        orange = colors[1]
        green = colors[2]
        red = colors[3]
        purple = colors[4]
        brown = colors[5]
        pink = colors[6]
        grey = colors[7]
        beis = colors[8]
        sky_blue = colors[9]
        plot.load_params()

        facecolor = tuple(list(beis)+[0.5])
        edgecolor = tuple(list((0,0,0))+[0.1])
        # cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
        # cmap = sns.color_palette("Dark2", as_cmap=True)
        # cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
        cmap = sns.dark_palette((50,50,90), input="husl", reverse=True, n_colors=10)# 0,0,74
        data_display = dict(
            marker=None,
            color=red,
            linewidth=1,
            linestyle="solid",
            mfc='none',
            label='Physical')
        model_display = dict(
            color="black",
            linestyle="dashed", 
            label=f"Mode",
            linewidth=1
            )
        interval_display = dict(alpha=None, edgecolor=edgecolor, linestyle="solid")
        ciset = dict(
            limits=[99],
            colors=[cmap[2]],
            # cmap=cmap,
            alpha=0.5)
        
        piset = dict(
            limits=[99],
            colors=[cmap[0]],
            # cmap=cmap,
            alpha=0.2)

        fig, axes = plt.subplots(len(intervals), ncols=1)
        for ii, (interval, ax) in enumerate(zip(intervals, axes)):
            fig, ax = plot.plot_intervals(intervals=interval,
                                            time=time,
                                            ydata=ydata[:,ii],
                                            data_display=data_display,
                                            model_display=model_display,
                                            interval_display=interval_display,
                                            ciset=ciset,
                                            piset=piset,
                                            fig=fig,
                                            ax=ax,
                                            adddata=True,
                                            addlegend=False,
                                            addmodel=True,
                                            addcredible=True,
                                            addprediction=True,
                                            figsize=(7, 5))
            myFmt = mdates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(myFmt)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.3), prop={'size': 12}, ncol=4)
        axes[-1].set_xlabel("Time")
        self.inference_fig = fig
        self.inference_axes = axes

    # @profile
    def run_emcee_estimation(self, 
                             n_sample=10000, 
                             n_temperature=15, 
                             fac_walker=2, 
                             T_max=np.inf, 
                             n_cores=multiprocessing.cpu_count(),
                             prior="uniform",
                             walker_initialization="uniform"):
        allowed_priors = ["uniform", "gaussian"]
        allowed_walker_initializations = ["uniform", "gaussian"]
        assert prior in allowed_priors, f"The \"prior\" argument must be one of the following: {', '.join(allowed_priors)} - \"{prior}\" was provided."
        assert walker_initialization in allowed_walker_initializations, f"The \"walker_initialization\" argument must be one of the following: {', '.join(allowed_walker_initializations)} - \"{walker_initialization}\" was provided."
        tol = 1e-5
        assert np.all(self.x0>=self.lb), "The provided x0 must be larger than the provided lower bound lb"
        assert np.all(self.x0<=self.ub), "The provided x0 must be smaller than the provided upper bound ub"
        assert np.all(np.abs(self.x0-self.lb)>tol), f"The difference between x0 and lb must be larger than {str(tol)}"
        assert np.all(np.abs(self.x0-self.ub)>tol), f"The difference between x0 and ub must be larger than {str(tol)}"
        self.model.make_pickable()
        ndim = len(self.flat_attr_list)
        n_walkers = int(ndim*fac_walker) #*4 #Round up to nearest even number and multiply by 2
        datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        savedir = str('{}_{}'.format(datestr, 'chain_log.pickle'))
        savedir = os.path.join(uppath(os.path.abspath(__file__), 1), "chain_logs", savedir)

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
        adaptive = False if n_temperature==1 else True
        betas = np.array([1]) if n_temperature==1 else make_ladder(ndim, n_temperature, Tmax=T_max)
        sampler = Sampler(n_walkers, 
                          ndim,
                          loglike,
                          logprior,
                          adaptive=adaptive,
                          betas=betas,
                          mapper=multiprocessing.Pool(n_cores, maxtasksperchild=100).imap) #maxtasksperchild is set because the FMUs are leaking memory
        for el in dir(self.model):
            print(el)
        chain = sampler.chain(x0_start)
        n_save_checkpoint = 50
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
                with open(savedir, 'wb') as handle:
                    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
                                startPeriod=self.startPeriod_train,
                                endPeriod=self.endPeriod_train,
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
                                startPeriod=self.startPeriod_train,
                                endPeriod=self.endPeriod_train,
                                trackGradients=self.trackGradients,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)

        # Non-zero flow filtering has to be constant size. Otherwise, scipy throws an error.
        waterFlowRate = np.array(self.model.component_dict["Supply air temperature setpoint"].savedInput["returnAirTemperature"])
        airFlowRate = np.array(self.model.component_dict["fan flow meter"].savedOutput["airFlowRate"])
        tol = 1e-4
        self.no_flow_mask = np.logical_and(waterFlowRate>tol,airFlowRate>tol)[self.n_initialization_steps:]

        self.n_adjusted = np.sum(self.no_flow_mask==True)
        res = np.zeros((self.n_adjusted, len(self.targetMeasuringDevices)))
        for j, (y_scale, measuring_device) in enumerate(zip(self.y_scale, self.targetMeasuringDevices)):
            
            simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
            actual_readings = self.actual_readings[measuring_device.id].to_numpy()
            # simulation_readings_min_max_normalized = (simulation_readings-self.min_actual_readings[measuring_device.id])/(self.max_actual_readings[measuring_device.id]-self.min_actual_readings[measuring_device.id])
            # res+=np.abs(simulation_readings-actual_readings)
            # res[k:k+self.n_adjusted] = simulation_readings_min_max_normalized[self.no_flow_mask]-self.actual_readings_min_max_normalized[measuring_device.id][self.no_flow_mask]
            res[:,j] = (simulation_readings[self.no_flow_mask]-actual_readings[self.no_flow_mask])/y_scale
        self.n_obj_eval+=1
        self.loss = np.sum(res**2, axis=0)
        # if self.loss<self.best_loss:
        #     self.best_loss = self.loss
        #     self.best_parameters = x

        # print("=================")
        # with np.printoptions(precision=3, suppress=True):
        #     print(x)
        #     print(f"Loss: {self.loss}")
        # # print("Best Loss: {:0.2f}".format(self.best_loss))
        # print("=================")
        # print("")
        return self.loss/self.T
    

    def _loglike_exeption_wrapper(self, theta):
        '''
            This function calculates the loss (residual) between the predicted and measured output using 
            the least_squares optimization method. It takes in an array x representing the parameters to be optimized, 
            sets these parameter values in the model and simulates the model to obtain the predictions. 
        '''
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
        verbose = True

        outsideBounds = np.any(theta<self.lb) or np.any(theta>self.ub)
        # if outsideBounds: #####################################################h
        #     return -np.inf
        
        self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
        self.simulator.simulate(self.model,
                                stepSize=self.stepSize,
                                startPeriod=self.startPeriod_train,
                                endPeriod=self.endPeriod_train,
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

        if verbose:
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

    
    