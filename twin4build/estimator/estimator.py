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
import math
import os
import sys
import seaborn as sns
from twin4build.simulator.simulator import Simulator
from twin4build.logger.Logging import Logging
from twin4build.monitor.monitor import Monitor
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
from twin4build.utils.plot.plot import get_fig_axes, load_params
from scipy.optimize import basinhopping
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import approx_fprime
# import pygad
import numpy as np
# import pymcmcstat
import matplotlib.dates as mdates
from pymcmcstat import mcmcplot as mcp
from pymcmcstat.MCMC import MCMC
from pymcmcstat.ParallelMCMC import ParallelMCMC
from pymcmcstat import propagation as up
from pymcmcstat.chain import ChainProcessing
from pymcmcstat.structures.ResultsStructure import ResultsStructure
from pymcmcstat.ParallelMCMC import load_parallel_simulation_results

from bayes_opt import BayesianOptimization

import pymc as pm
import pytensor
import arviz as az
import pytensor.tensor as pt

import matplotlib.pyplot as plt
from fmpy.fmi2 import FMICallException

import datetime

logger = Logging.get_logger("ai_logfile")

#Multiprocessing is used and messes up the logger due to race conditions and acces to write the logger file.
logger.disabled = True


pytensor.config.optimizer="None"
pytensor.config.exception_verbosity="high"

class LogLikeWithGrad(pt.Op):

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, lb, ub):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.loglike = loglike

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(loglike, lb, ub)

    def perform(self, node, inputs, outputs):
        (theta,) = inputs  # this will contain my variables
        logl = self.loglike(theta)
        outputs[0][0] = np.array(logl)  # output the log-likelihood
        print("AFTER outputs")

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        a = [g[0] * self.logpgrad(theta)]
        return a

def my_loglike(theta):
    return theta

class LogLikeGrad(pt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, loglike, lb, ub):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.loglike = loglike
        self.lb = lb
        self.ub = ub

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        # calculate gradients
        outputs[0][0] = theta#approx_fprime(theta, self.loglike)

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
        self.min_max_diff = self.max_actual_readings-self.min_actual_readings
        self.actual_readings_min_max_normalized = (self.actual_readings-self.min_actual_readings)/(self.max_actual_readings-self.min_actual_readings)
        self.x0 = [val for lst in x0.values() for val in lst]
        self.lb = [val for lst in lb.values() for val in lst]
        self.ub = [val for lst in ub.values() for val in lst]
        # x_scale = [val for lst in x_scale.values() for val in lst]
        self.y_scale = y_scale

        self.flat_component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
        self.flat_attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

        self.trackGradients = trackGradients
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
        bounds = (self.lb,self.ub)
        self.n_obj_eval = 0
        self.best_loss = math.inf

        if self.trackGradients:
            sol = least_squares(self.obj_fun, x0=self.x0, bounds=bounds, jac=self.get_jacobian, verbose=2, x_scale="jac", loss="linear", max_nfev=10000000, args=(stepSize, self.startPeriod_train, self.endPeriod_train)) #, x_scale="jac"
        else:
            ### LS
            # sol = least_squares(self.obj_fun, x0=self.x0, bounds=bounds, x_scale="jac", args=(stepSize, self.startPeriod_train, self.endPeriod_train))
            

            ### GA
            # ga_instance = pygad.GA(**self.get_GAoptions())
            # ga_instance.run()
            # solution, solution_fitness, solution_idx = ga_instance.best_solution()

            # ga_instance.plot_fitness()
            # print("Parameters of the best solution : {solution}".format(solution=solution))
            # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
            
            
            ### MCMC
            self.run_MCMC_estimation()
            # self.run_bayes_optimization()
            # self.run_pyMC_estimation()
            self.set_parameters()
            # print(sol)
        # try:
        #     if self.trackGradients:
        #         sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, jac=self.get_jacobian, verbose=2, x_scale="jac", loss="linear", max_nfev=10000000, xtol=1e-15, args=(stepSize, self.startPeriod_train, self.endPeriod_train)) #, x_scale="jac"
        #     else:
        #         sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, max_nfev=10000000, x_scale="jac", args=(stepSize, self.startPeriod_train, self.endPeriod_train))
        #     print(sol)
        # except Exception as e:
        #     print(e)
        #     self.set_parameters(self.best_parameters)
        #     print(self.best_parameters)

        
        self.monitor = Monitor(self.model)
        self.monitor.monitor(startPeriod=self.startPeriod_test,
                            endPeriod=self.endPeriod_test,
                            stepSize=stepSize,
                            do_plot=False)

    def run_bayes_optimization(self):

        pbounds = {}
        for attr,lb,ub in zip(self.flat_attr_list, self.lb, self.ub):
            pbounds[attr] = (lb, ub)

        optimizer = BayesianOptimization(
            f=self._loglike_exeption_wrapper,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(3000)

        print(optimizer.max)

    def run_pyMC_estimation(self):
        # create our Op
        lb = self.lb + [0 for i in range(len(self.targetMeasuringDevices))]
        ub = self.ub + [np.inf for i in range(len(self.targetMeasuringDevices))]
        logl = LogLikeWithGrad(self._loglike_exeption_wrapper, self.lb, self.ub)
        # logl = LogLikeWithGrad(my_loglike, lb, ub)

        # use PyMC to sampler from log-likelihood
        with pm.Model() as opmodel:
            theta = []
            for attr,lb,ub in zip(self.flat_attr_list, self.lb, self.ub):
                p = pm.Uniform(attr, lower=lb, upper=ub)
                theta.append(p)

            # for measuring_device in self.targetMeasuringDevices:
            #     p = pm.HalfNormal("sigma -- " + measuring_device.id, sigma=1)
            #     theta.append(p)

            theta = pt.as_tensor_variable(theta)

            # use a Potential
            pm.Potential("likelihood", logl(theta))

            idata_grad = pm.sample(10, tune=0, cores=1, chains=1)
        _ = az.plot_trace(idata_grad)
        _ = az.plot_posterior(idata_grad)
        # plt.show()

    def run_MCMC_estimation(self):
        load = False
        loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "chain_logs", "20230823_155959_chain_log")
        datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
        savedir = str('{}_{}'.format(datestr, 'chain_log'))
        savedir = os.path.join(uppath(os.path.abspath(__file__), 1), "chain_logs", savedir)
        print(savedir)
        y = []
        for measuring_device in self.targetMeasuringDevices:
            y.append(self.actual_readings[measuring_device.id].to_numpy())
        y = np.array(y).transpose()
        # initialize MCMC object
        mcstat = MCMC()
        # initialize data structure 
        mcstat.data.add_data_set(x=self.simulator.dateTimeSteps[self.n_initialization_steps:],
                                y=y)
        # initialize parameter array
        #theta = [0.5, 0.03, 0.1, 10, 0.02, 1.14, 0.77, 1.3, 10]
        # add model parameters


        for name,x0,lb,ub in zip(self.flat_attr_list,self.x0,self.lb,self.ub):
            mcstat.parameters.add_model_parameter(name=name, theta0=x0, minimum=lb, maximum=ub)

        # Generate options
        nsimu = 500
        mcstat.simulation_options.define_simulation_options(
            nsimu=nsimu, updatesigma=True, savedir=savedir, save_to_json=True, save_to_txt=True, savesize=100, doram=False, alphatarget=0.234, etaparam=0.7, ntry=5, method="dram")
        # Define model object:
        mcstat.model_settings.define_model_settings(
            sos_function=self._obj_fun_MCMC_exception_wrapper)
        
        parallel_MCMC = ParallelMCMC()
        parallel_MCMC.setup_parallel_simulation(mcset=mcstat,num_cores=4,num_chain=4)
        results_list = []
        if load==False:
            parallel_MCMC.run_parallel_simulation()
            parallel_MCMC.display_individual_chain_statistics()
            for mcmc_instance in parallel_MCMC.parmc:
                results_list.append(mcmc_instance.simulation_results.results)
        else:
            results = load_parallel_simulation_results(loaddir, extension='txt')
            for i in range(parallel_MCMC.num_chain):
                parallel_MCMC.parmc[i].simulation_results = ResultsStructure()
                parallel_MCMC.parmc[i].simulation_results.results = results[i]
                results_list.append(results[i])
                print(results[i].keys())

            # results_list = [ChainProcessing.read_in_savedir_files(savedir, extension='txt')]
            
        # display chain stats
        # 

        self.colors = sns.color_palette("deep")
        blue = self.colors[0]
        orange = self.colors[1]
        green = self.colors[2]
        red = self.colors[3]
        purple = self.colors[4]
        brown = self.colors[5]
        pink = self.colors[6]
        grey = self.colors[7]
        beis = self.colors[8]
        sky_blue = self.colors[9]
        load_params()

        nparam = len(self.flat_attr_list)
        ncols = 3
        nrows = math.ceil(nparam/ncols)
        fig_trace, axes_trace = plt.subplots(nrows=nrows, ncols=ncols)
        fig_trace.set_size_inches((7, 5))

        fig_loss, ax_loss = plt.subplots()
        fig_loss.set_size_inches((7, 5))

        for i_chain, results in enumerate(results_list):
            # results = mcmc_instance.simulation_results.results
            print(results.keys())
            burnin = 0#int(nsimu/2)
            sschain = np.sum(np.array(results['sschain'])[burnin:, :], axis=1)
            chain = np.array(results['chain'])[burnin:, :]
            s2chain = np.array(results['s2chain'])[burnin:, :]
            names = self.flat_attr_list # parameter names
            nsimu, nparam = chain.shape

            settings = dict(
                fig=dict(figsize=(7, 6))
            )
            # mcp.plot_density_panel(chain, names, settings, return_kde=True)
            # mcp.plot_histogram_panel(chain, names, settings)

            #Loss plots
            ax_loss.scatter(range(len(sschain)), sschain, label=f"Chain {i_chain}")
            
            # Trace plots
            for j, attr in enumerate(self.flat_attr_list):
                row = math.floor(j/ncols)
                col = int(j-ncols*row)
                axes_trace[row, col].scatter(range(len(chain[:,j])), chain[:,j], label=f"Chain {i_chain}")
                axes_trace[row, col].set_ylabel(attr)
            plt.pause(0.05)
            # mcp.plot_pairwise_correlation_panel(chain, names, settings)
            # mcp.plot_chain_metrics(chain, names, settings)
            # mcp.plot_joint_distributions(chain, names, settings)
            # mcp.plot_paired_density_matrix(chain, names, settings)



            ##################################################
            # pdata = mcstat.data
            # intervals = up.calculate_intervals(chain, results, pdata, self._sim_func_MCMC,
            #                       waitbar=True, s2chain=s2chain, nsample=500)

            # facecolor = tuple(list(beis)+[0.5])
            # edgecolor = tuple(list((0,0,0))+[0.1])
            # cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
                                
            # data_display = dict(
            #     marker=None,
            #     color=blue,
            #     linestyle="solid",
            #     mfc='none',
            #     label='Physical')
            # model_display = dict(
            #     color="black",
            #     linestyle="dashed", 
            #     label=f"Virtual",
            #     linewidth=2
            #     )
            # interval_display = dict(alpha=None, edgecolor=edgecolor, linestyle="solid")
            # ciset = dict(
            #     limits=[95],
            #     colors=[grey],
            #     # cmap=cmap,
            #     alpha=0.5)
            
            # piset = dict(
            #     limits=[95],
            #     colors=[facecolor],
            #     alpha=0.5)


            # for ii, interval in enumerate(intervals):
            #     fig, ax = up.plot_intervals(interval,
            #                                 time=mcstat.data.xdata[0],
            #                                 ydata=mcstat.data.ydata[0][:,ii],
            #                                 data_display=data_display,
            #                                 model_display=model_display,
            #                                 interval_display=interval_display,
            #                                 ciset=ciset,
            #                                 piset=piset,
            #                                 figsize=(7, 5),addcredible=True,addprediction=True)
            #     myFmt = mdates.DateFormatter('%H')
            #     ax.xaxis.set_major_formatter(myFmt)
            #     ax.xaxis.set_tick_params(rotation=45)
            #     ax.set_ylabel('')
            #     ax.set_title(str(ii))
            #     ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            #     ax.set_xlabel('Time (Days)')
            ##################################################
        
        # for component, attr_list in self.targetParameters.items():
        #     for attr in attr_list:
        #         rsetattr(component, attr, x)
        ax_loss.set_yscale('log')
        ax_loss.legend()
        plt.show()

        # mcstat.run_simulation()

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
            
    def get_jacobian(self, *args):
        jac = np.zeros((self.n_adjusted*len(self.targetMeasuringDevices), len(self.flat_attr_list)))
        k = 0
        # np.set_printoptions(threshold=sys.maxsize)
        for y_scale, measuring_device in zip(self.y_scale, self.targetMeasuringDevices):
            for i, (component, attr) in enumerate(zip(self.flat_component_list, self.flat_attr_list)):
                grad = np.array(component.savedParameterGradient[measuring_device][attr])[self.n_initialization_steps:]
                jac[k:k+self.n_adjusted, i] = grad[self.no_flow_mask]/y_scale
                # print("----")
                # print(measuring_device.id)
                # print(attr)
                # print(jac[k:k+self.n_adjusted, i])
                
            # jac[k:k+self.n_adjusted, :] = jac[k:k+self.n_adjusted, :]/(self.max_actual_readings[measuring_device.id]-self.min_actual_readings[measuring_device.id])
            k+=self.n_adjusted
        # print([el.id for el in self.targetMeasuringDevices])
        # np.set_printoptions(threshold=sys.maxsize)

        # print(self.targetMeasuringDevices[0].id)
        # print(jac[0:self.n_adjusted])

        # print(self.targetMeasuringDevices[1].id)
        # print(jac[self.n_adjusted:2*self.n_adjusted])

        # print(self.targetMeasuringDevices[2].id)
        # print(jac[2*self.n_adjusted:3*self.n_adjusted])
        # np.set_printoptions(precision=3)
        # np.set_printoptions(suppress=True)
        print(self.targetMeasuringDevices[0].id)
        print("Mean: ", np.mean(jac[0:self.n_adjusted], axis=0))
        print("Max: ", np.max(np.abs(jac[0:self.n_adjusted]), axis=0))

        print(self.targetMeasuringDevices[1].id)
        print("Mean: ", np.mean(jac[self.n_adjusted:2*self.n_adjusted], axis=0))
        print("Max: ", np.max(np.abs(jac[self.n_adjusted:2*self.n_adjusted]), axis=0))

        print(self.targetMeasuringDevices[2].id)
        print("Mean: ", np.mean(jac[2*self.n_adjusted:3*self.n_adjusted], axis=0))
        print("Max: ", np.max(np.abs(jac[2*self.n_adjusted:3*self.n_adjusted]), axis=0))
        return jac


    def set_parameters_from_array(self, x):
        for i, (obj, attr) in enumerate(zip(self.flat_component_list, self.flat_attr_list)):
            rsetattr(obj, attr, x[i])

    def set_parameters_from_dict(self, x):
        for (obj, attr) in zip(self.flat_component_list, self.flat_attr_list):
            rsetattr(obj, attr, x[attr])

    # def obj_fun(self, x, stepSize, startPeriod, endPeriod):
    #     '''
    #         This function calculates the loss (residual) between the predicted and measured output using 
    #         the least_squares optimization method. It takes in an array x representing the parameters to be optimized, 
    #         sets these parameter values in the model and simulates the model to obtain the predictions. 
    #     '''
    

    #     # Set parameters for the model
    #     self.set_parameters(x)

    #     self.simulator.simulate(self.model,
    #                             stepSize=stepSize,
    #                             startPeriod=startPeriod,
    #                             endPeriod=endPeriod,
    #                             trackGradients=self.trackGradients,
    #                             targetParameters=self.targetParameters,
    #                             targetMeasuringDevices=self.targetMeasuringDevices)
        
    #     ############################################################################################
    #     # import matplotlib.pyplot as plt
    #     # from matplotlib.pyplot import cm
    #     # import copy

    #     # coil = self.model.component_dict["coil"]
    #     # temp_joined = {key_input: [] for key_input in coil.FMUinputMap.values()}
    #     # temp_joined.update({key_input: [] for key_input in coil.FMUparameterMap.values()})
    #     # localGradientsSaved_re = {key_output: copy.deepcopy(temp_joined) for key_output in coil.FMUoutputMap.values()}
        
    #     # for i, localGradient in enumerate(coil.localGradientsSaved):
    #     #     for output_key, input_dict in localGradient.items():
    #     #         for input_key, value in input_dict.items():
    #     #             localGradientsSaved_re[output_key][input_key].append(value)
          
    #     # for output_key, input_dict in localGradient.items():
    #     #     fig, ax = plt.subplots()
    #     #     fig.suptitle(f"dy: {output_key}", fontsize=25)
    #     #     color = iter(cm.turbo(np.linspace(0, 1, len(localGradient[output_key]))))   
    #     #     for input_key, value in input_dict.items():
    #     #         label = f"dx: {input_key}"
    #     #         c = next(color)
    #     #         ax.plot(localGradientsSaved_re[output_key][input_key], label=label, color=c)
    #     #         print("-----------------------------")
    #     #         print(output_key)
    #     #         print(input_key)
    #     #         print(localGradientsSaved_re[output_key][input_key])
    #     #     ax.legend(prop={'size': 10})
    #     #     ax.set_ylim([-3, 7])
    #     # plt.show()
    #     ############################################################################################


    #     # Non-zero flow filtering has to be constant size. Otherwise, scipy throws an error.
    #     waterFlowRate = np.array(self.model.component_dict["Supply air temperature setpoint"].savedInput["exhaustAirTemperature"])
    #     airFlowRate = np.array(self.model.component_dict["fan flow meter"].savedOutput["airFlowRate"])
    #     tol = 1e-4
    #     self.no_flow_mask = np.logical_and(waterFlowRate>tol,airFlowRate>tol)[self.n_initialization_steps:]

    #     k = 0
    #     self.n_adjusted = np.sum(self.no_flow_mask==True)
    #     res = np.zeros((self.n_adjusted*len(self.targetMeasuringDevices)))
    #     for y_scale, measuring_device in zip(self.y_scale, self.targetMeasuringDevices):
    #         simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
    #         actual_readings = self.actual_readings[measuring_device.id].to_numpy()
    #         simulation_readings_min_max_normalized = (simulation_readings-self.min_actual_readings[measuring_device.id])/(self.max_actual_readings[measuring_device.id]-self.min_actual_readings[measuring_device.id])
    #         # res+=np.abs(simulation_readings-actual_readings)

    #         # res[k:k+self.n_adjusted] = simulation_readings_min_max_normalized[self.no_flow_mask]-self.actual_readings_min_max_normalized[measuring_device.id][self.no_flow_mask]
    #         res[k:k+self.n_adjusted] = (simulation_readings[self.no_flow_mask]-actual_readings[self.no_flow_mask])/y_scale
    #         k+=self.n_adjusted
    #     self.n_obj_eval+=1
    #     self.loss = np.sum(res**2)
    #     if self.loss<self.best_loss:
    #         self.best_loss = self.loss
    #         self.best_parameters = x

    #     print("=================")
    #     print("Loss: {:0.2f}".format(self.loss))
    #     print("Best Loss: {:0.2f}".format(self.best_loss))
    #     print(x)
    #     print()
    #     print("=================")
    #     # print(self.targetMeasuringDevices[0].id)
    #     # print(res[0:self.n_adjusted])

    #     # print(self.targetMeasuringDevices[1].id)
    #     # print(res[self.n_adjusted:2*self.n_adjusted])

    #     # print(self.targetMeasuringDevices[2].id)
    #     # print(res[2*self.n_adjusted:3*self.n_adjusted])
    #     # 
    #     return res

    # def obj_fun_GA_exception_wrapper(self, ga_instance, x, solution_idx):
    #     try:
    #         fitness = self.obj_fun_GA(ga_instance, x, solution_idx)
    #     except Exception as inst:
    #         print(inst)
    #         fitness = -1e+10
    #     return fitness
        
    # def obj_fun_GA(self, ga_instance, x, solution_idx):
    #     '''
    #         This function calculates the loss (residual) between the predicted and measured output using 
    #         the least_squares optimization method. It takes in an array x representing the parameters to be optimized, 
    #         sets these parameter values in the model and simulates the model to obtain the predictions. 
    #     '''
    

    #     # Set parameters for the model
    #     self.set_parameters(x)

    #     self.simulator.simulate(self.model,
    #                             stepSize=self.stepSize,
    #                             startPeriod=self.startPeriod_train,
    #                             endPeriod=self.endPeriod_train,
    #                             trackGradients=self.trackGradients,
    #                             targetParameters=self.targetParameters,
    #                             targetMeasuringDevices=self.targetMeasuringDevices,
    #                             show_progress_bar=False)

    #     # Non-zero flow filtering has to be constant size. Otherwise, scipy throws an error.
    #     waterFlowRate = np.array(self.model.component_dict["Supply air temperature setpoint"].savedInput["exhaustAirTemperature"])
    #     airFlowRate = np.array(self.model.component_dict["fan flow meter"].savedOutput["airFlowRate"])
    #     tol = 1e-4
    #     self.no_flow_mask = np.logical_and(waterFlowRate>tol,airFlowRate>tol)[self.n_initialization_steps:]

    #     k = 0
    #     self.n_adjusted = np.sum(self.no_flow_mask==True)
    #     res = np.zeros((self.n_adjusted*len(self.targetMeasuringDevices)))
    #     for y_scale, measuring_device in zip(self.y_scale, self.targetMeasuringDevices):
    #         simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
    #         actual_readings = self.actual_readings[measuring_device.id].to_numpy()
    #         res[k:k+self.n_adjusted] = (simulation_readings[self.no_flow_mask]-actual_readings[self.no_flow_mask])/y_scale
    #         k+=self.n_adjusted
    #     self.n_obj_eval+=1
    #     self.loss = np.sum(res**2)
    #     if self.loss<self.best_loss:
    #         self.best_loss = self.loss
    #         self.best_parameters = x

    #     print("=================")
    #     print(f"Population number: {solution_idx}")
    #     print("Loss: {:0.2f}".format(self.loss))
    #     # print("Best Loss: {:0.2f}".format(self.best_loss))
    #     print(x)
    #     print("=================")
    #     print("")

    #     return -self.loss


    def _obj_fun_MCMC_exception_wrapper(self, x, data):
        try:
            fitness = self._obj_fun_MCMC(x, data)
        except FMICallException as inst:
            fitness = 1e+10
        return fitness
    
    def _sim_func_MCMC(self, x, data):
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
        waterFlowRate = np.array(self.model.component_dict["Supply air temperature setpoint"].savedInput["exhaustAirTemperature"])
        airFlowRate = np.array(self.model.component_dict["fan flow meter"].savedOutput["airFlowRate"])
        tol = 1e-4
        self.no_flow_mask = np.logical_and(waterFlowRate>tol,airFlowRate>tol)[self.n_initialization_steps:]

        self.n_adjusted = np.sum(self.no_flow_mask==True)
        res = np.zeros((self.n_adjusted, len(self.targetMeasuringDevices)))
        for j, (y_scale, measuring_device) in enumerate(zip(self.y_scale, self.targetMeasuringDevices)):
            
            simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
            actual_readings = self.actual_readings[measuring_device.id].to_numpy()
            simulation_readings_min_max_normalized = (simulation_readings-self.min_actual_readings[measuring_device.id])/(self.max_actual_readings[measuring_device.id]-self.min_actual_readings[measuring_device.id])
            # res+=np.abs(simulation_readings-actual_readings)
            # res[k:k+self.n_adjusted] = simulation_readings_min_max_normalized[self.no_flow_mask]-self.actual_readings_min_max_normalized[measuring_device.id][self.no_flow_mask]
            res[:,j] = (simulation_readings[self.no_flow_mask]-actual_readings[self.no_flow_mask])/y_scale
        self.n_obj_eval+=1
        self.loss = np.sum(res**2, axis=0)
        # if self.loss<self.best_loss:
        #     self.best_loss = self.loss
        #     self.best_parameters = x

        print("=================")
        with np.printoptions(precision=3, suppress=True):
            print(x)
            print(f"Loss: {self.loss}")
        # print("Best Loss: {:0.2f}".format(self.best_loss))
        print("=================")
        print("")
        return self.loss
    

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
        n_sigma = len(self.targetMeasuringDevices)
        # x = theta[:-n_sigma]
        # sigma = theta[-n_sigma:]
        sigma = np.array([1,1,1,1])
        self.set_parameters_from_array(theta)
        self.simulator.simulate(self.model,
                                stepSize=self.stepSize,
                                startPeriod=self.startPeriod_train,
                                endPeriod=self.endPeriod_train,
                                trackGradients=self.trackGradients,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)

        # Non-zero flow filtering has to be constant size. Otherwise, scipy throws an error.
        waterFlowRate = np.array(self.model.component_dict["Supply air temperature setpoint"].savedInput["exhaustAirTemperature"])
        airFlowRate = np.array(self.model.component_dict["fan flow meter"].savedOutput["airFlowRate"])
        tol = 1e-4
        self.no_flow_mask = np.logical_and(waterFlowRate>tol,airFlowRate>tol)[self.n_initialization_steps:]
        self.n_adjusted = np.sum(self.no_flow_mask==True)
        res = np.zeros((self.n_adjusted, len(self.targetMeasuringDevices)))
        for j, (y_scale, measuring_device) in enumerate(zip(self.y_scale, self.targetMeasuringDevices)):
            simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
            actual_readings = self.actual_readings[measuring_device.id].to_numpy()
            res[:,j] = (simulation_readings[self.no_flow_mask]-actual_readings[self.no_flow_mask])/y_scale
        #np.random.normal(loc=0, scale=sigma[j], size=self.n_adjusted)
        self.n_obj_eval+=1
        ss = np.sum(res**2, axis=0)
        loglike = -0.5*np.sum(ss/(sigma**2))


        print("=================")
        with np.printoptions(precision=3, suppress=True):
            print(f"Theta: {theta}")
            print(f"Sum of squares: {ss}")
            print(f"Sigma: {sigma}")
            print(f"Loglikelihood: {loglike}")
        print("=================")
        print("")
        
        return loglike
        