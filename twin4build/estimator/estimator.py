import multiprocessing
import math
import os
from tqdm import tqdm
from twin4build.simulator.simulator import Simulator
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.uppath import uppath
import numpy as np
from ptemcee.sampler import Sampler, make_ladder
import datetime
import pickle
from twin4build.utils.mkdir_in_root import mkdir_in_root
import twin4build.base as base
from fmpy.fmi2 import FMICallException
from scipy.optimize import least_squares, OptimizeResult
import george
from george import kernels
from george.metrics import Metric
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Optional

class Estimator():
    def __init__(self,
                model=None):
        self.model = model
        self.simulator = Simulator(model)
        self.tol = 1e-10
    
    def estimate(self,
                 y_scale: np.ndarray = None,
                 trackGradients: bool = False,
                 targetParameters: Dict[str, Dict] = None,
                 targetMeasuringDevices: Dict[str, Dict] = None,
                 n_initialization_steps: int = 60,
                 startTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 endTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 stepSize: Union[float, List[float]] = None,
                 verbose: bool = False,
                 method: str = "MCMC",
                 options: Dict = None) -> None:
        """
        Perform parameter estimation using the specified method.

        This method sets up the estimation problem and calls the appropriate estimation
        method (MCMC or least squares) based on the 'method' argument.

        Args:
            y_scale (np.ndarray, optional): Scale factors for the output variables. Defaults to None.
            trackGradients (bool): Whether to track gradients during simulation. Defaults to False.
            targetParameters (Dict[str, Dict]): Dictionary of parameters to be estimated.
            targetMeasuringDevices (Dict[str, Dict]): Dictionary of measuring devices and their properties.
            n_initialization_steps (int): Number of initialization steps. Defaults to 60.
            startTime (Union[datetime.datetime, List[datetime.datetime]]): Start time(s) for simulation.
            endTime (Union[datetime.datetime, List[datetime.datetime]]): End time(s) for simulation.
            stepSize (Union[float, List[float]]): Step size(s) for simulation.
            verbose (bool): Whether to print verbose output. Defaults to False.
            method (str): Estimation method to use ("MCMC" or "least_squares"). Defaults to "MCMC".
            options (Dict, optional): Additional options for the estimation method. Defaults to None.

        Raises:
            AssertionError: If the provided method is not supported.
        """


        # Convert to lists
        if "private" not in targetParameters:
            targetParameters["private"] = {}
        
        if "shared" not in targetParameters:
            targetParameters["shared"] = {}

        for attr, par_dict in targetParameters["private"].items():
            if isinstance(par_dict["components"], list)==False:
                targetParameters["private"][attr]["components"] = [par_dict["components"]]
            
            if isinstance(par_dict["x0"], list)==False:
                targetParameters["private"][attr]["x0"] = [par_dict["x0"]]*len(par_dict["components"])
            else:
                assert len(par_dict["x0"])==len(par_dict["components"]), f"The number of elements in the \"x0\" list must be equal to the number of components in the private dictionary for attribute {attr}."
            
            if isinstance(par_dict["lb"], list)==False:
                targetParameters["private"][attr]["lb"] = [par_dict["lb"]]*len(par_dict["components"])
            else:
                assert len(par_dict["lb"])==len(par_dict["components"]), f"The number of elements in the \"lb\" list must be equal to the number of components in the private dictionary for attribute {attr}."
            
            if isinstance(par_dict["ub"], list)==False:
                targetParameters["private"][attr]["ub"] = [par_dict["ub"]]*len(par_dict["components"])
            else:
                assert len(par_dict["lb"])==len(par_dict["components"]), f"The number of elements in the \"lb\" list must be equal to the number of components in the private dictionary for attribute {attr}."
        
        members = ["x0", "lb", "ub"]
        for attr, par_dict in targetParameters["shared"].items():
            assert isinstance(par_dict["components"], list), f"The \"components\" key in the shared dictionary must be a list for attribute {attr}."
            assert len(par_dict["components"])>0, f"The \"components\" key in the shared dictionary must contain at least one element for attribute {attr}."
            if isinstance(par_dict["components"][0], list)==False:
                targetParameters["shared"][attr]["components"] = [par_dict["components"]]
            for m in members:
                if isinstance(par_dict[m], list)==False:
                    targetParameters["shared"][attr][m] = [[par_dict[m] for c in l] for l in par_dict["components"]]
                else:
                    assert len(par_dict[m])==len(targetParameters["shared"][attr]["components"]), f"The number of elements in the \"{m}\" list must be equal to the number of components in the shared dictionary for attribute {attr}."

            for key, list_ in par_dict.items():
                if isinstance(list_, list)==False:
                    targetParameters["shared"][attr][key] = [[list_]]
                elif isinstance(list_[0], list)==False:
                    targetParameters["shared"][attr][key] = [list_]
        

        allowed_methods = ["MCMC","LS"]
        assert method in allowed_methods, f"The \"method\" argument must be one of the following: {', '.join(allowed_methods)} - \"{method}\" was provided."
        
        self.verbose = verbose 
        self.n_initialization_steps = n_initialization_steps
        if isinstance(startTime, list)==False:
            startTime = [startTime]
        if isinstance(endTime, list)==False:
            endTime = [endTime]
        if isinstance(stepSize, list)==False:
            stepSize = [stepSize]
        for startTime_, endTime_, stepSize_  in zip(startTime, endTime, stepSize):
            assert endTime>startTime, "The endTime must be later than the startTime."
        self.startTime_train = startTime
        self.endTime_train = endTime
        self.stepSize_train = stepSize
        for startTime_, endTime_, stepSize_  in zip(self.startTime_train, self.endTime_train, self.stepSize_train):    
            self.model.cache(startTime=startTime_,
                            endTime=endTime_,
                            stepSize=stepSize_)

        self.standardDeviation = np.array([el["standardDeviation"] for el in targetMeasuringDevices.values()])
        self.flat_component_list_private = [obj for par_dict in targetParameters["private"].values() for obj in par_dict["components"]]
        self.flat_attr_list_private = [attr for attr, par_dict in targetParameters["private"].items() for obj in par_dict["components"]]
        self.flat_component_list_shared = [obj for par_dict in targetParameters["shared"].values() for obj_list in par_dict["components"] for obj in obj_list]
        self.flat_attr_list_shared = [attr for attr, par_dict in targetParameters["shared"].items() for obj_list in par_dict["components"] for obj in obj_list]
        private_mask = np.arange(len(self.flat_component_list_private), dtype=int)
        shared_mask = []
        n = len(self.flat_component_list_private)
        k = 0
        for attr, par_dict in targetParameters["shared"].items():
            for obj_list in par_dict["components"]:
                for obj in obj_list:
                    shared_mask.append(k+n)
                k += 1
        shared_mask = np.array(shared_mask)
        self.flat_component_list = self.flat_component_list_private + self.flat_component_list_shared
        self.theta_mask = np.concatenate((private_mask, shared_mask)).astype(int)
        self.flat_attr_list = self.flat_attr_list_private + self.flat_attr_list_shared
        self.simulator.flat_component_list = self.flat_component_list
        self.simulator.flat_attr_list = self.flat_attr_list
        self.simulator.theta_mask = self.theta_mask
        self.simulator.targetParameters = targetParameters
        self.simulator.targetMeasuringDevices = targetMeasuringDevices
        self.trackGradients = trackGradients
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
        self.n_obj_eval = 0
        self.best_loss = math.inf
        self.n_timesteps = 0
        for i, (startTime_, endTime_, stepSize_)  in enumerate(zip(self.startTime_train, self.endTime_train, self.stepSize_train)):
            self.simulator.get_simulation_timesteps(startTime_, endTime_, stepSize_)
            self.n_timesteps += len(self.simulator.secondTimeSteps)-self.n_initialization_steps
        x0 = []
        for par_dict in targetParameters["private"].values():
            if len(par_dict["components"])==len(par_dict["x0"]):
                x0 += par_dict["x0"]
            else:
                x0 += [par_dict["x0"][0]]*len(par_dict["components"])

        for par_dict in targetParameters["shared"].values():
            for l in par_dict["x0"]:
                x0.append(l[0])
            
        lb = []
        for par_dict in targetParameters["private"].values():
            if len(par_dict["components"])==len(par_dict["lb"]):
                lb += par_dict["lb"]
            else:
                lb += [par_dict["lb"][0]]*len(par_dict["components"])
        for par_dict in targetParameters["shared"].values():
            for l in par_dict["lb"]:
                lb.append(l[0])

        ub = []
        for par_dict in targetParameters["private"].values():
            if len(par_dict["components"])==len(par_dict["ub"]):
                ub += par_dict["ub"]
            else:
                ub += [par_dict["ub"][0]]*len(par_dict["components"])
        for par_dict in targetParameters["shared"].values():
            for l in par_dict["ub"]:
                ub.append(l[0])

        self.x0 = np.array(x0)
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        if y_scale is None:
            self.y_scale = np.array([1]*len(targetMeasuringDevices))
        else:
            self.y_scale = np.array(y_scale)

        if method == "MCMC":
            if options is None:
                options = {}
            self.mcmc(**options)
        elif method == "LS":
            self.ls(self.x0, self.lb, self.ub)

    def sample_cartesian_n_sphere(self, r: float, n_dim: int, n_samples: int) -> np.ndarray:
        """
        Sample points uniformly from the surface of an n-dimensional sphere.
        Based on https://stackoverflow.com/questions/20133318/n-sphere-coordinate-system-to-cartesian-coordinate-system

        Args:
            r (float): Radius of the sphere.
            n_dim (int): Number of dimensions.
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of sampled points with shape (n_samples, n_dim).
        
        
        """
        def sphere_to_cart(r, c):
            a = np.concatenate((np.array([2*np.pi]), c))
            si = np.sin(a)
            si[0] = 1
            si = np.cumprod(si)
            co = np.cos(a)
            co = np.roll(co, -1)
            return si*co*r
        c = np.random.random_sample((n_samples, n_dim-1))*2*np.pi
        x = np.array([sphere_to_cart(r, c_) for c_ in c])
        return x

    def sample_bounded_gaussian(self, n_temperature: int, n_walkers: int, ndim: int, 
                                x0: np.ndarray, lb: np.ndarray, ub: np.ndarray, 
                                standardDeviation_x0: np.ndarray) -> np.ndarray:
        """
        Sample from a bounded Gaussian distribution.

        Args:
            n_temperature (int): Number of temperatures.
            n_walkers (int): Number of walkers.
            ndim (int): Number of dimensions.
            x0 (np.ndarray): Mean of the Gaussian distribution.
            lb (np.ndarray): Lower bounds.
            ub (np.ndarray): Upper bounds.
            standardDeviation_x0 (np.ndarray): Standard deviation of the Gaussian distribution.

        Returns:
            np.ndarray: Sampled points with shape (n_temperature, n_walkers, ndim).
        """
        nrem = n_walkers*n_temperature
        x0 = np.resize(x0,(nrem, ndim))
        lb = np.resize(lb,(nrem, ndim))
        ub = np.resize(ub,(nrem, ndim))
        x0_start = np.zeros((nrem, ndim))
        not_found = np.ones((nrem, ndim), dtype=bool)
        while np.any(not_found):
            x0_ = np.random.normal(loc=x0, scale=standardDeviation_x0, size=(nrem, ndim))
            x0_start[not_found] = x0_[not_found]
            not_found = np.logical_or(x0_start<lb, x0_start>ub)
        x0_start = x0_start.reshape((n_temperature, n_walkers, ndim))
        return x0_start

    def mcmc(self, 
             n_sample: int = 10000, 
             n_temperature: int = 15, 
            fac_walker: int = 2, 
            T_max: float = np.inf, 
            n_cores: int = multiprocessing.cpu_count(), 
            prior: str = "uniform", model_prior: str = None, 
            noise_prior: str = None, walker_initialization: str = "uniform", 
            model_walker_initialization: str = None, 
            noise_walker_initialization: str = None, 
            add_gp: bool = False, gp_input_type: str = "closest", 
            gp_add_time: bool = True, gp_max_inputs: int = 3, 
            maxtasksperchild: int = 100, n_save_checkpoint: int = None, 
            use_pickle: bool = True, use_npz: bool = True) -> None:
        """
        Run the EMCEE estimation method.

        This method performs Markov Chain Monte Carlo (MCMC) estimation using the emcee sampler.

        Args:
            n_sample (int): Number of samples to draw. Defaults to 10000.
            n_temperature (int): Number of temperatures for parallel tempering. Defaults to 15.
            fac_walker (int): Factor to determine number of walkers. Defaults to 2.
            T_max (float): Maximum temperature. Defaults to np.inf.
            n_cores (int): Number of CPU cores to use. Defaults to all available cores.
            prior (str): Type of prior to use. Defaults to "uniform".
            model_prior (str, optional): Prior for the model parameters. Defaults to None.
            noise_prior (str, optional): Prior for the noise parameters. Defaults to None.
            walker_initialization (str): Method to initialize walkers. Defaults to "uniform".
            model_walker_initialization (str, optional): Initialization method for model parameters. Defaults to None.
            noise_walker_initialization (str, optional): Initialization method for noise parameters. Defaults to None.
            add_gp (bool): Whether to add Gaussian Process modeling. Defaults to False.
            gp_input_type (str): Type of input for Gaussian Process. Defaults to "closest".
            gp_add_time (bool): Whether to add time as an input to GP. Defaults to True.
            gp_max_inputs (int): Maximum number of inputs for GP. Defaults to 3.
            maxtasksperchild (int): Maximum number of tasks per child process. Defaults to 100.
            n_save_checkpoint (int, optional): Number of iterations between checkpoints. Defaults to None.
            use_pickle (bool): Whether to save results using pickle. Defaults to True.
            use_npz (bool): Whether to save results using numpy's npz format. Defaults to True.

        Raises:
            AssertionError: If the provided arguments are invalid or incompatible.
        """
        assert n_cores>=1, "The argument \"n_cores\" must be larger than or equal to 1"
        assert fac_walker>=2, "The argument \"fac_walker\" must be larger than or equal to 2"
        allowed_priors = ["uniform", "gaussian", "sample_gaussian"]
        allowed_walker_initializations = ["uniform", "gaussian", "hypersphere", "hypercube", "sample", "sample_hypercube", "sample_gaussian"]
        assert prior in allowed_priors, f"The \"prior\" argument must be one of the following: {', '.join(allowed_priors)} - \"{prior}\" was provided."
        assert model_prior is None or model_prior in allowed_priors, f"The \"model_prior\" argument must be one of the following: {', '.join(allowed_priors)} - \"{model_prior}\" was provided."
        assert noise_prior is None or noise_prior in allowed_priors, f"The \"noise_prior\" argument must be one of the following: {', '.join(allowed_priors)} - \"{noise_prior}\" was provided."
        assert walker_initialization in allowed_walker_initializations, f"The \"walker_initialization\" argument must be one of the following: {', '.join(allowed_walker_initializations)} - \"{walker_initialization}\" was provided."
        assert (model_walker_initialization is None and noise_walker_initialization is None) or (model_walker_initialization is not None and noise_walker_initialization is not None), "\"model_walker_initialization\" and \"noise_walker_initialization\" must both be either None or set to one of the general options."
        assert model_walker_initialization is None or model_walker_initialization in allowed_walker_initializations, f"The \"model_walker_initialization\" argument must be one of the following: {', '.join(allowed_walker_initializations)} - \"{model_walker_initialization}\" was provided."
        assert noise_walker_initialization is None or noise_walker_initialization in allowed_walker_initializations, f"The \"noise_walker_initialization\" argument must be one of the following: {', '.join(allowed_walker_initializations)} - \"{noise_walker_initialization}\" was provided."
        
        if prior!="uniform" or (model_prior is not None and model_prior!="uniform") or (walker_initialization is not None and walker_initialization!="uniform") or (model_walker_initialization is not None and model_walker_initialization!="uniform"):
            assert np.all(self.x0>=self.lb), "The provided x0 must be larger than the provided lower bound lb"
            assert np.all(self.x0<=self.ub), "The provided x0 must be smaller than the provided upper bound ub"
            a = (np.abs(self.x0-self.lb)>self.tol)==False
            b = (np.abs(self.x0-self.ub)>self.tol)==False
            c = a[self.theta_mask]
            d = b[self.theta_mask]
            assert np.all(np.abs(self.x0-self.lb)>self.tol), f"The difference between x0 and lb must be larger than {str(self.tol)}. {np.array(self.flat_attr_list)[c]} violates this condition." 
            assert np.all(np.abs(self.x0-self.ub)>self.tol), f"The difference between x0 and ub must be larger than {str(self.tol)}. {np.array(self.flat_attr_list)[d]} violates this condition."

        
        for startTime_, endTime_, stepSize_  in zip(self.startTime_train, self.endTime_train, self.stepSize_train):    
            self.model.cache(startTime=startTime_,
                            endTime=endTime_,
                            stepSize=stepSize_)
        
        datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_pickle = str('{}{}'.format(datestr, '.pickle'))
        filename_npz = str('{}{}'.format(datestr, '.npz'))
        self.chain_savedir_pickle, isfile = self.model.get_dir(folder_list=["model_parameters", "estimation_results", "chain_logs"], filename=filename_pickle)
        self.chain_savedir_npz, isfile = self.model.get_dir(folder_list=["model_parameters", "estimation_results", "chain_logs"], filename=filename_npz)
        
        self.gp_input_type = gp_input_type
        self.gp_add_time = gp_add_time
        self.gp_max_inputs = gp_max_inputs

        assert (model_prior is None and noise_prior is None) or (model_prior is not None and noise_prior is not None), "\"model_prior\" and \"noise_prior\" must both be either None or set to one of the available priors."
        if model_prior=="gaussian" and noise_prior=="uniform":
            logprior = self.gaussian_model_uniform_noise_logprior
        elif model_prior=="sample_gaussian" and noise_prior=="uniform":
            x = self.model.chain_log["chain.x"][:,0,:,:]
            logl = self.model.chain_log["chain.logl"][:,0,:]
            best_tuple = np.unravel_index(logl.argmax(), logl.shape)
            self.x0 = x[best_tuple + (slice(None),)]
            logprior = self.gaussian_model_uniform_noise_logprior
        elif model_prior=="uniform" and noise_prior=="gaussian":
            raise Exception("Not implemented")
        elif prior=="uniform":
            logprior = self.uniform_logprior
        elif prior=="gaussian":
            logprior = self.gaussian_logprior


        self.gp_input_map = None
        ndim = int(self.theta_mask[-1]+1)
        add_par = 1 # We add the following parameters: "a"
        self.n_par = 0
        self.n_par_map = {}
        lower_bound = -3
        upper_bound = 5

        x0_time = 8
        lower_bound_time = 0 #1 second
        upper_bound_time = 10 #3600 seconds

        diff_lower = np.abs(self.x0-self.lb)
        diff_upper = np.abs(self.ub-self.x0)
        self.standardDeviation_x0 = np.minimum(diff_lower, diff_upper)/2 #Set the standard deviation such that around 95% of the values are within the bounds

        if add_gp:
            n_par = self.model.chain_log["n_par"]
            self.gp_input_map = self.model.chain_log["gp_input_map"]
            if hasattr(self.model, "chain_log") and (self.model.chain_log["chain.x"].shape[3]==ndim or self.model.chain_log["chain.x"].shape[3]==ndim+n_par):
                x = self.model.chain_log["chain.x"][:,0,:,:]
                r = 1e-5
                logl = self.model.chain_log["chain.logl"][:,0,:]
                best_tuple = np.unravel_index(logl.argmax(), logl.shape)
                x0_ = x[best_tuple + (slice(None),)]
            else:
                raise Exception("The model does not contain the required chain_log attribute or the dimensions are wrong.")


            for i, (startTime_, endTime_, stepSize_)  in enumerate(zip(self.startTime_train, self.endTime_train, self.stepSize_train)):
                if self.gp_input_type=="closest":
                    # This is a temporary solution. The fmu.freeInstance() method fails with a segmentation fault. 
                    # The following ensures that we run the simulation in a separate process.
                    args = (self.targetMeasuringDevices, startTime_, endTime_, stepSize_)
                    kwargs = {"input_type":gp_input_type,
                            "add_time":self.gp_add_time,
                            "max_inputs":self.gp_max_inputs,
                            "run_simulation":True,
                            "x0_":x0_,
                            "gp_input_map": self.gp_input_map}
                    a = [(args, kwargs)]
                    pool = multiprocessing.Pool(1)
                    chunksize = 1
                    self.model.make_pickable()
                    y_list = list(pool.imap(self.simulator._get_gp_input_wrapped, a, chunksize=chunksize))
                    pool.close()
                    y_list = [el for el in y_list if el is not None]
                    if len(y_list)>0:
                        gp_input, self.gp_input_map = y_list[0]
                    else:
                        raise(Exception("get_gp_input failed."))
                else:
                    gp_input, self.gp_input_map = self.simulator.get_gp_input(self.targetMeasuringDevices, startTime_, endTime_, stepSize_, gp_input_type, self.gp_add_time, self.gp_max_inputs, False, None, None)
                
                actual_readings = self.simulator.get_actual_readings(startTime=startTime_, endTime=endTime_, stepSize=stepSize_)
                if i==0:
                    self.gp_input = gp_input
                    self.actual_readings = {}
                    for measuring_device in self.targetMeasuringDevices:
                        self.gp_input[measuring_device.id] = self.gp_input[measuring_device.id][self.n_initialization_steps:,:]
                        self.actual_readings[measuring_device.id] = actual_readings[measuring_device.id].to_numpy()[self.n_initialization_steps:]
                else:
                    for measuring_device in self.targetMeasuringDevices:
                        x = gp_input[measuring_device.id][self.n_initialization_steps:,:]
                        self.gp_input[measuring_device.id] = np.concatenate((self.gp_input[measuring_device.id], x), axis=0)
                        self.actual_readings[measuring_device.id] = np.concatenate((self.actual_readings[measuring_device.id], actual_readings[measuring_device.id].to_numpy()[self.n_initialization_steps:]), axis=0)

            self.gp_lengthscale = self.simulator.get_gp_lengthscale(self.targetMeasuringDevices, self.gp_input)
            for j, measuring_device in enumerate(self.targetMeasuringDevices):
                n = self.gp_input[measuring_device.id].shape[1] ######################################################
                self.n_par += n+add_par
                self.n_par_map[measuring_device.id] = n+add_par
            self.gp_variance = self.simulator.get_gp_variance(self.targetMeasuringDevices, x0_, self.startTime_train, self.endTime_train, self.stepSize_train)
            for j, measuring_device in enumerate(self.targetMeasuringDevices):
                
                add_x0 = np.zeros((self.n_par_map[measuring_device.id],))
                add_lb = lower_bound*np.ones((self.n_par_map[measuring_device.id],))
                add_ub = upper_bound*np.ones((self.n_par_map[measuring_device.id],))

                a_x0 = np.log(self.gp_variance[measuring_device.id])
                a_lb = a_x0-5
                a_ub = a_x0+5
                add_x0[0] = a_x0
                add_lb[0] = a_lb
                add_ub[0] = a_ub
                if self.gp_add_time:
                    scale_lengths = self.gp_lengthscale[measuring_device.id][:-1]
                    add_x0[1:-1] = np.log(scale_lengths)
                    add_lb[1:-1] = np.log(scale_lengths)-5
                    add_ub[1:-1] = np.log(scale_lengths)+5
                    add_x0[-1] = x0_time
                    add_lb[-1] = lower_bound_time
                    add_ub[-1] = upper_bound_time
                else:
                    scale_lengths = self.gp_lengthscale[measuring_device.id]
                    add_x0[1:] = np.log(scale_lengths)
                    add_lb[1:] = np.log(scale_lengths)-5
                    add_ub[1:] = np.log(scale_lengths)+5

                self.x0 = np.append(self.x0, add_x0)
                self.lb = np.append(self.lb, add_lb)
                self.ub = np.append(self.ub, add_ub)


            
            

            loglike = self._loglike_gaussian_process_wrapper
            ndim = ndim+self.n_par
            
            self.standardDeviation_x0 = np.append(self.standardDeviation_x0, (upper_bound-lower_bound)/2*np.ones((self.n_par,))) ###################################################
        else:
            loglike = self._loglike_wrapper


        n_walkers = int(ndim*fac_walker) #*4 #Round up to nearest even number and multiply by 2
        
        if add_gp and model_walker_initialization=="hypercube" and noise_walker_initialization=="uniform":
            r = 1e-5
            low = np.zeros((ndim-self.n_par,))

            x0_start = np.random.uniform(low=self.x0[:-self.n_par]-r, high=self.x0[:-self.n_par]+r, size=(n_temperature, n_walkers, ndim-self.n_par))
            # lb = np.resize(self.lb[:-self.n_par],(x0_start.shape))
            # ub = np.resize(self.ub[:-self.n_par],(x0_start.shape))
            # x0_start[x0_start<self.lb[:-self.n_par]] = lb[x0_start<self.lb[:-self.n_par]]
            # x0_start[x0_start>self.ub[:-self.n_par]] = ub[x0_start>self.ub[:-self.n_par]]
            model_x0_start = x0_start

            x0_start = np.random.uniform(low=self.lb[-self.n_par:], high=self.ub[-self.n_par:], size=(n_temperature, n_walkers, self.n_par))
            noise_x0_start = x0_start

            x0_start = np.append(model_x0_start, noise_x0_start, axis=2)
        elif add_gp and model_walker_initialization=="uniform" and noise_walker_initialization=="gaussian":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="uniform" and noise_walker_initialization=="hypersphere":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="uniform" and noise_walker_initialization=="hypercube":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="gaussian" and noise_walker_initialization=="hypersphere":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="gaussian" and noise_walker_initialization=="hypercube":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="hypersphere" and noise_walker_initialization=="uniform":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="hypersphere" and noise_walker_initialization=="gaussian":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="hypersphere" and noise_walker_initialization=="hypercube":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="hypercube" and noise_walker_initialization=="gaussian":
            r = 1e-5
            # low = self.lb if self.x0[:-self.n_par]-r*np.abs(self.x0[:-self.n_par])<self.lb else self.x0[:-self.n_par]-r*np.abs(self.x0[:-self.n_par])
            # high = self.ub if self.x0[:-self.n_par]+r*np.abs(self.x0[:-self.n_par])>self.ub else self.x0[:-self.n_par]+r*np.abs(self.x0[:-self.n_par])
            x0 = self.x0[:-self.n_par]
            lb = self.lb[:-self.n_par]
            low = np.zeros((ndim-self.n_par,))
            cond = (self.x0[:-self.n_par]-r*np.abs(self.x0[:-self.n_par]))<lb
            low[cond] = lb[cond]
            low[cond==False] = x0[cond==False]-r*np.abs(x0[cond==False])
            ub = self.ub[:-self.n_par]
            high = np.zeros((ndim-self.n_par,))
            cond = (self.x0[:-self.n_par]+r*np.abs(self.x0[:-self.n_par]))>ub
            high[cond] = ub[cond]
            high[cond==False] = x0[cond==False]+r*np.abs(x0[cond==False])
            model_x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))
            
            x0 = self.x0[-self.n_par:]
            lb = self.lb[-self.n_par:]
            low = np.zeros((self.n_par,))
            cond = (self.x0[-self.n_par:]-r*np.abs(self.x0[-self.n_par:]))<lb
            low[cond] = lb[cond]
            low[cond==False] = x0[cond==False]-r*np.abs(x0[cond==False])
            ub = self.ub[-self.n_par:]
            high = np.zeros((self.n_par,))
            cond = (self.x0[-self.n_par:]+r*np.abs(self.x0[-self.n_par:]))>ub
            high[cond] = ub[cond]
            high[cond==False] = x0[cond==False]+r*np.abs(x0[cond==False])

            noise_x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))
            x0_start = np.append(model_x0_start, noise_x0_start, axis=2)
            
        elif add_gp and model_walker_initialization=="hypercube" and noise_walker_initialization=="hypersphere":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="sample" and noise_walker_initialization=="uniform":
            assert hasattr(self.model, "chain_log") and "chain.x" in self.model.chain_log, "Model object has no chain log. Please load before starting estimation."
            assert self.model.chain_log["chain.x"].shape[3]==ndim-self.n_par, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.chain_log["chain.x"][-1,0,:,:]
            del self.model.chain_log #We delete the chain log before initiating multiprocessing to save memory
            r = 1e-5
            if x.shape[0]==n_walkers:
                print("Using provided sample for initial walkers")
                # model_x0_start = np.random.uniform(low=x-r, high=x+r, size=(n_temperature, n_walkers, ndim-self.n_par))
                model_x0_start = x.reshape((n_temperature, n_walkers, ndim-self.n_par))
            elif x.shape[0]>n_walkers: #downsample
                print("Downsampling initial walkers")
                ind = np.arange(x.shape[0])
                ind_sample = np.random.choice(ind, n_walkers)
                model_x0_start = x[ind_sample,:]
                model_x0_start = model_x0_start.reshape((n_temperature, n_walkers, ndim-self.n_par))
                # model_x0_start = np.random.uniform(low=model_x0_start-r, high=model_x0_start+r, size=(n_temperature, n_walkers, ndim-self.n_par))
            else: #upsample
                print("Upsampling initial walkers")
                # diff = n_walkers-x.shape[0]
                ind = np.arange(x.shape[0])
                ind_sample = np.random.choice(ind, n_walkers)
                model_x0_start = x[ind_sample,:]
                lb = np.resize(self.lb,(model_x0_start.shape))
                low = np.zeros(model_x0_start.shape)
                cond = (model_x0_start-r*np.abs(model_x0_start))<lb
                low[cond] = lb[cond]
                low[cond==False] = model_x0_start[cond==False]-r*np.abs(model_x0_start[cond==False])
                ub = np.resize(self.ub,(model_x0_start.shape))
                high = np.zeros(model_x0_start.shape)
                cond = (model_x0_start+r*np.abs(model_x0_start))>ub
                high[cond] = ub[cond]
                high[cond==False] = model_x0_start[cond==False]+r*np.abs(model_x0_start[cond==False])
                model_x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim-self.n_par))
            
            x0_start = np.random.uniform(low=self.lb[-self.n_par:], high=self.ub[-self.n_par:], size=(n_temperature, n_walkers, self.n_par))
            noise_x0_start = x0_start

            x0_start = np.append(model_x0_start, noise_x0_start, axis=2)

        elif add_gp and model_walker_initialization=="sample" and noise_walker_initialization=="gaussian":
            assert hasattr(self.model, "chain_log") and "chain.x" in self.model.chain_log, "Model object has no chain log. Please load before starting estimation."
            assert self.model.chain_log["chain.x"].shape[3]==ndim-self.n_par or self.model.chain_log["chain.x"].shape[3]==ndim, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.chain_log["chain.x"][-1,0,:,:] if self.model.chain_log["chain.x"].shape[3]==ndim-self.n_par else self.model.chain_log["chain.x"][-1,0,:,:-self.n_par]
            del self.model.chain_log #We delete the chain log before initiating multiprocessing to save memory
            r = 1e-5
            if x.shape[0]==n_walkers:
                print("Using provided sample for initial walkers")
                # model_x0_start = np.random.uniform(low=x-r*np.abs(x), high=x+r*np.abs(x), size=(n_temperature, n_walkers, ndim-self.n_par))
                model_x0_start = x.reshape((n_temperature, n_walkers, ndim-self.n_par))
            elif x.shape[0]>n_walkers: #downsample
                print("Downsampling initial walkers")
                ind = np.arange(x.shape[0])
                ind_sample = np.random.choice(ind, n_walkers)
                model_x0_start = x[ind_sample,:]
                model_x0_start = model_x0_start.reshape((n_temperature, n_walkers, ndim-self.n_par))
                # model_x0_start = np.random.uniform(low=model_x0_start-r*np.abs(model_x0_start), high=model_x0_start+r*np.abs(model_x0_start), size=(n_temperature, n_walkers, ndim-self.n_par))
            else: #upsample
                print("Upsampling initial walkers")
                # diff = n_walkers-x.shape[0]
                ind = np.arange(x.shape[0])
                ind_sample = np.random.choice(ind, n_walkers)
                model_x0_start = x[ind_sample,:]
                lb = np.resize(self.lb[:-self.n_par],(model_x0_start.shape))
                low = np.zeros(model_x0_start.shape)
                cond = (model_x0_start-r*np.abs(model_x0_start))<lb
                low[cond] = lb[cond]
                low[cond==False] = model_x0_start[cond==False]-r*np.abs(model_x0_start[cond==False])
                ub = np.resize(self.ub[:-self.n_par],(model_x0_start.shape))
                high = np.zeros(model_x0_start.shape)
                cond = (model_x0_start+r*np.abs(model_x0_start))>ub
                high[cond] = ub[cond]
                high[cond==False] = model_x0_start[cond==False]+r*np.abs(model_x0_start[cond==False])
                model_x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim-self.n_par))
            
            noise_x0_start = self.sample_bounded_gaussian(n_temperature, n_walkers, self.n_par, self.x0[-self.n_par:], self.lb[-self.n_par:], self.ub[-self.n_par:], self.standardDeviation_x0[-self.n_par:])
            x0_start = np.append(model_x0_start, noise_x0_start, axis=2)

        elif add_gp and model_walker_initialization=="sample_hypercube" and noise_walker_initialization=="hypercube":
            assert hasattr(self.model, "chain_log") and "chain.x" in self.model.chain_log, "Model object has no chain log. Please load before starting estimation."
            assert self.model.chain_log["chain.x"].shape[3]==ndim-self.n_par, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.chain_log["chain.x"][:,0,:,:]
            r = 1e-5
            logl = self.model.chain_log["chain.logl"][:,0,:]
            best_tuple = np.unravel_index(logl.argmax(), logl.shape)
            x0_ = x[best_tuple + (slice(None),)]
            x0_ = np.concatenate((x0_, self.x0[-self.n_par:]))
            low = np.zeros((ndim,))
            cond = (x0_-r*np.abs(x0_))<self.lb
            low[cond] = self.lb[cond]
            low[cond==False] = x0_[cond==False]-r*np.abs(x0_[cond==False])
            high = np.zeros((ndim,))
            cond = (x0_+r*np.abs(x0_))>self.ub
            high[cond] = self.ub[cond]
            high[cond==False] = x0_[cond==False]+r*np.abs(x0_[cond==False])
            x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))
            del self.model.chain_log #We delete the chain log before initiating multiprocessing to save memory
            del x

        elif walker_initialization=="uniform":
            x0_start = np.random.uniform(low=self.lb, high=self.ub, size=(n_temperature, n_walkers, ndim))
        elif walker_initialization=="gaussian":
            x0_start = self.sample_bounded_gaussian(n_temperature, n_walkers, ndim, self.x0, self.lb, self.ub, self.standardDeviation_x0)
        elif walker_initialization=="hypersphere":
            r = 1e-5
            nrem = n_walkers*n_temperature
            x0_ = np.resize(self.x0,(nrem, ndim))
            lb = np.resize(self.lb,(nrem, ndim))
            ub = np.resize(self.ub,(nrem, ndim))
            cond = np.ones((nrem, ndim), dtype=bool)
            cond_1 = np.any(cond, axis=1)
            x0_start = np.zeros((nrem, ndim))
            while nrem>0:
                x0_origo = self.sample_cartesian_n_sphere(r, ndim, nrem)
                x0_start[cond_1,:] = x0_origo+x0_[:nrem]
                cond = np.logical_or(x0_start<lb, x0_start>ub)
                cond_1 = np.any(cond, axis=1)
                nrem = np.sum(cond_1)
            x0_start = x0_start.reshape((n_temperature, n_walkers, ndim))

        elif walker_initialization=="hypercube":
            r = 1e-5
            low = np.zeros((ndim,))
            cond = (self.x0-r*np.abs(self.x0))<self.lb
            low[cond] = self.lb[cond]
            low[cond==False] = self.x0[cond==False]-r*np.abs(self.x0[cond==False])
            high = np.zeros((ndim,))
            cond = (self.x0+r*np.abs(self.x0))>self.ub
            high[cond] = self.ub[cond]
            high[cond==False] = self.x0[cond==False]+r*np.abs(self.x0[cond==False])
            x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))


        elif walker_initialization=="sample_hypercube":
            assert hasattr(self.model, "chain_log") and "chain.x" in self.model.chain_log, "Model object has no chain log. Please load before starting estimation."
            assert self.model.chain_log["chain.x"].shape[3]==ndim, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.chain_log["chain.x"][:,0,:,:]
            r = 1e-5
            logl = self.model.chain_log["chain.logl"][:,0,:]
            best_tuple = np.unravel_index(logl.argmax(), logl.shape)
            x0_ = x[best_tuple + (slice(None),)]
            assert np.all(x0_>=self.lb), "The provided x0 must be larger than the provided lower bound lb"
            low = np.zeros((ndim,))
            cond = (x0_-r*np.abs(x0_))<self.lb
            low[cond] = self.lb[cond]
            low[cond==False] = x0_[cond==False]-r*np.abs(x0_[cond==False])
            assert np.all(low>=self.lb), "The provided x0 must be larger than the provided lower bound lb"
            high = np.zeros((ndim,))
            cond = (x0_+r*np.abs(x0_))>self.ub
            high[cond] = self.ub[cond]
            high[cond==False] = x0_[cond==False]+r*np.abs(x0_[cond==False])
            x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))
            del self.model.chain_log #We delete the chain log before initiating multiprocessing to save memory
            del x

        elif walker_initialization=="sample_gaussian":
            assert hasattr(self.model, "chain_log") and "chain.x" in self.model.chain_log, "Model object has no chain log. Please load before starting estimation."
            assert self.model.chain_log["chain.x"].shape[3]==ndim, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.chain_log["chain.x"][:,0,:,:]
            logl = self.model.chain_log["chain.logl"][:,0,:]
            best_tuple = np.unravel_index(logl.argmax(), logl.shape)
            x0_ = x[best_tuple + (slice(None),)]
            diff_lower = np.abs(x0_-self.lb)
            diff_upper = np.abs(self.ub-x0_)
            self.standardDeviation_x0 = np.minimum(diff_lower, diff_upper)/2 #Set the standard deviation such that around 95% of the values are within the bounds
            x0_start = np.random.normal(loc=x0_, scale=self.standardDeviation_x0, size=(n_temperature, n_walkers, ndim))
            del self.model.chain_log #We delete the chain log before initiating multiprocessing to save memory
            del x

        elif walker_initialization=="sample":
            assert hasattr(self.model, "chain_log") and "chain.x" in self.model.chain_log, "Model object has no chain log. Please load before starting estimation."
            assert self.model.chain_log["chain.x"].shape[3]==ndim-self.n_par, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.chain_log["chain.x"][-1,0,:,:]
            if x.shape[0]==n_walkers:
                x0_start = x
            elif x.shape[0]>n_walkers: #downsample
                print("Downsampling initial walkers")
                ind = np.arange(x.shape[0])
                ind_sample = np.random.choice(ind, n_walkers)
                x0_start = x[ind,:]
            else: #upsample
                print("Upsampling initial walkers")
                diff = n_walkers-x.shape[0]
                ind = np.arange(x.shape[0])
                ind_sample = np.random.choice(ind, diff)
                x_add = x[ind_sample,:]
                r = 1e-5
                x0_start = np.concatenate(x, x_add, axis=0)
        lb = np.resize(self.lb,(x0_start.shape))
        ub = np.resize(self.ub,(x0_start.shape))
        model_x0_start = x0_start[:,:,:-self.n_par]
        model_lb = lb[:,:,:-self.n_par]
        model_ub = ub[:,:,:-self.n_par]
        attr_list = np.array(self.flat_attr_list)
        idx_below = np.argwhere(model_x0_start<model_lb)
        idx_above = np.argwhere(model_x0_start>model_ub)
        for i, attr in enumerate(attr_list):
            if i in idx_below[:,2]:
                d = idx_below[idx_below[:,2]==i,:]
                values = model_x0_start[d[:,0],d[:,1],d[:,2]]
                print(f"{attr} is below the lower bound with values:")
                print(values)
            if i in idx_above:
                d = idx_above[idx_above[:,2]==i,:]
                values = model_x0_start[d[:,0],d[:,1],d[:,2]]
                print(f"{attr} is above the upper bound with values:")
                print(values)
        assert np.all(model_x0_start>=model_lb), f"The initial values must be larger than the lower bound."
        assert np.all(model_x0_start<=model_ub), f"The initial values must be larger than the lower bound."
        cond_below = x0_start<lb
        cond_above = x0_start>ub
        assert np.all(x0_start>=lb), f"The initial values must be larger than the lower bound. {x0_start[cond_below][0]}<{lb[cond_below][0]} violates this."
        assert np.all(x0_start<=ub), f"The initial values must be larger than the lower bound. {x0_start[cond_above][0]}>{ub[cond_above][0]} violates this."

        
        print(f"Number of cores: {n_cores}")
        print(f"Number of estimated parameters: {ndim}")
        print(f"Number of temperatures: {n_temperature}")
        print(f"Number of ensemble walkers per chain: {n_walkers}")
        self.model.make_pickable()
        adaptive = False if n_temperature==1 else True
        betas = np.array([1]) if n_temperature==1 else make_ladder(ndim, n_temperature, Tmax=T_max)

        if n_cores==1:
            sampler = Sampler(n_walkers,
                            ndim,
                            loglike,
                            logprior,
                            adaptive=adaptive,
                            betas=betas)
        else:
            pool = multiprocessing.Pool(n_cores, maxtasksperchild=maxtasksperchild) #maxtasksperchild is set because the FMUs are leaking memory
            sampler = Sampler(n_walkers,
                            ndim,
                            loglike,
                            logprior,
                            adaptive=adaptive,
                            betas=betas,
                            mapper=pool.imap)

        chain = sampler.chain(x0_start)
        n_save_checkpoint = 50 if n_save_checkpoint is None else n_save_checkpoint
        result = {"integratedAutoCorrelatedTime": [],
                    "chain.swap_acceptance": None,
                    "chain.jump_acceptance": None,
                    "chain.logl": None,
                    "chain.logP": None,
                    "chain.x": None,
                    "chain.betas": None,
                    "component_id": [com.id for com in self.flat_component_list],
                    "component_attr": [attr for attr in self.flat_attr_list],
                    "theta_mask": self.theta_mask,
                    "standardDeviation": self.standardDeviation,
                    "startTime_train": self.startTime_train,
                    "endTime_train": self.endTime_train,
                    "stepSize_train": self.stepSize_train,
                    "gp_input_map": self.gp_input_map,
                    "n_par": self.n_par,
                    "n_par_map": self.n_par_map
                    }
        swap_acceptance = np.zeros((n_sample, n_temperature))
        jump_acceptance = np.zeros((n_sample, n_temperature))
        pbar = tqdm(enumerate(chain.iterate(n_sample)), total=n_sample)
        for i, ensemble in pbar:
            datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            des = f"Date: {datestr} logl: {str(int(np.max(chain.logl[:i+1,0,:])))}"
            pbar.set_description(des)
            result["integratedAutoCorrelatedTime"].append(chain.get_acts())
            if n_temperature>1:
                swap_acceptance[i] = np.sum(ensemble.swaps_accepted)/np.sum(ensemble.swaps_proposed)
            else:
                swap_acceptance[i] = np.nan
            jump_acceptance[i] = np.sum(ensemble.jumps_accepted)/np.sum(ensemble.jumps_proposed)
            if (i+1) % n_save_checkpoint == 0:
                result["chain.logl"] = chain.logl[:i+1]
                result["chain.logP"] = chain.logP[:i+1]
                result["chain.x"] = chain.x[:i+1]
                result["chain.betas"] = chain.betas[:i+1]
                result["chain.swap_acceptance"] = swap_acceptance[:i+1]
                result["chain.jump_acceptance"] = jump_acceptance[:i+1]
                if use_pickle:
                    with open(self.chain_savedir_pickle, 'wb') as handle:
                        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                if use_npz:
                    np.savez_compressed(self.chain_savedir_npz, **result)
        if n_cores>1:
            pool.close()

    def get_solution(self) -> Dict:
        """
        Get the current solution of the estimation problem.

        Returns:
            Dict: A dictionary containing the current solution, including MSE, RMSE,
                  number of objective function evaluations, and estimated parameter values.
        """
        sol_dict = {}
        sol_dict["MSE"] = self.monitor.get_MSE()
        sol_dict["RMSE"] = self.monitor.get_RMSE()
        sol_dict["n_obj_eval"] = self.n_obj_eval
        for component, attr_list in self.targetParameters.items():
            sol_dict[component.id] = []
            for attr in attr_list:
                sol_dict[component.id].append(rgetattr(component, attr))
        return sol_dict

    def _loglike_wrapper(self, theta: np.ndarray) -> float:
        """
        Wrapper for the log-likelihood function to handle boundary conditions and exceptions.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-likelihood value.
        """
        outsideBounds = np.any(theta<self.lb) or np.any(theta>self.ub)
        if outsideBounds:
            return -1e+10
        try:
            loglike = self._loglike(theta)
        except FMICallException as inst:
            return -1e+10
        return loglike

    def _loglike(self, theta: np.ndarray) -> float:
        """
        Calculate the log-likelihood for given parameters.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-likelihood value.
        """
        theta = theta[self.theta_mask]
        self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
        n_time_prev = 0
        self.simulation_readings = {com.id: np.zeros((self.n_timesteps)) for com in self.targetMeasuringDevices}
        for startTime_, endTime_, stepSize_  in zip(self.startTime_train, self.endTime_train, self.stepSize_train):
            self.simulator.simulate(self.model,
                                    stepSize=stepSize_,
                                    startTime=startTime_,
                                    endTime=endTime_,
                                    trackGradients=self.trackGradients,
                                    targetParameters=self.targetParameters,
                                    targetMeasuringDevices=self.targetMeasuringDevices,
                                    show_progress_bar=False)
            n_time = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            for measuring_device in self.targetMeasuringDevices:
                y_model = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]#/self.targetMeasuringDevices[measuring_device]["scale_factor"]
                self.simulation_readings[measuring_device.id][n_time_prev:n_time_prev+n_time] = y_model
            n_time_prev += n_time
        loglike = 0
        self.loglike_dict = {}
        for measuring_device in self.targetMeasuringDevices:
            simulation_readings = self.simulation_readings[measuring_device.id]
            actual_readings = self.actual_readings[measuring_device.id]
            res = (actual_readings-simulation_readings)/self.targetMeasuringDevices[measuring_device]["scale_factor"]
            ss = np.sum(res**2, axis=0)
            sd = self.targetMeasuringDevices[measuring_device]["standardDeviation"]/self.targetMeasuringDevices[measuring_device]["scale_factor"]
            loglike_ = -0.5*np.sum(ss/(sd**2))
            loglike += loglike_
            self.loglike_dict[measuring_device.id] = loglike_
        return loglike
    
    
    def _loglike_gaussian_process_wrapper(self, theta: np.ndarray) -> float:
        """
        Wrapper for the Gaussian Process log-likelihood function to handle boundary conditions and exceptions.

        Args:
            theta (np.ndarray): Parameter vector including GP hyperparameters.

        Returns:
            float: Log-likelihood value.
        """
        outsideBounds = np.any(theta<self.lb) or np.any(theta>self.ub)
        if outsideBounds:
            return -1e+10
        try:
            loglike = self._loglike_gaussian_process(theta)
        except FMICallException as inst:
            return -1e+10
        except np.linalg.LinAlgError as inst:
            return -1e+10
        if loglike==np.inf:
            return -1e+10
        elif loglike==-np.inf:
            return -1e+10
        elif np.isnan(loglike):
            return -1e+10
        return loglike

    def _loglike_gaussian_process(self, theta: np.ndarray) -> float:
        """
        Calculate the log-likelihood for given parameters using Gaussian Process modeling.

        Args:
            theta (np.ndarray): Parameter vector including GP hyperparameters.

        Returns:
            float: Log-likelihood value.
        """
        theta_kernel = np.exp(theta[-self.n_par:])
        theta = theta[:-self.n_par]
        theta = theta[self.theta_mask]
        self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list) #Some parameters are shared - therefore, we use a mask to select and expand the correct parameters
        n_time_prev = 0
        self.simulation_readings = {com.id: np.zeros((self.n_timesteps)) for com in self.targetMeasuringDevices}
        for i, (startTime_, endTime_, stepSize_) in enumerate(zip(self.startTime_train, self.endTime_train, self.stepSize_train)):
            self.simulator.simulate(self.model,
                                    stepSize=stepSize_,
                                    startTime=startTime_,
                                    endTime=endTime_,
                                    trackGradients=self.trackGradients,
                                    targetParameters=self.targetParameters,
                                    targetMeasuringDevices=self.targetMeasuringDevices,
                                    show_progress_bar=False)
            n_time = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            for measuring_device in self.targetMeasuringDevices:
                y_model = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]#/self.targetMeasuringDevices[measuring_device]["scale_factor"]
                self.simulation_readings[measuring_device.id][n_time_prev:n_time_prev+n_time] = y_model
            n_time_prev += n_time

            if self.gp_input_type=="closest":
                self.simulator.get_gp_input(self.targetMeasuringDevices, startTime_, endTime_, stepSize_, input_type="closest", add_time=self.gp_add_time, max_inputs=self.gp_max_inputs)
                if i==0:
                    self.gp_input = self.simulator.gp_input
                    for measuring_device in self.targetMeasuringDevices:
                        self.gp_input[measuring_device.id] = self.gp_input[measuring_device.id][self.n_initialization_steps:,:]
                else:
                    gp_input = self.simulator.gp_input
                    for measuring_device in self.targetMeasuringDevices:
                        x = gp_input[measuring_device.id][self.n_initialization_steps:,:]
                        self.gp_input[measuring_device.id] = np.concatenate((self.gp_input[measuring_device.id], x), axis=0)
        
        loglike = 0
        n_prev = 0
        for measuring_device in self.targetMeasuringDevices:
            x = self.gp_input[measuring_device.id]
            simulation_readings = self.simulation_readings[measuring_device.id]
            actual_readings = self.actual_readings[measuring_device.id]
            res = (actual_readings-simulation_readings)/self.targetMeasuringDevices[measuring_device]["scale_factor"]
            n = self.n_par_map[measuring_device.id]
            scale_lengths = theta_kernel[n_prev:n_prev+n]
            a = scale_lengths[0]
            scale_lengths = scale_lengths[1:]
            s = int(scale_lengths.size)
            scale_lengths_base = scale_lengths[:s]
            axes = list(range(s))
            std = self.targetMeasuringDevices[measuring_device]["standardDeviation"]/self.targetMeasuringDevices[measuring_device]["scale_factor"]
            kernel1 = kernels.Matern32Kernel(metric=scale_lengths_base, ndim=s, axes=axes)
            kernel = kernel1# + kernel2*kernel3
            gp = george.GP(a*kernel, solver=george.HODLRSolver, tol=1e-8, min_size=500)#, white_noise=np.log(var))#(tol=0.01))
            gp.compute(x, std)
            loglike_ = gp.lnlikelihood(res)
            loglike += loglike_
            n_prev += n
        return loglike
    
    def uniform_logprior(self, theta: np.ndarray) -> float:
        """
        Calculate the log-prior probability assuming uniform prior distribution.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-prior probability.
        """
        outsideBounds = np.any(theta<self.lb) or np.any(theta>self.ub)
        p = np.sum(np.log(1/(self.ub-self.lb)))
        return -np.inf if outsideBounds else p
    
    def gaussian_logprior(self, theta: np.ndarray) -> float:
        """
        Calculate the log-prior probability assuming Gaussian prior distribution.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-prior probability.
        """
        const = np.log(1/(self.standardDeviation_x0*np.sqrt(2*np.pi)))
        p = -0.5*((self.x0-theta)/self.standardDeviation_x0)**2
        return np.sum(const+p)

    def gaussian_model_uniform_noise_logprior(self, theta: np.ndarray) -> float:
        """
        Calculate the log-prior probability assuming Gaussian prior for model parameters
        and uniform prior for noise parameters.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-prior probability.
        """
        theta_noise = theta[self.n_par:]
        lb_noise = self.lb[self.n_par:]
        ub_noise = self.ub[self.n_par:]
        outsideBounds = np.any(theta_noise<lb_noise) or np.any(theta_noise>ub_noise)
        if outsideBounds:
            return -np.inf
        p_noise = np.sum(np.log(1/(ub_noise-lb_noise)))
        
        theta_model = theta[:-self.n_par]
        x0_model = self.x0[:-self.n_par]
        standardDeviation_x0_model = self.standardDeviation_x0[:-self.n_par]
        const = np.log(1/(standardDeviation_x0_model*np.sqrt(2*np.pi)))
        p = -0.5*((x0_model-theta_model)/standardDeviation_x0_model)**2
        p_model = np.sum(const+p)

        return p_model+p_noise

    def ls(self, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> OptimizeResult:
        """
        Run least squares estimation.

        Args:
            x0 (np.ndarray): Initial guess for the parameters.
            lb (np.ndarray): Lower bounds for the parameters.
            ub (np.ndarray): Upper bounds for the parameters.

        Returns:
            OptimizeResult: The optimization result returned by scipy.optimize.least_squares.
        """
        assert np.all(self.x0>=self.lb), "The provided x0 must be larger than the provided lower bound lb"
        assert np.all(self.x0<=self.ub), "The provided x0 must be smaller than the provided upper bound ub"
        assert np.all(np.abs(self.x0-self.lb)>self.tol), f"The difference between x0 and lb must be larger than {str(self.tol)}"
        assert np.all(np.abs(self.x0-self.ub)>self.tol), f"The difference between x0 and ub must be larger than {str(self.tol)}"
        datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = str('{}{}'.format(datestr, '.pickle'))
        self.ls_res_savedir, isfile = self.model.get_dir(folder_list=["model_parameters", "estimation_results", "LS_result"], filename=filename)
        ls_result = least_squares(self._res_fun_LS_exception_wrapper, x0, bounds=(lb, ub), verbose=2) #Change verbose to 2 to see the optimization progress
        with open(self.ls_res_savedir, 'wb') as handle:
            pickle.dump(ls_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return ls_result
    
    def _res_fun_ls(self, theta: np.ndarray) -> np.ndarray:
        """
        Residual function for least squares estimation.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            np.ndarray: Array of residuals.
        """
        self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
        n_time_prev = 0
        self.simulation_readings = {com.id: np.zeros((self.n_timesteps)) for com in self.targetMeasuringDevices}
        for startTime_, endTime_, stepSize_  in zip(self.startTime_train, self.endTime_train, self.stepSize_train):
            self.simulator.simulate(self.model,
                                    stepSize=stepSize_,
                                    startTime=startTime_,
                                    endTime=endTime_,
                                    trackGradients=self.trackGradients,
                                    targetParameters=self.targetParameters,
                                    targetMeasuringDevices=self.targetMeasuringDevices,
                                    show_progress_bar=False)
            n_time = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            for measuring_device in self.targetMeasuringDevices:
                y_model = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]#/self.targetMeasuringDevices[measuring_device]["scale_factor"]
                self.simulation_readings[measuring_device.id][n_time_prev:n_time_prev+n_time] = y_model
            n_time_prev += n_time
        res = np.zeros((self.n_timesteps, len(self.targetMeasuringDevices)))
        for j, measuring_device in enumerate(self.targetMeasuringDevices):
            simulation_readings = self.simulation_readings[measuring_device.id]
            actual_readings = self.actual_readings[measuring_device.id]
            res[:,j] = (actual_readings-simulation_readings)/self.targetMeasuringDevices[measuring_device]["scale_factor"]
        res = res.flatten()
        return res
    
    def _res_fun_ls_exception_wrapper(self, theta: np.ndarray) -> np.ndarray:
        """
        Wrapper for the residual function to handle exceptions.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            np.ndarray: Array of residuals or a large value if an exception occurs.
        """
        try:
            res = self._res_fun_ls(theta)
        except FMICallException as inst:
            res = 10e+10*np.ones((len(self.targetMeasuringDevices)))
        return res