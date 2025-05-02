from __future__ import annotations
import multiprocessing
import math
from tqdm import tqdm
from twin4build.utils.rgetattr import rgetattr
import numpy as np
from ptemcee.sampler import Sampler, make_ladder
import datetime
import pickle
from fmpy.fmi2 import FMICallException
from scipy.optimize import least_squares, OptimizeResult
import george
from george import kernels
# from george.metrics import Metric
# import twin4build.simulator.simulator as simulator
import twin4build.core as core
import pygad
import functools
from scipy._lib._array_api import atleast_nd, array_namespace

from typing import Union, List, Dict, Optional, Callable, TYPE_CHECKING, Any

# if TYPE_CHECKING:
#     import twin4build.model.model as model


class MCMCEstimationResult(dict):
    def __init__(self,
                 integratedAutoCorrelatedTime: List=None,
                 chain_swap_acceptance: np.array=None,
                 chain_jump_acceptance: np.array=None,
                 chain_logl: np.array=None,
                 chain_logP: np.array=None,
                 chain_x: np.array=None,
                 chain_betas: np.array=None,
                 chain_T: np.array=None,
                 component_id: List[str]=None,
                 component_attr: List[str]=None,
                 theta_mask: np.array=None,
                 standardDeviation: np.array=None,
                 startTime_train: List[datetime.datetime]=None,
                 endTime_train: List[datetime.datetime]=None,
                 stepSize_train: List[int]=None,
                 gp_input_map: np.array=None,
                 n_par: int=None,
                 n_par_map: dict=None):
        super().__init__(integratedAutoCorrelatedTime=integratedAutoCorrelatedTime,
                         chain_swap_acceptance=chain_swap_acceptance,
                         chain_jump_acceptance=chain_jump_acceptance,
                         chain_logl=chain_logl,
                         chain_logP=chain_logP,
                         chain_x=chain_x,
                         chain_betas=chain_betas,
                         chain_T=chain_T,
                         component_id=component_id,
                         component_attr=component_attr,
                         theta_mask=theta_mask,
                         standardDeviation=standardDeviation,
                         startTime_train=startTime_train,
                         endTime_train=endTime_train,
                         stepSize_train=stepSize_train,
                         gp_input_map=gp_input_map,
                         n_par=n_par,
                         n_par_map=n_par_map)

    def __copy__(self):
        return MCMCEstimationResult(**self)

    def copy(self):
        return self.__copy__()

class LSEstimationResult(dict):
    def __init__(self,
                 result_x: np.array=None,
                 component_id: List[str]=None,
                 component_attr: List[str]=None,
                 theta_mask: np.array=None,
                 startTime_train: List[datetime.datetime]=None,
                 endTime_train: List[datetime.datetime]=None,
                 stepSize_train: List[int]=None):
        super().__init__(result_x=result_x,
                         component_id=component_id,
                         component_attr=component_attr,
                         theta_mask=theta_mask,
                         startTime_train=startTime_train,
                         endTime_train=endTime_train,
                         stepSize_train=stepSize_train)

    def __copy__(self):
        return LSEstimationResult(**self)

    
    def copy(self):
        return self.__copy__()

class Estimator():
    """
    A class for parameter estimation in the twin4build framework.

    This class provides methods for estimating model parameters using various
    approaches, with a focus on Markov Chain Monte Carlo (MCMC) methods and
    Gaussian Process (GP) modeling.

    Attributes:
        model (Model): The model to perform estimation on.
        simulator (Simulator): The simulator instance for running simulations.
        x0 (np.ndarray): Initial parameter values.
        lb (np.ndarray): Lower bounds for parameters.
        ub (np.ndarray): Upper bounds for parameters.
        tol (float): Tolerance for parameter bounds checking.
        ndim (int): Number of dimensions/parameters.
        standardDeviation_x0 (np.ndarray): Standard deviation for parameter initialization.
        n_par (int): Number of additional parameters (e.g., for GP).
        n_par_map (Dict): Mapping of parameter indices.
        gp_input_map (Dict): Mapping of GP input features.
        targetMeasuringDevices (Dict): Target devices for estimation.
    """

    def __init__(self,
                model: Optional[core.Model] = None):
        self.model = model
        self.simulator = core.Simulator(model)
        self.tol = 1e-10
    
    def estimate(self,
                 targetParameters: Dict[str, Dict] = None,
                 targetMeasuringDevices: Dict[str, Dict] = None,
                 n_initialization_steps: int = 60,
                 startTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 endTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 stepSize: Union[float, List[float]] = None,
                 verbose: bool = False,
                 method: str = "MCMC",
                 options: Dict = None) -> None:
        """Perform parameter estimation using specified method and configuration.

        This method sets up and executes the parameter estimation process, supporting multiple
        estimation methods including MCMC, least squares (LS), and genetic algorithms (GA).

        Parameters:
            targetParameters (Dict[str, Dict], optional): 
            
                Dictionary containing:

                    - "private": Parameters unique to each component
                    - "shared": Parameters shared across components

                    Each parameter entry contains:

                        - "components": List of components or single component
                        - "x0": List of initial values or single initial value
                        - "lb": List of lower bounds or single lower bound
                        - "ub": List of upper bounds or single upper bound

            targetMeasuringDevices (Dict[str, Dict], optional):

                Dictionary mapping measuringdevice IDs to their configuration:

                    - "standardDeviation": Measurement uncertainty
                    - "scale_factor": Scaling factor for measurements

            n_initialization_steps (int, optional):
                Number of steps to skip during initialization. Defaults to 60.

            startTime (Union[datetime.datetime, List[datetime]], optional):
                Start time(s) for estimation period(s).

            endTime (Union[datetime.datetime, List[datetime]], optional):
                End time(s) for estimation period(s).

            stepSize (Union[float, List[float]], optional):
                Step size(s) for simulation.

            verbose (bool, optional):
                Whether to print detailed output. Defaults to False.

            method (str, optional):
                Estimation method to use ("MCMC", "LS", or "GA"). Defaults to "MCMC".

            options (Dict, optional):
                Additional options for the chosen method:
                    For MCMC:
                        - "n_sample": Number of samples
                        - "n_temperature": Number of temperature chains
                        - "fac_walker": Walker scaling factor
                        - "prior": Prior distribution type
                        - "add_gp": Whether to use Gaussian processes

        Raises:
            AssertionError: If method is not one of ["MCMC", "LS", "GA"] or if input
                parameters are invalid.
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
                assert len(par_dict["ub"])==len(par_dict["components"]), f"The number of elements in the \"ub\" list must be equal to the number of components in the private dictionary for attribute {attr}."
        
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
        

        allowed_methods = ["MCMC","LS","GA"]
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
            assert endTime_>startTime_, "The endTime must be later than the startTime."
        self._startTime_train = startTime
        self._endTime_train = endTime
        self._stepSize_train = stepSize
        for startTime_, endTime_, stepSize_  in zip(self._startTime_train, self._endTime_train, self._stepSize_train):    
            self.model.cache(startTime=startTime_,
                            endTime=endTime_,
                            stepSize=stepSize_)

        # self.standardDeviation = np.array([el["standardDeviation"] for el in targetMeasuringDevices.values()])
        self._flat_component_list_private = [obj for par_dict in targetParameters["private"].values() for obj in par_dict["components"]]
        self._flat_attr_list_private = [attr for attr, par_dict in targetParameters["private"].items() for obj in par_dict["components"]]
        self._flat_component_list_shared = [obj for par_dict in targetParameters["shared"].values() for obj_list in par_dict["components"] for obj in obj_list]
        self._flat_attr_list_shared = [attr for attr, par_dict in targetParameters["shared"].items() for obj_list in par_dict["components"] for obj in obj_list]
        private_mask = np.arange(len(self._flat_component_list_private), dtype=int)
        shared_mask = []
        n = len(self._flat_component_list_private)
        k = 0
        for attr, par_dict in targetParameters["shared"].items():
            for obj_list in par_dict["components"]:
                for obj in obj_list:
                    shared_mask.append(k+n)
                k += 1
        shared_mask = np.array(shared_mask)
        self.flat_component_list = self._flat_component_list_private + self._flat_component_list_shared
        self.theta_mask = np.concatenate((private_mask, shared_mask)).astype(int)
        self.flat_attr_list = self._flat_attr_list_private + self._flat_attr_list_shared
        self.simulator.flat_component_list = self.flat_component_list
        self.simulator.flat_attr_list = self.flat_attr_list
        self.simulator.theta_mask = self.theta_mask
        self.simulator.targetParameters = targetParameters
        self.simulator.targetMeasuringDevices = targetMeasuringDevices
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
        self.n_obj_eval = 0
        self.best_loss = math.inf
        self.n_timesteps = 0
        for i, (startTime_, endTime_, stepSize_)  in enumerate(zip(self._startTime_train, self._endTime_train, self._stepSize_train)):
            self.simulator.get_simulation_timesteps(startTime_, endTime_, stepSize_)
            self.n_timesteps += len(self.simulator.secondTimeSteps)-self.n_initialization_steps
            actual_readings = self.simulator.get_actual_readings(startTime=startTime_, endTime=endTime_, stepSize=stepSize_)
            if i==0:
                self.actual_readings = {}
                for measuring_device in self.targetMeasuringDevices:
                    self.actual_readings[measuring_device.id] = actual_readings[measuring_device.id].to_numpy()[self.n_initialization_steps:]
            else:
                for measuring_device in self.targetMeasuringDevices:
                    self.actual_readings[measuring_device.id] = np.concatenate((self.actual_readings[measuring_device.id], actual_readings[measuring_device.id].to_numpy()[self.n_initialization_steps:]), axis=0)

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

        self._x0 = np.array(x0)
        self._lb = np.array(lb)
        self._ub = np.array(ub)
        self.ndim = int(self.theta_mask[-1]+1)

        if method == "MCMC":
            if options is None:
                options = {}
            self.mcmc(**options)
        elif method == "LS":
            if options is None:
                options = {}
            self.ls(**options)
        elif method == "GA":
            if options is None:
                options = {}
            self.ga(**options)

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
             prior: str = "uniform",
             model_prior: str = None, 
             noise_prior: str = None,
             walker_initialization: str = "uniform",
             model_walker_initialization: str = None,
             noise_walker_initialization: str = None,
             add_gp: bool = False,
             gp_input_type: str = "closest",
             gp_add_time: bool = True,
             gp_max_inputs: int = 3,
             maxtasksperchild: int = 100,
             n_save_checkpoint: int = None,
             use_pickle: bool = True,
             use_npz: bool = True) -> None:
        """
        Perform MCMC parameter estimation with optional Gaussian Process modeling.

        This method implements parallel tempering MCMC with ensemble sampling,
        supporting both standard parameter estimation and GP-based inference.

        Args:
            n_sample (int): Number of MCMC samples to generate. Defaults to 10000.
            n_temperature (int): Number of temperature chains. Defaults to 15.
            fac_walker (int): Factor for number of walkers. Defaults to 2.
            T_max (float): Maximum temperature. Defaults to infinity.
            n_cores (int): Number of CPU cores to use. Defaults to all available.
            prior (str): Prior distribution type ("uniform" or "gaussian").
            model_prior (str, optional): Specific prior for model parameters.
            noise_prior (str, optional): Specific prior for noise parameters.
            walker_initialization (str): How to initialize walkers ("uniform", "hypercube", etc.).
            model_walker_initialization (str, optional): Specific initialization for model parameters.
            noise_walker_initialization (str, optional): Specific initialization for noise parameters.
            add_gp (bool): Whether to use Gaussian Process modeling. Defaults to False.
            gp_input_type (str): Type of GP inputs ("closest", "boundary", "time").
            gp_add_time (bool): Whether to add time as GP input. Defaults to True.
            gp_max_inputs (int): Maximum number of GP input features. Defaults to 3.
            maxtasksperchild (int): Max tasks per child process. Defaults to 100.
            n_save_checkpoint (int, optional): Save checkpoints every N steps.
            use_pickle (bool): Whether to save results as pickle. Defaults to True.
            use_npz (bool): Whether to save results as npz. Defaults to True.

        Raises:
            AssertionError: If parameter bounds or initialization conditions are violated.
            Exception: If GP initialization fails or chain log is missing.

        Notes:
            - Uses parallel tempering for better exploration of parameter space
            - Supports both standard and GP-based inference
            - Handles parameter bounds and initialization carefully
            - Implements checkpointing for long runs
            - Memory management for FMU simulations
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
            assert np.all(self._x0>=self._lb), "The provided x0 must be larger than the provided lower bound lb"
            assert np.all(self._x0<=self._ub), "The provided x0 must be smaller than the provided upper bound ub"
            a = (np.abs(self._x0-self._lb)>self.tol)==False
            b = (np.abs(self._x0-self._ub)>self.tol)==False
            c = a[self.theta_mask]
            d = b[self.theta_mask]
            assert np.all(np.abs(self._x0-self._lb)>self.tol), f"The difference between x0 and lb must be larger than {str(self.tol)}. {np.array(self.flat_attr_list)[c]} violates this condition." 
            assert np.all(np.abs(self._x0-self._ub)>self.tol), f"The difference between x0 and ub must be larger than {str(self.tol)}. {np.array(self.flat_attr_list)[d]} violates this condition."

        
        for startTime_, endTime_, stepSize_  in zip(self._startTime_train, self._endTime_train, self._stepSize_train):    
            self.model.cache(startTime=startTime_,
                            endTime=endTime_,
                            stepSize=stepSize_)
        
        datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_pickle = str('{}{}'.format(datestr, '_mcmc.pickle'))
        filename_npz = str('{}{}'.format(datestr, '_mcmc.npz'))
        self.result_savedir_pickle, isfile = self.model.get_dir(folder_list=["model_parameters", "estimation_results", "chain_logs"], filename=filename_pickle)
        self.result_savedir_npz, isfile = self.model.get_dir(folder_list=["model_parameters", "estimation_results", "chain_logs"], filename=filename_npz)
        
        self.gp_input_type = gp_input_type
        self.gp_add_time = gp_add_time
        self.gp_max_inputs = gp_max_inputs

        assert (model_prior is None and noise_prior is None) or (model_prior is not None and noise_prior is not None), "\"model_prior\" and \"noise_prior\" must both be either None or set to one of the available priors."
        if model_prior=="gaussian" and noise_prior=="uniform":
            logprior = self._gaussian_model_uniform_noise_logprior
        elif model_prior=="sample_gaussian" and noise_prior=="uniform":
            x = self.model.result["chain_x"][:,0,:,:]
            logl = self.model.result["chain.logl"][:,0,:]
            best_tuple = np.unravel_index(logl.argmax(), logl.shape)
            self._x0 = x[best_tuple + (slice(None),)]
            logprior = self._gaussian_model_uniform_noise_logprior
        elif model_prior=="uniform" and noise_prior=="gaussian":
            raise Exception("Not implemented")
        elif prior=="uniform":
            logprior = self._uniform_logprior
        elif prior=="gaussian":
            logprior = self._gaussian_logprior


        self.gp_input_map = None
        ndim = self.ndim
        add_par = 1 # We add the following parameters: "a"
        self.n_par = 0
        self.n_par_map = {}
        lower_bound = -3
        upper_bound = 5

        x0_time = 8
        lower_bound_time = 0 #1 second
        upper_bound_time = 10 #3600 seconds

        diff_lower = np.abs(self._x0-self._lb)
        diff_upper = np.abs(self._ub-self._x0)
        self.standardDeviation_x0 = np.minimum(diff_lower, diff_upper)/2 #Set the standard deviation such that around 95% of the values are within the bounds

        if add_gp:
            n_par = self.model.result["n_par"]
            self.gp_input_map = self.model.result["gp_input_map"]
            if hasattr(self.model, "chain_log") and (self.model.result["chain_x"].shape[3]==ndim or self.model.result["chain_x"].shape[3]==ndim+n_par):
                x = self.model.result["chain_x"][:,0,:,:]
                r = 1e-5
                logl = self.model.result["chain.logl"][:,0,:]
                best_tuple = np.unravel_index(logl.argmax(), logl.shape)
                x0_ = x[best_tuple + (slice(None),)]
            else:
                raise Exception("The model does not contain the required chain_log attribute or the dimensions are wrong.")


            for i, (startTime_, endTime_, stepSize_)  in enumerate(zip(self._startTime_train, self._endTime_train, self._stepSize_train)):
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
                
                # actual_readings = self.simulator.get_actual_readings(startTime=startTime_, endTime=endTime_, stepSize=stepSize_)
                if i==0:
                    self.gp_input = gp_input
                    # self.actual_readings = {}
                    for measuring_device in self.targetMeasuringDevices:
                        self.gp_input[measuring_device.id] = self.gp_input[measuring_device.id][self.n_initialization_steps:,:]
                        # self.actual_readings[measuring_device.id] = actual_readings[measuring_device.id].to_numpy()[self.n_initialization_steps:]
                else:
                    for measuring_device in self.targetMeasuringDevices:
                        x = gp_input[measuring_device.id][self.n_initialization_steps:,:]
                        self.gp_input[measuring_device.id] = np.concatenate((self.gp_input[measuring_device.id], x), axis=0)
                        # self.actual_readings[measuring_device.id] = np.concatenate((self.actual_readings[measuring_device.id], actual_readings[measuring_device.id].to_numpy()[self.n_initialization_steps:]), axis=0)

            self.gp_lengthscale = self.simulator.get_gp_lengthscale(self.targetMeasuringDevices, self.gp_input)
            for j, measuring_device in enumerate(self.targetMeasuringDevices):
                n = self.gp_input[measuring_device.id].shape[1] ######################################################
                self.n_par += n+add_par
                self.n_par_map[measuring_device.id] = n+add_par
            self.gp_variance = self.simulator.get_gp_variance(self.targetMeasuringDevices, x0_, self._startTime_train, self._endTime_train, self._stepSize_train)
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

                self._x0 = np.append(self._x0, add_x0)
                self._lb = np.append(self._lb, add_lb)
                self._ub = np.append(self._ub, add_ub)
            loglike = self._loglike_gaussian_process_wrapper
            ndim = ndim+self.n_par
            self.standardDeviation_x0 = np.append(self.standardDeviation_x0, (upper_bound-lower_bound)/2*np.ones((self.n_par,))) ###################################################
        else:
            loglike = self._loglike_mcmc_wrapper


        n_walkers = int(ndim*fac_walker) #*4 #Round up to nearest even number and multiply_const by 2
        if add_gp and model_walker_initialization=="hypercube" and noise_walker_initialization=="uniform":
            r = 1e-5
            low = np.zeros((ndim-self.n_par,))

            x0_start = np.random.uniform(low=self._x0[:-self.n_par]-r, high=self._x0[:-self.n_par]+r, size=(n_temperature, n_walkers, ndim-self.n_par))
            # lb = np.resize(self._lb[:-self.n_par],(x0_start.shape))
            # ub = np.resize(self._ub[:-self.n_par],(x0_start.shape))
            # x0_start[x0_start<self._lb[:-self.n_par]] = lb[x0_start<self._lb[:-self.n_par]]
            # x0_start[x0_start>self._ub[:-self.n_par]] = ub[x0_start>self._ub[:-self.n_par]]
            model_x0_start = x0_start

            x0_start = np.random.uniform(low=self._lb[-self.n_par:], high=self._ub[-self.n_par:], size=(n_temperature, n_walkers, self.n_par))
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
            # low = self._lb if self._x0[:-self.n_par]-r*np.abs(self._x0[:-self.n_par])<self._lb else self._x0[:-self.n_par]-r*np.abs(self._x0[:-self.n_par])
            # high = self._ub if self._x0[:-self.n_par]+r*np.abs(self._x0[:-self.n_par])>self._ub else self._x0[:-self.n_par]+r*np.abs(self._x0[:-self.n_par])
            x0 = self._x0[:-self.n_par]
            lb = self._lb[:-self.n_par]
            low = np.zeros((ndim-self.n_par,))
            cond = (self._x0[:-self.n_par]-r*np.abs(self._x0[:-self.n_par]))<lb
            low[cond] = lb[cond]
            low[cond==False] = x0[cond==False]-r*np.abs(x0[cond==False])
            ub = self._ub[:-self.n_par]
            high = np.zeros((ndim-self.n_par,))
            cond = (self._x0[:-self.n_par]+r*np.abs(self._x0[:-self.n_par]))>ub
            high[cond] = ub[cond]
            high[cond==False] = x0[cond==False]+r*np.abs(x0[cond==False])
            model_x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))
            
            x0 = self._x0[-self.n_par:]
            lb = self._lb[-self.n_par:]
            low = np.zeros((self.n_par,))
            cond = (self._x0[-self.n_par:]-r*np.abs(self._x0[-self.n_par:]))<lb
            low[cond] = lb[cond]
            low[cond==False] = x0[cond==False]-r*np.abs(x0[cond==False])
            ub = self._ub[-self.n_par:]
            high = np.zeros((self.n_par,))
            cond = (self._x0[-self.n_par:]+r*np.abs(self._x0[-self.n_par:]))>ub
            high[cond] = ub[cond]
            high[cond==False] = x0[cond==False]+r*np.abs(x0[cond==False])

            noise_x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))
            x0_start = np.append(model_x0_start, noise_x0_start, axis=2)
            
        elif add_gp and model_walker_initialization=="hypercube" and noise_walker_initialization=="hypersphere":
            raise Exception("Not implemented")
        elif add_gp and model_walker_initialization=="sample" and noise_walker_initialization=="uniform":
            assert hasattr(self.model, "chain_log") and "chain_x" in self.model.result, "Model object has no chain log. Please load before starting estimation."
            assert self.model.result["chain_x"].shape[3]==ndim-self.n_par, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.result["chain_x"][-1,0,:,:]
            del self.model.result #We delete the chain log before initiating multiprocessing to save memory
            r = 1e-5
            if x.shape[0]==n_walkers:
                print("Using provided sample for initial walkers")
                # model_x0_start = np.random.uniform(low=x-r, high=x+r, size=(n_temperature, n_walkers, ndim-self.n_par))
                model_x0_start = x.reshape((n_temperature, n_walkers, ndim-self.n_par))
                # model_x0_start = np.random.uniform(low=model_x0_start-r, high=model_x0_start+r, size=(n_temperature, n_walkers, ndim-self.n_par))
            else: #upsample
                print("Upsampling initial walkers")
                # diff = n_walkers-x.shape[0]
                ind = np.arange(x.shape[0])
                ind_sample = np.random.choice(ind, n_walkers)
                model_x0_start = x[ind_sample,:]
                lb = np.resize(self._lb,(model_x0_start.shape))
                low = np.zeros(model_x0_start.shape)
                cond = (model_x0_start-r*np.abs(model_x0_start))<lb
                low[cond] = lb[cond]
                low[cond==False] = model_x0_start[cond==False]-r*np.abs(model_x0_start[cond==False])
                ub = np.resize(self._ub,(model_x0_start.shape))
                high = np.zeros(model_x0_start.shape)
                cond = (model_x0_start+r*np.abs(model_x0_start))>ub
                high[cond] = ub[cond]
                high[cond==False] = model_x0_start[cond==False]+r*np.abs(model_x0_start[cond==False])
                model_x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim-self.n_par))
            
            x0_start = np.random.uniform(low=self._lb[-self.n_par:], high=self._ub[-self.n_par:], size=(n_temperature, n_walkers, self.n_par))
            noise_x0_start = x0_start

            x0_start = np.append(model_x0_start, noise_x0_start, axis=2)

        elif add_gp and model_walker_initialization=="sample" and noise_walker_initialization=="gaussian":
            assert hasattr(self.model, "chain_log") and "chain_x" in self.model.result, "Model object has no chain log. Please load before starting estimation."
            assert self.model.result["chain_x"].shape[3]==ndim-self.n_par or self.model.result["chain_x"].shape[3]==ndim, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.result["chain_x"][-1,0,:,:] if self.model.result["chain_x"].shape[3]==ndim-self.n_par else self.model.result["chain_x"][-1,0,:,:-self.n_par]
            del self.model.result #We delete the chain log before initiating multiprocessing to save memory
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
                lb = np.resize(self._lb[:-self.n_par],(model_x0_start.shape))
                low = np.zeros(model_x0_start.shape)
                cond = (model_x0_start-r*np.abs(model_x0_start))<lb
                low[cond] = lb[cond]
                low[cond==False] = model_x0_start[cond==False]-r*np.abs(model_x0_start[cond==False])
                ub = np.resize(self._ub[:-self.n_par],(model_x0_start.shape))
                high = np.zeros(model_x0_start.shape)
                cond = (model_x0_start+r*np.abs(model_x0_start))>ub
                high[cond] = ub[cond]
                high[cond==False] = model_x0_start[cond==False]+r*np.abs(model_x0_start[cond==False])
                model_x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim-self.n_par))
            
            noise_x0_start = self.sample_bounded_gaussian(n_temperature, n_walkers, self.n_par, self._x0[-self.n_par:], self._lb[-self.n_par:], self._ub[-self.n_par:], self.standardDeviation_x0[-self.n_par:])
            x0_start = np.append(model_x0_start, noise_x0_start, axis=2)

        elif add_gp and model_walker_initialization=="sample_hypercube" and noise_walker_initialization=="hypercube":
            assert hasattr(self.model, "chain_log") and "chain_x" in self.model.result, "Model object has no chain log. Please load before starting estimation."
            assert self.model.result["chain_x"].shape[3]==ndim-self.n_par, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.result["chain_x"][:,0,:,:]
            r = 1e-5
            logl = self.model.result["chain.logl"][:,0,:]
            best_tuple = np.unravel_index(logl.argmax(), logl.shape)
            x0_ = x[best_tuple + (slice(None),)]
            x0_ = np.concatenate((x0_, self._x0[-self.n_par:]))
            low = np.zeros((ndim,))
            cond = (x0_-r*np.abs(x0_))<self._lb
            low[cond] = self._lb[cond]
            low[cond==False] = x0_[cond==False]-r*np.abs(x0_[cond==False])
            high = np.zeros((ndim,))
            cond = (x0_+r*np.abs(x0_))>self._ub
            high[cond] = self._ub[cond]
            high[cond==False] = x0_[cond==False]+r*np.abs(x0_[cond==False])
            x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))
            del self.model.result #We delete the chain log before initiating multiprocessing to save memory
            del x

        elif walker_initialization=="uniform":
            x0_start = np.random.uniform(low=self._lb, high=self._ub, size=(n_temperature, n_walkers, ndim))
        elif walker_initialization=="gaussian":
            x0_start = self.sample_bounded_gaussian(n_temperature, n_walkers, ndim, self._x0, self._lb, self._ub, self.standardDeviation_x0)
        elif walker_initialization=="hypersphere":
            r = 1e-5
            nrem = n_walkers*n_temperature
            x0_ = np.resize(self._x0,(nrem, ndim))
            lb = np.resize(self._lb,(nrem, ndim))
            ub = np.resize(self._ub,(nrem, ndim))
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
            cond = (self._x0-r*np.abs(self._x0))<self._lb
            low[cond] = self._lb[cond]
            low[cond==False] = self._x0[cond==False]-r*np.abs(self._x0[cond==False])
            high = np.zeros((ndim,))
            cond = (self._x0+r*np.abs(self._x0))>self._ub
            high[cond] = self._ub[cond]
            high[cond==False] = self._x0[cond==False]+r*np.abs(self._x0[cond==False])
            x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))


        elif walker_initialization=="sample_hypercube":
            assert hasattr(self.model, "chain_log") and "chain_x" in self.model.result, "Model object has no chain log. Please load before starting estimation."
            assert self.model.result["chain_x"].shape[3]==ndim, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.result["chain_x"][:,0,:,:]
            r = 1e-5
            logl = self.model.result["chain.logl"][:,0,:]
            best_tuple = np.unravel_index(logl.argmax(), logl.shape)
            x0_ = x[best_tuple + (slice(None),)]
            assert np.all(x0_>=self._lb), "The provided x0 must be larger than the provided lower bound lb"
            low = np.zeros((ndim,))
            cond = (x0_-r*np.abs(x0_))<self._lb
            low[cond] = self._lb[cond]
            low[cond==False] = x0_[cond==False]-r*np.abs(x0_[cond==False])
            assert np.all(low>=self._lb), "The provided x0 must be larger than the provided lower bound lb"
            high = np.zeros((ndim,))
            cond = (x0_+r*np.abs(x0_))>self._ub
            high[cond] = self._ub[cond]
            high[cond==False] = x0_[cond==False]+r*np.abs(x0_[cond==False])
            x0_start = np.random.uniform(low=low, high=high, size=(n_temperature, n_walkers, ndim))
            del self.model.result #We delete the chain log before initiating multiprocessing to save memory
            del x

        elif walker_initialization=="sample_gaussian":
            assert hasattr(self.model, "chain_log") and "chain_x" in self.model.result, "Model object has no chain log. Please load before starting estimation."
            assert self.model.result["chain_x"].shape[3]==ndim, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.result["chain_x"][:,0,:,:]
            logl = self.model.result["chain.logl"][:,0,:]
            best_tuple = np.unravel_index(logl.argmax(), logl.shape)
            x0_ = x[best_tuple + (slice(None),)]
            diff_lower = np.abs(x0_-self._lb)
            diff_upper = np.abs(self._ub-x0_)
            self.standardDeviation_x0 = np.minimum(diff_lower, diff_upper)/2 #Set the standard deviation such that around 95% of the values are within the bounds
            x0_start = np.random.normal(loc=x0_, scale=self.standardDeviation_x0, size=(n_temperature, n_walkers, ndim))
            del self.model.result #We delete the chain log before initiating multiprocessing to save memory
            del x

        elif walker_initialization=="sample":
            assert hasattr(self.model, "chain_log") and "chain_x" in self.model.result, "Model object has no chain log. Please load before starting estimation."
            assert self.model.result["chain_x"].shape[3]==ndim-self.n_par, "The amount of estimated parameters in the chain log is not equal to the number of estimated parameters in the given estimation problem."
            x = self.model.result["chain_x"][-1,0,:,:]
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
        lb = np.resize(self._lb,(x0_start.shape))
        ub = np.resize(self._ub,(x0_start.shape))
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
        result = MCMCEstimationResult(integratedAutoCorrelatedTime=[],
                                      component_id=[c.id for c in self.flat_component_list],
                                      component_attr=self.flat_attr_list,
                                      theta_mask=self.theta_mask,
                                      standardDeviation=self.standardDeviation_x0,
                                      startTime_train=self._startTime_train,
                                      endTime_train=self._endTime_train,
                                      stepSize_train=self._stepSize_train,
                                      gp_input_map=self.gp_input_map,
                                      n_par=self.n_par,
                                      n_par_map=self.n_par_map,
                                      )
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
                result["chain_x"] = chain.x[:i+1]
                result["chain.betas"] = chain.betas[:i+1]
                result["chain.T"] = 1/chain.betas[:i+1]
                result["chain.swap_acceptance"] = swap_acceptance[:i+1]
                result["chain.jump_acceptance"] = jump_acceptance[:i+1]
                if use_pickle:
                    with open(self.result_savedir_pickle, 'wb') as handle:
                        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                if use_npz:
                    np.savez_compressed(self.result_savedir_npz, **result)
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

    def _loglike_mcmc_wrapper(self, theta: np.ndarray) -> float:
        """
        Wrapper for the log-likelihood function to handle boundary conditions and exceptions.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-likelihood value.
        """
        outsideBounds = np.any(theta<self._lb) or np.any(theta>self._ub)
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
        for startTime_, endTime_, stepSize_  in zip(self._startTime_train, self._endTime_train, self._stepSize_train):
            self.simulator.simulate(self.model,
                                    stepSize=stepSize_,
                                    startTime=startTime_,
                                    endTime=endTime_,
                                    show_progress_bar=False)
            n_time = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            for measuring_device in self.targetMeasuringDevices:
                y_model = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
                self.simulation_readings[measuring_device.id][n_time_prev:n_time_prev+n_time] = y_model
            n_time_prev += n_time
        loglike = 0
        self.loglike_dict = {}
        for measuring_device in self.targetMeasuringDevices:
            simulation_readings = self.simulation_readings[measuring_device.id]
            actual_readings = self.actual_readings[measuring_device.id]
            res = (actual_readings-simulation_readings)
            sd = self.targetMeasuringDevices[measuring_device]["standardDeviation"]
            loglike_ = -1*np.sum(((0.5)**0.5*res/sd)**2)
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
        outsideBounds = np.any(theta<self._lb) or np.any(theta>self._ub)
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
        for i, (startTime_, endTime_, stepSize_) in enumerate(zip(self._startTime_train, self._endTime_train, self._stepSize_train)):
            self.simulator.simulate(self.model,
                                    stepSize=stepSize_,
                                    startTime=startTime_,
                                    endTime=endTime_,
                                    show_progress_bar=False)
            n_time = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            for measuring_device in self.targetMeasuringDevices:
                y_model = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
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
            res = (actual_readings-simulation_readings)
            n = self.n_par_map[measuring_device.id]
            scale_lengths = theta_kernel[n_prev:n_prev+n]
            a = scale_lengths[0]
            scale_lengths = scale_lengths[1:]
            s = int(scale_lengths.size)
            scale_lengths_base = scale_lengths[:s]
            axes = list(range(s))
            std = self.targetMeasuringDevices[measuring_device]["standardDeviation"]
            kernel1 = kernels.Matern32Kernel(metric=scale_lengths_base, ndim=s, axes=axes)
            kernel = kernel1# + kernel2*kernel3
            gp = george.GP(a*kernel, solver=george.HODLRSolver, tol=1e-8, min_size=500)#, white_noise=np.log(var))#(tol=0.01))
            gp.compute(x, std)
            loglike_ = gp.lnlikelihood(res)
            loglike += loglike_
            n_prev += n
        return loglike
    
    def _uniform_logprior(self, theta: np.ndarray) -> float:
        """
        Calculate the log-prior probability assuming uniform prior distribution.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-prior probability.
        """
        outsideBounds = np.any(theta<self._lb) or np.any(theta>self._ub)
        p = np.sum(np.log(1/(self._ub-self._lb)))
        return -np.inf if outsideBounds else p
    
    def _gaussian_logprior(self, theta: np.ndarray) -> float:
        """
        Calculate the log-prior probability assuming Gaussian prior distribution.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-prior probability.
        """
        const = np.log(1/(self.standardDeviation_x0*np.sqrt(2*np.pi)))
        p = -0.5*((self._x0-theta)/self.standardDeviation_x0)**2
        return np.sum(const+p)

    def _gaussian_model_uniform_noise_logprior(self, theta: np.ndarray) -> float:
        """
        Calculate the log-prior probability assuming Gaussian prior for model parameters
        and uniform prior for noise parameters.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            float: Log-prior probability.
        """
        theta_noise = theta[self.n_par:]
        lb_noise = self._lb[self.n_par:]
        ub_noise = self._ub[self.n_par:]
        outsideBounds = np.any(theta_noise<lb_noise) or np.any(theta_noise>ub_noise)
        if outsideBounds:
            return -np.inf
        p_noise = np.sum(np.log(1/(ub_noise-lb_noise)))
        
        theta_model = theta[:-self.n_par]
        x0_model = self._x0[:-self.n_par]
        standardDeviation_x0_model = self.standardDeviation_x0[:-self.n_par]
        const = np.log(1/(standardDeviation_x0_model*np.sqrt(2*np.pi)))
        p = -0.5*((x0_model-theta_model)/standardDeviation_x0_model)**2
        p_model = np.sum(const+p)

        return p_model+p_noise




    def numerical_jac(self, x0):
        def _prepare_bounds(bounds, x0):
            """
            Prepares new-style bounds from a two-tuple specifying the lower and upper
            limits for values in x0. If a value is not bound then the lower/upper bound
            will be expected to be -np.inf/np.inf.

            Examples
            --------
            >>> _prepare_bounds([(0, 1, 2), (1, 2, np.inf)], [0.5, 1.5, 2.5])
            (array([0., 1., 2.]), array([ 1.,  2., inf]))
            """
            lb, ub = (np.asarray(b, dtype=float) for b in bounds)
            if lb.ndim == 0:
                lb = np.resize(lb, x0.shape)

            if ub.ndim == 0:
                ub = np.resize(ub, x0.shape)

            return lb, ub

        def _adjust_scheme_to_bounds(x0, h, num_steps, scheme, lb, ub):
            """Adjust final difference scheme to the presence of bounds.

            Parameters
            ----------
            x0 : ndarray, shape (n,)
                Point at which we wish to estimate derivative.
            h : ndarray, shape (n,)
                Desired absolute finite difference steps.
            num_steps : int
                Number of `h` steps in one direction required to implement finite
                difference scheme. For example, 2 means that we need to evaluate
                f(x0 + 2 * h) or f(x0 - 2 * h)
            scheme : {'1-sided', '2-sided'}
                Whether steps in one or both directions are required. In other
                words '1-sided' applies to forward and backward schemes, '2-sided'
                applies to center schemes.
            lb : ndarray, shape (n,)
                Lower bounds on independent variables.
            ub : ndarray, shape (n,)
                Upper bounds on independent variables.

            Returns
            -------
            h_adjusted : ndarray, shape (n,)
                Adjusted absolute step sizes. Step size decreases only if a sign flip
                or switching to one-sided scheme doesn't allow to take a full step.
            use_one_sided : ndarray of bool, shape (n,)
                Whether to switch to one-sided scheme. Informative only for
                ``scheme='2-sided'``.
            """
            if scheme == '1-sided':
                use_one_sided = np.ones_like(h, dtype=bool)
            elif scheme == '2-sided':
                h = np.abs(h)
                use_one_sided = np.zeros_like(h, dtype=bool)
            else:
                raise ValueError("`scheme` must be '1-sided' or '2-sided'.")

            if np.all((lb == -np.inf) & (ub == np.inf)):
                return h, use_one_sided

            h_total = h * num_steps
            h_adjusted = h.copy()

            lower_dist = x0 - lb
            upper_dist = ub - x0

            if scheme == '1-sided':
                x = x0 + h_total
                violated = (x < lb) | (x > ub)
                fitting = np.abs(h_total) <= np.maximum(lower_dist, upper_dist)
                h_adjusted[violated & fitting] *= -1

                forward = (upper_dist >= lower_dist) & ~fitting
                h_adjusted[forward] = upper_dist[forward] / num_steps
                backward = (upper_dist < lower_dist) & ~fitting
                h_adjusted[backward] = -lower_dist[backward] / num_steps
            elif scheme == '2-sided':
                central = (lower_dist >= h_total) & (upper_dist >= h_total)

                forward = (upper_dist >= lower_dist) & ~central
                h_adjusted[forward] = np.minimum(
                    h[forward], 0.5 * upper_dist[forward] / num_steps)
                use_one_sided[forward] = True

                backward = (upper_dist < lower_dist) & ~central
                h_adjusted[backward] = -np.minimum(
                    h[backward], 0.5 * lower_dist[backward] / num_steps)
                use_one_sided[backward] = True

                min_dist = np.minimum(upper_dist, lower_dist) / num_steps
                adjusted_central = (~central & (np.abs(h_adjusted) <= min_dist))
                h_adjusted[adjusted_central] = min_dist[adjusted_central]
                use_one_sided[adjusted_central] = False

            return h_adjusted, use_one_sided


        # def fun_wrapped(x):
        #     # send user function same fp type as x0. (but only if cs is not being
        #     # used
        #     if xp.isdtype(x.dtype, "real floating"):
        #         x = xp.astype(x, x0.dtype)

        #     f = np.atleast_1d(self._res_fun_ls_exception_wrapper(x))
        #     if f.ndim > 1:
        #         raise RuntimeError("`fun` return value has "
        #                         "more than 1 dimension.")
        #     return f
        
        def _dense_difference(fun, x0, f0, h, use_one_sided, method):
            m = f0.size
            n = x0.size
            J_transposed = np.empty((n, m))
            x1 = x0.copy()
            x2 = x0.copy()
            xc = x0.astype(complex, copy=True)

            x1_ = np.empty((n, n))
            x2_ = np.empty((n, n))


            for i in range(h.size):
                if method == '2-point':
                    x1[i] += h[i]
                elif method == '3-point' and use_one_sided[i]:
                    x1[i] += h[i]
                    x2[i] += 2 * h[i]
                elif method == '3-point' and not use_one_sided[i]:
                    x1[i] -= h[i]
                    x2[i] += h[i]
                else:
                    raise RuntimeError("Never be here.")

                x1_[i,:] = x1
                x2_[i,:] = x2
                x1[i] = x2[i] = xc[i] = x0[i]

            if method == '2-point':
                args = [(x) for x in x1_]
                f = np.array(list(self.jac_pool.imap(self._res_fun_ls_exception_wrapper, args, chunksize=self.jac_chunksize)))
                df = f-f0
                dx = np.diag(x1_)-x0
            elif method == '3-point':
                args = [(x) for x in x1_]
                f1 = np.array(list(self.jac_pool.imap(self._res_fun_ls_exception_wrapper, args, chunksize=self.jac_chunksize)))
                args = [(x) for x in x2_]
                f2 = np.array(list(self.jac_pool.imap(self._res_fun_ls_exception_wrapper, args, chunksize=self.jac_chunksize)))
                df[use_one_sided,:] = -3.0 * f0[use_one_sided] + 4 * f1[use_one_sided,:] - f2[use_one_sided,:]
                df[~use_one_sided] = f2[~use_one_sided,:]-f1[~use_one_sided,:]
                dx = np.diag(x2_)-x0
                dx[~use_one_sided] = np.diag(x2_)[~use_one_sided]-np.diag(x1_)[~use_one_sided]

            J_transposed = df / dx.reshape((dx.shape[0], 1))

            if m == 1:
                J_transposed = np.ravel(J_transposed)

            return J_transposed.T
        
        def _compute_absolute_step(rel_step, x0, f0, method):
            """
            Computes an absolute step from a relative step for finite difference
            calculation.

            Parameters
            ----------
            rel_step: None or array-like
                Relative step for the finite difference calculation
            x0 : np.ndarray
                Parameter vector
            f0 : np.ndarray or scalar
            method : {'2-point', '3-point', 'cs'}

            Returns
            -------
            h : float
                The absolute step size

            Notes
            -----
            `h` will always be np.float64. However, if `x0` or `f0` are
            smaller floating point dtypes (e.g. np.float32), then the absolute
            step size will be calculated from the smallest floating point size.
            """
            # this is used instead of np.sign(x0) because we need
            # sign_x0 to be 1 when x0 == 0.
            sign_x0 = (x0 >= 0).astype(float) * 2 - 1

            rstep = _eps_for_method(x0.dtype, f0.dtype, method)

            if rel_step is None:
                abs_step = rstep * sign_x0 * np.maximum(1.0, np.abs(x0))
            else:
                # User has requested specific relative steps.
                # Don't multiply by max(1, abs(x0) because if x0 < 1 then their
                # requested step is not used.
                abs_step = rel_step * sign_x0 * np.abs(x0)

                # however we don't want an abs_step of 0, which can happen if
                # rel_step is 0, or x0 is 0. Instead, substitute a realistic step
                dx = ((x0 + abs_step) - x0)
                abs_step = np.where(dx == 0,
                                    rstep * sign_x0 * np.maximum(1.0, np.abs(x0)),
                                    abs_step)

            return abs_step

        @functools.lru_cache
        def _eps_for_method(x0_dtype, f0_dtype, method):
            """
            Calculates relative EPS step to use for a given data type
            and numdiff step method.

            Progressively smaller steps are used for larger floating point types.

            Parameters
            ----------
            f0_dtype: np.dtype
                dtype of function evaluation

            x0_dtype: np.dtype
                dtype of parameter vector

            method: {'2-point', '3-point', 'cs'}

            Returns
            -------
            EPS: float
                relative step size. May be np.float16, np.float32, np.float64

            Notes
            -----
            The default relative step will be np.float64. However, if x0 or f0 are
            smaller floating point types (np.float16, np.float32), then the smallest
            floating point type is chosen.
            """
            # the default EPS value
            EPS = np.finfo(np.float64).eps

            x0_is_fp = False
            if np.issubdtype(x0_dtype, np.inexact):
                # if you're a floating point type then over-ride the default EPS
                EPS = np.finfo(x0_dtype).eps
                x0_itemsize = np.dtype(x0_dtype).itemsize
                x0_is_fp = True

            if np.issubdtype(f0_dtype, np.inexact):
                f0_itemsize = np.dtype(f0_dtype).itemsize
                # choose the smallest itemsize between x0 and f0
                if x0_is_fp and f0_itemsize < x0_itemsize:
                    EPS = np.finfo(f0_dtype).eps

            if method in ["2-point", "cs"]:
                return EPS**0.5
            elif method in ["3-point"]:
                return EPS**(1/3)
            else:
                raise RuntimeError("Unknown step method, should be one of "
                                "{'2-point', '3-point', 'cs'}")
        


        method="2-point" 
        rel_step=None
        f0 = None

        if method not in ['2-point', '3-point', 'cs']:
            raise ValueError("Unknown method '%s'. " % method)

        xp = array_namespace(x0)
        _x = atleast_nd(x0, ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, "real floating"):
            _dtype = _x.dtype

        # promotes to floating
        x0 = xp.astype(_x, _dtype)

        if x0.ndim > 1:
            raise ValueError("`x0` must have at most 1 dimension.")

        lb, ub = _prepare_bounds(self.bounds, x0)

        if lb.shape != x0.shape or ub.shape != x0.shape:
            raise ValueError("Inconsistent shapes between bounds and `x0`.")

        if f0 is None:
            f0 = self._res_fun_ls_separate_process(x0)
        else:
            f0 = np.atleast_1d(f0)
            if f0.ndim > 1:
                raise ValueError("`f0` passed has more than 1 dimension.")

        if np.any((x0 < lb) | (x0 > ub)):
            raise ValueError("`x0` violates bound constraints.")

        
        # by default we use rel_step
        h = _compute_absolute_step(rel_step, x0, f0, method)

        if method == '2-point':
            h, use_one_sided = _adjust_scheme_to_bounds(
                x0, h, 1, '1-sided', lb, ub)
        elif method == '3-point':
            h, use_one_sided = _adjust_scheme_to_bounds(
                x0, h, 1, '2-sided', lb, ub)
        elif method == 'cs':
            use_one_sided = False

        return _dense_difference(self._res_fun_ls_exception_wrapper, x0, f0, h,
                                    use_one_sided, method)

    

    def ls(self,
           n_cores=multiprocessing.cpu_count(),
           method: str="trf",
           ftol: float = 1e-8,
           xtol: float = 1e-8,
           gtol: float = 1e-8,
           x_scale: float = 1,
           loss: str = 'linear',
           f_scale: float = 1,
           diff_step: Any | None = None,
           tr_solver: Any | None = None,
           tr_options: Any = {},
           jac_sparsity: Any | None = None,
           max_nfev: Any | None = None,
           verbose: int = 0,
           **kwargs) -> LSEstimationResult:
        """
        Run least squares estimation.

        Returns:
            OptimizeResult: The optimization result returned by scipy.optimize.least_squares.
        """
        assert np.all(self._x0>=self._lb), "The provided x0 must be larger than the provided lower bound lb"
        assert np.all(self._x0<=self._ub), "The provided x0 must be smaller than the provided upper bound ub"
        assert np.all(np.abs(self._x0-self._lb)>self.tol), f"The difference between x0 and lb must be larger than {str(self.tol)}"
        assert np.all(np.abs(self._x0-self._ub)>self.tol), f"The difference between x0 and ub must be larger than {str(self.tol)}"
        datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = str('{}{}'.format(datestr, '_ls.pickle'))
        res_fail = np.zeros((self.n_timesteps, len(self.targetMeasuringDevices)))
        for j, measuring_device in enumerate(self.targetMeasuringDevices):
            res_fail[:,j] = self.targetMeasuringDevices[measuring_device]["standardDeviation"]*np.ones((self.n_timesteps))*100
        self.res_fail = res_fail.flatten()
        self.result_savedir_pickle, isfile = self.model.get_dir(folder_list=["model_parameters", "estimation_results", "LS_result"], filename=filename)


        self.model.set_save_simulation_result(flag=False)
        self.model.set_save_simulation_result(flag=True, c=list(self.targetMeasuringDevices.keys()))
        self.fun_pool = multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=30)
        self.jac_pool = multiprocessing.get_context("spawn").Pool(n_cores, maxtasksperchild=10)
        self.jac_chunksize = 1
        self.model.make_pickable()

        self.bounds = (self._lb, self._ub)


        ls_result = least_squares(self._res_fun_ls_separate_process,
                                  self._x0,
                                  jac=self.numerical_jac,
                                  bounds=self.bounds,
                                  method=method,
                                  ftol=ftol,
                                  xtol=xtol,
                                  gtol=gtol,
                                  x_scale=x_scale,
                                  loss=loss,
                                  f_scale=f_scale,
                                  diff_step=diff_step,
                                  tr_solver=tr_solver,
                                  tr_options=tr_options,
                                  jac_sparsity=jac_sparsity,
                                  max_nfev=max_nfev,
                                  verbose=verbose) #Change verbose to 2 to see the optimization progress
    

        ls_result = LSEstimationResult(result_x=ls_result.x,
                                      component_id=[com.id for com in self.flat_component_list],
                                      component_attr=[attr for attr in self.flat_attr_list],
                                      theta_mask=self.theta_mask,
                                      startTime_train=self._startTime_train,
                                      endTime_train=self._endTime_train,
                                      stepSize_train=self._stepSize_train)
        with open(self.result_savedir_pickle, 'wb') as handle:
            pickle.dump(ls_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return ls_result
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if hasattr(self, 'fun_pool'):
            del self_dict['fun_pool']
        if hasattr(self, 'jac_pool'):
            del self_dict['jac_pool']
        return self_dict
    
    def _res_fun_ls(self, theta: np.ndarray) -> np.ndarray:
        """
        Residual function for least squares estimation.

        Args:
            theta (np.ndarray): Parameter vector.

        Returns:
            np.ndarray: Array of residuals.
        """
        theta = theta[self.theta_mask]
        self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
        n_time_prev = 0
        self.simulation_readings = {com.id: np.zeros((self.n_timesteps)) for com in self.targetMeasuringDevices}
        for startTime_, endTime_, stepSize_  in zip(self._startTime_train, self._endTime_train, self._stepSize_train):
            self.simulator.simulate(self.model,
                                    stepSize=stepSize_,
                                    startTime=startTime_,
                                    endTime=endTime_,
                                    show_progress_bar=False)
            n_time = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            for measuring_device in self.targetMeasuringDevices:
                y_model = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
                self.simulation_readings[measuring_device.id][n_time_prev:n_time_prev+n_time] = y_model
            n_time_prev += n_time
        res = np.zeros((self.n_timesteps, len(self.targetMeasuringDevices)))
        for j, measuring_device in enumerate(self.targetMeasuringDevices):
            simulation_readings = self.simulation_readings[measuring_device.id]
            actual_readings = self.actual_readings[measuring_device.id]
            res[:,j] = (actual_readings-simulation_readings)
            sd = self.targetMeasuringDevices[measuring_device]["standardDeviation"]
            res[:,j] = (0.5)**0.5*res[:,j]/sd
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
            # res = np.array(list(self.jac_pool.imap(self._res_fun_ls, [(theta)], chunksize=self.jac_chunksize)))
            res = self._res_fun_ls(theta)
        except FMICallException as inst:
            res = self.res_fail
        return res

    def _res_fun_ls_separate_process(self, theta: np.ndarray):
        res = np.array(list(self.fun_pool.imap(self._res_fun_ls_exception_wrapper, [(theta)], chunksize=self.jac_chunksize))[0])
        return res

    

    
    def _loglike_ga_wrapper(self, 
                            ga_instance: pygad.GA, 
                            theta: np.ndarray, 
                            solution_idx: int) -> float:
        """
        Wrapper for the GA log-likelihood function to handle boundary conditions and exceptions.

        Args:
            ga_instance (pygad.GA): The GA instance.
            theta (np.ndarray): The parameter vector.
            solution_idx (int): The index of the solution.

        Returns:
            float: The log-likelihood value.
        """
        outsideBounds = np.any(theta<self._lb) or np.any(theta>self._ub)
        if outsideBounds:
            return -1e+10
        try:
            loglike = self._loglike(theta)
        except FMICallException as inst:
            return -1e+10
        return loglike
    

    def callback_generation(self, ga_instance):
        print(f"Generation = {ga_instance.generations_completed}")
        print(f"Fitness    = {ga_instance.best_solution()[1]}")
        print(f"Change     = {ga_instance.best_solution()[1] - self.last_fitness}")
        self.last_fitness = ga_instance.best_solution()[1]


    def ga(self, 
           num_generations: int=10,
           num_parents_mating: int=25,
           sol_per_pop: int=1000,
           on_start: Callable=None,
           on_fitness: Callable=None,
           on_parents: Callable=None,
           on_crossover: Callable=None,
           on_mutation: Callable=None,
           on_generation: Callable=None,
           on_stop: Callable=None,
           parallel_processing: List[str, int]=['process', 4],
           **kwargs) -> None:
        """
        Run genetic algorithm (GA) estimation.

        Args:
            num_generations (int): Number of generations.
            num_parents_mating (int): Number of parents for mating.
            sol_per_pop (int): Number of solutions per population.
            on_start (Callable): Callback function for start of the GA.
            on_fitness (Callable): Callback function for fitness evaluation.
            on_parents (Callable): Callback function for parent selection.
            on_crossover (Callable): Callback function for crossover.
            on_mutation (Callable): Callback function for mutation.
            on_generation (Callable): Callback function for generation.
            on_stop (Callable): Callback function for stopping condition.
            parallel_processing (List[str, int]): Parallel processing options.
        """
        self.last_fitness = -np.inf
        # self.pbar = tqdm(total=num_generations)
        init_range_low = self._lb
        init_range_high = self._ub
        gene_space = [{"low": lb, "high": ub} for lb, ub in zip(self._lb, self._ub)]
        ga_instance = pygad.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=self._loglike_ga_wrapper,
                                sol_per_pop=sol_per_pop,
                                num_genes=self.ndim,
                                init_range_low=init_range_low,
                                init_range_high=init_range_high,
                                gene_space=gene_space,
                                on_start=on_start,
                                on_fitness=on_fitness,
                                on_parents=on_parents,
                                on_crossover=on_crossover,
                                on_mutation=on_mutation,
                                on_generation=self.callback_generation,
                                on_stop=on_stop,
                                parallel_processing=parallel_processing,
                                suppress_warnings=True)
        ga_instance.run()
        ga_instance.plot_fitness()
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        

# class on_generation_callback:
#     def __init__(self, pbar: tqdm):
#         self.pbar = pbar

#     def __call__(self, ga_instance):
#         solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
#         des = f"Date: {datestr} logl: {str(int(solution_fitness))}"
#         self.pbar.set_description(des)

