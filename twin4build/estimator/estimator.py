from __future__ import annotations
import multiprocessing
import math
import numpy as np
import datetime
import pickle
from fmpy.fmi2 import FMICallException
from scipy.optimize import least_squares
import twin4build.core as core
import functools
from scipy._lib._array_api import array_namespace
from scipy._lib._array_api.array_api_extra import atleast_nd
import torch
import torch.nn as nn

from typing import Union, List, Dict, Optional, Any


def atleast_nd(x, /, *, ndim: int, xp) -> Array:
    """
    Recursively expand the dimension of an array to at least `ndim`.

    Parameters
    ----------
    x : array
    ndim : int
        The minimum number of dimensions for the result.
    xp : array_namespace
        The standard-compatible namespace for `x`.

    Returns
    -------
    res : array
        An array with ``res.ndim`` >= `ndim`.
        If ``x.ndim`` >= `ndim`, `x` is returned.
        If ``x.ndim`` < `ndim`, `x` is expanded by prepending new axes
        until ``res.ndim`` equals `ndim`.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([1])
    >>> xpx.atleast_nd(x, ndim=3, xp=xp)
    Array([[[1]]], dtype=array_api_strict.int64)

    >>> x = xp.asarray([[[1, 2],
    ...                  [3, 4]]])
    >>> xpx.atleast_nd(x, ndim=1, xp=xp) is x
    True

    """
    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        x = atleast_nd(x, ndim=ndim, xp=xp)
    return x

class Estimator:
    r"""
    A class for parameter estimation in the twin4build framework.

    This class provides methods for estimating model parameters using maximum likelihood
    estimation (MLE), with two different optimization approaches: Least Squares (LS) and
    PyTorch-based gradient optimization (TORCH).

    Mathematical Formulation:

    The general parameter estimation problem is formulated as a maximum likelihood estimation:

        .. math::

            \hat{\theta} = \underset{\theta \in \Theta}{\operatorname{arg\,max}} \mathcal{L}(\theta; \mathbf{y})

    where:
       - :math:`\hat{\theta}` is the maximum likelihood estimate
       - :math:`\theta` is the parameter vector
       - :math:`\Theta` is the parameter space
       - :math:`\mathcal{L}(\theta; \mathbf{y})` is the likelihood function
       - :math:`\mathbf{y}` are the observed measurements

    For normally distributed measurement errors, the log-likelihood function becomes:

        .. math::

            \ell(\theta; \mathbf{y}) = \ln \mathcal{L}(\theta; \mathbf{y}) = -\frac{1}{2} \sum_{i=1}^{n} \left(\frac{y_i - f(x_i, \theta)}{\sigma_i}\right)^2 + C

    where:
       - :math:`y_i` are the measured values
       - :math:`f(x_i, \theta)` is the model prediction
       - :math:`\sigma_i` is the measurement uncertainty
       - :math:`C` is a constant term

    Since maximizing the log-likelihood is equivalent to minimizing the negative log-likelihood,
    the problem can be reformulated as a minimization problem:

        .. math::

            \hat{\theta} = \underset{\theta \in \Theta}{\operatorname{arg\,min}} \sum_{i=1}^{n} \left(\frac{y_i - f(x_i, \theta)}{\sigma_i}\right)^2

    This class provides two different approaches to solve this optimization problem:

       1. PyTorch-based Gradient Optimization (TORCH) - Preferred Method:
          Uses PyTorch's automatic differentiation and gradient-based optimization
          to solve the minimization problem. This is the preferred method when all
          model components are implemented using PyTorch.

          Key characteristics:
             - Uses automatic differentiation to compute exact gradients
             - Can ONLY be used when ALL components are implemented as torch.nn.Module
             - Requires the entire model to be differentiable
             - Much faster for large models due to efficient gradient computation
             - Single model evaluation per gradient computation
             - Better convergence properties due to exact gradient computation

       2. Least Squares (LS) - Fallback Method:
          Uses scipy's least_squares optimization algorithm, which employs trust-region
          methods to solve the minimization problem. This method is necessary when
          working with mixed model types or non-PyTorch components.

          Key characteristics:
             - Computes gradients numerically using finite differences
             - Can be used with any type of model, regardless of implementation
             - More robust to non-differentiable or discontinuous model behavior
             - Slower for large models due to numerical gradient computation
             - Requires multiple model evaluations per gradient computation
             - Useful as a fallback when TORCH cannot be used

    Both methods solve the same underlying optimization problem, but use different
    numerical approaches:
       - TORCH (preferred): Uses gradient descent with automatic differentiation
       - LS (fallback): Uses trust-region methods with numerical Jacobian computation

    Model Compatibility and Usage Guidelines:
       - TORCH: Preferred method when all components are torch.nn.Module
       - LS: Necessary fallback for mixed model types or non-PyTorch components
       - When possible, convert models to PyTorch to use the more efficient TORCH method

    Parameter Bounds:
    For each parameter :math:`\theta_i`:

        .. math::

            \theta_i^{lb} \leq \theta_i \leq \theta_i^{ub}

    where:
       - :math:`\theta_i^{lb}` is the lower bound
       - :math:`\theta_i^{ub}` is the upper bound

    Attributes:
       - model (Model): The model to perform estimation on.
       - simulator (Simulator): The simulator instance for running simulations.
       - x0 (np.ndarray): Initial parameter values.
       - lb (np.ndarray): Lower bounds for parameters.
       - ub (np.ndarray): Upper bounds for parameters.
       - tol (float): Tolerance for parameter bounds checking.
       - ndim (int): Number of dimensions/parameters.
       - targetMeasuringDevices (Dict): Target devices for estimation.

    Examples
    --------
    >>> import twin4build as tb
    >>> model = tb.SimulationModel(id="my_model")
    >>> estimator = tb.Estimator(model)
    >>> targetParameters = {
    ...     "private": {},
    ...     "shared": {}
    ... }
    >>> targetMeasuringDevices = {}
    >>> import datetime, pytz
    >>> start = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    >>> end = datetime.datetime(2024, 1, 2, tzinfo=pytz.UTC)
    >>> step = 3600
    >>> estimator.estimate(targetParameters=targetParameters, targetMeasuringDevices=targetMeasuringDevices, startTime=start, endTime=end, stepSize=step)
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
                 method: str = "LS",
                 options: Dict = None) -> None:
        """Perform parameter estimation using specified method and configuration.

        This method sets up and executes the parameter estimation process, supporting
        least squares (LS) estimation and PyTorch-based gradient optimization.

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

            n_initialization_steps (int, optional):
                Number of steps to skip during initialization. Defaults to 60.

            startTime (Union[datetime.datetime, List[datetime.datetime]], optional):
                Start time(s) for estimation period(s).

            endTime (Union[datetime.datetime, List[datetime.datetime]], optional):
                End time(s) for estimation period(s).

            stepSize (Union[float, List[float]], optional):
                Step size(s) for simulation.

            verbose (bool, optional):
                Whether to print detailed output. Defaults to False.

            method (str, optional):
                Estimation method to use ("LS" or "TORCH"). Defaults to "LS".

            options (Dict, optional):
                Additional options for the chosen method:
                    For LS:
                        - "method": Optimization method for least_squares
                        - "ftol": Function tolerance
                        - "xtol": Parameter tolerance
                        - "gtol": Gradient tolerance
                        - "x_scale": Parameter scaling
                        - "loss": Loss function type
                        - "f_scale": Function scaling
                        - "diff_step": Step size for numerical derivatives
                        - "tr_solver": Trust region solver
                        - "tr_options": Trust region options
                        - "jac_sparsity": Jacobian sparsity pattern
                        - "max_nfev": Maximum function evaluations
                        - "verbose": Verbosity level
                    For TORCH:
                        - "lr": Learning rate
                        - "iterations": Number of optimization iterations
                        - "scheduler_type": Learning rate scheduler type
                        - "scheduler_params": Learning rate scheduler parameters

        Raises:
            AssertionError: If method is not one of ["LS", "TORCH"] or if input
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
        

        allowed_methods = ["LS", "TORCH"]
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

        self._flat_component_list_private = [obj for par_dict in targetParameters["private"].values() for obj in par_dict["components"]]
        self._flat_attr_list_private = [attr for attr, par_dict in targetParameters["private"].items() for obj in par_dict["components"]]
        self._flat_component_list_shared = [obj for par_dict in targetParameters["shared"].values() for obj_list in par_dict["components"] for obj in obj_list]
        self._flat_attr_list_shared = [attr for attr, par_dict in targetParameters["shared"].items() for obj_list in par_dict["components"] for obj in obj_list]
        self.flat_attr_list = self._flat_attr_list_private + self._flat_attr_list_shared
        self.flat_component_list = self._flat_component_list_private + self._flat_component_list_shared

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
        self.theta_mask = np.concatenate((private_mask, shared_mask)).astype(int)
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
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

        if method == "LS":
            if options is None:
                options = {}
            self._ls(**options)
        elif method == "TORCH":
            if options is None:
                options = {}
            return self._estimate_torch(**options)

    def _numerical_jac(self, x0):
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

    

    def _ls(self,
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
                                  jac=self._numerical_jac,
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
    

        ls_result = EstimationResult(result_x=ls_result.x,
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
                y_model = np.array(measuring_device.input["measuredValue"].history.detach())[self.n_initialization_steps:]
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

    def _estimate_torch(self,
                     lr: float = 0.01,
                     iterations: int = 100,
                     scheduler_type: str = "step",
                     scheduler_params: Dict = None) -> LSEstimationResult:
        """Perform parameter estimation using PyTorch gradient-based optimization.

        This method sets up and executes the parameter estimation process using PyTorch's
        automatic differentiation capabilities. It supports learning rate scheduling and
        various optimization options.

        Parameters:
            lr (float, optional):
                Learning rate for optimizer. Defaults to 0.01.

            iterations (int, optional):
                Number of optimization iterations. Defaults to 100.

            scheduler_type (str, optional):
                Type of learning rate scheduler. Options: "step", "exponential", "cosine", "reduce_on_plateau".
                Defaults to "step".

            scheduler_params (Dict, optional):
                Parameters for learning rate scheduler.

        Returns:
            LSEstimationResult: The estimation result containing optimized parameters.
        """
        # Enable gradients for parameters to be estimated
        opt_params = []
        for component in self.flat_component_list:
            if isinstance(component, nn.Module):
                for name, param in component.named_parameters():
                    if name in self.flat_attr_list:
                        param.requires_grad_(True)
                        opt_params.append(param)

        # Initialize optimizer
        optimizer = torch.optim.SGD(opt_params, lr=lr)
        
        # Initialize scheduler
        if scheduler_params is None:
            scheduler_params = {}
            
        if scheduler_type == "step":
            step_size = scheduler_params.get("step_size", 30)
            gamma = scheduler_params.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "exponential":
            gamma = scheduler_params.get("gamma", 0.95)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == "cosine":
            T_max = scheduler_params.get("T_max", 100)
            eta_min = scheduler_params.get("eta_min", 0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == "reduce_on_plateau":
            mode = scheduler_params.get("mode", "min")
            factor = scheduler_params.get("factor", 0.1)
            patience = scheduler_params.get("patience", 10)
            threshold = scheduler_params.get("threshold", 1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, 
                patience=patience, threshold=threshold
            )
        else:
            scheduler = None

        # Optimization loop
        best_loss = float('inf')
        best_params = None
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Run simulation
            loss = 0
            for startTime_, endTime_, stepSize_ in zip(self._startTime_train, self._endTime_train, self._stepSize_train):
                self.simulator.simulate(self.model,
                                      stepSize=stepSize_,
                                      startTime=startTime_,
                                      endTime=endTime_,
                                      show_progress_bar=False)
                
                # Compute loss for each measuring device
                for measuring_device in self.targetMeasuringDevices:
                    y_model = measuring_device.input["measuredValue"].history[self.n_initialization_steps:]
                    y_actual = torch.tensor(self.actual_readings[measuring_device.id])
                    sd = self.targetMeasuringDevices[measuring_device]["standardDeviation"]
                    loss += torch.mean(((y_actual - y_model) / sd) ** 2)

            # Backpropagate and update parameters
            loss.backward()
            optimizer.step()
            
            # Apply parameter bounds
            with torch.no_grad():
                for component in self.flat_component_list:
                    if isinstance(component, nn.Module):
                        for name, param in component.named_parameters():
                            if name in self.flat_attr_list:
                                # Find corresponding bounds
                                for par_dict in self.targetParameters["private"].values():
                                    if name in par_dict and component in par_dict["components"]:
                                        idx = par_dict["components"].index(component)
                                        lb = par_dict["lb"][idx]
                                        ub = par_dict["ub"][idx]
                                        param.clamp_(min=lb, max=ub)
                                for par_dict in self.targetParameters["shared"].values():
                                    if name in par_dict:
                                        for comp_list, lb_list, ub_list in zip(par_dict["components"], par_dict["lb"], par_dict["ub"]):
                                            if component in comp_list:
                                                idx = comp_list.index(component)
                                                lb = lb_list[idx]
                                                ub = ub_list[idx]
                                                param.clamp_(min=lb, max=ub)
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()
            
            # Track best parameters
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {name: param.clone().detach() for component in self.flat_component_list 
                              if isinstance(component, nn.Module)
                              for name, param in component.named_parameters()
                              if name in self.flat_attr_list}
            
            if self.verbose:
                print(f"Iteration {i+1}/{iterations}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Restore best parameters
        with torch.no_grad():
            for component in self.flat_component_list:
                if isinstance(component, nn.Module):
                    for name, param in component.named_parameters():
                        if name in self.flat_attr_list:
                            param.copy_(best_params[name])

        # Convert parameters to numpy array for result
        result_x = []
        for component in self.flat_component_list:
            if isinstance(component, nn.Module):
                for name, param in component.named_parameters():
                    if name in self.flat_attr_list:
                        result_x.append(param.detach().cpu().numpy())

        # Create and return result
        result = EstimationResult(
            result_x=np.array(result_x),
            component_id=[com.id for com in self.flat_component_list],
            component_attr=[attr for attr in self.flat_attr_list],
            theta_mask=self.theta_mask,
            startTime_train=self._startTime_train,
            endTime_train=self._endTime_train,
            stepSize_train=self._stepSize_train
        )

        return result

    
class EstimationResult(dict):
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
        return EstimationResult(**self)

    
    def copy(self):
        return self.__copy__()