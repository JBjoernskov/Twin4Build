from __future__ import annotations

# Standard library imports
import datetime
import functools

# import multiprocessing
import math
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import torch.nn as nn
from fmpy.fmi2 import FMICallException
from scipy._lib._array_api import array_namespace
from scipy.optimize import Bounds, least_squares, minimize

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.rgetattr import rgetattr


def _atleast_nd(x, /, *, ndim: int, xp) -> Any:
    """
    Recursively expand the dimension of an array to at least `ndim`.

    Parameters
    ----------
    x : array
        Input array to expand.
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
    >>> xpx._atleast_nd(x, ndim=3, xp=xp)
    Array([[[1]]], dtype=array_api_strict.int64)

    >>> x = xp.asarray([[[1, 2],
    ...                  [3, 4]]])
    >>> xpx._atleast_nd(x, ndim=1, xp=xp) is x
    True
    """
    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        x = _atleast_nd(x, ndim=ndim, xp=xp)
    return x


class Estimator:
    r"""
    A class for parameter estimation in the twin4build framework.

    This class provides methods for estimating model parameters using maximum likelihood
    estimation (MLE), with two different optimization approaches: Automatic Differentiation (AD)
    and Finite Difference (FD) methods.

    Mathematical Formulation:
    =========================

    The general parameter estimation problem is formulated as a maximum likelihood estimation:

        .. math::

            \hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta} \in \Theta}{\operatorname{argmax}} \; \mathcal{L}(\boldsymbol{\theta} | \boldsymbol{Y})

    where:

        - :math:`\hat{\boldsymbol{\theta}}` is the maximum likelihood estimate
        - :math:`\boldsymbol{\theta}` is the parameter vector
        - :math:`\Theta \subseteq \mathbb{R}^{n_p}` is the parameter space
        - :math:`\mathcal{L}(\boldsymbol{\theta} | \boldsymbol{Y})` is the likelihood function
        - :math:`\boldsymbol{Y}` are the observed measurements

    **Dimensions:**

    - :math:`n_t`: Number of time steps in the simulation period
    - :math:`n_p`: Number of parameters to estimate
    - :math:`n_x`: Number of input variables (disturbances, setpoints, etc.)
    - :math:`n_y`: Number of output variables (measurements, performance metrics)

    **Model Structure:**

    The building model :math:`\mathcal{M}` is represented as a directed graph where nodes are dynamic components
    and edges represent input/output connections.

    .. figure:: /_static/estimator_graph_.png
       :alt: System overview showing components and their relationships
       :align: center
       :width: 80%


    The model takes input variables :math:`\boldsymbol{X} \in \mathbb{R}^{n_x \times n_t}`
    along with parameters :math:`\boldsymbol{\theta} \in \mathbb{R}^{n_p}`, and produces system outputs
    :math:`\boldsymbol{\hat{Y}} \in \mathbb{R}^{n_y \times n_t}` with timesteps :math:`\boldsymbol{t} \in \mathbb{R}^{n_t}`:

    .. math::

            \boldsymbol{\hat{Y}} = \mathcal{M}(\boldsymbol{X}, \boldsymbol{t}, \boldsymbol{\theta})

    where :math:`\mathcal{M}` represents the complete simulation model. See :class:`~twin4build.simulator.simulator.Simulator`
    for detailed explanation of the simulation process.

    **Likelihood Function:**

    Using the Kennedy-O'Hagan (KOH) Bayesian model formulation, the relationship between observations
    :math:`\boldsymbol{Y}`, model response :math:`\boldsymbol{\hat{Y}}`, and measurement errors :math:`\boldsymbol{\epsilon}` is:

    .. math::

            \boldsymbol{Y}_j = \boldsymbol{\hat{Y}}_j + \boldsymbol{\epsilon}_j \quad \forall j \in \{1, \ldots, n_y\}

    For normally distributed measurement errors, where :math:`\boldsymbol{\epsilon}_j \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_j)`, the likelihood function becomes:

    .. math::

            \mathcal{L}(\boldsymbol{\theta} | \boldsymbol{Y}) = \prod_{j=1}^{n_y} (2\pi)^{-n_t/2} \det(\boldsymbol{\Sigma}_j)^{-1/2} \exp\left(-\frac{1}{2}(\boldsymbol{Y}_j - \boldsymbol{\hat{Y}}_j)^T \boldsymbol{\Sigma}_j^{-1} (\boldsymbol{Y}_j - \boldsymbol{\hat{Y}}_j)\right)

    where:

        - :math:`\boldsymbol{Y}_j \in \mathbb{R}^{n_t}`: Measured values for output :math:`j` across all time steps
        - :math:`\boldsymbol{\hat{Y}}_j \in \mathbb{R}^{n_t}`: Model predictions for output :math:`j` across all time steps
        - :math:`\boldsymbol{\Sigma}_j \in \mathbb{R}^{n_t \times n_t}`: Covariance matrix for output :math:`j`

    Taking the negative log-likelihood (for minimization) gives:

    .. math::

            -\ln\mathcal{L}(\boldsymbol{\theta} | \boldsymbol{Y}) = \frac{n_t n_y}{2} \ln(2\pi) + \frac{1}{2} \sum_{j=1}^{n_y} \ln\det(\boldsymbol{\Sigma}_j) + \frac{1}{2} \sum_{j=1}^{n_y} (\boldsymbol{Y}_j - \boldsymbol{\hat{Y}}_j)^T \boldsymbol{\Sigma}_j^{-1} (\boldsymbol{Y}_j - \boldsymbol{\hat{Y}}_j)

    With i.i.d. assumption and diagonal covariance matrices :math:`\boldsymbol{\Sigma}_j = \sigma_j^2 \boldsymbol{I}_{n_t}`, this simplifies to:

    .. math::

            -\ln\mathcal{L}(\boldsymbol{\theta} | \boldsymbol{Y}) = \frac{n_t n_y}{2} \ln(2\pi) + \frac{n_t}{2} \sum_{j=1}^{n_y} \ln(\sigma_j^2) + \frac{1}{2} \sum_{j=1}^{n_y} \sum_{t=1}^{n_t} \left(\frac{Y_{j,t} - \hat{Y}_{j,t}}{\sigma_j}\right)^2

    This is the form we use in twin4build for parameter estimation, meaning that we solve the following optimization problem:

    .. math::

            \hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta} \in \Theta}{\operatorname{argmin}} \; \sum_{j=1}^{n_y} \sum_{t=1}^{n_t} \left(\frac{Y_{j,t} - \hat{Y}_{j,t}}{\sigma_j}\right)^2

    where the constant terms have been dropped since they do not affect the optimization.



    **Parameter Bounds:**

    For each parameter :math:`\theta_{i}`:

    .. math::

            \theta_{i}^{lb} \leq \theta_{i} \leq \theta_{i}^{ub}

    where:

        - :math:`\theta_{i}^{lb}` is the lower bound
        - :math:`\theta_{i}^{ub}` is the upper bound

    See method docstrings for details on the specific optimization algorithms and implementation.

    Attributes
    ----------
    simulator : Optional[core.Simulator]
        The simulator instance for running simulations.
    tol : float
        Tolerance for parameter bounds checking. Defaults to 1e-10.
    n_initialization_steps : int
        Number of steps to skip during initialization.
    parameters : Dict[str, Dict]
        Dictionary containing private and shared parameters to estimate.
    measurements : List[core.System]
        List of measuring devices used for estimation.
    best_loss : float
        Best objective function value achieved during optimization.
    n_timesteps : int
        Total number of timesteps across all simulation periods.

    Examples
    --------
    Basic usage with automatic differentiation (recommended):

    >>> import twin4build as tb
    >>> import datetime
    >>> import pytz
    >>>
    >>> # Create model and simulator
    >>> model = tb.SimulationModel(id="my_model")
    >>> simulator = tb.Simulator(model)
    >>> estimator = tb.Estimator(simulator)
    >>>
    >>> # Define parameters to estimate
    >>> parameters = {
    ...     "private": {
    ...         "efficiency": {
    ...             "components": [component1, component2],
    ...             "x0": [0.8, 0.85],
    ...             "lb": [0.5, 0.6],
    ...             "ub": [1.0, 1.0]
    ...         }
    ...     },
    ...     "shared": {
    ...         "heatTransferCoefficient": {
    ...             "components": [[component1, component2]],
    ...             "x0": [[0.5]],
    ...             "lb": [[0.1]],
    ...             "ub": [[2.0]]
    ...         }
    ...     }
    ... }
    >>>
    >>> # Define measuring devices
    >>> measurements = [measuring_device1, measuring_device2]
    >>>
    >>> # Set time period
    >>> start = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    >>> end = datetime.datetime(2024, 1, 2, tzinfo=pytz.UTC)
    >>> step = 3600
    >>>
    >>> # Run estimation with automatic differentiation (recommended)
    >>> result = estimator.estimate(
    ...     parameters=parameters,
    ...     measurements=measurements,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("scipy", "SLSQP", "ad")  # Preferred for most problems
    ... )

    >>> # Alternative: Use L-BFGS-B with automatic differentiation
    >>> result = estimator.estimate(
    ...     parameters=parameters,
    ...     measurements=measurements,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("scipy", "L-BFGS-B", "ad")
    ... )

    >>> # For non-PyTorch models: Use finite difference method
    >>> result = estimator.estimate(
    ...     parameters=parameters,
    ...     measurements=measurements,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("scipy", "trf", "fd"),
    ...     n_cores=4  # Required for FD mode
    ... )

    >>> # Legacy string format (still supported)
    >>> result = estimator.estimate(
    ...     parameters=parameters,
    ...     measurements=measurements,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method="scipy"  # Defaults to SLSQP with AD
    ... )
    """

    def __init__(self, simulator: Optional[core.Simulator] = None):
        """
        Initialize the Estimator.

        Parameters
        ----------
        simulator : Optional[core.Simulator], optional
            The simulator instance for running simulations. If None, must be provided
            when calling estimate().
        """
        self.simulator = simulator
        self.tol = 1e-10

    def estimate(
        self,
        parameters: Optional[Dict[str, Dict]] = None,
        measurements: Optional[List[core.System]] = None,
        n_initialization_steps: int = 60,
        startTime: Optional[Union[datetime.datetime, List[datetime.datetime]]] = None,
        endTime: Optional[Union[datetime.datetime, List[datetime.datetime]]] = None,
        stepSize: Optional[Union[float, List[float]]] = None,
        method: Union[str, Tuple[str, str, str]] = "scipy",
        n_cores: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> EstimationResult:
        """
        Perform parameter estimation using specified method and configuration.

        This method sets up and executes the parameter estimation process, supporting
        both automatic differentiation (AD) and finite difference (FD) optimization methods.

        Parameters
        ----------
        parameters : Optional[Dict[str, Dict]], optional
            Dictionary containing parameter specifications:

            - "private": Parameters unique to each component
            - "shared": Parameters shared across components

            Each parameter entry contains:
                - "components": List of components or single component
                - "x0": List of initial values or single initial value
                - "lb": List of lower bounds or single lower bound
                - "ub": List of upper bounds or single upper bound

        measurements : Optional[List[core.System]], optional
            List of measuring devices used for estimation. Each device should have
            an "input" attribute with a "measuredValue" that contains historical data.

        n_initialization_steps : int, default=60
            Number of simulation steps to skip during initialization phase.

        startTime : Optional[Union[datetime.datetime, List[datetime.datetime]]], optional
            Start time(s) for estimation period(s). Can be a single datetime or list
            of datetimes for multiple periods.

        endTime : Optional[Union[datetime.datetime, List[datetime.datetime]]], optional
            End time(s) for estimation period(s). Must be later than corresponding startTime.

        stepSize : Optional[Union[float, List[float]]], optional
            Step size(s) for simulation in seconds. Can be a single value or list
            of values for multiple periods.

        method : Union[str, Tuple[str, str, str]], default="scipy"
            Estimation method specification. Can be specified in two formats:

            1. String format (legacy):
               - "scipy": Uses default SLSQP optimizer with automatic differentiation
               - Other valid strings: Any optimizer name that matches the supported algorithms
                 (e.g., "L-BFGS-B", "TNC", "SLSQP", "trust-constr", "trf", "dogbox")

            2. Tuple format (recommended):
               - (library, optimizer, mode) where:
                 - library: "scipy" (currently the only supported library)
                 - optimizer: The specific optimization algorithm
                 - mode: "ad" (automatic differentiation) or "fd" (finite difference)

            Supported optimizers by mode:

            Automatic Differentiation (AD) mode:
               - "SLSQP": Sequential Least Squares Programming (preferred for most problems)
               - "L-BFGS-B": Limited-memory BFGS with bounds
               - "TNC": Truncated Newton algorithm with bounds
               - "trust-constr": Trust-region constrained optimization
               - "trf": Trust Region Reflective (for least-squares problems)
               - "dogbox": Dogleg algorithm (for least-squares problems)

            Finite Difference (FD) mode:
               - "trf": Trust Region Reflective (for least-squares problems)
               - "dogbox": Dogleg algorithm (for least-squares problems)

            Mode selection guidelines:
               - "ad": Use when all components are torch.nn.Module (preferred, faster)
               - "fd": Use for non-PyTorch models or mixed model types (requires n_cores)

            Examples:
               - ("scipy", "SLSQP", "ad"): Preferred for most PyTorch models
               - ("scipy", "L-BFGS-B", "ad"): Good alternative for unconstrained problems
               - ("scipy", "trf", "fd"): For non-PyTorch models with least-squares formulation
               - "scipy": Legacy format, defaults to ("scipy", "SLSQP", "ad")

        n_cores : Optional[int], optional
            Number of CPU cores to use for parallel computation. Required when using
            finite difference (FD) mode for Jacobian computation. Not used in automatic
            differentiation (AD) mode.

            - For FD mode: Must be specified (typically 2-8 cores depending on system)
            - For AD mode: Ignored (not needed for automatic differentiation)
            - Default: None (will raise error if FD mode is used without specifying)

        options : Optional[Dict], optional
            Additional options for the chosen optimization method:

            For scipy optimizers:
                - "ftol": Function tolerance (default: 1e-8)
                - "xtol": Parameter tolerance (default: 1e-8)
                - "gtol": Gradient tolerance (default: 1e-8)
                - "maxiter": Maximum iterations
                - "verbose": Verbosity level

        Returns
        -------
        EstimationResult
            Object containing the estimation results including optimized parameters,
            component information, and metadata.

        Raises
        ------
        AssertionError
            If method specification is invalid or input parameters are inconsistent.
        ValueError
            If method format is incorrect or unsupported.
        FMICallException
            If simulation fails during parameter evaluation.

        Notes
        -----
        - The method automatically handles parameter normalization and bounds checking.
        - For AD mode, all components must be torch.nn.Module instances.
        - For FD mode, n_cores must be specified for parallel Jacobian computation.
        - Results are automatically saved to disk in the model's estimation_results directory.
        - Multiple time periods are supported by providing lists for startTime, endTime, and stepSize.

        Examples
        --------
        >>> # Simple estimation with default settings (legacy format)
        >>> result = estimator.estimate(
        ...     parameters=params,
        ...     measurements=devices,
        ...     startTime=start,
        ...     endTime=end,
        ...     stepSize=3600
        ... )

        >>> # Recommended: Use SLSQP with automatic differentiation (preferred)
        >>> result = estimator.estimate(
        ...     parameters=params,
        ...     measurements=devices,
        ...     startTime=start,
        ...     endTime=end,
        ...     stepSize=3600,
        ...     method=("scipy", "SLSQP", "ad")
        ... )

        >>> # Alternative: Use L-BFGS-B with automatic differentiation
        >>> result = estimator.estimate(
        ...     parameters=params,
        ...     measurements=devices,
        ...     startTime=start,
        ...     endTime=end,
        ...     stepSize=3600,
        ...     method=("scipy", "L-BFGS-B", "ad")
        ... )

        >>> # For constrained optimization: Use SLSQP with custom options
        >>> result = estimator.estimate(
        ...     parameters=params,
        ...     measurements=devices,
        ...     startTime=start,
        ...     endTime=end,
        ...     stepSize=3600,
        ...     method=("scipy", "SLSQP", "ad"),
        ...     options={"maxiter": 1000, "ftol": 1e-10}
        ... )

        >>> # For non-PyTorch models: Use finite difference method
        >>> result = estimator.estimate(
        ...     parameters=params,
        ...     measurements=devices,
        ...     startTime=start,
        ...     endTime=end,
        ...     stepSize=3600,
        ...     method=("scipy", "trf", "fd"),
        ...     n_cores=4  # Required for FD mode
        ... )

        >>> # String format examples (legacy support)
        >>> result = estimator.estimate(
        ...     parameters=params,
        ...     measurements=devices,
        ...     startTime=start,
        ...     endTime=end,
        ...     stepSize=3600,
        ...     method="L-BFGS-B"  # Automatically uses AD mode
        ... )
        """

        # Input validation and preprocessing
        if parameters is None:
            parameters = {"private": {}, "shared": {}}

        # Ensure required dictionary structure
        if "private" not in parameters:
            parameters["private"] = {}

        if "shared" not in parameters:
            parameters["shared"] = {}

        # Process private parameters
        for attr, par_dict in parameters["private"].items():
            # Ensure components is a list
            if not isinstance(par_dict["components"], list):
                parameters["private"][attr]["components"] = [par_dict["components"]]

            # Ensure x0 is a list with correct length
            if not isinstance(par_dict["x0"], list):
                parameters["private"][attr]["x0"] = [par_dict["x0"]] * len(
                    par_dict["components"]
                )
            else:
                assert len(par_dict["x0"]) == len(
                    par_dict["components"]
                ), f'The number of elements in the "x0" list must be equal to the number of components in the private dictionary for attribute {attr}.'

            # Ensure lb is a list with correct length
            if not isinstance(par_dict["lb"], list):
                parameters["private"][attr]["lb"] = [par_dict["lb"]] * len(
                    par_dict["components"]
                )
            else:
                assert len(par_dict["lb"]) == len(
                    par_dict["components"]
                ), f'The number of elements in the "lb" list must be equal to the number of components in the private dictionary for attribute {attr}.'

            # Ensure ub is a list with correct length
            if not isinstance(par_dict["ub"], list):
                parameters["private"][attr]["ub"] = [par_dict["ub"]] * len(
                    par_dict["components"]
                )
            else:
                assert len(par_dict["ub"]) == len(
                    par_dict["components"]
                ), f'The number of elements in the "ub" list must be equal to the number of components in the private dictionary for attribute {attr}.'

        # Process shared parameters
        members = ["x0", "lb", "ub"]
        for attr, par_dict in parameters["shared"].items():
            assert isinstance(
                par_dict["components"], list
            ), f'The "components" key in the shared dictionary must be a list for attribute {attr}.'
            assert (
                len(par_dict["components"]) > 0
            ), f'The "components" key in the shared dictionary must contain at least one element for attribute {attr}.'

            # Ensure components is a list of lists
            if not isinstance(par_dict["components"][0], list):
                parameters["shared"][attr]["components"] = [par_dict["components"]]

            # Process x0, lb, ub for shared parameters
            for m in members:
                if not isinstance(par_dict[m], list):
                    parameters["shared"][attr][m] = [
                        [par_dict[m] for c in l] for l in par_dict["components"]
                    ]
                else:
                    assert len(par_dict[m]) == len(
                        parameters["shared"][attr]["components"]
                    ), f'The number of elements in the "{m}" list must be equal to the number of components in the shared dictionary for attribute {attr}.'

            # Ensure all keys are properly formatted as lists
            for key, list_ in par_dict.items():
                if not isinstance(list_, list):
                    parameters["shared"][attr][key] = [[list_]]
                elif not isinstance(list_[0], list):
                    parameters["shared"][attr][key] = [list_]

        # Define allowed optimization methods
        allowed_methods = [
            ("scipy", "trf", "fd"),
            ("scipy", "dogbox", "fd"),
            ("scipy", "trf", "ad"),
            ("scipy", "dogbox", "ad"),
            ("scipy", "L-BFGS-B", "ad"),
            ("scipy", "TNC", "ad"),
            ("scipy", "SLSQP", "ad"),
            ("scipy", "trust-constr", "ad"),
        ]
        default_none_method = ("scipy", "SLSQP", "ad")
        default_methods = [("scipy", "SLSQP", "ad")]
        default_mode = (
            "ad"  # Always choose automatic differentiation mode when ambiguous
        )

        # Process method specification
        if isinstance(method, str):
            valid_methods = list(
                set([l[0] for l in allowed_methods] + [l[1] for l in allowed_methods])
            )
            assert (
                method in valid_methods
            ), f"If a string is provided, the \"method\" argument must be one of the following: {', '.join(valid_methods)} - \"{method}\" was provided."

            # Try to match with default methods first
            matched = False
            for t in default_methods:
                if t[0] == method:
                    method = t
                    matched = True
                    break

            # If no match found, look for candidates
            if not matched:
                candidates = []
                for m in allowed_methods:
                    if m[1] == method:
                        candidates.append(m)

                if len(candidates) == 1:
                    method = candidates[0]
                elif len(candidates) > 1:
                    # Choose the one with default mode
                    for c in candidates:
                        if c[2] == default_mode:
                            method = c
                            break

        elif isinstance(method, tuple):
            assert (
                len(method) == 3
            ), f'If a tuple is provided, it must contain three elements, corresponding to the library, method, and mode (e.g. ("scipy", "SLSQP", "ad")) - "{method}" was provided.'
            assert method[0] in [
                l[0] for l in allowed_methods
            ], f"If a tuple is provided, the first element must be one of the following: {', '.join(list(set([l[0] for l in allowed_methods])))} - \"{method}\" was provided."
            assert method[1] in [
                l[1] for l in allowed_methods
            ], f"If a tuple is provided, the second element must be one of the following: {', '.join(list(set([l[1] for l in allowed_methods])))} - \"{method}\" was provided."
            assert method[2] in [
                l[2] for l in allowed_methods
            ], f"If a tuple is provided, the third element must be one of the following: {', '.join(list(set([l[2] for l in allowed_methods])))} - \"{method}\" was provided."

            # Validate the method tuple
            method_ = None
            for t in allowed_methods:
                if t[0] == method[0] and t[1] == method[1] and t[2] == method[2]:
                    method_ = t
                    break
            assert (
                method_ is not None
            ), f"The method {method} is not valid. Only the following methods are supported: {', '.join([str(t) for t in allowed_methods])}"
            method = method_
        elif method is None:
            method = default_none_method
        else:
            raise ValueError(
                f'The "method" argument must be a string or a tuple - "{method}" was provided.'
            )

        # Set up time periods
        self.n_initialization_steps = n_initialization_steps
        if not isinstance(startTime, list):
            startTime = [startTime]
        if not isinstance(endTime, list):
            endTime = [endTime]
        if not isinstance(stepSize, list):
            stepSize = [stepSize]

        # Validate time periods
        for startTime_, endTime_, stepSize_ in zip(startTime, endTime, stepSize):
            assert (
                endTime_ > startTime_
            ), "The endTime must be later than the startTime."

        self._startTime_train = startTime
        self._endTime_train = endTime
        self._stepSize_train = stepSize

        # Extract component and parameter information
        self._flat_components_private = [
            obj
            for par_dict in parameters["private"].values()
            for obj in par_dict["components"]
        ]
        self._parameter_names_private = [
            attr
            for attr, par_dict in parameters["private"].items()
            for obj in par_dict["components"]
        ]
        self._flat_components_shared = [
            obj
            for par_dict in parameters["shared"].values()
            for obj_list in par_dict["components"]
            for obj in obj_list
        ]
        self._parameter_names_shared = [
            attr
            for attr, par_dict in parameters["shared"].items()
            for obj_list in par_dict["components"]
            for obj in obj_list
        ]
        self._parameter_names = (
            self._parameter_names_private + self._parameter_names_shared
        )
        self._flat_components = (
            self._flat_components_private + self._flat_components_shared
        )

        self._parameters = [
            rgetattr(component, attr)
            for component, attr in zip(self._flat_components, self._parameter_names)
        ]

        # Create parameter masks
        private_mask = np.arange(len(self._flat_components_private), dtype=int)
        shared_mask = []
        n = len(self._flat_components_private)
        k = 0
        for attr, par_dict in parameters["shared"].items():
            for obj_list in par_dict["components"]:
                for obj in obj_list:
                    shared_mask.append(k + n)
                k += 1
        shared_mask = np.array(shared_mask)
        self.theta_mask = np.concatenate((private_mask, shared_mask)).astype(int)

        # Store configuration
        self.parameters = parameters
        self.measurements = measurements
        self._obj_scale = None
        self.best_loss = math.inf
        self.n_timesteps = 0

        # Load actual measurements
        for i, (startTime_, endTime_, stepSize_) in enumerate(
            zip(self._startTime_train, self._endTime_train, self._stepSize_train)
        ):
            self.simulator.get_simulation_timesteps(startTime_, endTime_, stepSize_)
            self.n_timesteps += (
                len(self.simulator.secondTimeSteps) - self.n_initialization_steps
            )
            actual_readings = self.simulator.get_actual_readings(
                startTime=startTime_, endTime=endTime_, stepSize=stepSize_
            )
            if i == 0:
                self.actual_readings = {}
                for measuring_device, sd in self.measurements:
                    self.actual_readings[measuring_device.id] = actual_readings[
                        measuring_device.id
                    ].to_numpy()
            else:
                for measuring_device, sd in self.measurements:
                    self.actual_readings[measuring_device.id] = np.concatenate(
                        (
                            self.actual_readings[measuring_device.id],
                            actual_readings[measuring_device.id].to_numpy(),
                        ),
                        axis=0,
                    )

        # Extract initial values, bounds
        x0 = []
        for par_dict in parameters["private"].values():
            assert np.all(
                np.array(par_dict["x0"]) is not None
            ), "The x0 must be provided for all components."
            if len(par_dict["components"]) == len(par_dict["x0"]):
                x0 += par_dict["x0"]
            else:
                x0 += [par_dict["x0"][0]] * len(par_dict["components"])

        for par_dict in parameters["shared"].values():
            for l in par_dict["x0"]:
                x0.append(l[0])

        lb = []
        for par_dict in parameters["private"].values():
            lb_ = [i if i is not None else -np.inf for i in par_dict["lb"]]
            if len(par_dict["components"]) == len(par_dict["lb"]):
                lb += lb_
            else:
                lb += [lb_[0]] * len(par_dict["components"])
        for par_dict in parameters["shared"].values():
            lb_ = [i if i is not None else -np.inf for i in par_dict["lb"]]
            for l in par_dict["lb"]:
                lb.append(l[0])

        ub = []
        for par_dict in parameters["private"].values():
            ub_ = [i if i is not None else np.inf for i in par_dict["ub"]]
            if len(par_dict["components"]) == len(par_dict["ub"]):
                ub += ub_
            else:
                ub += [ub_[0]] * len(par_dict["components"])
        for par_dict in parameters["shared"].values():
            ub_ = [i if i is not None else np.inf for i in par_dict["ub"]]
            for l in par_dict["ub"]:
                ub.append(l[0])

        self._x0 = np.array(x0)
        self._lb = np.array(lb)
        self._ub = np.array(ub)

        # Validate bounds
        assert np.all(
            self._x0 >= self._lb
        ), f"The provided x0 must be larger than the provided lower bound lb for parameter {np.array(self._parameter_names)[self._x0 < self._lb][0]}"
        assert np.all(
            self._x0 <= self._ub
        ), f"The provided x0 must be smaller than the provided upper bound ub for parameter {np.array(self._parameter_names)[self._x0 > self._ub][0]}"

        # Set up parameter bounds and normalization
        self._set_bounds(normalize=True)

        # Run optimization based on method
        if method[0] == "scipy":
            if options is None:
                options = {}
            return self._scipy_solver(method=method, n_cores=n_cores, **options)
        else:
            raise ValueError(f"Unsupported library: {method[0]}")

    def _jac_fd(self, x0: np.ndarray, output: str) -> np.ndarray:
        """
        Compute the Jacobian matrix using finite differences.

        This method implements numerical differentiation using finite difference schemes
        to compute the Jacobian matrix for optimization algorithms that require gradient
        information but cannot use automatic differentiation.

        Parameters
        ----------
        x0 : np.ndarray
            Parameter vector at which to compute the Jacobian.

        Returns
        -------
        np.ndarray
            Jacobian matrix with shape (n_residuals, n_parameters).

        Notes
        -----
        This method uses a 2-point finite difference scheme by default, with automatic
        adjustment for bound constraints. The step size is computed based on the
        parameter values and machine precision.
        """

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
            if scheme == "1-sided":
                use_one_sided = np.ones_like(h, dtype=bool)
            elif scheme == "2-sided":
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

            if scheme == "1-sided":
                x = x0 + h_total
                violated = (x < lb) | (x > ub)
                fitting = np.abs(h_total) <= np.maximum(lower_dist, upper_dist)
                h_adjusted[violated & fitting] *= -1

                forward = (upper_dist >= lower_dist) & ~fitting
                h_adjusted[forward] = upper_dist[forward] / num_steps
                backward = (upper_dist < lower_dist) & ~fitting
                h_adjusted[backward] = -lower_dist[backward] / num_steps
            elif scheme == "2-sided":
                central = (lower_dist >= h_total) & (upper_dist >= h_total)

                forward = (upper_dist >= lower_dist) & ~central
                h_adjusted[forward] = np.minimum(
                    h[forward], 0.5 * upper_dist[forward] / num_steps
                )
                use_one_sided[forward] = True

                backward = (upper_dist < lower_dist) & ~central
                h_adjusted[backward] = -np.minimum(
                    h[backward], 0.5 * lower_dist[backward] / num_steps
                )
                use_one_sided[backward] = True

                min_dist = np.minimum(upper_dist, lower_dist) / num_steps
                adjusted_central = ~central & (np.abs(h_adjusted) <= min_dist)
                h_adjusted[adjusted_central] = min_dist[adjusted_central]
                use_one_sided[adjusted_central] = False

            return h_adjusted, use_one_sided

        def _dense_difference(fun, x0, f0, h, use_one_sided, method):
            """Compute finite differences for dense Jacobian computation."""
            m = f0.size
            n = x0.size
            J_transposed = np.empty((n, m))
            x1 = x0.copy()
            x2 = x0.copy()
            xc = x0.astype(complex, copy=True)

            x1_ = np.empty((n, n))
            x2_ = np.empty((n, n))

            for i in range(h.size):
                if method == "2-point":
                    x1[i] += h[i]
                elif method == "3-point" and use_one_sided[i]:
                    x1[i] += h[i]
                    x2[i] += 2 * h[i]
                elif method == "3-point" and not use_one_sided[i]:
                    x1[i] -= h[i]
                    x2[i] += h[i]
                else:
                    raise RuntimeError("Never be here.")

                x1_[i, :] = x1
                x2_[i, :] = x2
                x1[i] = x2[i] = xc[i] = x0[i]

            if method == "2-point":
                args = [(x, output) for x in x1_]
                f = np.array(
                    list(
                        self.jac_pool.starmap(
                            self._obj_fd, args, chunksize=self.jac_chunksize
                        )
                    )
                )
                df = f - f0
                dx = np.diag(x1_) - x0
            elif method == "3-point":
                args = [(x, output) for x in x1_]
                f1 = np.array(
                    list(
                        self.jac_pool.starmap(
                            self._obj_fd, args, chunksize=self.jac_chunksize
                        )
                    )
                )
                args = [(x, output) for x in x2_]
                f2 = np.array(
                    list(
                        self.jac_pool.starmap(
                            self._obj_fd, args, chunksize=self.jac_chunksize
                        )
                    )
                )
                df = np.empty_like(f1)
                df[use_one_sided, :] = (
                    -3.0 * f0[use_one_sided]
                    + 4 * f1[use_one_sided, :]
                    - f2[use_one_sided, :]
                )
                df[~use_one_sided] = f2[~use_one_sided, :] - f1[~use_one_sided, :]
                dx = np.diag(x2_) - x0
                dx[~use_one_sided] = (
                    np.diag(x2_)[~use_one_sided] - np.diag(x1_)[~use_one_sided]
                )

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
                dx = (x0 + abs_step) - x0
                abs_step = np.where(
                    dx == 0, rstep * sign_x0 * np.maximum(1.0, np.abs(x0)), abs_step
                )

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
                return EPS ** (1 / 3)
            else:
                raise RuntimeError(
                    "Unknown step method, should be one of "
                    "{'2-point', '3-point', 'cs'}"
                )

        method = "2-point"
        rel_step = None
        f0 = None

        if method not in ["2-point", "3-point", "cs"]:
            raise ValueError("Unknown method '%s'. " % method)

        xp = array_namespace(x0)
        _x = _atleast_nd(x0, ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, "real floating"):
            _dtype = _x.dtype

        # promotes to floating
        x0 = xp.astype(_x, _dtype)

        if x0.ndim > 1:
            raise ValueError("`x0` must have at most 1 dimension.")

        lb, ub = self.bounds.lb, self.bounds.ub
        bounds = (lb, ub)
        lb, ub = _prepare_bounds(bounds, x0)

        if lb.shape != x0.shape or ub.shape != x0.shape:
            raise ValueError("Inconsistent shapes between bounds and `x0`.")

        if f0 is None:
            f0 = self._obj_fd(x0, output)
        else:
            f0 = np.atleast_1d(f0)
            if f0.ndim > 1:
                raise ValueError("`f0` passed has more than 1 dimension.")

        if np.any((x0 < lb) | (x0 > ub)):
            raise ValueError("`x0` violates bound constraints.")

        # by default we use rel_step
        h = _compute_absolute_step(rel_step, x0, f0, method)

        if method == "2-point":
            h, use_one_sided = _adjust_scheme_to_bounds(x0, h, 1, "1-sided", lb, ub)
        elif method == "3-point":
            h, use_one_sided = _adjust_scheme_to_bounds(x0, h, 1, "2-sided", lb, ub)
        elif method == "cs":
            use_one_sided = False

        jac = _dense_difference(self._obj_fd, x0, f0, h, use_one_sided, method)

        return jac

    def __getstate__(self):
        """Prepare object for pickling by removing non-serializable attributes."""

        self_dict = self.__dict__.copy()
        if hasattr(self, "fun_pool"):
            del self_dict["fun_pool"]
        if hasattr(self, "jac_pool"):
            del self_dict["jac_pool"]

        if hasattr(self, "obj"):
            del self_dict["obj"]
            del self_dict["_theta_obj"]
        if hasattr(self, "jac"):
            del self_dict["jac"]
            del self_dict["_theta_jac"]
        if hasattr(self, "hes"):
            del self_dict["hes"]
            del self_dict["_theta_hes"]
        return self_dict

    def _obj_fd(self, theta: np.ndarray, output: str) -> np.ndarray:
        """
        Objective function wrapper for finite difference methods.

        This method handles exceptions during objective function evaluation
        and returns a large penalty value if the simulation fails.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Objective function value or penalty value if evaluation fails.
        """
        try:
            theta_tensor = torch.tensor(theta, dtype=torch.float64)
            res = self._obj(theta_tensor, output).detach().numpy()
        except FMICallException:
            res = self.res_fail
        except Exception as e:
            # Handle any other exceptions, including TensorWrapper issues
            print(f"Warning: Objective function evaluation failed: {e}")
            res = self.res_fail
        return res

    def _obj_fd_separate_process(self, theta: np.ndarray, output: str) -> np.ndarray:
        """
        Evaluate objective function in a separate process.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Objective function value.
        """
        # res = np.array(list(self.fun_pool.imap(self._obj_fd, [(theta, output)], chunksize=self.jac_chunksize))[0])
        res = list(
            self.fun_pool.starmap(
                self._obj_fd, [(theta, output)], chunksize=self.jac_chunksize
            )
        )[0]

        return res

    def _set_bounds(self, normalize: bool = True) -> None:
        """
        Set up parameter bounds and enable gradients for optimization.

        This method configures the parameter bounds and enables gradient computation
        for parameters that will be estimated.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to normalize parameter bounds to [0, 1] range.

        Notes
        -----
        - All components must be torch.nn.Module instances for gradient computation.
        - Parameters must be subclasses of tps.Parameter.
        - Bounds are set on the parameter objects for constraint enforcement.
        """
        # Enable gradients for parameters to be estimated
        for component, attr in zip(self._flat_components, self._parameter_names):
            assert isinstance(
                component, nn.Module
            ), "All components must be subclasses of nn.Module when using PyTorch-based optimization"
            param = rgetattr(component, attr)
            assert isinstance(
                param, (tps.Parameter)
            ), "All parameters must be subclasses of tps.Parameter when using PyTorch-based optimization"
            param.requires_grad_(True)

            if (
                attr in self.parameters["private"]
                and component in self.parameters["private"][attr]["components"]
            ):
                idx = self.parameters["private"][attr]["components"].index(component)
                if normalize:
                    lb = self.parameters["private"][attr]["lb"][idx]
                    ub = self.parameters["private"][attr]["ub"][idx]
                else:
                    lb = 0  # Do nothing
                    ub = 1  # Do nothing
                param.min_value = lb
                param.max_value = ub

            elif (
                attr in self.parameters["shared"]
                and component in self.parameters["shared"][attr]["components"]
            ):

                if normalize:
                    lb = self.parameters["shared"][attr]["lb"]
                    ub = self.parameters["shared"][attr]["ub"]
                else:
                    lb = 0  # Do nothing
                    ub = 1  # Do nothing
                param.min_value = lb
                param.max_value = ub

        self._lb_norm = np.array(
            [param.normalize(lb) for param, lb in zip(self._parameters, self._lb)]
        )
        self._ub_norm = np.array(
            [param.normalize(ub) for param, ub in zip(self._parameters, self._ub)]
        )
        self._x0_norm = np.array(
            [param.normalize(x0) for param, x0 in zip(self._parameters, self._x0)]
        )

    def _scipy_solver(
        self, method: tuple, n_cores: Optional[int] = None, **options
    ) -> EstimationResult:
        """
        Perform optimization using SciPy's optimization algorithms.

        This method handles both automatic differentiation and finite difference
        optimization using various SciPy optimizers.

        Parameters
        ----------
        method : tuple
            Tuple of (library, optimizer, mode) specifying the optimization method.
        **options
            Additional options for the optimization algorithm.

        Returns
        -------
        EstimationResult
            Object containing the estimation results.

        Raises
        ------
        ValueError
            If the optimization method is not supported.
        """
        datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str("{}{}".format(datestr, f"_{str(method)}.pickle"))
        self.result_savedir_pickle, isfile = self.simulator.model.get_dir(
            folder_list=["model_parameters", "estimation_results"], filename=filename
        )

        # Disable gradients for non-estimated parameters
        for component in self.simulator.model.components.values():
            if isinstance(component, nn.Module):
                for name, param in component.named_parameters():
                    param.requires_grad_(False)

        # Validate that all components are torch.nn.Module
        for component in self._flat_components:
            assert isinstance(
                component, nn.Module
            ), "All components must be subclasses of nn.Module when using PyTorch-based optimization"

        # Set initial parameters
        self.simulator.model.set_parameters_from_array(
            self._x0_norm,
            self._flat_components,
            self._parameter_names,
            normalized=True,
            overwrite=True,
            save_original=True,
        )

        assert len(self._parameters) > 0, "No parameters to optimize"

        # Initialize simulator
        self.simulator.get_simulation_timesteps(
            self._startTime_train[0], self._endTime_train[0], self._stepSize_train[0]
        )
        self.simulator.model.initialize(
            startTime=self._startTime_train[0],
            endTime=self._endTime_train[0],
            stepSize=self._stepSize_train[0],
            simulator=self.simulator,
        )

        # Disable gradients for history to save memory
        for component in self.simulator.model.components.values():
            for output in component.output.values():
                if isinstance(output, tps.Scalar):
                    output.set_requires_grad(False)

        # Create bounds object for SciPy
        self.bounds = Bounds(lb=self._lb_norm, ub=self._ub_norm)

        assert np.all(self.bounds.lb <= self._x0_norm) and np.all(
            self._x0_norm <= self.bounds.ub
        ), "Initial guess must be within bounds"

        # Initialize caching variables for AD
        self._theta_obj = torch.nan * torch.ones_like(
            torch.tensor(self._x0_norm, dtype=torch.float64)
        )
        self._theta_jac = torch.nan * torch.ones_like(
            torch.tensor(self._x0_norm, dtype=torch.float64)
        )
        self._theta_hes = torch.nan * torch.ones_like(
            torch.tensor(self._x0_norm, dtype=torch.float64)
        )

        # Run optimization based on method
        if method[1] in ["trf", "dogbox"]:
            if method[2] == "ad":
                result = least_squares(
                    self._obj_ad,
                    x0=self._x0_norm,
                    args=("vector",),
                    jac=self._jac_ad,
                    bounds=self.bounds,
                    method=method[1],
                    **options,
                )
            else:
                # Clean up torch objects before setting up FD method
                # self.cleanup_torch_objects() # Removed as per edit hint

                res_fail = np.zeros((self.n_timesteps, len(self.measurements)))
                for j, measuring_device in enumerate(self.measurements):
                    res_fail[:, j] = np.ones((self.n_timesteps)) * 100
                self.res_fail = res_fail.flatten()

                assert (
                    n_cores is not None
                ), "n_cores must be provided when using FD method"

                # Set up multiprocessing pools
                self.fun_pool = multiprocessing.get_context("spawn").Pool(
                    1, maxtasksperchild=30
                )
                self.jac_pool = multiprocessing.get_context("spawn").Pool(
                    n_cores, maxtasksperchild=10
                )
                self.jac_chunksize = 1

                # Make model pickable and ensure all tensors are properly handled
                self.simulator.model.make_pickable()

                result = least_squares(
                    self._obj_fd_separate_process,
                    x0=self._x0_norm,
                    args=("vector",),
                    jac=self._jac_fd,
                    bounds=self.bounds,
                    method=method[1],
                    **options,
                )
        else:
            if method[1] in [
                "newton-cg",
                "dogleg",
                "trust-ncg",
                "trust-constr",
                "trust-krylov",
                "trust-exact",
                "_custom",
            ]:  # See optimize._minimize for these options
                hess = self._hes_ad
            else:
                hess = None

            # Ensure all arrays are float64
            self._x0_norm = np.asarray(self._x0_norm, dtype=np.float64)
            if self.bounds is not None:
                self.bounds.lb = np.asarray(self.bounds.lb, dtype=np.float64)
                self.bounds.ub = np.asarray(self.bounds.ub, dtype=np.float64)

            result = minimize(
                self._obj_ad,
                self._x0_norm,
                args=("scalar",),
                method=method[1],
                jac=self._jac_ad,
                hess=hess,
                bounds=self.bounds,
                options=options,
            )

        if method[0] == "scipy":
            self.simulator.model.restore_parameters(keep_values=False)

        # Create and save result
        result = EstimationResult(
            result_x=result.x,
            component_id=[com.id for com in self._flat_components],
            component_attr=[attr for attr in self._parameter_names],
            theta_mask=self.theta_mask,
            startTime_train=self._startTime_train,
            endTime_train=self._endTime_train,
            stepSize_train=self._stepSize_train,
            x0=self._x0,
            lb=self._lb,
            ub=self._ub,
            iterations=getattr(result, "nit", None),
            nfev=getattr(result, "nfev", None),
            final_objective=getattr(result, "fun", None),
            success=getattr(result, "success", None),
            message=getattr(result, "message", None),
        )

        with open(self.result_savedir_pickle, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return result

    def _obj(self, theta: torch.Tensor, output: str) -> torch.Tensor:
        """
        Objective function for automatic differentiation.

        This method computes the objective function value by running a simulation
        with the given parameters and comparing the results with actual measurements.

        Parameters
        ----------
        theta : torch.Tensor
            Flattened parameter vector.
        output : str, default="scalar"
            Output format: "scalar" for mean squared error, "vector" for residuals.

        Returns
        -------
        torch.Tensor
            Objective function value.

        Raises
        ------
        ValueError
            If output format is invalid.
        """
        theta = theta[self.theta_mask]
        self.simulator.model.set_parameters_from_array(
            theta,
            self._flat_components,
            self._parameter_names,
            normalized=True,
            overwrite=True,
        )

        n_time_prev = 0
        simulation_readings = {
            com.id: torch.zeros((self.n_timesteps), dtype=torch.float64)
            for com, sd in self.measurements
        }
        actual_readings = {
            com.id: torch.zeros((self.n_timesteps), dtype=torch.float64)
            for com, sd in self.measurements
        }

        # Run simulations for all time periods
        for startTime_, endTime_, stepSize_ in zip(
            self._startTime_train, self._endTime_train, self._stepSize_train
        ):
            self.simulator.simulate(
                stepSize=stepSize_,
                startTime=startTime_,
                endTime=endTime_,
                show_progress_bar=False,
            )
            n_time = len(self.simulator.dateTimeSteps) - self.n_initialization_steps

            # Extract and normalize measurements
            for measuring_device, sd in self.measurements:
                y_model = measuring_device.input["measuredValue"].history[
                    self.n_initialization_steps :
                ]
                y_actual = torch.tensor(
                    self.actual_readings[measuring_device.id], dtype=torch.float64
                )[self.n_initialization_steps :]
                # y_model_norm = measuring_device.input["measuredValue"].normalize(y_model)
                # y_actual_norm = measuring_device.input["measuredValue"].normalize(y_actual)
                y_model_norm = y_model
                y_actual_norm = y_actual

                simulation_readings[measuring_device.id][
                    n_time_prev : n_time_prev + n_time
                ] = y_model_norm
                actual_readings[measuring_device.id][
                    n_time_prev : n_time_prev + n_time
                ] = y_actual_norm

            n_time_prev += n_time

        # Compute residuals
        res = torch.zeros((self.n_timesteps, len(self.measurements)))
        for j, (measuring_device, sd) in enumerate(self.measurements):
            simulation_readings_ = simulation_readings[measuring_device.id]
            actual_readings_ = actual_readings[measuring_device.id]
            res[:, j] = (actual_readings_ - simulation_readings_) / sd

        # Return appropriate output format
        if output == "scalar":
            obj = torch.mean(res.flatten() ** 2)
            if self._obj_scale is None:
                self._obj_scale = obj.detach().item()
        elif output == "vector":
            obj = res.flatten()
            if self._obj_scale is None:
                self._obj_scale = torch.mean(obj**2).detach().item()
        else:
            raise ValueError(f"Invalid output: {output}")

        # Scale objective function to 100 initially for numerical stability
        self.obj = 100 * obj / self._obj_scale
        return self.obj

    def _obj_ad(self, theta: torch.Tensor, output: str = "scalar") -> torch.Tensor:
        """
        Wrapper function for SciPy interface that converts numpy to torch and returns numpy.

        This method provides caching to avoid redundant computations when the same
        parameter vector is evaluated multiple times.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector.
        output : str, default="scalar"
            Output format.

        Returns
        -------
        torch.Tensor
            Objective value as numpy array.
        """
        theta = torch.tensor(theta, dtype=torch.float64)
        if torch.equal(theta, self._theta_obj):
            return np.asarray(self.obj.detach().numpy(), dtype=np.float64)
        else:
            self._theta_obj = theta
            self.obj = self._obj(theta, output)
            return np.asarray(self.obj.detach().numpy(), dtype=np.float64)

    def __jac_ad(self, theta: torch.Tensor, output: str) -> torch.Tensor:
        """
        Compute the Jacobian matrix using automatic differentiation.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector.

        Returns
        -------
        torch.Tensor
            Jacobian matrix.

        Notes
        -----
        Uses torch.func.jacfwd for forward-mode automatic differentiation.
        """
        self.jac = torch.func.jacfwd(self._obj, argnums=0)(theta, output)
        assert not torch.any(torch.isnan(self.jac)), "JAC contains NaNs"
        return self.jac

    def _jac_ad(self, theta: torch.Tensor, output: str) -> torch.Tensor:
        """
        Compute the Jacobian matrix using automatic differentiation with caching.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector.
        *args
            Additional arguments (ignored).

        Returns
        -------
        torch.Tensor
            Jacobian matrix as numpy array.
        """
        theta = torch.tensor(theta, dtype=torch.float64)

        if torch.equal(theta, self._theta_jac):
            return np.asarray(self.jac.detach().numpy(), dtype=np.float64)
        else:
            self._theta_jac = theta
            self.jac = self.__jac_ad(theta, output)
            return np.asarray(self.jac.detach().numpy(), dtype=np.float64)

    def __hes_ad(self, theta: torch.Tensor, output: str) -> torch.Tensor:
        """
        Compute the Hessian matrix using automatic differentiation.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector.
        output : str, default="scalar"
            Output format for the objective function.

        Returns
        -------
        torch.Tensor
            Hessian matrix.

        Notes
        -----
        Uses torch.func.jacfwd applied to the Jacobian function.
        """
        self.hes = torch.func.jacfwd(self.__jac_ad, argnums=0)(theta, output)
        return self.hes

    def _hes_ad(self, theta: torch.Tensor, output: str) -> torch.Tensor:
        """
        Compute the Hessian matrix using automatic differentiation with caching.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector.
        *args
            Additional arguments (ignored).

        Returns
        -------
        torch.Tensor
            Hessian matrix as numpy array.
        """
        theta = torch.tensor(theta, dtype=torch.float64)

        if torch.equal(theta, self._theta_hes):
            return np.asarray(self.hes.detach().numpy(), dtype=np.float64)
        else:
            self._theta_hes = theta
            self.hes = self.__hes_ad(theta, output)
            return np.asarray(self.hes.detach().numpy(), dtype=np.float64)


class EstimationResult(dict):
    """
    A dictionary-like object containing parameter estimation results.

    This class stores the results of parameter estimation including optimized
    parameters, component information, and metadata about the estimation process.

    Attributes
    ----------
    result_x : np.ndarray
        Optimized parameter values.
    component_id : List[str]
        List of component IDs corresponding to the parameters.
    component_attr : List[str]
        List of attribute names corresponding to the parameters.
    theta_mask : np.ndarray
        Mask indicating which parameters were estimated.
    startTime_train : List[datetime.datetime]
        Start times of training periods.
    endTime_train : List[datetime.datetime]
        End times of training periods.
    stepSize_train : List[int]
        Step sizes used in training periods.
    x0 : np.ndarray
        Initial parameter values.
    lb : np.ndarray
        Lower bounds for parameters.
    ub : np.ndarray
        Upper bounds for parameters.
    iterations : Optional[int]
        Number of iterations performed by the optimizer.
    nfev : Optional[int]
        Number of function evaluations performed by the optimizer.
    final_objective : Optional[float]
        Final objective function value achieved.
    success : Optional[bool]
        Whether the optimization was successful.
    message : Optional[str]
        Optimization result message.

    Examples
    --------
    >>> result = EstimationResult(
    ...     result_x=np.array([0.8, 0.9]),
    ...     component_id=["comp1", "comp2"],
    ...     component_attr=["efficiency", "efficiency"],
    ...     theta_mask=np.array([0, 1]),
    ...     startTime_train=[datetime.datetime(2024, 1, 1)],
    ...     endTime_train=[datetime.datetime(2024, 1, 2)],
    ...     stepSize_train=[3600],
    ...     x0=np.array([0.7, 0.8]),
    ...     lb=np.array([0.5, 0.6]),
    ...     ub=np.array([1.0, 1.0]),
    ...     iterations=15,
    ...     nfev=45,
    ...     final_objective=0.00123,
    ...     success=True,
    ...     message="Optimization terminated successfully"
    ... )
    >>> print(result["result_x"])
    [0.8 0.9]
    >>> print(result["iterations"])
    15
    """

    def __init__(
        self,
        result_x: Optional[np.ndarray] = None,
        component_id: Optional[List[str]] = None,
        component_attr: Optional[List[str]] = None,
        theta_mask: Optional[np.ndarray] = None,
        startTime_train: Optional[List[datetime.datetime]] = None,
        endTime_train: Optional[List[datetime.datetime]] = None,
        stepSize_train: Optional[List[int]] = None,
        x0: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        iterations: Optional[int] = None,
        nfev: Optional[int] = None,
        final_objective: Optional[float] = None,
        success: Optional[bool] = None,
        message: Optional[str] = None,
    ):
        """
        Initialize the EstimationResult object.

        Parameters
        ----------
        result_x : Optional[np.ndarray], optional
            Optimized parameter values.
        component_id : Optional[List[str]], optional
            List of component IDs.
        component_attr : Optional[List[str]], optional
            List of attribute names.
        theta_mask : Optional[np.ndarray], optional
            Parameter mask.
        startTime_train : Optional[List[datetime.datetime]], optional
            Training start times.
        endTime_train : Optional[List[datetime.datetime]], optional
            Training end times.
        stepSize_train : Optional[List[int]], optional
            Training step sizes.
        x0 : Optional[np.ndarray], optional
            Initial parameter values.
        lb : Optional[np.ndarray], optional
            Lower bounds.
        ub : Optional[np.ndarray], optional
            Upper bounds.
        iterations : Optional[int], optional
            Number of iterations performed by the optimizer.
        nfev : Optional[int], optional
            Number of function evaluations performed by the optimizer.
        final_objective : Optional[float], optional
            Final objective function value achieved.
        success : Optional[bool], optional
            Whether the optimization was successful.
        message : Optional[str], optional
            Optimization result message.
        """
        super().__init__(
            result_x=result_x,
            component_id=component_id,
            component_attr=component_attr,
            theta_mask=theta_mask,
            startTime_train=startTime_train,
            endTime_train=endTime_train,
            stepSize_train=stepSize_train,
            x0=x0,
            lb=lb,
            ub=ub,
            iterations=iterations,
            nfev=nfev,
            final_objective=final_objective,
            success=success,
            message=message,
        )

    def __copy__(self):
        """Create a shallow copy of the EstimationResult."""
        return EstimationResult(**self)

    def copy(self):
        """Create a shallow copy of the EstimationResult."""
        return self.__copy__()
