from __future__ import annotations

# Standard library imports
import datetime
import functools

# import multiprocessing
import math
import os
import pickle
import time as time_module
import warnings
from contextlib import nullcontext as _nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import torch.nn as nn
from fmpy.fmi1 import FMICallException
from scipy._lib._array_api import array_namespace
from scipy.optimize import Bounds, basinhopping, dual_annealing, least_squares, minimize

# Local application imports
import twin4build.core as core
import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.systems.utils.smooth_saturation import saturation_mode
from twin4build.utils.deprecation import deprecate_args
from twin4build.utils.print_progress import LOGGER
from twin4build.utils.rgetattr import rgetattr

# Per-sensor lower bound on the standard deviation used inside
# ``measurements="auto"`` (both for the placeholder when sensor data
# isn't loaded yet and for the data-driven path).  Equal to 5 % of a
# normalized actuator full scale, 0.05 K on temperatures, and 0.05 kg/s
# on supply-air flows -- all below typical instrument resolution but
# large enough that ``1/sd**2`` weighting cannot let one channel dwarf
# the rest.  Hand-built measurement lists pass their own ``sd`` and are
# **not** affected by this floor (see ``estimate`` -- only the IDs
# returned by :meth:`_auto_measurements` are refreshed against the
# loaded ``actual_readings``).
AUTO_SD_FLOOR = 0.05


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

    This class estimates model parameters from measured data using maximum
    likelihood estimation (MLE). Gradients are computed either by automatic
    differentiation (AD, preferred for torch-based models) or by finite
    differences (FD, for FMU/non-torch models).

    Args:
        simulator: The simulator instance for running simulations.

    Overview
    --------

    Two *transcriptions* of the estimation problem are supported:

    - **Single-shooting** (default): the model is simulated over the full
      horizon from a fixed initial state, and only the physical parameters
      are decision variables. Available with all optimizer backends.
    - **Collocation**: the state at every timestep boundary is promoted to a
      decision variable and the dynamics are enforced as sparse equality
      constraints. Available with the CasADi/IPOPT backend only; robust to
      poor initial parameter guesses and returns the estimated initial state
      as part of the fit.

    Two optimizer *backends* are supported:

    - **SciPy** (``method=("scipy", <optimizer>, <mode>)``): local optimizers
      (SLSQP, L-BFGS-B, TNC, trust-constr, trf, dogbox) and global optimizers
      (dual_annealing, basinhopping).
    - **CasADi/IPOPT** (``method=("casadi", "ipopt", "ad")``): the IPOPT
      interior-point solver, optionally with the collocation transcription
      (``method=("casadi", "ipopt", "ad", "collocation")``).

    For composable torch models, ``options={"fast": True}`` replaces the
    object-graph objective with an equivalent composed one-step map that
    skips the per-step Python dispatch (see the Single-Shooting section
    below); values and gradients are identical by construction.

    Mathematical Formulation
    ------------------------

    Maximum Likelihood Estimation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The general parameter estimation problem is formulated as a maximum likelihood estimation:

        .. math::

            \hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta} \in \Theta}{\operatorname{argmax}} \; \mathcal{L}(\boldsymbol{\theta} | \boldsymbol{Y})

    where:

        - :math:`\hat{\boldsymbol{\theta}}` is the maximum likelihood estimate
        - :math:`\boldsymbol{\theta}` is the parameter vector
        - :math:`\Theta \subseteq \mathbb{R}^{n_p}` is the parameter space
        - :math:`\mathcal{L}(\boldsymbol{\theta} | \boldsymbol{Y})` is the likelihood function
        - :math:`\boldsymbol{Y}` are the observed measurements

    Dimensions
    ^^^^^^^^^^

    - :math:`n_t`: Number of time steps in the simulation period
    - :math:`n_p`: Number of parameters to estimate
    - :math:`n_x`: Number of input variables (disturbances, setpoints, etc.)
    - :math:`n_y`: Number of output variables (measurements, performance metrics)

    Model Structure
    ^^^^^^^^^^^^^^^

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

    Likelihood Function
    ^^^^^^^^^^^^^^^^^^^

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



    Parameter Bounds
    ^^^^^^^^^^^^^^^^

    For each parameter :math:`\theta_{i}`:

    .. math::

            \theta_{i}^{lb} \leq \theta_{i} \leq \theta_{i}^{ub}

    where:

        - :math:`\theta_{i}^{lb}` is the lower bound
        - :math:`\theta_{i}^{ub}` is the upper bound

    Single-Shooting (default)
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Single-shooting evaluates the objective by simulating the *entire*
    horizon from a fixed initial state :math:`\boldsymbol{x}_0`. With the
    one-step state-transition map :math:`f` (one simulator step of the full
    model) the predicted trajectory is the :math:`n_t`-fold composition of
    :math:`f` with itself:

    .. math::

            \boldsymbol{x}_{t+1} = f(\boldsymbol{x}_t, \boldsymbol{X}_t, \boldsymbol{\theta}),
            \qquad
            \hat{\boldsymbol{Y}}_t = g(\boldsymbol{x}_t, \boldsymbol{X}_t, \boldsymbol{\theta}),
            \qquad t = 0, \ldots, n_t - 1

    and only :math:`\boldsymbol{\theta} \in \mathbb{R}^{n_p}` is a decision
    variable. The problem is solved as the unconstrained (bound-constrained)
    minimization of the negative log-likelihood above.

    Gradients are obtained either by backpropagating through the unrolled
    trajectory (AD mode, requires all components to be torch modules) or by
    parallel finite differences (FD mode, requires ``n_cores``). The gradient
    flows through the full composition
    :math:`f \circ f \circ \cdots \circ f`, i.e. a product of per-step
    Jacobians. For unstable or oscillatory dynamics this product is badly
    conditioned on long horizons (the exploding/vanishing-gradient problem of
    backprop through time) -- the classical argument for simultaneous
    transcriptions. Dissipative building models are largely insensitive to
    this: the per-step Jacobians contract, so single-shooting remains well
    behaved even on multi-week horizons.

    For composable torch models, ``options={"fast": True}`` builds a pure
    one-step map :math:`F_{\text{aug}}` by composing the components'
    ``forward`` methods, captures the exogenous inputs once from a reference
    rollout, and evaluates the same objective as a plain sequential torch
    rollout -- removing the per-step object-graph dispatch. Every composable
    component's ``do_step`` delegates to the same ``forward`` the composed
    map threads, so the fast objective is exact by construction; the
    estimator silently falls back to the object-graph objective for
    non-composable models.

    Collocation
    ~~~~~~~~~~~

    The collocation (simultaneous) transcription promotes the state at every
    timestep boundary :math:`\boldsymbol{s}_i` to a decision variable,
    stacked alongside the physical parameters. The dynamics are enforced as
    hard equality *continuity defects*:

    .. math::

            \boldsymbol{d}_i = f(\boldsymbol{s}_i, \boldsymbol{X}_i, \boldsymbol{\theta}) - \boldsymbol{s}_{i+1} = \boldsymbol{0},
            \qquad i = 0, \ldots, n_t - 2

    yielding the equality-constrained problem:

    .. math::

            \underset{\boldsymbol{\theta}, \boldsymbol{s}_0, \ldots, \boldsymbol{s}_{n_t-1}}{\operatorname{minimize}}
            \; \sum_{j=1}^{n_y} \sum_{t=1}^{n_t} \left(\frac{Y_{j,t} - \hat{Y}_{j,t}}{\sigma_j}\right)^2
            \quad \text{subject to} \quad \boldsymbol{d}_i = \boldsymbol{0} \; \forall i

    Gradients only ever flow through a *single* simulation step. In
    practice the benefits of this transcription are robustness to poor
    initial parameter guesses (the state decision variables can stay close
    to the data even while :math:`\boldsymbol{\theta}` is far off), the
    sparse block-bidiagonal NLP structure that IPOPT exploits, and the
    estimated initial state coming out of the fit for free. The boundary
    states are nuisance variables: after the fit only
    :math:`\boldsymbol{\theta}` is reported (the per-period initial states
    are additionally returned as ``estimated_initial_state``).

    The solve hands IPOPT the defects as sparse equality constraints with an
    explicit block-bidiagonal Jacobian, a Gauss-Newton Hessian of the
    least-squares objective, and patience-based early stopping. It requires
    the CasADi/IPOPT backend
    (``method=("casadi", "ipopt", "ad", "collocation")``). A defect audit is
    returned as
    ``transcription_audit`` so the quality of the converged solution can be
    inspected (max defect, per-sensor RMSE consistency between the NLP
    solution and a forward rollout).

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
    >>> # Define parameters to estimate: (component, attribute, x0, lb, ub)
    >>> parameters = [
    ...     (space, "thermal.C_air", 2e+6, 1e+6, 1e+7),
    ...     (space, "thermal.C_wall", 2e+6, 1e+6, 1e+7),
    ...     ([controller1, controller2], "kp", 0.001, 1e-5, 1, "shared"),
    ... ]
    >>>
    >>> # Define measuring devices (sensors with historical readings)
    >>> measurements = [temperature_sensor, co2_sensor]
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
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("scipy", "SLSQP", "ad")  # Preferred for most problems
    ... )

    Fast single-shooting for composable torch models (same result, faster):

    >>> result = estimator.estimate(
    ...     parameters=parameters,
    ...     measurements=measurements,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("scipy", "SLSQP", "ad"),
    ...     options={"fast": True}
    ... )

    IPOPT single-shooting via CasADi:

    >>> result = estimator.estimate(
    ...     parameters=parameters,
    ...     measurements=measurements,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("casadi", "ipopt", "ad")
    ... )

    Collocation transcription:

    >>> result = estimator.estimate(
    ...     parameters=parameters,
    ...     measurements=measurements,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("casadi", "ipopt", "ad", "collocation"),
    ...     options={"maxiter": 500, "early_stopping": {"patience": 20}}
    ... )

    For non-PyTorch models, use the finite difference mode:

    >>> result = estimator.estimate(
    ...     parameters=parameters,
    ...     measurements=measurements,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("scipy", "trf", "fd"),
    ...     n_cores=4  # Required for FD mode
    ... )
    """

    def __init__(self, simulator: core.Simulator):
        """
        Initialize the Estimator.

        Args:
            simulator : The simulator instance for running simulations.
        """
        assert isinstance(
            simulator, core.Simulator
        ), "Simulator must be a twin4build.core.Simulator instance"
        self.simulator = simulator
        self.tol = 1e-10

    def estimate(
        self,
        start_time: Union[datetime.datetime, List[datetime.datetime]] = None,
        end_time: Union[datetime.datetime, List[datetime.datetime]] = None,
        step_size: Union[float, List[float]] = None,
        parameters: Union[Dict[str, Dict], List[Tuple]] = None,
        measurements: List[core.System] = None,
        n_warmup: int = 60,
        method: Union[str, Tuple[str, str, str]] = "scipy",
        n_cores: Optional[int] = None,
        options: Optional[Dict] = None,
        schedule: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Dict,
    ) -> EstimationResult:
        """
        Perform parameter estimation using specified method and configuration.

        This method sets up and executes the parameter estimation process, supporting
        both automatic differentiation (AD) and finite difference (FD) optimization methods.

        Args:
            start_time: Start time(s) for estimation period(s). Can be a single date_time or list
                of date_times for multiple periods.

            end_time: End time(s) for estimation period(s). Can be a single date_time or list
                of date_times for multiple periods. Must be later than corresponding start_time.

            step_size: Step size(s) for simulation in seconds. Can be a single value or list
                of values for multiple periods.

            parameters: Parameter specifications. Either the string ``"auto"`` or
                a list/dict as described below.

                **Auto-discovery**: Passing ``parameters="auto"`` walks every
                component on the model and collects parameter tuples from those
                that implement ``get_estimable_parameters()``.

                **New format (recommended)**: List of tuples
                ``(component, attr, x0, lb, ub[, parameter_type])`` where
                ``component`` is a component object or list of component
                objects, ``attr`` is the parameter attribute name, ``x0`` is
                the initial value, ``lb``/``ub`` are the bounds, and the
                optional ``parameter_type`` is ``"private"`` (each listed
                component gets its own independent parameter; default) or
                ``"shared"`` (all listed components share one parameter
                value).

                Example::

                    # Private parameters (default)
                    parameters = [
                        (space, "thermal.C_air", 2e+6, 1e+6, 1e+7),  # implicit "private"
                        (space, "thermal.C_wall", 2e+6, 1e+6, 1e+7, "private"),  # explicit
                        ([controller1, controller2], "kp", 0.001, 1e-5, 1, "private"),
                    ]

                    # Shared parameters
                    parameters = [
                        ([space1, space2], "thermal.C_air", 2e+6, 1e+6, 1e+7, "shared"),
                        ([controller1, controller2], "kp", 0.001, 1e-5, 1, "shared"),
                    ]

                **Legacy format (deprecated)**: Dictionary with ``"private"``
                and ``"shared"`` keys, where each parameter entry contains
                ``"components"``, ``"x0"``, ``"lb"``, and ``"ub"`` lists (or
                single values).

            measurements: Measurement specification. Either the string ``"auto"``
                or a list of ``(sensor, sd)`` tuples, where ``sensor`` is a
                measuring device whose ``input["measuredValue"]`` holds historical
                data and ``sd`` is the measurement standard deviation used to
                weight that sensor's residuals (:math:`\\sigma_j` in the likelihood).

                Passing ``measurements="auto"`` includes every sensor that is
                driven by a non-sensor upstream component and has a wired data
                source, with ``sd = max(0.1 * data_std, 0.05)``. Build the list
                manually for custom weighting or sensor selection.

            n_warmup: Number of simulation steps used to initialize the model. These are not included in the likelihood calculation.

            method: Estimation method specification. Either a legacy string
                (``"scipy"`` or any supported optimizer name such as
                ``"SLSQP"``, ``"L-BFGS-B"``, ``"TNC"``, ``"trust-constr"``,
                ``"trf"``, ``"dogbox"``) or, recommended, a tuple
                ``(library, optimizer, mode)`` or
                ``(library, optimizer, mode, transcription)`` where
                ``library`` is ``"scipy"`` or ``"casadi"``, ``optimizer`` is
                the algorithm name, ``mode`` is ``"ad"`` (automatic
                differentiation) or ``"fd"`` (finite difference), and the
                optional ``transcription`` is ``"single_shooting"`` (default)
                or ``"collocation"`` (requires the CasADi/IPOPT backend).

                Supported optimizers by backend and mode:

                - SciPy backend, AD mode, local optimizers: ``"SLSQP"``
                  (preferred for most problems), ``"L-BFGS-B"``, ``"TNC"``,
                  ``"trust-constr"``, and the least-squares solvers ``"trf"``
                  and ``"dogbox"``.
                - SciPy backend, AD mode, global optimizers: ``"dual_annealing"``
                  (generalized simulated annealing; explores broadly at high
                  temperature to find the right basin, then anneals and
                  polishes with a local gradient-based minimizer; options
                  ``initial_temp``, ``restart_temp_ratio``, ``visit``,
                  ``accept``, ``maxiter``, ``maxfun``, ``no_local_search``,
                  ``local_search_method``) and ``"basinhopping"`` (random
                  perturbation plus local minimization with Metropolis
                  acceptance; options ``niter``, ``T``, ``stepsize``,
                  ``local_search_method``). Good for non-convex landscapes
                  with many local minima.
                - SciPy backend, FD mode (all require ``n_cores``): ``"trf"``,
                  ``"dogbox"``, ``"SLSQP"``, ``"L-BFGS-B"``, ``"TNC"``,
                  ``"trust-constr"`` -- same algorithms with the Jacobian
                  computed by parallel finite differences.
                - CasADi backend:
                  ``("casadi", "ipopt", "ad")`` is an IPOPT interior-point
                  solve of the same single-shooting objective as the SciPy
                  backends -- only the optimizer changes.
                  ``("casadi", "ipopt", "ad", "collocation")`` is the
                  simultaneous (collocation) transcription -- every
                  timestep-boundary state becomes a decision variable tied by
                  sparse hard continuity constraints. Robust to poor initial
                  parameter guesses. See the class docstring's Collocation
                  section for the formulation.

                Mode selection: use ``"ad"`` when all components are
                ``torch.nn.Module`` (preferred, faster); use ``"fd"`` for
                non-PyTorch or mixed models (requires ``n_cores``).

                Examples: ``("scipy", "SLSQP", "ad")`` for most PyTorch
                models; ``("scipy", "dual_annealing", "ad")`` for non-convex
                problems; ``("casadi", "ipopt", "ad", "collocation")`` for
                long horizons; ``("scipy", "trf", "fd")`` for non-PyTorch
                least-squares problems; ``"scipy"`` defaults to
                ``("scipy", "SLSQP", "ad")``.

            n_cores: Number of CPU cores to use for parallel computation. Required when using
                finite difference (FD) mode for Jacobian computation. Not used in automatic
                differentiation (AD) mode.

                - For FD mode: Must be specified (typically 2-8 cores depending on system)
                - For AD mode: Ignored (not needed for automatic differentiation)
                - Default: None (will raise error if FD mode is used without specifying)

            options: Additional options for the chosen optimization method.

                Common keys (all backends):

                - "maxiter": Maximum iterations
                - "ftol": Function tolerance (SciPy: solver default applies
                  when omitted; CasADi: mapped to IPOPT's ``tol``)
                - "verbose": Verbosity level

                Fast single-shooting (SciPy/CasADi backends, torch models only):

                - "fast" (bool, default False): Replace the object-graph
                  objective with the composed one-step-map rollout described
                  in the class docstring's Single-Shooting section. Values
                  and gradients are identical by construction; the estimator
                  silently falls back to the object-graph objective when the
                  model is not composable (components without ``forward``,
                  ``n_c > 1`` states or multi-branch parameters, or a
                  measurement the composed map cannot produce). Shared
                  parameters are supported.
                - "fast_validate" (bool, default False): Additionally
                  cross-check the fast objective against the object-graph
                  objective on the initial iterate (debugging aid).

                Collocation transcription only:

                - "gauss_newton" (bool, default True): Supply IPOPT with a
                  Gauss-Newton Hessian of the least-squares objective instead
                  of the default limited-memory BFGS approximation. Turns a
                  >1000-iteration L-BFGS crawl into a Newton-type solve.
                - "early_stopping" (bool or dict, default: enabled when
                  ``gauss_newton`` is on): Patience-based stagnation stop
                  with a best-feasible-iterate checkpoint. A dict overrides
                  the defaults: ``patience`` (10), ``feas_tol`` (1e-2),
                  ``min_delta_rel`` (1e-3), ``theta_tol`` (1e-4).
                - "pin_initial_state" (bool, default False): Fix each
                  period's initial boundary state at its warm-start value so
                  the feasible set is exactly the single-shooting trajectory
                  manifold (mainly for equivalence testing).

            schedule: Multi-phase continuation schedule -- the single,
                self-contained way to drive parameter estimation.

                A list of dicts, one per phase.  Each phase warm-starts
                from the converged solution of the previous phase.
                When ``None`` (default), a single phase with all
                defaults is run -- equivalent to ``schedule=[{}]``.

                Recognised keys per phase (all optional):

                - ``regularization_lambda`` (float, default ``0.0``):
                  Weight for the binarization penalty
                  ``P(x) = x(1 - x)`` summed over the phase's
                  ``regularization_components``.  Set ``0.0`` to
                  disable.
                - ``regularization_components`` (list of
                  :class:`core.System`, default ``None``): Components
                  whose ``compute_binarization_penalty()`` is summed.
                  When ``None`` and ``regularization_lambda > 0``,
                  components with that method are auto-detected from
                  the parameter components.
                - ``saturation_mode`` (``"smooth"`` | ``"hard"``,
                  default = current process-global mode): Scopes
                  :func:`twin4build.systems.utils.smooth_saturation.saturation_mode`
                  around the phase's solver call.  Use ``"smooth"``
                  for cold-start exploration (gradient flows through
                  deep windup) and ``"hard"`` for bias-correction
                  refinement (forward exact at bounds).
                - ``options`` (dict): Per-phase solver-option overrides
                  merged on top of the top-level ``options`` dict
                  (top-level ``options`` is the base; per-phase keys
                  win on conflict).

                Example -- "explore then refine" workflow::

                    schedule = [
                        {"saturation_mode": "smooth", "regularization_lambda": 0.0},
                        {"saturation_mode": "hard",   "regularization_lambda": 0.1,
                         "options": {"ftol": 1e-9}},
                    ]

                Example -- binarization annealing with final hard
                refinement::

                    schedule = [
                        {"regularization_lambda": 0.0,   "saturation_mode": "smooth"},
                        {"regularization_lambda": 0.01,  "saturation_mode": "smooth"},
                        {"regularization_lambda": 0.1,   "saturation_mode": "smooth"},
                        {"regularization_lambda": 0.1,   "saturation_mode": "hard",
                         "options": {"ftol": 1e-9}},
                    ]

        Returns:
            EstimationResult: Dict-like object containing the optimized parameters
                (``result_x``), component information, bounds, iteration metadata,
                and convergence status. Additional fields:

                - ``estimated_initial_state``: Per-component initial states
                  recovered from the fit (collocation also estimates the
                  boundary states; single-shooting reports the warm-up result).
                - ``transcription_audit`` (collocation only): Solution-quality
                  audit with the maximum continuity defect, per-sensor RMSE
                  consistency (NLP solution vs. forward rollout vs. object-graph
                  ``do_step`` rollout), and active-bound counts.

        Raises:
            AssertionError: If method specification is invalid or input parameters are inconsistent.
            ValueError: If method format is incorrect or unsupported.
            FMICallException: If simulation fails during parameter evaluation.

        Notes:
            - The method automatically handles parameter normalization and bounds checking.
            - For AD mode, all components must be torch.nn.Module instances.
            - For FD mode, n_cores must be specified for parallel Jacobian computation.
            - Results are automatically saved to disk in the model's estimation_results
              directory and can be reloaded with
              :meth:`~twin4build.model.simulation_model.simulation_model.SimulationModel.load_estimation_result`.
            - Multiple time periods are supported by providing lists for start_time, end_time, and step_size.

        Examples:
            >>> # New list format (recommended)
            >>> parameters = [
            ...     (space, "thermal.C_air", 2e+6, 1e+6, 1e+7),  # private (default)
            ...     ([space1, space2], "thermal.C_wall", 2e+6, 1e+6, 1e+7, "shared"),  # shared
            ...     (heating_controller, "kp", 0.001, 1e-5, 1, "private"),  # explicit private
            ... ]
            >>> result = estimator.estimate(
            ...     parameters=parameters,
            ...     measurements=[(temperature_sensor, 0.1)],
            ...     start_time=start,
            ...     end_time=end,
            ...     step_size=3600,
            ...     method=("scipy", "SLSQP", "ad")
            ... )

            >>> # Collocation transcription for long horizons
            >>> result = estimator.estimate(
            ...     parameters=parameters,
            ...     measurements=[(temperature_sensor, 0.1)],
            ...     start_time=start,
            ...     end_time=end,
            ...     step_size=3600,
            ...     method=("casadi", "ipopt", "ad", "collocation"),
            ...     options={"maxiter": 500}
            ... )
        """
        deprecated_args = ["startTime", "endTime", "stepSize", "n_initialization_steps"]
        new_args = ["start_time", "end_time", "step_size", "n_warmup"]
        position = [1, 2, 3, None]
        value_map = deprecate_args(deprecated_args, new_args, position, kwargs)
        start_time = value_map.get("start_time", start_time)
        end_time = value_map.get("end_time", end_time)
        step_size = value_map.get("step_size", step_size)
        n_warmup = value_map.get("n_warmup", n_warmup)

        # Reject the removed top-level estimation-config kwargs with a
        # clear migration message.  These were all promoted into
        # per-phase entries of ``schedule`` so config lives in exactly
        # one place.
        for legacy_key in ("regularization_lambda", "regularization_components"):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` is no longer a top-level argument of "
                    f"Estimator.estimate().  Move it into a per-phase entry "
                    f"of `schedule=[...]`, e.g. "
                    f"`schedule=[{{'{legacy_key}': ...}}]`.  See the "
                    f"`schedule` argument's docstring for the full set of "
                    f"recognised per-phase keys."
                )
        if "lambda_schedule" in kwargs:
            raise TypeError(
                "`lambda_schedule` has been removed.  It was a special case "
                "of the unified `schedule` argument.  Migrate "
                "`lambda_schedule=[(lam, opts), ...]` to "
                "`schedule=[{'regularization_lambda': lam, 'options': opts}, ...]`. "
                "Per-phase entries also accept 'saturation_mode' and "
                "'regularization_components'; see the `schedule` argument's "
                "docstring."
            )

        # Input validation and preprocessing
        if parameters is None:
            parameters = []

        # -- Pre-expand multi-branch parameters -------------------------------
        # Multi-branch sub-components (AHU supply / exhaust dampers,
        # heat-recovery effectiveness vector, fan polynomial
        # coefficients, ...) only get ``Parameter.expand_to_n_c(
        # n_branches)`` called from inside *their owner's*
        # ``initialize``.  If we let that happen later, inside the
        # first ``simulator.simulate(...)`` of the optimization loop,
        # then ``_process_parameters_list`` below has already read the
        # pre-expand ``n_c=1`` and allocated a single shared theta
        # slot per parameter -- ``Parameter.set``'s
        # ``_broadcast_for_n_c`` then fans the optimizer's lone update
        # out to every branch, locking all branches at the same value
        # regardless of how the per-room flow rates / effectiveness
        # coefficients actually differ.  Running ``model.initialize``
        # once here forces every component through its ``expand_to_n_c``
        # path so ``_process_parameters_list`` sees the post-expand
        # ``n_c`` and the solver gets one theta slot per branch.  A
        # bare ``initialize`` (no simulate) is cheap;
        # ``time_series_input`` caches by ``(start, end, step_size)``,
        # so the per-device init below + the simulate loop's own
        # ``model.initialize`` skip the historian fetch.  We normalize
        # ``start_time`` / ``end_time`` / ``step_size`` to lists up
        # front so this and the per-device init both see the same
        # batched shape -- the later normalization block at "Set up
        # time periods" then becomes idempotent.
        if not isinstance(start_time, list):
            start_time = [start_time]
        if not isinstance(end_time, list):
            end_time = [end_time]
        if not isinstance(step_size, list):
            step_size = [step_size] * len(start_time)

        # Guard against degenerate / inverted periods up front -- otherwise the
        # eager ``model.initialize`` below crashes inside ``ScheduleSystem``
        # with an opaque ``IndexError`` ("index -7 is out of bounds for axis 1
        # with size 0") before the per-period validation block can produce a
        # readable error.
        for s, e, ss in zip(start_time, end_time, step_size):
            if not isinstance(ss, int) or ss <= 0:
                raise ValueError(
                    f"step_size must be a positive integer, got {ss!r}"
                )
            if s >= e:
                raise ValueError(
                    f"start_time ({s}) must be strictly less than end_time ({e})"
                )

        self.simulator.model.initialize(start_time, end_time, step_size)

        # ``parameters="auto"`` / ``measurements="auto"`` sentinels.
        # Both walk every component on the underlying simulation model
        # once and assemble the standard estimation set from anything
        # that satisfies the per-side contract:
        #
        #   * ``parameters="auto"`` -> call ``c.get_estimable_parameters()``
        #     on every component that implements it.  The contract returns
        #     a list of ``(comp, attr, x0, lb, ub)`` tuples (see
        #     :meth:`ControllerIdentificationTorchSystem.get_estimable_parameters`
        #     for the canonical implementation).
        #
        #   * ``measurements="auto"`` -> walk every :class:`SensorSystem`
        #     with a wired data source and include it as a measurement
        #     with sd = ``max(0.1 * data_std, 1e-3)``.  If users want
        #     anything else (skip sensors, custom sd, instrument-specific
        #     weighting) they should build the list manually -- ``"auto"``
        #     is deliberately opinionated.
        # Reset the auto-discovery set so a previous ``estimate`` call's
        # state can't bleed into an explicit measurement list on a
        # subsequent call (which would otherwise refresh and overwrite
        # the caller's hand-picked ``sd``).
        self._auto_measurement_ids = set()
        if isinstance(parameters, str) and parameters == "auto":
            parameters = self._auto_parameters()
        if isinstance(measurements, str) and measurements == "auto":
            measurements = self._auto_measurements()

        # Convert old dict format to new list format if needed
        if isinstance(parameters, dict):
            # Issue deprecation warning for dict format
            warnings.warn(
                "The dictionary format for the 'parameters' argument is deprecated and will be "
                "removed in a future version. Please use the new list format: "
                "parameters = [(component, attr, x0, lb, ub), ...]. "
                "See the documentation for examples of the new format.",
                DeprecationWarning,
                stacklevel=2,
            )
            parameters = self._convert_dict_to_list_format(parameters)
        elif isinstance(parameters, list):
            # Validate the new list format
            parameters = self._validate_list_format(parameters)
        else:
            raise ValueError(
                "The 'parameters' argument must be either a list of tuples "
                "[(component, attr, x0, lb, ub), ...] or a dictionary (deprecated format)."
            )

        # Process parameters in new list format
        self._process_parameters_list(parameters)

        # Optional per-iteration parameter dump.  Enabled via
        # ``estimate(..., log_parameters=True)``; when on, every new
        # objective evaluation logs the full denormalized theta vector as
        # ``compID.attr=value`` pairs so the caller can see *which* parameter
        # the solver is moving (or not moving) when convergence stalls.
        self._log_parameters = bool(kwargs.pop("log_parameters", False))

        LOGGER.task("Estimating parameters")
        LOGGER.add_level()

        # Define allowed optimization methods
        allowed_methods = [
            ("scipy", "trf", "fd"),
            ("scipy", "dogbox", "fd"),
            ("scipy", "trf", "ad"),
            ("scipy", "dogbox", "ad"),
            ("scipy", "L-BFGS-B", "fd"),
            ("scipy", "L-BFGS-B", "ad"),
            ("scipy", "TNC", "fd"),
            ("scipy", "TNC", "ad"),
            ("scipy", "SLSQP", "fd"),
            ("scipy", "SLSQP", "ad"),
            ("scipy", "trust-constr", "fd"),
            ("scipy", "trust-constr", "ad"),
            ("scipy", "dual_annealing", "ad"),
            ("scipy", "basinhopping", "ad"),
            # IPOPT via CasADi (optional dependency).  Single-shooting: same
            # objective as the SciPy backends, only the optimizer changes.
            ("casadi", "ipopt", "ad"),
        ]
        default_none_method = ("scipy", "SLSQP", "ad")
        default_methods = [("scipy", "SLSQP", "ad")]
        default_mode = (
            "ad"  # Always choose automatic differentiation mode when ambiguous
        )

        # Transcription mode (4th, optional tuple element).  Governs *how* the
        # dynamics enter the NLP, orthogonally to the (library, optimizer, mode)
        # optimizer choice:
        #   - "single_shooting" (default): parameters -> full forward simulation
        #     -> residuals.  The classic sequential approach.
        #   - "collocation": every timestep-boundary state becomes a decision
        #     variable tied by hard continuity constraints (full simultaneous
        #     transcription; requires the CasADi/IPOPT backend -- see
        #     twin4build.estimator._transcription).
        # A soft-penalty "multiple_shooting" mode was removed: it wandered
        # without converging on exactly the problems collocation handles.
        allowed_transcriptions = (
            "single_shooting",
            "collocation",
        )
        self._transcription = "single_shooting"
        if isinstance(method, tuple) and len(method) == 4:
            transcription = method[3]
            assert transcription in allowed_transcriptions, (
                "The 4th (transcription) element of the method tuple must be one "
                f"of {allowed_transcriptions} - \"{transcription}\" was provided."
            )
            self._transcription = transcription
            method = tuple(method[:3])

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

        LOGGER.config("Method: %s", method)

        # Set up time periods
        self._n_warmup = n_warmup
        if not isinstance(start_time, list):
            start_time = [start_time]
        if not isinstance(end_time, list):
            end_time = [end_time]
        if not isinstance(step_size, list):
            step_size = [step_size] * len(start_time)

        # Initialise regularization state to defaults.  The schedule
        # loop overwrites both per-phase from the entry dict, so these
        # values are only ever observed for the trivial 1-phase
        # ``schedule=[{}]`` default.
        self._regularization_lambda = 0.0
        self._regularization_components = None

        # Validate time periods
        for startTime_, endTime_, stepSize_ in zip(start_time, end_time, step_size):
            assert (
                endTime_ > startTime_
            ), "The end_time must be later than the start_time."

        self._start_time = start_time
        self._end_time = end_time
        self._stepSize = step_size

        n_periods = len(start_time)
        LOGGER.config("Time periods: %d", n_periods)
        LOGGER.add_level()
        for i, (s, e, ss) in enumerate(zip(start_time, end_time, step_size)):
            LOGGER.config(
                "Period %d: start=%s | end=%s | step=%ss", i + 1, s, e, ss
            )
        LOGGER.remove_level()

        # Store configuration
        self._parameters_list = parameters  # Store the list format
        self._measurements = measurements
        self._mse_scaled = None
        self._n_timesteps = 0

        LOGGER.task("Initializing measurement devices")
        self.actual_readings = {}
        for measuring_device, sd in self._measurements:
            measuring_device.initialize(start_time, end_time, step_size)
            df = measuring_device.get_physical_readings(start_time, end_time, step_size)
            self.actual_readings[measuring_device.id] = df  # list of

        # -- Refresh ``sd`` for auto-discovered measurements -------------------
        # ``_auto_measurements`` runs *before* the devices are
        # initialized, so its only data source is whatever
        # ``time_series_input.values`` happens to be cached on each
        # sensor at that point.  In practice almost everything falls
        # through to the ``AUTO_SD_FLOOR`` placeholder, which gives the
        # ``1/sd**2`` loss weight wildly different magnitudes per
        # sensor (e.g. AHU supply-air-temp ends up with the floor while
        # zone-air-temp gets data-driven 0.16 K -> SAT carries 10x the
        # weight even though both are temperatures in the same units).
        # Now that ``actual_readings`` are loaded for *all* measurements
        # we can compute ``0.1 * data_std`` directly and replace the
        # placeholder, capped at ``AUTO_SD_FLOOR`` so a near-constant
        # window still produces a sane weight.  We only touch the
        # measurements ``_auto_measurements`` returned -- hand-built
        # measurement lists keep whatever ``sd`` the caller chose.
        auto_ids = getattr(self, "_auto_measurement_ids", set())
        if auto_ids:
            refreshed: List[Tuple[core.System, float]] = []
            for md, sd in self._measurements:
                if md.id in auto_ids:
                    try:
                        dfs = self.actual_readings.get(md.id, [])
                        vals = np.concatenate(
                            [np.asarray(d.values).flatten() for d in dfs]
                        )
                        vals = vals[np.isfinite(vals)]
                        if vals.size > 1:
                            data_std = float(np.nanstd(vals))
                            if np.isfinite(data_std):
                                sd = max(0.1 * data_std, AUTO_SD_FLOOR)
                    except Exception:  # noqa: BLE001
                        sd = max(sd, AUTO_SD_FLOOR)
                refreshed.append((md, sd))
            self._measurements = refreshed

        measuring_devices = [
            measuring_device for measuring_device, sd in self._measurements
        ]
        self.simulator.model.set_save_simulation_result(flag=False)
        self.simulator.model.set_save_simulation_result(flag=True, c=measuring_devices)

        for df_ in df:
            self._n_timesteps += len(df_.index)

        LOGGER.config("Measurements: %d devices, %d total timesteps", len(self._measurements), self._n_timesteps)

        # -- Per-measurement summary ------------------------------------------
        # Cheap diagnostic that tells "no data" vs "saturated" vs "well-excited"
        # apart at a glance, before the solver hides these in RMSE aggregates.
        # For each sensor log: min, max, mean, std, and fraction of samples
        # near the observed min/max (saturation detection).  A measurement
        # that is (say) 98% at 0 carries almost no information about PID
        # gains / integral time, and its parameters will look "stuck" for
        # reasons that have nothing to do with the optimizer.
        LOGGER.task("Measurement statistics")
        LOGGER.add_level()
        for md, sd in self._measurements:
            dfs = self.actual_readings.get(md.id, [])
            vals_list = []
            for df_ in dfs:
                try:
                    vals_list.append(np.asarray(df_.values).flatten())
                except AttributeError:
                    vals_list.append(np.asarray(df_).flatten())
            if not vals_list:
                LOGGER.iter("%s  sd=%.4g  <no data>", md.id, sd)
                continue
            v = np.concatenate(vals_list)
            v = v[np.isfinite(v)]
            if v.size == 0:
                LOGGER.iter("%s  sd=%.4g  <all NaN>", md.id, sd)
                continue
            vmin, vmax = float(np.min(v)), float(np.max(v))
            vmean, vstd = float(np.mean(v)), float(np.std(v))
            vrange = vmax - vmin
            if vrange > 0:
                tol = 0.01 * vrange
                frac_lo = 100.0 * float(np.mean(v <= vmin + tol))
                frac_hi = 100.0 * float(np.mean(v >= vmax - tol))
                sat_str = f" near_min={frac_lo:.0f}% near_max={frac_hi:.0f}%"
            else:
                sat_str = "  CONSTANT"
            LOGGER.iter(
                "%s  sd=%.4g  min=%.3g max=%.3g mean=%.3g std=%.3g%s",
                md.id, sd, vmin, vmax, vmean, vstd, sat_str,
            )
        LOGGER.remove_level()

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
        if method[0] not in ("scipy", "casadi"):
            raise ValueError(f"Unsupported library: {method[0]}")

        if options is None:
            options = {}

        # Fast-path for notebook example tests: keep every cell exercising
        # the full Estimator API (so we still catch wiring / API
        # regressions) but stop the solver after a single iteration.
        # Honors the env var set by ``utils.test_notebook.test_notebook``;
        # the regular ``test_estimator.py`` suite already passes
        # ``maxiter=2`` explicitly so this is a no-op for them.
        #
        # The cap is applied here (top-level ``options``) *and* again
        # inside :meth:`_run_schedule` after the per-phase merge -- so a
        # notebook that defines a multi-phase schedule with explicit
        # per-phase ``maxiter`` values (e.g. ``[{"options": {"maxiter":
        # 100}}, {"options": {"maxiter": 50}}]``) still collapses to
        # one iteration per phase in tests instead of overriding the
        # top-level cap during the schedule merge.
        if os.environ.get("TWIN4BUILD_TESTING", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            options = {**options, "maxiter": 1}

        # --- Schedule dispatch ---
        # ``schedule=None`` collapses to a single phase with all
        # defaults so the trivial single-phase API stays terse:
        # ``estimator.estimate(parameters=..., ...)``.
        if schedule is None:
            schedule = [{}]

        LOGGER.task("Running schedule with %d phase(s)", len(schedule))
        LOGGER.add_level()
        result = self._run_schedule(
            schedule=schedule,
            method=method,
            n_cores=n_cores,
            base_options=options,
        )
        LOGGER.remove_level()
        LOGGER.ok(
            "Running schedule with %d phase(s)",
            len(schedule),
            change_status=True,
            ignore_no_match=True,
        )
        LOGGER.remove_level()
        LOGGER.ok(
            "Estimating parameters",
            change_status=True,
            ignore_no_match=True,
        )
        return result

    # Recognised keys for entries in the unified ``schedule`` argument.
    # Centralised here so a typo in a phase dict raises early instead of
    # being silently ignored.
    _SCHEDULE_PHASE_KEYS = frozenset(
        {
            "regularization_lambda",
            "saturation_mode",
            "options",
            "regularization_components",
        }
    )

    def _run_schedule(
        self,
        schedule: List[Dict[str, Any]],
        method: tuple,
        n_cores: Optional[int],
        base_options: Dict,
    ) -> "EstimationResult":
        """Run a multi-phase optimisation schedule.

        Each phase is a dict.  Recognised keys (all optional):

        - ``regularization_lambda`` (float, default ``0.0``)
        - ``regularization_components`` (list, default ``None``)
        - ``saturation_mode`` (``"smooth"`` | ``"hard"``, default =
          inherited process-global mode)
        - ``options`` (dict, merged onto ``base_options``)

        Phases warm-start from the previous phase's solution by
        copying ``self._last_x_norm`` into ``self._x0_norm``; the
        ``_mse_scaled`` cache is reset per phase so each rescales from
        its own initial MSE.

        Parameters
        ----------
        schedule : list of dicts
            Per-phase config.  Empty dicts run the phase with all
            defaults (no regularization, inherited saturation mode,
            ``base_options`` only).
        method : tuple
            Solver method tuple, forwarded to :meth:`_scipy_solver`.
        n_cores : int or None
            Cores for FD mode, forwarded to :meth:`_scipy_solver`.
        base_options : dict
            Top-level ``options`` dict; per-phase ``options`` are
            merged on top.

        Returns
        -------
        EstimationResult
            Result from the **final** phase.  The penultimate phase's
            result is also written to disk by ``_scipy_solver`` but is
            overwritten by the final phase.
        """
        n_phases = len(schedule)
        result = None

        for phase_idx, raw_entry in enumerate(schedule):
            entry = self._normalize_schedule_entry(raw_entry, phase_idx)

            lam = entry.get("regularization_lambda", 0.0)
            mode = entry.get("saturation_mode")  # None => keep current global
            phase_opts = entry.get("options") or {}
            phase_reg_comps = entry.get("regularization_components", None)

            merged_options = {**base_options, **phase_opts}

            # Re-apply the notebook-test fast-path cap after the
            # per-phase merge so an explicit per-phase ``maxiter`` in
            # the schedule cannot accidentally undo the env-var cap
            # (see :meth:`estimate` for the top-level pass).
            if os.environ.get("TWIN4BUILD_TESTING", "").lower() in (
                "1",
                "true",
                "yes",
            ):
                merged_options["maxiter"] = 1

            self._regularization_lambda = lam
            self._regularization_components = phase_reg_comps
            self._mse_scaled = None  # rescale per phase

            LOGGER.config(
                "Phase %d/%d | lambda=%s | saturation_mode=%s",
                phase_idx + 1,
                n_phases,
                lam,
                mode if mode is not None else "<inherit>",
            )

            # Scope the saturation-mode override to this phase only;
            # nullcontext() preserves the inherited global mode when no
            # override is requested.
            ctx = saturation_mode(mode) if mode is not None else _nullcontext()
            with ctx:
                result = self._scipy_solver(
                    method=method, n_cores=n_cores, **merged_options
                )

            # Warm-start next phase from this phase's converged x.
            self._x0_norm = self._last_x_norm.copy()

            obj_val = result.get("final_objective", None)
            LOGGER.ok(
                "Phase %d/%d | objective=%s",
                phase_idx + 1,
                n_phases,
                obj_val,
            )

        return result

    def _normalize_schedule_entry(
        self, entry: Any, phase_idx: int
    ) -> Dict[str, Any]:
        """Validate one schedule entry.  Unknown keys raise so typos
        surface early instead of silently being ignored.
        """
        if not isinstance(entry, dict):
            raise ValueError(
                f"Schedule entry at index {phase_idx} must be a dict; "
                f"got {type(entry).__name__}: {entry!r}"
            )

        unknown = set(entry) - self._SCHEDULE_PHASE_KEYS
        if unknown:
            raise ValueError(
                f"Schedule entry at index {phase_idx} contains unknown "
                f"keys {sorted(unknown)!r}; allowed: "
                f"{sorted(self._SCHEDULE_PHASE_KEYS)!r}"
            )

        mode = entry.get("saturation_mode")
        if mode is not None and mode not in ("smooth", "hard"):
            raise ValueError(
                f"Schedule entry at index {phase_idx}: saturation_mode "
                f"must be 'smooth' or 'hard', got {mode!r}"
            )
        return entry

    def _auto_parameters(self) -> List[Tuple]:
        """Build the estimation parameter list automatically.

        Implements the ``parameters="auto"`` sentinel of :meth:`estimate`.
        Walks every component on the underlying simulation model and
        collects parameter tuples from those that implement the
        ``get_estimable_parameters()`` contract -- the canonical example
        being :class:`ControllerIdentificationTorchSystem`, which returns
        ``(comp, attr, x0, lb, ub)`` tuples seeded from its post-rewire
        state.

        Returns:
            The concatenated parameter list, ready to be passed through
            :meth:`_validate_list_format`.  Empty when no component
            implements the contract (so the caller will hit the existing
            "no parameters" error path naturally).
        """
        model = self.simulator.model
        params: List[Tuple] = []
        for component in model.components.values():
            getter = getattr(component, "get_estimable_parameters", None)
            if not callable(getter):
                continue
            try:
                params.extend(getter())
            except Exception as ex:  # noqa: BLE001
                LOGGER.warning(
                    "[Estimator] component %r get_estimable_parameters() "
                    "raised: %s -- skipping.",
                    getattr(component, "id", component),
                    ex,
                )
        return params

    def _auto_measurements(
        self,
    ) -> List[Tuple[core.System, float]]:
        """Build the measurement list automatically.

        Implements the ``measurements="auto"`` sentinel of
        :meth:`estimate`.  A :class:`SensorSystem` is included iff:

        * its ``input["measuredValue"]`` receives a connection from a
          **non-sensor** upstream component (typically a controller
          or a physics system) -- so the simulation actually produces
          a fresh prediction on it, rather than the sensor just being
          a pass-through wrapper over another sensor, and
        * it has a wired data source (database / spreadsheet / df) so
          ground truth is available for the loss.

        The standard deviation is ``max(0.1 * data_std, AUTO_SD_FLOOR)``
        with ``AUTO_SD_FLOOR = 0.05`` -- the cross-domain noise floor we
        adopt for the canonical sensor families ``"auto"`` discovers:
        actuator commands on [0, 1] (5 % of full scale), zone air
        temperatures (0.05 K is below typical wall-mount thermistor
        precision so it cannot dominate the residual), and supply air
        flows (0.05 kg/s ~ a few %  of typical AHU branch design flow).
        Anything tighter risks the ``1/sd**2`` weighting making one
        channel dwarf the rest.  ``data_std`` is computed from
        ``time_series_input.values`` when already loaded; otherwise the
        floor is used as a *placeholder* and ``estimate`` refreshes
        every auto-discovered ``sd`` right after
        ``measuring_device.initialize`` has populated
        ``actual_readings``.

        The ``0.1 * data_std`` heuristic (rather than a tighter
        fraction) is deliberate: the estimator scales the loss
        gradient as ``1/sd**2``, so an over-tight ``sd`` makes SLSQP /
        L-BFGS line searches overshoot and bounce.  ``0.1 * data_std``
        matches the empirical sweet spot used by the legacy hand-tuned
        Mortar examples (``sd=0.02`` absolute on 0-1 actuator signals
        with ``std ~ 0.2``) and yields well-behaved descent on the
        building-controls test suite.

        Sensors excluded by this filter:

        * **Leaf** sensors (``connects_at == []``) -- they *are* the
          data sources for loop inputs (room temp, setpoint, ...);
          their ``input["measuredValue"]`` is never populated.
        * **Sensor-to-sensor pass-throughs** (e.g.
          BRICK ``hasRef`` wrappers around a leaf data sensor) -- the
          simulation just propagates the upstream sensor's value, so
          fitting model parameters to make them match the data is
          ill-posed.

        Any user need for finer-grained control (skip sensors, custom
        sd per instrument, alternate noise model) should drop ``"auto"``
        and build the list manually.

        Returns:
            A list of ``(sensor, sd)`` tuples.  Empty when no qualifying
            sensor is present, which will surface naturally via the
            existing "no measurements" assertion.
        """
        model = self.simulator.model
        out: List[Tuple[core.System, float]] = []
        for component in model.components.values():
            if not isinstance(component, systems.SensorSystem):
                continue
            # Pure leaf sensors have no measuredValue input history at
            # all -- skip them up front.
            if len(component.connects_at) == 0:
                continue
            # Look for at least one upstream component on the
            # ``measuredValue`` input that is *not* itself a sensor.
            # This is what distinguishes a CITS-output actuator (CITS
            # -> sensor.measuredValue, included) from a BRICK
            # ``hasRef`` pass-through (leaf-sensor ->
            # wrapper-sensor.measuredValue, excluded).
            has_non_sensor_driver = False
            for cp in component.connects_at:
                if cp.input_port != "measuredValue":
                    continue
                for conn in cp.connects_system_through:
                    upstream = conn.connects_system
                    if upstream is None:
                        continue
                    if not isinstance(upstream, systems.SensorSystem):
                        has_non_sensor_driver = True
                        break
                if has_non_sensor_driver:
                    break
            if not has_non_sensor_driver:
                continue
            # ``has_data_source`` covers spreadsheet/database/df without
            # depending on private attrs; ``time_series_input`` is set
            # by ``initialize`` and may not exist yet on the first
            # ``"auto"`` resolution.
            has_data = any(
                bool(getattr(component, flag, False))
                for flag in ("use_database", "use_spreadsheet", "use_df")
            )
            if not has_data:
                continue

            sd = AUTO_SD_FLOOR
            ts = getattr(component, "time_series_input", None)
            if ts is not None and hasattr(ts, "values"):
                try:
                    arr = np.asarray(ts.values).flatten()
                    if arr.size > 1:
                        std = float(np.nanstd(arr))
                        if np.isfinite(std):
                            sd = max(0.1 * std, AUTO_SD_FLOOR)
                except Exception:  # noqa: BLE001
                    sd = AUTO_SD_FLOOR
            out.append((component, sd))
        # Stash the auto-discovered IDs so ``estimate`` can recompute
        # their ``sd`` from the freshly-loaded ``actual_readings``
        # *without* clobbering any user-supplied ``sd`` values in
        # measurements lists that callers built by hand.
        self._auto_measurement_ids = {c.id for c, _ in out}
        return out

    def _validate_list_format(self, parameters_list: List[Tuple]) -> List[Tuple]:
        """
        Validate and clean the new list format parameters.

        Args:
            parameters_list: List of tuples in format:
                (component(s), attr, x0, lb, ub) or
                (component(s), attr, x0, lb, ub, parameter_type)

        Returns:
            Validated list of parameter tuples with explicit parameter_type

        Raises:
            ValueError: If tuple format is invalid
        """
        if not isinstance(parameters_list, list):
            raise ValueError("Parameters must be a list of tuples")

        validated_params = []

        for i, param_tuple in enumerate(parameters_list):
            if not isinstance(param_tuple, tuple):
                raise ValueError(
                    f"Each parameter must be a tuple. Got {type(param_tuple)} at index {i}"
                )

            # Handle both 5-element and 6-element tuples
            if len(param_tuple) == 5:
                component_s, attr, x0, lb, ub = param_tuple
                parameter_type = "private"  # default
            elif len(param_tuple) == 6:
                component_s, attr, x0, lb, ub, parameter_type = param_tuple
            else:
                raise ValueError(
                    f"Each parameter tuple must have either 5 or 6 elements: "
                    f"(component(s), attr, x0, lb, ub[, parameter_type]). "
                    f"Got {len(param_tuple)} elements at index {i}: {param_tuple}"
                )

            # Validate parameter_type
            if parameter_type not in ["private", "shared"]:
                raise ValueError(
                    f"Parameter type must be 'private' or 'shared'. "
                    f"Got '{parameter_type}' at index {i}"
                )

            # Ensure component_s is a list for consistent processing
            if not isinstance(component_s, list):
                components = [component_s]
            else:
                components = component_s
                if len(components) == 0:
                    raise ValueError(f"Component list cannot be empty at index {i}")

            for c in components:
                if not isinstance(c, core.System):
                    raise ValueError(
                        f"Component must be a System object at index {i}. Got: {type(c)}"
                    )

            # Validate attribute name
            if not isinstance(attr, str) or not attr:
                raise ValueError(
                    f"Attribute must be a non-empty string at index {i}. Got: {attr}"
                )

            # Validate numeric values
            if x0 is None:
                raise ValueError(f"Initial value (x0) cannot be None at index {i}")

            # Convert None bounds to infinity
            if lb is None:
                lb = -np.inf
            if ub is None:
                ub = np.inf

            # For shared parameters, validate that we have multiple components
            if parameter_type == "shared" and len(components) == 1:
                warnings.warn(
                    f"Parameter at index {i} is marked as 'shared' but only has one component. "
                    f"Consider using 'private' instead.",
                    UserWarning,
                    stacklevel=3,
                )

            validated_params.append((components, attr, x0, lb, ub, parameter_type))

        return validated_params

    def _convert_dict_to_list_format(
        self, parameters_dict: Dict[str, Dict]
    ) -> List[Tuple]:
        """
        Convert old dict format to new list format.

        Args:
            parameters_dict: Dictionary in legacy format with "private" and "shared" keys

        Returns:
            List of tuples in format (components, attr, x0, lb, ub, parameter_type)

        Raises:
            ValueError: If dict format is invalid
        """
        if not isinstance(parameters_dict, dict):
            raise ValueError("Parameters dict must be a dictionary")

        # Ensure required dictionary structure
        if "private" not in parameters_dict:
            parameters_dict["private"] = {}
        if "shared" not in parameters_dict:
            parameters_dict["shared"] = {}

        parameters_list = []

        # Process private parameters
        for attr, par_dict in parameters_dict["private"].items():
            # Ensure components is a list
            components = par_dict["components"]
            if not isinstance(components, list):
                components = [components]

            # Ensure x0, lb, ub are lists with correct length
            x0_list = par_dict["x0"]
            if not isinstance(x0_list, list):
                x0_list = [x0_list] * len(components)
            elif len(x0_list) != len(components):
                raise ValueError(
                    f'The number of elements in the "x0" list must be equal to the number '
                    f"of components in the private dictionary for attribute {attr}."
                )

            lb_list = par_dict["lb"]
            if not isinstance(lb_list, list):
                lb_list = [lb_list] * len(components)
            elif len(lb_list) != len(components):
                raise ValueError(
                    f'The number of elements in the "lb" list must be equal to the number '
                    f"of components in the private dictionary for attribute {attr}."
                )

            ub_list = par_dict["ub"]
            if not isinstance(ub_list, list):
                ub_list = [ub_list] * len(components)
            elif len(ub_list) != len(components):
                raise ValueError(
                    f'The number of elements in the "ub" list must be equal to the number '
                    f"of components in the private dictionary for attribute {attr}."
                )

            # Add each component as a separate private parameter
            for component, x0, lb, ub in zip(components, x0_list, lb_list, ub_list):
                parameters_list.append(([component], attr, x0, lb, ub, "private"))

        # Process shared parameters
        for attr, par_dict in parameters_dict["shared"].items():
            components_lists = par_dict["components"]
            if not isinstance(components_lists, list):
                raise ValueError(
                    f'The "components" key in the shared dictionary must be a list for attribute {attr}.'
                )

            # Ensure components is a list of lists
            if components_lists and not isinstance(components_lists[0], list):
                components_lists = [components_lists]

            x0_lists = par_dict["x0"]
            if not isinstance(x0_lists, list):
                x0_lists = [
                    [x0_lists for _ in components_list]
                    for components_list in components_lists
                ]
            elif x0_lists and not isinstance(x0_lists[0], list):
                x0_lists = [x0_lists]

            lb_lists = par_dict["lb"]
            if not isinstance(lb_lists, list):
                lb_lists = [
                    [lb_lists for _ in components_list]
                    for components_list in components_lists
                ]
            elif lb_lists and not isinstance(lb_lists[0], list):
                lb_lists = [lb_lists]

            ub_lists = par_dict["ub"]
            if not isinstance(ub_lists, list):
                ub_lists = [
                    [ub_lists for _ in components_list]
                    for components_list in components_lists
                ]
            elif ub_lists and not isinstance(ub_lists[0], list):
                ub_lists = [ub_lists]

            # Each group of components shares the same parameter values
            for components_list, x0_list, lb_list, ub_list in zip(
                components_lists, x0_lists, lb_lists, ub_lists
            ):
                # All components in this group get the same parameter values
                shared_x0 = x0_list[0] if isinstance(x0_list, list) else x0_list
                shared_lb = lb_list[0] if isinstance(lb_list, list) else lb_list
                shared_ub = ub_list[0] if isinstance(ub_list, list) else ub_list

                # Create one shared parameter entry for this group
                parameters_list.append(
                    (components_list, attr, shared_x0, shared_lb, shared_ub, "shared")
                )

        return parameters_list

    def _process_parameters_list(self, parameters_list: List[Tuple]) -> None:
        """
        Process the parameter list and extract component and parameter information.

        Args:
            parameters_list: List of tuples in format (components, attr, x0, lb, ub, parameter_type)
        """
        if not parameters_list:
            # Initialize empty lists for no parameters case
            self._flat_components = []
            self._parameter_names = []
            self._flat_parameters = []
            self._x0 = np.array([])
            self._lb = np.array([])
            self._ub = np.array([])
            self._theta_mask = np.array([], dtype=int)
            self._theta_slices = []  # (start, end) for each unique param in theta
            self._unique_param_n_c = []  # n_c for each unique param
            self._flat_components_private = []
            self._parameter_names_private = []
            self._flat_components_shared = []
            self._parameter_names_shared = []
            LOGGER.warning("No parameters to estimate.")
            return

        # Separate private and shared parameters
        private_params = []
        shared_params = []

        for components, attr, x0, lb, ub, parameter_type in parameters_list:
            if parameter_type == "private":
                # For private parameters, each component gets its own parameter
                for component in components:
                    private_params.append((component, attr, x0, lb, ub))
            elif parameter_type == "shared":
                # For shared parameters, all components share one parameter
                shared_params.append((components, attr, x0, lb, ub))

        LOGGER.config("Parameters: %d private, %d shared", len(private_params), len(shared_params))
        LOGGER.add_level()
        for comp, attr, x0, lb, ub in private_params:
            LOGGER.debug("Private: %s.%s (x0=%s, lb=%s, ub=%s)", comp.id, attr, x0, lb, ub)
        for comps, attr, x0, lb, ub in shared_params:
            comp_ids = [c.id for c in comps]
            LOGGER.debug("Shared: %s.%s (x0=%s, lb=%s, ub=%s)", comp_ids, attr, x0, lb, ub)
        LOGGER.remove_level()

        # Build flat lists for private parameters
        self._flat_components_private = [param[0] for param in private_params]
        self._parameter_names_private = [param[1] for param in private_params]

        # Build flat lists for shared parameters
        self._flat_components_shared = []
        self._parameter_names_shared = []

        for components, attr, x0, lb, ub in shared_params:
            for component in components:
                self._flat_components_shared.append(component)
                self._parameter_names_shared.append(attr)

        # Combine all components and parameters
        self._flat_components = (
            self._flat_components_private + self._flat_components_shared
        )
        self._parameter_names = (
            self._parameter_names_private + self._parameter_names_shared
        )

        # Get parameter objects
        self._flat_parameters = [
            rgetattr(component, attr)
            for component, attr in zip(self._flat_components, self._parameter_names)
        ]

        # Build theta with n_c support
        # theta is flat: [p0_v0, p0_v1, ..., p0_vn_c0, p1_v0, ..., p1_vn_c1, ...]
        # _theta_slices[i] = (start, end) for unique parameter i
        # _theta_mask[j] = which unique parameter index flat_parameter[j] maps to

        self._unique_param_n_c = []  # n_c for each unique parameter
        self._theta_slices = []  # (start, end) for each unique param in theta
        theta_offset = 0

        # Process private parameters (each is unique)
        private_x0_flat = []
        private_lb_flat = []
        private_ub_flat = []

        for i, (component, attr, x0, lb, ub) in enumerate(private_params):
            param = rgetattr(component, attr)
            n_c = param.n_c if hasattr(param, "n_c") else 1
            self._unique_param_n_c.append(n_c)
            self._theta_slices.append((theta_offset, theta_offset + n_c))
            theta_offset += n_c

            # Flatten x0, lb, ub for this parameter
            if isinstance(x0, (list, np.ndarray, torch.Tensor)):
                x0_vals = (
                    np.array(x0).flatten()
                    if not isinstance(x0, torch.Tensor)
                    else x0.detach().numpy().flatten()
                )
            else:
                x0_vals = np.full(n_c, x0)

            lb_val = lb if lb is not None else -np.inf
            ub_val = ub if ub is not None else np.inf
            if isinstance(lb_val, (list, np.ndarray, torch.Tensor)):
                lb_vals = (
                    np.array(lb_val).flatten()
                    if not isinstance(lb_val, torch.Tensor)
                    else lb_val.detach().numpy().flatten()
                )
            else:
                lb_vals = np.full(n_c, lb_val)
            if isinstance(ub_val, (list, np.ndarray, torch.Tensor)):
                ub_vals = (
                    np.array(ub_val).flatten()
                    if not isinstance(ub_val, torch.Tensor)
                    else ub_val.detach().numpy().flatten()
                )
            else:
                ub_vals = np.full(n_c, ub_val)

            private_x0_flat.extend(x0_vals)
            private_lb_flat.extend(lb_vals)
            private_ub_flat.extend(ub_vals)

        # Process shared parameters
        shared_x0_flat = []
        shared_lb_flat = []
        shared_ub_flat = []
        n_private_unique = len(private_params)

        for components, attr, x0, lb, ub in shared_params:
            # All members of a shared group MUST use the same normalization
            # scaling: the objective denormalizes each member with its own
            # parameter's scaling, so a mismatch would silently assign
            # DIFFERENT physical values (and gradients) to the "shared"
            # parameter for the same normalized theta.
            scalings = {
                getattr(rgetattr(c, attr), "scaling", "linear") for c in components
            }
            if len(scalings) > 1:
                raise ValueError(
                    f"Shared parameter group {[c.id for c in components]} attr "
                    f"'{attr}' mixes normalization scalings {sorted(scalings)}; "
                    "all members of a shared group must use the same "
                    "tps.Parameter scaling."
                )
            # Get n_c from first component (all shared components should have same n_c)
            param = rgetattr(components[0], attr)
            n_c = param.n_c if hasattr(param, "n_c") else 1
            self._unique_param_n_c.append(n_c)
            self._theta_slices.append((theta_offset, theta_offset + n_c))
            theta_offset += n_c

            # Flatten x0, lb, ub for this shared parameter
            if isinstance(x0, (list, np.ndarray, torch.Tensor)):
                x0_vals = (
                    np.array(x0).flatten()
                    if not isinstance(x0, torch.Tensor)
                    else x0.detach().numpy().flatten()
                )
            else:
                x0_vals = np.full(n_c, x0)

            lb_val = lb if lb is not None else -np.inf
            ub_val = ub if ub is not None else np.inf
            if isinstance(lb_val, (list, np.ndarray, torch.Tensor)):
                lb_vals = (
                    np.array(lb_val).flatten()
                    if not isinstance(lb_val, torch.Tensor)
                    else lb_val.detach().numpy().flatten()
                )
            else:
                lb_vals = np.full(n_c, lb_val)
            if isinstance(ub_val, (list, np.ndarray, torch.Tensor)):
                ub_vals = (
                    np.array(ub_val).flatten()
                    if not isinstance(ub_val, torch.Tensor)
                    else ub_val.detach().numpy().flatten()
                )
            else:
                ub_vals = np.full(n_c, ub_val)

            shared_x0_flat.extend(x0_vals)
            shared_lb_flat.extend(lb_vals)
            shared_ub_flat.extend(ub_vals)

        # Combine flattened values
        self._x0 = (
            np.array(private_x0_flat + shared_x0_flat)
            if (private_x0_flat or shared_x0_flat)
            else np.array([])
        )
        self._lb = (
            np.array(private_lb_flat + shared_lb_flat)
            if (private_lb_flat or shared_lb_flat)
            else np.array([])
        )
        self._ub = (
            np.array(private_ub_flat + shared_ub_flat)
            if (private_ub_flat or shared_ub_flat)
            else np.array([])
        )

        # Create theta_mask: maps flat_parameters index -> unique parameter index
        # Private parameters: one-to-one mapping (indices 0, 1, 2, ...)
        private_mask = list(range(len(self._flat_components_private)))

        # Shared parameters: components share unique parameter indices
        shared_mask = []
        unique_idx = n_private_unique
        for components, attr, x0, lb, ub in shared_params:
            for _ in components:
                shared_mask.append(unique_idx)
            unique_idx += 1

        self._theta_mask = np.array(private_mask + shared_mask, dtype=int)

    def _theta_to_param_values(self, theta: np.ndarray) -> List[np.ndarray]:
        """
        Convert flat theta array to list of parameter values.

        Args:
            theta: Flat array of all parameter values

        Returns:
            List of arrays, one per flat_parameter, with values from theta
            (shared parameters get the same values)
        """
        values = []
        for i, param_idx in enumerate(self._theta_mask):
            start, end = self._theta_slices[param_idx]
            values.append(theta[start:end])
        return values

    def _composer_theta_spec(self) -> Tuple[List[Tuple], List]:
        """Indexed theta spec + representative parameters for the composed map.

        Returns ``(theta_spec, unique_parameters)``:

        - ``theta_spec``: one ``(component, attr, theta_index)`` entry per flat
          parameter, with ``theta_index`` pointing into the *unique* theta
          vector (``_theta_slices[_theta_mask[j]]``).  Shared parameters route
          several entries to the same index, which is exactly how
          ``OneStepComposer`` composes them.
        - ``unique_parameters``: one representative ``tps.Parameter`` per theta
          entry (the first flat occurrence), for bounds/scaling extraction
          (all members of a shared group are configured with identical
          bounds).

        Raises:
            RuntimeError: If any parameter is multi-branch (``n_c > 1``) --
                the composed map only supports scalar theta entries.
        """
        if any(int(n) != 1 for n in self._unique_param_n_c):
            raise RuntimeError("multi-branch (n_c > 1) parameters")
        theta_spec = []
        rep: Dict[int, object] = {}
        for j, (comp, attr) in enumerate(
            zip(self._flat_components, self._parameter_names)
        ):
            idx = int(self._theta_slices[int(self._theta_mask[j])][0])
            owner, owner_attr = self._composed_owner(comp, attr)
            theta_spec.append((owner, owner_attr, idx))
            rep.setdefault(idx, self._flat_parameters[j])
        unique_parameters = [rep[i] for i in range(len(rep))]
        assert len(unique_parameters) == len(self._x0_norm)
        return theta_spec, unique_parameters

    def _composed_owner(self, comp, attr) -> Tuple[object, str]:
        """Remap a theta entry on a nested sub-object onto its owning model
        component with a prefixed attribute path.

        Users may put theta directly on an owned sub-object -- e.g. the
        ``OccupancySystem``'s internal ``supply_damper`` (shared with a model
        damper).  The composer routes parameters by *model component*, so such
        entries must become ``(owner, "supply_damper.a")``.  Components that
        are themselves in the model pass through unchanged (including
        composites addressed with dotted attrs like ``(office,
        "thermal.C_air")``).
        """
        components = self.simulator.model.components
        cid = getattr(comp, "id", None)
        if cid is not None and components.get(cid) is comp:
            return comp, attr
        for owner in components.values():
            # nn.Module owners keep sub-module attributes in ``_modules``
            # (e.g. OccupancySystem.supply_damper), not ``__dict__``.
            attrs = dict(vars(owner))
            attrs.update(getattr(owner, "_modules", None) or {})
            for name, val in attrs.items():
                if val is comp:
                    return owner, f"{name}.{attr}"
        return comp, attr

    def _param_values_to_theta(self, values: List[np.ndarray]) -> np.ndarray:
        """
        Convert list of parameter values to flat theta array.

        Only uses the first occurrence of each unique parameter (for shared params).

        Args:
            values: List of parameter value arrays

        Returns:
            Flat theta array
        """
        theta = np.zeros(sum(self._unique_param_n_c))
        seen_unique = set()
        for i, (value, param_idx) in enumerate(zip(values, self._theta_mask)):
            if param_idx not in seen_unique:
                start, end = self._theta_slices[param_idx]
                theta[start:end] = value
                seen_unique.add(param_idx)
        return theta

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
            LOGGER.warning("Objective function evaluation failed: %s.", e)
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
        # Get per-parameter bounds from flat theta arrays
        lb_values = self._theta_to_param_values(self._lb)
        ub_values = self._theta_to_param_values(self._ub)
        x0_values = self._theta_to_param_values(self._x0)

        # Enable gradients for parameters to be estimated
        for i, (component, attr) in enumerate(
            zip(self._flat_components, self._parameter_names)
        ):
            assert isinstance(
                component, nn.Module
            ), "All components must be subclasses of nn.Module when using PyTorch-based optimization"
            param = rgetattr(component, attr)
            assert isinstance(
                param, (tps.Parameter)
            ), "All parameters must be subclasses of tps.Parameter when using PyTorch-based optimization"
            param.requires_grad_(True)

            if normalize == False:
                lb = 0  # Do nothing
                ub = 1  # Do nothing
            else:
                lb = lb_values[i]
                ub = ub_values[i]

            param.min_value = lb
            param.max_value = ub

        # Normalize each parameter's values
        lb_norm_list = []
        ub_norm_list = []
        x0_norm_list = []

        seen_unique = set()
        for i, (param, param_idx) in enumerate(
            zip(self._flat_parameters, self._theta_mask)
        ):
            if param_idx not in seen_unique:
                # Normalize the values for this unique parameter
                lb_norm = param.normalize(
                    torch.tensor(lb_values[i], dtype=torch.float64)
                )
                ub_norm = param.normalize(
                    torch.tensor(ub_values[i], dtype=torch.float64)
                )
                x0_norm = param.normalize(
                    torch.tensor(x0_values[i], dtype=torch.float64)
                )

                # Convert to numpy and flatten
                lb_norm_list.extend(lb_norm.detach().numpy().flatten())
                ub_norm_list.extend(ub_norm.detach().numpy().flatten())
                x0_norm_list.extend(x0_norm.detach().numpy().flatten())
                seen_unique.add(param_idx)

        self._lb_norm = np.array(lb_norm_list)
        self._ub_norm = np.array(ub_norm_list)
        self._x0_norm = np.array(x0_norm_list)

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
        self._eval_count = 0
        self._solver_start_time = time_module.time()
        # Track best objective seen for this solver run so the per-sensor
        # RMSE diagnostic only emits on strictly-improving evaluations.
        self._best_obj_logged = float("inf")

        LOGGER.task("Running %s solver: %s (%s mode)", method[0], method[1], method[2])
        LOGGER.add_level()

        datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        method_str = "_".join(list(method))
        filename = str("{}{}".format(datestr, f"_{method_str}.pickle"))
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

        assert len(self._flat_parameters) > 0, "No parameters to optimize"

        LOGGER.config("Parameters to optimize: %d (theta size: %d)", len(self._flat_parameters), len(self._x0_norm))

        # Initialize simulator
        LOGGER.task("Initializing model")
        self.simulator.model.initialize(
            start_time=self._start_time,
            end_time=self._end_time,
            step_size=self._stepSize,
        )

        # Set initial parameters - convert flat theta to per-parameter values
        x0_param_values = self._theta_to_param_values(self._x0_norm)
        self.simulator.model.set_parameters(
            x0_param_values,
            self._flat_components,
            self._parameter_names,
            normalized=True,
            overwrite=True,
            save_original=True,
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

        # Setup for FD method
        if method[2] == "fd":
            if method[1] in ["trf", "dogbox"]:
                res_fail = np.zeros((self._n_timesteps, len(self._measurements)))
                for j, measuring_device in enumerate(self._measurements):
                    res_fail[:, j] = np.ones((self._n_timesteps)) * 100
                self.res_fail = res_fail.flatten()
            else:
                # scalar output
                self.res_fail = 100

            assert n_cores is not None, "n_cores must be provided when using FD method"

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

        # -- Initial Jacobian diagnostic --------------------------------------
        # Compute J(x0) *once* before scipy starts so we can log per-parameter
        # column norms.  A parameter with ||J[:, i]|| ≈ 0 is structurally
        # unidentifiable from x0 -- its residual gradient is zero, so the
        # solver will shuffle it around aimlessly (or freeze it) regardless
        # of trust-region settings.  This call is cached via ``_theta_jac``,
        # so when scipy queries J(x0) a moment later the cache hits and no
        # work is wasted.
        if method[1] in ["trf", "dogbox"] and method[2] == "ad":
            self._log_initial_jacobian_diagnostic()

        # -- Optional fast single-shooting objective ---------------------------
        # options={"fast": True} replaces the object-graph objective with a
        # sequential rollout of the composed pure one-step map (F_aug) --
        # exogenous inputs captured once, no per-eval model.initialize / CSV
        # re-reads.  Structural compatibility is always checked at build time
        # (un-composable models fall back to the exact path); numerical
        # equivalence holds BY CONSTRUCTION (each component's do_step is a thin
        # wrapper delegating to the same forward the composer threads) and is
        # regression-checked by tests/estimator/test_fast_shooting.py, not
        # re-proven per run.
        # options={"fast_validate": True} additionally cross-checks value and
        # gradient against the object-graph objective at runtime (debugging
        # aid; costs ~3 object-graph evaluations).
        self._fast_obj = None
        fast_requested = bool(options.pop("fast", False))
        fast_validate = bool(options.pop("fast_validate", False))
        if (
            fast_requested
            and self._transcription == "single_shooting"
            and method[2] == "ad"
        ):
            self._setup_fast_objective(validate=fast_validate)

        # Run optimization based on method
        LOGGER.task("Running optimization")
        if self._transcription != "single_shooting":
            # Collocation transcription: every timestep-boundary state becomes
            # a decision variable alongside theta, tied by hard continuity
            # constraints.  Reuses the shared setup above
            # (params/bounds/measurements) and the result-building teardown
            # below; only the NLP itself differs.
            from twin4build.estimator._transcription import solve_transcription

            result = solve_transcription(self, method, dict(options))
        elif method[0] == "casadi":
            # IPOPT (via CasADi) single-shooting solve.  Reuses the exact
            # AD objective / gradient the SciPy backends use -- only the
            # optimizer changes.  CasADi is imported lazily so it stays an
            # optional dependency.
            from twin4build.estimator._casadi_ipopt import solve_ipopt

            result = solve_ipopt(
                x0=np.asarray(self._x0_norm, dtype=np.float64),
                lb=np.asarray(self.bounds.lb, dtype=np.float64),
                ub=np.asarray(self.bounds.ub, dtype=np.float64),
                fun=lambda x: float(self._obj_ad(x, "scalar")),
                jac=lambda x: self._jac_ad(x, "scalar"),
                options=dict(options),
            )
            # Mirror the SciPy backends' ``nfev`` bookkeeping (CasADi does not
            # report objective evaluations, but the Estimator counts them).
            result.nfev = self._eval_count
        elif method[1] in ["trf", "dogbox"]:
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
                result = least_squares(
                    self._obj_fd_separate_process,
                    x0=self._x0_norm,
                    args=("vector",),
                    jac=self._jac_fd,
                    bounds=self.bounds,
                    method=method[1],
                    **options,
                )
        elif method[1] == "dual_annealing":
            # Global optimization via generalized simulated annealing.
            # High temperature explores broadly to find the right basin,
            # then anneals and polishes with a local gradient-based minimizer.
            bounds_list = list(zip(self.bounds.lb, self.bounds.ub))

            # Separate dual_annealing kwargs from local minimizer options
            da_keys = {
                "maxiter",
                "initial_temp",
                "restart_temp_ratio",
                "visit",
                "accept",
                "maxfun",
                "seed",
                "no_local_search",
                "callback",
            }
            da_kwargs = {k: v for k, v in options.items() if k in da_keys}
            local_opts = {k: v for k, v in options.items() if k not in da_keys}

            # The local minimizer uses SLSQP with AD gradients by default.
            # Users can override via local_search_method in options.
            local_method = local_opts.pop("local_search_method", "SLSQP")
            minimizer_kwargs = {
                "method": local_method,
                "jac": self._jac_ad,
                "args": ("scalar",),
                "bounds": self.bounds,
                "options": local_opts if local_opts else None,
            }

            result = dual_annealing(
                func=self._obj_ad,
                bounds=bounds_list,
                args=("scalar",),
                x0=self._x0_norm,
                minimizer_kwargs=minimizer_kwargs,
                **da_kwargs,
            )

        elif method[1] == "basinhopping":
            # Global optimization via random perturbation + local minimization.
            # At each step: perturb current solution, run local optimizer,
            # accept/reject via Metropolis criterion at temperature T.

            # Separate basinhopping kwargs from local minimizer options
            bh_keys = {
                "niter",
                "T",
                "stepsize",
                "seed",
                "niter_success",
                "target_accept_rate",
                "stepwise_factor",
                "callback",
            }
            bh_kwargs = {k: v for k, v in options.items() if k in bh_keys}
            local_opts = {k: v for k, v in options.items() if k not in bh_keys}

            local_method = local_opts.pop("local_search_method", "SLSQP")
            minimizer_kwargs = {
                "method": local_method,
                "jac": self._jac_ad,
                "args": ("scalar",),
                "bounds": self.bounds,
                "options": local_opts if local_opts else None,
            }

            # Bounded perturbation step function.
            # The default scipy RandomDisplacement adds uniform noise in
            # [-stepsize, +stepsize] but does NOT clip to bounds, so the
            # local optimizer starts from infeasible points.  This wrapper
            # clips the perturbed vector back to [lb, ub], preserving the
            # "explore nearby basins" philosophy of basinhopping while
            # keeping all starts feasible.
            #
            # Tuning stepsize:
            #   - Parameters are normalized to [0, 1].
            #   - stepsize=0.5  → conservative, stays close to current best
            #   - stepsize=1.0  → moderate, can flip most schedule weights
            #   - stepsize=2.0+ → aggressive, nearly a random restart but
            #                     still biased toward the current best for
            #                     parameters near the interior of [0, 1]
            lb_arr = np.asarray(self.bounds.lb, dtype=np.float64)
            ub_arr = np.asarray(self.bounds.ub, dtype=np.float64)
            seed = bh_kwargs.pop("seed", None)
            rng = np.random.default_rng(seed)
            _stepsize = bh_kwargs.pop("stepsize", 0.5)

            class _BoundedStep:
                """Uniform perturbation clipped to parameter bounds."""

                def __init__(self, stepsize, lb, ub, rng):
                    self.stepsize = stepsize
                    self.lb = lb
                    self.ub = ub
                    self.rng = rng

                def __call__(self, x):
                    x_new = x + self.rng.uniform(
                        -self.stepsize, self.stepsize, size=x.shape
                    )
                    return np.clip(x_new, self.lb, self.ub)

            take_step = _BoundedStep(_stepsize, lb_arr, ub_arr, rng)

            result = basinhopping(
                func=self._obj_ad,
                x0=self._x0_norm,
                minimizer_kwargs=minimizer_kwargs,
                take_step=take_step,
                **bh_kwargs,
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

            if method[2] == "fd":
                result = minimize(
                    self._obj_fd_separate_process,
                    self._x0_norm,
                    args=("scalar",),
                    method=method[1],
                    jac=self._jac_fd,
                    hess=hess,
                    bounds=self.bounds,
                    options=options,
                )
            else:
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

        elapsed = time_module.time() - self._solver_start_time
        LOGGER.task("Finishing optimization")
        LOGGER.result(
            "Elapsed: %.1fs | function evaluations: %d", elapsed, self._eval_count
        )

        opt_success = getattr(result, "success", None)
        opt_message = getattr(result, "message", None)
        opt_nit = getattr(result, "nit", None)
        opt_fun = getattr(result, "fun", None)
        if opt_success is not None:
            if opt_success:
                LOGGER.ok(
                    "success=%s | iterations=%s | objective=%s",
                    opt_success,
                    opt_nit,
                    opt_fun,
                )
            else:
                LOGGER.warning(
                    "success=%s | iterations=%s | objective=%s.",
                    opt_success,
                    opt_nit,
                    opt_fun,
                )
        if opt_message:
            LOGGER.result("Solver message: %s", opt_message)

        if method[0] in ("scipy", "casadi"):
            # Leave the model at the OPTIMUM, not at the last objective
            # evaluation: the solver's final evaluation is a line-search probe
            # (and for the transcription/collocation backends the returned x
            # is a restored best iterate the model never saw), so without this
            # a subsequent ``simulator.simulate()`` runs with junk parameters.
            theta_opt = torch.tensor(
                np.asarray(result.x, dtype=np.float64), dtype=torch.float64
            )
            self.simulator.model.set_parameters(
                self._theta_to_param_values(theta_opt),
                self._flat_components,
                self._parameter_names,
                normalized=True,
                overwrite=True,
            )
            self.simulator.model.restore_parameters(keep_values=True)

        # Store the normalised solution for warm-starting (used by lambda scheduling)
        self._last_x_norm = result.x.copy()

        # Denormalize result using parameter's denormalize method
        # result.x is flat array of all unique parameter values
        result_x_list = []
        seen_unique = set()
        for i, (param, param_idx) in enumerate(
            zip(self._flat_parameters, self._theta_mask)
        ):
            if param_idx not in seen_unique:
                start, end = self._theta_slices[param_idx]
                x_norm = torch.tensor(result.x[start:end], dtype=torch.float64)
                x_denorm = param.denormalize(x_norm)
                result_x_list.extend(x_denorm.detach().numpy().flatten())
                seen_unique.add(param_idx)
        result_x = np.array(result_x_list)

        # Transcription (collocation) also estimates the boundary states;
        # carry the optimised initial state through so callers can seed a
        # continuous prediction from it (see EstimationResult).
        estimated_initial_state = getattr(result, "estimated_initial_state", None)
        transcription_audit = getattr(result, "transcription_audit", None)

        result = EstimationResult(
            result_x=result_x,
            component_id=[com.id for com in self._flat_components],
            component_attr=[attr for attr in self._parameter_names],
            theta_mask=self._theta_mask,
            theta_slices=self._theta_slices,
            unique_param_n_c=self._unique_param_n_c,
            start_time=self._start_time,
            end_time=self._end_time,
            step_size=self._stepSize,
            x0=self._x0,
            lb=self._lb,
            ub=self._ub,
            iterations=getattr(result, "nit", None),
            nfev=getattr(result, "nfev", None),
            final_objective=getattr(result, "fun", None),
            success=getattr(result, "success", None),
            message=getattr(result, "message", None),
        )
        if estimated_initial_state is not None:
            result["estimated_initial_state"] = estimated_initial_state
        if transcription_audit is not None:
            result["transcription_audit"] = transcription_audit

        with open(self.result_savedir_pickle, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        LOGGER.ok("Results saved to %s", self.result_savedir_pickle)
        LOGGER.remove_level()
        LOGGER.ok(
            "Running %s solver: %s (%s mode)",
            method[0],
            method[1],
            method[2],
            change_status=True,
            ignore_no_match=True,
        )
        return result

    def _setup_fast_objective(self, validate: bool = False) -> None:
        """Build the composed-map single-shooting objective.

        On success sets ``self._fast_obj`` (consumed by :meth:`_obj`); on any
        incompatibility leaves it ``None`` and the exact object-graph objective
        is used.  Construction itself performs the structural checks (every
        cone component composable, no shared theta, every measurement
        producible by the composed map).  Numerical equivalence with the
        object-graph objective holds by construction -- each composable
        component's ``do_step`` delegates to the same ``forward`` the composer
        threads -- and is regression-checked by
        ``tests/estimator/test_fast_shooting.py``.

        With ``validate=True`` (``options={"fast_validate": True}``) the
        objective value AND gradient are additionally cross-checked against
        the object-graph objective at ``x0`` and a perturbed theta before the
        fast path is enabled -- a runtime debugging aid costing ~3
        object-graph evaluations.
        """
        t0 = time_module.time()
        try:
            from twin4build.estimator._shooting import FastSingleShooting

            fast = FastSingleShooting(self)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Fast single-shooting unavailable (%s) -- using the "
                "object-graph objective.", exc,
            )
            return
        if not validate:
            LOGGER.ok(
                "Fast single-shooting objective enabled (setup %.1fs).",
                time_module.time() - t0,
            )
            self._fast_obj = fast
            return
        try:
            rng = np.random.default_rng(0)
            lbn = np.asarray(self._lb_norm, dtype=np.float64)
            ubn = np.asarray(self._ub_norm, dtype=np.float64)
            x0 = np.asarray(self._x0_norm, dtype=np.float64)
            x1 = np.clip(x0 + 0.05 * (rng.random(x0.shape) - 0.5), lbn, ubn)
            worst_val = 0.0
            for xi in (x0, x1):
                zt = torch.tensor(xi, dtype=torch.float64)
                self._mse_scaled = None
                self._obj(zt, "scalar")
                rmse_slow = float(self._last_rmse)
                fast.loglike(zt, "scalar")
                rmse_fast = float(self._last_rmse)
                worst_val = max(
                    worst_val,
                    abs(rmse_fast - rmse_slow) / max(1e-12, abs(rmse_slow)),
                )
            # Gradient agreement at x0 (this is what steers the solver).
            self._mse_scaled = None
            z = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            (g_slow,) = torch.autograd.grad(self._obj(z, "scalar"), z)
            self._mse_scaled = None
            z = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            (g_fast,) = torch.autograd.grad(fast.loglike(z, "scalar"), z)
            g_slow, g_fast = g_slow.numpy(), g_fast.numpy()
            gscale = max(1e-12, float(np.abs(g_slow).max()))
            worst_grad = float(np.abs(g_fast - g_slow).max()) / gscale
            self._mse_scaled = None
            if worst_val > 1e-4 or worst_grad > 1e-3:
                LOGGER.warning(
                    "Fast single-shooting objective DISAGREES with the "
                    "object-graph objective (rel value err %.2e, rel grad err "
                    "%.2e) -- using the object-graph objective.",
                    worst_val, worst_grad,
                )
                return
            LOGGER.ok(
                "Fast single-shooting objective enabled (validated: rel value "
                "err %.2e, rel grad err %.2e; setup %.1fs).",
                worst_val, worst_grad, time_module.time() - t0,
            )
            self._fast_obj = fast
        except Exception as exc:  # noqa: BLE001
            self._mse_scaled = None
            LOGGER.warning(
                "Fast single-shooting validation failed (%s) -- using the "
                "object-graph objective.", exc,
            )

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
        # Fast path: composed one-step map rollout (see _shooting.py), built
        # in _scipy_solver when options={"fast": True}.  Both paths end in
        # _loglike_from_residuals, so value/diagnostics contracts are shared
        # by construction.
        if getattr(self, "_fast_obj", None) is not None and output == "scalar":
            return self._fast_obj.loglike(theta, output)

        # Convert flat theta to per-parameter values
        param_values = self._theta_to_param_values(theta)
        self.simulator.model.set_parameters(
            param_values,
            self._flat_components,
            self._parameter_names,
            normalized=True,
            overwrite=True,
        )

        #

        simulation_readings = {
            com.id: torch.zeros((self._n_timesteps), dtype=torch.float64)
            for com, sd in self._measurements
        }
        actual_readings = {
            com.id: torch.zeros((self._n_timesteps), dtype=torch.float64)
            for com, sd in self._measurements
        }

        # Run parallelized simulation for all time periods
        self.simulator.simulate(
            start_time=self._start_time,
            end_time=self._end_time,
            step_size=self._stepSize,
            show_progress_bar=False,
        )

        # Extract and concatenate measurements from all periods
        n_time_prev = 0
        for batch_idx, (startTime_, endTime_, stepSize_) in enumerate(
            zip(self._start_time, self._end_time, self._stepSize)
        ):
            second_time_steps, date_time_steps, max_timesteps, _ = (
                core.Simulator.get_simulation_timesteps(startTime_, endTime_, stepSize_)
            )
            n_time = max_timesteps - self._n_warmup

            # Extract measurements for this period
            for measuring_device, sd in self._measurements:
                # Get simulation results for this period
                # History uses time-first layout (n_t, n_s, n_c), extract first component
                y_model_period = measuring_device.input["measuredValue"].history(
                    i_t=slice(self._n_warmup, max_timesteps), i_s=batch_idx, i_c=0
                )

                # Filter out NaN values (padding) for shorter periods
                # valid_mask = ~torch.isnan(y_model_period)
                # y_model_valid = y_model_period[valid_mask]

                y_actual_period = self.actual_readings[measuring_device.id][batch_idx]
                y_actual_period = y_actual_period.to_numpy()
                y_actual_period = y_actual_period[self._n_warmup :]
                y_actual_period = torch.tensor(y_actual_period, dtype=torch.float64)

                # Store in concatenated arrays
                end_idx = n_time_prev + len(y_model_period)
                simulation_readings[measuring_device.id][
                    n_time_prev:end_idx
                ] = y_model_period
                actual_readings[measuring_device.id][
                    n_time_prev:end_idx
                ] = y_actual_period

            n_time_prev += n_time

        # Raw residuals over the FULL padded horizon: rows past the filled
        # range stay zero (the historical normalization contract -- see
        # _loglike_from_residuals).
        res_raw = torch.zeros(
            (self._n_timesteps, len(self._measurements)), dtype=torch.float64
        )
        for j, (measuring_device, sd) in enumerate(self._measurements):
            res_raw[:, j] = (
                actual_readings[measuring_device.id]
                - simulation_readings[measuring_device.id]
            )

        return self._loglike_from_residuals(res_raw, output)

    def _loglike_from_residuals(
        self, res_raw: torch.Tensor, output: str
    ) -> torch.Tensor:
        """THE data-fit objective from raw residuals (single source of truth).

        Both objective implementations end here: the object-graph :meth:`_obj`
        and the composed-map fast path
        (:meth:`twin4build.estimator._shooting.FastSingleShooting.loglike`).
        Everything downstream of the residuals -- sd weighting, MSE
        normalization, the rescale-to-100 trick, regularization, and the
        ``_last_*`` diagnostics -- therefore cannot diverge between the two.

        ``res_raw`` is ``(N, n_meas)`` raw residuals ``actual - model``.  ``N``
        may be the full padded horizon (object-graph path: unfilled rows are
        zero) or only the scored rows (fast path); either way the mean is
        taken over ``_n_timesteps * n_meas`` -- identical values, matching the
        historical contract the rescaled objective was tuned on.

        Pure tensor math on the differentiable path (torch.func-safe); the
        diagnostic side effects detach first.
        """
        n_meas = res_raw.shape[1]
        denom = float(self._n_timesteps) * n_meas
        sd = torch.tensor(
            [float(sd_) for _, sd_ in self._measurements], dtype=torch.float64
        )
        res = res_raw / sd

        # Diagnostics (identical for both output modes): raw MSE / RMSE in
        # measurement units, per-sensor RMSE.
        raw_sq = res_raw.detach() ** 2
        raw_mse = (torch.sum(raw_sq) / denom).item()
        self._last_mse = raw_mse
        self._last_rmse = raw_mse**0.5
        self._last_rmse_per_sensor = {
            md.id: (torch.sum(raw_sq[:, j]).item() / self._n_timesteps) ** 0.5
            for j, (md, _sd) in enumerate(self._measurements)
        }

        # We scale the objective to 100 on the first evaluation of a phase for
        # numerical stability (_mse_scaled is reset per phase).
        if output == "scalar":
            mse = torch.sum(res**2) / denom
            if self._mse_scaled is None:
                self._mse_scaled = mse.detach().item() / 100
            self._loglike = mse / self._mse_scaled

            # Add binarization penalty if regularization is enabled
            if self._regularization_lambda > 0:
                penalty = self._compute_regularization_penalty()
                self._last_penalty = penalty.detach().item()
                self._loglike = self._loglike + self._regularization_lambda * penalty
            else:
                self._last_penalty = 0.0
        elif output == "vector":
            res_flat = res.flatten()
            if self._mse_scaled is None:
                self._mse_scaled = (
                    torch.sum(res.detach() ** 2).item() / denom / 100
                ) ** 0.5  # We take squareroot because of the scipy least squares method which expects a residual vector which will later be squared
            self._loglike = res_flat / self._mse_scaled
            # Note: Regularization not supported for vector output (least squares methods)
            self._last_penalty = 0.0
        else:
            raise ValueError(f"Invalid output: {output}")

        return self._loglike

    def _compute_regularization_penalty(self) -> torch.Tensor:
        """
        Compute the binarization penalty P(x) = x(1-x) for all regularization components.

        The penalty encourages selection weights toward discrete values (0 or 1).
        Components must implement a `compute_binarization_penalty()` method.

        Returns
        -------
        torch.Tensor
            Total binarization penalty summed across all regularization components.
        """
        penalty = torch.tensor(0.0, dtype=torch.float64)

        # If no specific components provided, auto-detect from parameter components
        if self._regularization_components is None:
            components_to_check = set()
            for comp in self._flat_components:
                if hasattr(comp, "compute_binarization_penalty"):
                    components_to_check.add(comp)
        else:
            components_to_check = self._regularization_components

        # Sum penalties from all components
        for comp in components_to_check:
            if hasattr(comp, "compute_binarization_penalty"):
                penalty = penalty + comp.compute_binarization_penalty()

        return penalty

    @staticmethod
    def _short_component_label(component_id: str) -> str:
        """Compress a (possibly composite) component id into a short log tag.

        Composite ``ModeledNode`` ids have the shape
        ``[memA][memB]..._<16-hex-fingerprint>`` where the bracketed members
        are per-member *tail slices* of semantic short names plus boilerplate;
        they're designed to guarantee uniqueness under Windows MAX_PATH, not
        to be read.  Per-eval theta dumps repeat this prefix on every
        parameter which makes the output unreadable.

        This collapses to ``[firstMember]_<8hex>`` when the id ends in an
        underscore + >=8 hex chars (the fingerprint tail), otherwise returns
        the id unchanged.  The first bracketed segment plus 8-char fingerprint
        is already globally unique in practice for any reasonable run.
        """
        import re as _re

        m = _re.match(r"^(\[[^\]]+\]).*_([0-9a-f]{8})[0-9a-f]*$", component_id)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
        return component_id

    def _log_initial_jacobian_diagnostic(self) -> None:
        """Compute and log the residual Jacobian column norms at ``x0``.

        For each estimated parameter, prints ``||J[:, i]||_2`` (absolute and
        relative to the max column norm).  Parameters whose relative norm is
        below 1e-6 are flagged: they do not measurably affect any residual
        at the initial point, so the solver has essentially no information
        about them until *other* parameters move to bring them into play.

        This is a one-shot call at the start of estimation; the Jacobian is
        cached in :attr:`_jac` / :attr:`_theta_jac` so scipy's own initial
        Jacobian evaluation at ``x0`` hits the cache.  If the AD pass
        fails for any reason the diagnostic is skipped -- we don't want a
        diagnostic hook to crash the solver run.
        """
        try:
            LOGGER.task("Initial Jacobian diagnostic (at x0)")
            LOGGER.add_level()
            t0 = time_module.time()
            jac = self._jac_ad(self._x0_norm, "vector")  # shape (n_res, n_theta)
            elapsed = time_module.time() - t0
            jac = np.asarray(jac)
            col_norms = np.linalg.norm(jac, axis=0)
            max_norm = float(np.max(col_norms)) if col_norms.size else 0.0
            if max_norm <= 0.0:
                LOGGER.iter(
                    "Jacobian is identically zero at x0 -- solver cannot make progress "
                    "(check that measurements are actually affected by the parameters)"
                )
                LOGGER.remove_level()
                return
            n_dead = int(np.sum(col_norms / max_norm < 1e-6))
            LOGGER.iter(
                "n_theta=%d | n_residuals=%d | max ||J[:,i]||=%.4g | "
                "zero-gradient params=%d | elapsed=%.1fs",
                jac.shape[1], jac.shape[0], max_norm, n_dead, elapsed,
            )
            # Per-parameter breakdown, grouped by component, matching the
            # format of the theta-dump so the user can correlate with
            # iteration logs by eye.
            seen_unique: set = set()
            comp_order: List[str] = []
            comp_parts: Dict[str, List[str]] = {}
            for component, attr, param, param_idx in zip(
                self._flat_components,
                self._parameter_names,
                self._flat_parameters,
                self._theta_mask,
            ):
                if param_idx in seen_unique:
                    continue
                seen_unique.add(param_idx)
                start, end = self._theta_slices[param_idx]
                col_slice = col_norms[start:end]
                col_norm = float(np.max(col_slice)) if col_slice.size else 0.0
                rel = col_norm / max_norm if max_norm > 0 else 0.0
                tag = " DEAD" if rel < 1e-6 else (" weak" if rel < 1e-3 else "")
                part = f"{attr}=|J|={col_norm:.3g} (rel={rel:.2g}){tag}"
                cid = component.id
                if cid not in comp_parts:
                    comp_parts[cid] = []
                    comp_order.append(cid)
                comp_parts[cid].append(part)
            for cid in comp_order:
                label = self._short_component_label(cid)
                LOGGER.iter("%s: %s", label, "  ".join(comp_parts[cid]))
            LOGGER.remove_level()
        except Exception as exc:  # pragma: no cover -- diagnostic must never break the run
            LOGGER.warning("Initial Jacobian diagnostic failed: %s", exc)
            try:
                LOGGER.remove_level()
            except Exception:
                pass

    def _format_theta_dump(self, theta: np.ndarray) -> List[str]:
        """Format a flat *normalized* theta vector as a human-readable dump.

        For each unique parameter, denormalizes ``theta[start:end]`` using the
        corresponding :class:`tps.Parameter` so the logged values are in the
        user's native units (not the [0, 1] normalized solver coordinates).

        Returns a *list of lines*, one per component, of the form::

            [VRM103]_1427052e: kp=0.5 Ti=900 output_min=1e-10 gate_0.threshold=19 gate_0.band=4

        Multi-valued (``n_c > 1``) parameters are emitted as bracketed lists.
        The caller is responsible for emitting each line to the log so the
        logger's prefix/status columns align on each row.
        """
        theta_np = np.asarray(theta, dtype=np.float64).flatten()

        seen_unique: set = set()
        comp_order: List[str] = []
        comp_parts: Dict[str, List[str]] = {}
        for component, attr, param, param_idx in zip(
            self._flat_components,
            self._parameter_names,
            self._flat_parameters,
            self._theta_mask,
        ):
            if param_idx in seen_unique:
                continue
            seen_unique.add(int(param_idx))
            start, end = self._theta_slices[int(param_idx)]
            x_norm = torch.tensor(theta_np[start:end], dtype=torch.float64)
            try:
                x_user = param.denormalize(x_norm).detach().cpu().numpy().flatten()
            except Exception:
                x_user = x_norm.detach().cpu().numpy().flatten()
            if x_user.size == 1:
                rendered = f"{attr}={x_user[0]:.4g}"
            else:
                vals = ", ".join(f"{v:.4g}" for v in x_user)
                rendered = f"{attr}=[{vals}]"

            cid = component.id
            if cid not in comp_parts:
                comp_parts[cid] = []
                comp_order.append(cid)
            comp_parts[cid].append(rendered)

        lines: List[str] = []
        for cid in comp_order:
            label = self._short_component_label(cid)
            lines.append(f"{label}: {' '.join(comp_parts[cid])}")
        return lines

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
            return np.asarray(self._loglike.detach().numpy(), dtype=np.float64)
        else:
            self._theta_obj = theta
            self._eval_count += 1
            # -- Numerical-failure recovery ---------------------------------------
            # The torch graph inside ``self._obj`` runs the full
            # simulation; if any sub-system produces a NaN (damper at
            # nominalAirFlowRate ~ 0, log-scaled valve hitting its
            # bound, heat-recovery effectiveness saturating, etc.) the
            # simulator raises ``ValueError("Input ... is NaN")``
            # downstream when the bad value propagates into the next
            # component.  Crashing the whole estimator on that wastes
            # everything the solver has learned so far; ``_obj_fd``
            # already handles this with a large-penalty ``res_fail``.
            # Mirror that here and dump the offending theta in user
            # units so the *next* run can see exactly which parameter
            # combination is unstable instead of guessing.
            try:
                self._loglike = self._obj(theta, output)
                obj_val = self._loglike.detach().numpy()
                eval_failed = False
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "eval=%d objective failed (%s: %s) -- returning penalty",
                    self._eval_count, type(exc).__name__, exc,
                )
                try:
                    dump_lines = self._format_theta_dump(theta.detach().numpy())
                    LOGGER.warning(
                        "theta[%d] at failure  (%d components, denormalized)",
                        self._eval_count, len(dump_lines),
                    )
                    LOGGER.add_level()
                    for line in dump_lines:
                        LOGGER.warning("%s", line)
                    LOGGER.remove_level()
                except Exception as dump_exc:  # noqa: BLE001
                    LOGGER.warning(
                        "could not dump failing theta: %s", dump_exc,
                    )
                penalty = 1e10
                if output == "scalar":
                    obj_val = np.array(penalty, dtype=np.float64)
                else:
                    obj_val = np.full(
                        self._n_timesteps * len(self._measurements),
                        penalty,
                        dtype=np.float64,
                    )
                # Keep the cache consistent with what we hand back to
                # the solver so the next ``torch.equal`` short-circuit
                # returns the same penalty until the solver moves
                # ``theta``.
                self._loglike = torch.tensor(obj_val, dtype=torch.float64)
                self._last_rmse = float("nan")
                self._last_rmse_per_sensor = {}
                self._last_penalty = 0.0
                eval_failed = True
            elapsed = time_module.time() - self._solver_start_time
            obj_f = float(np.sum(obj_val))
            if output == "scalar":
                rmse_f = (
                    float(self._last_rmse) if hasattr(self, "_last_rmse") else float("nan")
                )
                if hasattr(self, "_last_penalty") and self._last_penalty > 0:
                    LOGGER.iter(
                        "eval=%d | obj=%.6f | rmse=%.4f | penalty=%.4f | elapsed=%.1fs",
                        self._eval_count,
                        obj_f,
                        rmse_f,
                        float(self._last_penalty),
                        elapsed,
                    )
                else:
                    LOGGER.iter(
                        "eval=%d | obj=%.6f | rmse=%.4f | elapsed=%.1fs",
                        self._eval_count,
                        obj_f,
                        rmse_f,
                        elapsed,
                    )
            else:
                # Vector mode (least_squares): report the scalar LS objective
                # 0.5 * ||r||^2 alongside the raw RMSE so progress is
                # directly comparable to the scalar-mode log line.
                ls_obj = 0.5 * float(np.dot(obj_val, obj_val))
                rmse_f = (
                    float(self._last_rmse) if hasattr(self, "_last_rmse") else float("nan")
                )
                LOGGER.iter(
                    "eval=%d | obj=%.6f | rmse=%.4f | elapsed=%.1fs",
                    self._eval_count,
                    ls_obj,
                    rmse_f,
                    elapsed,
                )
            if getattr(self, "_log_parameters", False):
                dump_lines = self._format_theta_dump(theta.detach().numpy())
                LOGGER.iter("theta[%d]  (%d components)", self._eval_count, len(dump_lines))
                LOGGER.add_level()
                for line in dump_lines:
                    LOGGER.iter("%s", line)
                LOGGER.remove_level()

            # Per-sensor RMSE: lets us see which zones are driving the
            # aggregate ``rmse`` field above.  Emitted on every *new best*
            # iteration regardless of the ``log_parameters`` flag -- it's
            # the only honest physical-units view when
            # ``measurements="auto"`` pools sensors of different units
            # (temperatures, damper / valve positions, flows...) into the
            # single pooled ``rmse`` number.  Trust-region solvers
            # evaluate many rejected trial steps, so gating on new-best
            # keeps the log focused on real progress instead of 34x-ing
            # the volume on every probe.  Sorted worst-first so the eye
            # lands on problem sensors immediately.
            per_sensor = getattr(self, "_last_rmse_per_sensor", None)
            best = getattr(self, "_best_obj_logged", float("inf"))
            if per_sensor and obj_f < best:
                self._best_obj_logged = obj_f
                sorted_items = sorted(
                    per_sensor.items(), key=lambda kv: kv[1], reverse=True
                )
                LOGGER.iter(
                    "rmse_per_sensor[%d]  (%d sensors, worst first, new best)",
                    self._eval_count, len(sorted_items),
                )
                LOGGER.add_level()
                for sid, v in sorted_items:
                    LOGGER.iter("%s: %.4f", sid, v)
                LOGGER.remove_level()
            return np.asarray(obj_val, dtype=np.float64)

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
        Chooses the AD mode based on output dimensionality:

        - ``output == "scalar"``:   reverse-mode (``jacrev``) is optimal --
          one backward pass through the simulation regardless of ``dim(theta)``.
        - ``output == "vector"``:   forward-mode (``jacfwd``) is optimal --
          cost scales with ``dim(theta)`` (tens) instead of ``dim(residuals)``
          (thousands to tens of thousands).  Using ``jacrev`` here would
          require one backward pass *per residual* and keep a large autograd
          graph alive, which is orders of magnitude slower and far more
          memory-hungry for tall-skinny residual Jacobians (as in the
          least_squares / trf / dogbox solvers).
        """
        # Save and restore state that _obj overwrites during the AD call's
        # internal forward pass, so the cached values in _obj_ad stay valid.
        # Use getattr so this is safe when called before _obj_ad has ever
        # been invoked (e.g. from the pre-solver Jacobian diagnostic).
        saved_loglike = getattr(self, "_loglike", None)
        if output == "vector":
            self._jac = torch.func.jacfwd(self._obj, argnums=0)(theta, output)
        else:
            self._jac = torch.func.jacrev(self._obj, argnums=0)(theta, output)
        if saved_loglike is not None:
            self._loglike = saved_loglike

        if torch.any(torch.isnan(self._jac)):
            raise ValueError("JAC contains NaNs")
        return self._jac

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
            return np.asarray(self._jac.detach().numpy(), dtype=np.float64)
        elif getattr(self, "_fast_obj", None) is not None and output == "scalar":
            # Fast path: plain reverse-mode autograd through the composed-map
            # rollout.  jacrev would work too but activates functorch, which
            # forces the state-space components onto the unrolled
            # scaling-and-squaring matrix exponential (no vmap/functorch rule
            # for torch.matrix_exp) -- about 2x slower end to end.
            self._theta_jac = theta
            try:
                z = theta.detach().clone().requires_grad_(True)
                (g,) = torch.autograd.grad(self._fast_obj.loglike(z, output), z)
                self._jac = g.detach()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "fast jacobian eval failed (%s: %s) -- returning zero "
                    "gradient", type(exc).__name__, exc,
                )
                self._jac = torch.zeros(theta.numel(), dtype=torch.float64)
            jac_np = np.asarray(self._jac.numpy(), dtype=np.float64)
            LOGGER.debug("grad_norm=%.4f", float(np.linalg.norm(jac_np.ravel())))
            return jac_np
        else:
            self._theta_jac = theta
            # Mirror the simulation-failure recovery added to ``_obj_ad``.
            # The scipy SLSQP / TRF / L-BFGS-B drivers always call the
            # Jacobian at the same theta where the objective was just
            # evaluated, so a NaN-producing iterate that ``_obj_ad``
            # converted to a penalty value will crash here on the
            # subsequent ``jacrev(self._obj)`` call.  Returning a
            # zero / NaN Jacobian without the matching obj penalty
            # confuses the solver, so we substitute a zero gradient: the
            # QP step inside SLSQP then degenerates to "no descent
            # direction", scipy's line search backtracks, and the next
            # accepted iterate steps away from this bad region.
            try:
                self._jac = self.__jac_ad(theta, output)
                jac_np = np.asarray(self._jac.detach().numpy(), dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "jacobian eval failed (%s: %s) -- returning zero gradient",
                    type(exc).__name__, exc,
                )
                if output == "scalar":
                    jac_np = np.zeros(theta.numel(), dtype=np.float64)
                else:
                    jac_np = np.zeros(
                        (self._n_timesteps * len(self._measurements),
                         theta.numel()),
                        dtype=np.float64,
                    )
                self._jac = torch.tensor(jac_np, dtype=torch.float64)
            LOGGER.debug(
                "grad_norm=%.4f",
                float(np.linalg.norm(jac_np.ravel())),
            )
            return jac_np

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
        self._hes = torch.func.jacfwd(self.__jac_ad, argnums=0)(theta, output)
        return self._hes

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
            return np.asarray(self._hes.detach().numpy(), dtype=np.float64)
        else:
            self._theta_hes = theta
            self._hes = self.__hes_ad(theta, output)
            return np.asarray(self._hes.detach().numpy(), dtype=np.float64)


class EstimationResult(dict):
    """
    A dictionary-like object containing parameter estimation results.

    This class stores the results of parameter estimation including optimized
    parameters, component information, and metadata about the estimation process.

    Args:
        result_x: Optimized parameter values.
        component_id: List of component IDs.
        component_attr: List of attribute names.
        theta_mask: Parameter mask mapping flat theta entries to unique parameters.
        theta_slices: Per-parameter ``(start, stop)`` slices into the flat theta
            vector (multi-branch parameters occupy more than one slot).
        unique_param_n_c: Number of branches (``n_c``) per unique parameter.
        start_time: Training start times.
        end_time: Training end times.
        step_size: Training step sizes.
        x0: Initial parameter values.
        lb: Lower bounds.
        ub: Upper bounds.
        iterations: Number of iterations performed by the optimizer.
        nfev: Number of function evaluations performed by the optimizer.
        final_objective: Final objective function value achieved.
        success: Whether the optimization was successful.
        message: Optimization result message.

    Notes:
        Depending on the estimation configuration, additional keys may be
        present on the result dict: ``estimated_initial_state`` (per-component
        initial states) and ``transcription_audit`` (collocation
        solution-quality audit). Results saved to disk can be reloaded with
        :meth:`~twin4build.model.simulation_model.simulation_model.SimulationModel.load_estimation_result`.

    Examples:
        >>> result = EstimationResult(
        ...     result_x=np.array([0.8, 0.9]),
        ...     component_id=["comp1", "comp2"],
        ...     component_attr=["efficiency", "efficiency"],
        ...     theta_mask=np.array([0, 1]),
        ...     theta_slices=[(0, 1), (1, 2)],
        ...     unique_param_n_c=[1, 1],
        ...     start_time=[datetime.datetime(2024, 1, 1)],
        ...     end_time=[datetime.datetime(2024, 1, 2)],
        ...     step_size=[3600],
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
        theta_slices: Optional[List[tuple]] = None,
        unique_param_n_c: Optional[List[int]] = None,
        start_time: Optional[List[datetime.datetime]] = None,
        end_time: Optional[List[datetime.datetime]] = None,
        step_size: Optional[List[int]] = None,
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

        Args:
            result_x: Optimized parameter values.
            component_id: List of component IDs.
            component_attr: List of attribute names.
            theta_mask: Parameter mask mapping flat theta entries to unique parameters.
            theta_slices: Per-parameter ``(start, stop)`` slices into the flat theta vector.
            unique_param_n_c: Number of branches (``n_c``) per unique parameter.
            start_time: Training start times.
            end_time: Training end times.
            step_size: Training step sizes.
            x0: Initial parameter values.
            lb: Lower bounds.
            ub: Upper bounds.
            iterations: Number of iterations performed by the optimizer.
            nfev: Number of function evaluations performed by the optimizer.
            final_objective: Final objective function value achieved.
            success: Whether the optimization was successful.
            message: Optimization result message.
        """
        super().__init__(
            result_x=result_x,
            component_id=component_id,
            component_attr=component_attr,
            theta_mask=theta_mask,
            theta_slices=theta_slices,
            unique_param_n_c=unique_param_n_c,
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
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
