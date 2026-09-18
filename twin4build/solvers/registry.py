"""Pluggable solvers for :class:`~twin4build.estimator.estimator.Estimator`
and Pareto routes for :class:`~twin4build.optimizer.optimizer.Optimizer`.

The built-in backends (SciPy, IPOPT, the batched torch solvers) are wired by
name inside the estimator.  Anything else -- a solver developed outside the
library, an experimental step, a wrapper around another optimizer -- plugs in
through the two small contracts below.  Nothing in the library needs to know
the solver's name in advance: the estimator prepares the problem exactly as
it does for the built-in backends and hands it over.

**Estimation solver.**  An object with

- ``method``: the ``(library, name, mode)`` tuple users pass as ``method=``,
- ``solve(problem, options)``: returns a SciPy-like result whose ``x`` is
  the solution in the estimator's *normalized* coordinates, plus ``fun``,
  ``success``, ``status``, ``message``, ``nit`` and ``nfev``.  Optional
  attributes the estimator records when present: ``multistart_audit``,
  ``derivative_stats``, ``iteration_history``, ``aggregate_nfev`` /
  ``aggregate_njev`` and ``curvature_block_sizes``.

``problem`` is an :class:`EstimationProblem`: the normalized start and
bounds, the composed (functional) objective with its batched value /
gradient / residual / Jacobian bundles, the device and dtype, the
transcription, and the estimator itself for solvers that need its host
callbacks.  A solver can be used in two ways::

    from twin4build.solvers.registry import register_solver

    register_solver(MySolver())                # by name from now on
    estimator.estimate(..., method=("mylib", "my-step", "ad"))

    estimator.estimate(..., method=MySolver())  # or the instance itself

A registered method takes precedence over a built-in one of the same name,
so a plug-in can replace a built-in backend without touching the library.

**Pareto route.**  :meth:`Optimizer.pareto_front` solves the payoff-table
anchors and the epsilon-subproblems of the augmented epsilon-constraint
sweep with a host solver by default.  A route replaces both stages: an
object with ``method`` and

- ``anchors(opt, x0, bounds, delta, options) -> (X, audit)``: both anchors,
  ``X`` of shape ``(2, n_theta)`` in normalized coordinates and one audit
  dict per row with at least ``success`` and ``nit``;
- ``sweep(opt, eps_grid, x_a1, x_a2, ideal2, range2, delta, bounds, mu=None,
  options=None) -> (X, audit)``: one row per epsilon value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

MethodTuple = Tuple[str, str, str]


@dataclass
class EstimationProblem:
    """What the estimator hands to a plug-in solver.

    Coordinates are the estimator's normalized ones: ``x0``, ``lb`` and
    ``ub`` are float64 arrays of length ``n_theta`` and the objective's
    bundles take and return normalized theta.  ``objective`` is the composed
    functional objective (``None`` when the model is not fully functional;
    a solver that needs it should raise a clear error).  ``estimator`` is
    the calling :class:`Estimator`, for solvers that use its host paths
    (``_obj_ad`` / ``_jac_ad``) or its logging.
    """

    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    objective: Any
    device: Any
    dtype: Any
    transcription: str
    estimator: Any

    @property
    def n_theta(self) -> int:
        return int(len(self.x0))

    def multistart(self, options: Dict[str, Any]) -> np.ndarray:
        """Starts ``(n_starts, n_theta)`` from the multistart options.

        Consumes ``normalized_starts``, ``n_starts``, ``start_strategy``
        (``"uniform_bounds"`` or ``"local"``), ``start_spread`` and
        ``start_seed`` from ``options`` (in place) so a solver can pass the
        remainder on as its own options.  The first row is always ``x0``.
        """
        normalized_starts = options.pop("normalized_starts", None)
        n_starts = int(options.pop("n_starts", 1))
        start_strategy = options.pop("start_strategy", "uniform_bounds")
        start_spread = float(options.pop("start_spread", 0.15))
        start_seed = options.pop("start_seed", 0)
        x0 = np.asarray(self.x0, dtype=np.float64)
        lb = np.asarray(self.lb, dtype=np.float64)
        ub = np.asarray(self.ub, dtype=np.float64)
        if normalized_starts is None:
            rng = np.random.default_rng(start_seed)
            starts = np.repeat(x0[None, :], n_starts, axis=0)
            if n_starts > 1:
                if start_strategy == "uniform_bounds":
                    starts[1:] = rng.uniform(lb, ub, size=starts[1:].shape)
                elif start_strategy == "local":
                    starts[1:] += rng.uniform(
                        -start_spread, start_spread, size=starts[1:].shape
                    )
                    starts = np.clip(starts, lb, ub)
                else:
                    raise ValueError(
                        "start_strategy must be 'uniform_bounds' or 'local'"
                    )
            return starts
        starts = np.asarray(normalized_starts, dtype=np.float64)
        if starts.ndim == 1:
            starts = starts[None, :]
        if starts.shape[1] != len(x0):
            raise ValueError(
                f"normalized_starts must have shape (n_starts, {len(x0)})"
            )
        return starts


@runtime_checkable
class EstimationSolver(Protocol):
    """The plug-in solver contract (see the module docstring)."""

    method: MethodTuple

    def solve(self, problem: EstimationProblem, options: Dict[str, Any]) -> Any: ...


@runtime_checkable
class ParetoRoute(Protocol):
    """The plug-in Pareto route contract (see the module docstring)."""

    method: MethodTuple

    def anchors(self, opt, x0, bounds, delta, options) -> Tuple[np.ndarray, list]: ...

    def sweep(
        self, opt, eps_grid, x_a1, x_a2, ideal2, range2, delta, bounds, mu=None, options=None
    ) -> Tuple[np.ndarray, list]: ...


def _method_key(method) -> MethodTuple:
    method = tuple(method)
    if len(method) != 3 or not all(isinstance(m, str) for m in method):
        raise ValueError(
            "A solver method must be a (library, name, mode) tuple of strings; "
            f"got {method!r}."
        )
    return method  # type: ignore[return-value]


def is_solver(obj) -> bool:
    """True for an object that satisfies :class:`EstimationSolver`."""
    return (
        not isinstance(obj, (str, tuple, list))
        and hasattr(obj, "method")
        and callable(getattr(obj, "solve", None))
    )


def is_pareto_route(obj) -> bool:
    """True for an object that satisfies :class:`ParetoRoute`."""
    return (
        not isinstance(obj, (str, tuple, list))
        and hasattr(obj, "method")
        and callable(getattr(obj, "anchors", None))
        and callable(getattr(obj, "sweep", None))
    )


_SOLVERS: Dict[MethodTuple, Any] = {}
_PARETO_ROUTES: Dict[MethodTuple, Any] = {}


def register_solver(solver, *, replace: bool = True) -> MethodTuple:
    """Make ``solver`` available to every estimator under ``solver.method``.

    ``replace=False`` raises if that method is already registered.
    Returns the method tuple.
    """
    if not is_solver(solver):
        raise TypeError(
            "register_solver expects an object with a 'method' tuple and a "
            f"'solve(problem, options)' method; got {type(solver).__name__}."
        )
    key = _method_key(solver.method)
    if not replace and key in _SOLVERS:
        raise ValueError(f"A solver is already registered for {key!r}.")
    _SOLVERS[key] = solver
    return key


def unregister_solver(method) -> None:
    _SOLVERS.pop(_method_key(method), None)


def registered_solvers() -> Dict[MethodTuple, Any]:
    return dict(_SOLVERS)


def find_solver(method) -> Optional[Any]:
    """The registered solver for ``method``, or ``None``."""
    try:
        return _SOLVERS.get(_method_key(method))
    except ValueError:
        return None


def register_pareto_route(route, *, replace: bool = True) -> MethodTuple:
    """Make ``route`` available to :meth:`Optimizer.pareto_front` under
    ``route.method``."""
    if not is_pareto_route(route):
        raise TypeError(
            "register_pareto_route expects an object with a 'method' tuple and "
            f"'anchors' / 'sweep' methods; got {type(route).__name__}."
        )
    key = _method_key(route.method)
    if not replace and key in _PARETO_ROUTES:
        raise ValueError(f"A Pareto route is already registered for {key!r}.")
    _PARETO_ROUTES[key] = route
    return key


def unregister_pareto_route(method) -> None:
    _PARETO_ROUTES.pop(_method_key(method), None)


def registered_pareto_routes() -> Dict[MethodTuple, Any]:
    return dict(_PARETO_ROUTES)


def find_pareto_route(method) -> Optional[Any]:
    """The registered Pareto route for ``method`` (3- or 4-tuple), or ``None``."""
    method = tuple(method)
    if len(method) == 4:
        method = method[:3]
    try:
        return _PARETO_ROUTES.get(_method_key(method))
    except ValueError:
        return None


__all__ = [
    "EstimationProblem",
    "EstimationSolver",
    "ParetoRoute",
    "MethodTuple",
    "is_solver",
    "is_pareto_route",
    "register_solver",
    "unregister_solver",
    "registered_solvers",
    "find_solver",
    "register_pareto_route",
    "unregister_pareto_route",
    "registered_pareto_routes",
    "find_pareto_route",
]
