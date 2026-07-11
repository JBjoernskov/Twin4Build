"""IPOPT optimization backend for :class:`~twin4build.estimator.estimator.Estimator`, via CasADi.

Why CasADi
----------
IPOPT (COIN-OR interior-point NLP solver) is the solver the building-model
parameter-estimation literature benchmarks against (e.g. Blum et al.'s
JModelica/CasADi collocation pipeline).  Rather than depend on ``cyipopt`` --
which ships no Windows pip wheel and needs a from-source build against a system
IPOPT -- we reach IPOPT through **CasADi**, which installs as a single pip wheel
on every platform (``pip install casadi``) with IPOPT + MUMPS bundled.  This
keeps IPOPT an *optional* dependency: core twin4build never imports CasADi, and
only ``Estimator.estimate(method=("casadi", "ipopt", ...))`` pulls it in.

Black-box objective bridge
--------------------------
The estimation objective is opaque to CasADi -- each evaluation runs a full
PyTorch forward simulation of the building model.  We therefore expose it to
IPOPT through :class:`casadi.Callback` wrappers:

* an *objective* callback ``f(x) -> R`` backed by the Estimator's
  ``_obj_ad(x, "scalar")`` (the negative-log-likelihood / SSE), and
* a *gradient* callback ``grad f(x)`` backed by ``_jac_ad(x, "scalar")``
  (exact reverse-mode autodiff through the simulation).

IPOPT approximates the Hessian with limited-memory BFGS, so no second-order
information is required from the model.  This is a **single-shooting** solve --
identical problem to the SciPy backends, only the optimizer changes -- and is
the foundation the collocation (simultaneous-transcription) backends build on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict, Optional

import numpy as np


def _require_casadi():
    """Import CasADi, raising an actionable error if it is missing."""
    try:
        import casadi as ca  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only w/o casadi
        raise ImportError(
            'The ("casadi", "ipopt", ...) estimation backend requires CasADi, '
            "which bundles the IPOPT solver.  Install it with `pip install "
            "casadi` (a single pip wheel with IPOPT + MUMPS, no compiler or "
            "conda needed)."
        ) from exc
    return ca


# IPOPT option names differ from the SciPy-style keys twin4build's public API
# uses.  Translate the common ones so callers can keep passing the same
# ``options={"maxiter": ..., "ftol": ...}`` dict regardless of backend.
def _map_options(options: Optional[Dict]) -> Dict:
    """Translate twin4build/SciPy-style option keys to IPOPT option names.

    Recognised source keys: ``maxiter`` -> ``max_iter``, ``ftol`` -> ``tol``,
    ``verbose`` -> ``print_level``.  Any key already prefixed ``ipopt.`` (or the
    passthrough keys ``print_level`` / ``max_iter`` / ``tol``) is forwarded
    verbatim, so power users can set arbitrary IPOPT options.
    """
    options = dict(options or {})
    ipopt_opts: Dict[str, object] = {}

    if "maxiter" in options:
        ipopt_opts["max_iter"] = int(options.pop("maxiter"))
    if "ftol" in options:
        ipopt_opts["tol"] = float(options.pop("ftol"))
    if "verbose" in options:
        # SciPy ``verbose`` is 0/1/2; IPOPT ``print_level`` is 0..12.
        ipopt_opts["print_level"] = 5 if int(options.pop("verbose")) else 0

    # Forward anything the caller addressed to IPOPT explicitly.
    for key in list(options):
        if key.startswith("ipopt."):
            ipopt_opts[key[len("ipopt.") :]] = options.pop(key)
        elif key in ("max_iter", "tol", "print_level", "acceptable_tol",
                     "mu_strategy", "linear_solver", "hessian_approximation"):
            ipopt_opts[key] = options.pop(key)

    # ``options`` may still hold SciPy-only keys (xtol, gtol, ...) that IPOPT
    # does not understand -- silently drop them rather than crash the solve.
    return ipopt_opts


def solve_ipopt(
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fun: Callable[[np.ndarray], float],
    jac: Callable[[np.ndarray], np.ndarray],
    options: Optional[Dict] = None,
    *,
    print_level: int = 0,
    quiet: bool = True,
) -> SimpleNamespace:
    """Minimize a black-box objective over box bounds with IPOPT (via CasADi).

    Parameters
    ----------
    x0, lb, ub : np.ndarray
        Initial guess and box bounds, each shape ``(n,)`` (the normalized
        parameter vector, so bounds are typically 0/1).
    fun : callable
        ``fun(x) -> float``, the scalar objective (e.g. ``_obj_ad(x, "scalar")``).
    jac : callable
        ``jac(x) -> np.ndarray`` of shape ``(n,)``, the objective gradient
        (e.g. ``_jac_ad(x, "scalar")``).
    options : dict, optional
        twin4build/SciPy-style options; see :func:`_map_options`.
    print_level : int
        Default IPOPT ``print_level`` when not overridden via ``options``.
    quiet : bool
        Suppress IPOPT's per-iteration wall-clock banner (``print_time``).

    Returns
    -------
    types.SimpleNamespace
        SciPy-``OptimizeResult``-compatible: ``x``, ``fun``, ``success``,
        ``nit``, ``message`` (plus ``status`` = raw IPOPT return string).  This
        lets the Estimator's existing result-building tail consume it unchanged.
    """
    ca = _require_casadi()

    x0 = np.asarray(x0, dtype=np.float64).flatten()
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    n = x0.size

    class _GradCB(ca.Callback):
        """Gradient of the scalar objective: returns a (1, n) row Jacobian."""

        def __init__(self, name, opts={}):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        def get_n_in(self):
            return 2  # (x, nominal f) -- CasADi's Jacobian calling convention

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1) if i == 0 else ca.Sparsity.dense(1, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(1, n)

        def eval(self, arg):
            x = np.asarray(arg[0]).flatten()
            g = np.asarray(jac(x), dtype=np.float64).flatten()
            return [ca.DM(g).T]

    class _ObjCB(ca.Callback):
        """Scalar objective ``f(x)`` with an analytic gradient via ``_GradCB``."""

        def __init__(self, name, opts={}):
            ca.Callback.__init__(self)
            # Keep a strong ref to the gradient callback: CasADi holds only a
            # weak link, so it must outlive the solver call to avoid a segfault.
            self._grad_cb = None
            self.construct(name, opts)

        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(1, 1)

        def has_jacobian(self):
            return True

        def get_jacobian(self, name, inames, onames, opts):
            self._grad_cb = _GradCB(name, opts)
            return self._grad_cb

        def eval(self, arg):
            x = np.asarray(arg[0]).flatten()
            return [float(fun(x))]

    obj_cb = _ObjCB("t4b_estimation_objective")

    X = ca.MX.sym("x", n)
    nlp = {"x": X, "f": obj_cb(X)}

    ipopt_opts = {"hessian_approximation": "limited-memory", "print_level": print_level}
    ipopt_opts.update(_map_options(options))

    solver_opts = {"ipopt": ipopt_opts}
    if quiet:
        solver_opts["print_time"] = False

    solver = ca.nlpsol("t4b_ipopt", "ipopt", nlp, solver_opts)
    sol = solver(x0=x0, lbx=lb, ubx=ub)

    stats = solver.stats()
    return_status = stats.get("return_status", "")
    x_opt = np.asarray(sol["x"]).flatten()
    return SimpleNamespace(
        x=x_opt,
        fun=float(sol["f"]),
        success=bool(stats.get("success", False)),
        nit=stats.get("iter_count", None),
        message=str(return_status),
        status=str(return_status),
    )


def solve_ipopt_constrained(
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    n_g: int,
    g_fun: Callable[[np.ndarray], np.ndarray],
    g_jac_vals: Callable[[np.ndarray], np.ndarray],
    jac_rows: np.ndarray,
    jac_cols: np.ndarray,
    options: Optional[Dict] = None,
    *,
    print_level: int = 0,
    quiet: bool = True,
) -> SimpleNamespace:
    """Minimize ``fun`` s.t. box bounds and equality constraints ``g(x) = 0``.

    This is the engine for **simultaneous transcription (collocation)**: the
    dynamics enter as hard equality defects ``g`` whose Jacobian is *sparse and
    block-bidiagonal* (each defect row touches only one segment's start state,
    the next segment's start state, and ``theta``).  Supplying that sparsity to
    IPOPT is what lets its sparse linear solver factorize the KKT system in
    ~linear time in the horizon -- the source of collocation's speed.

    Parameters
    ----------
    x0, lb, ub : np.ndarray, shape (n,)
        Initial guess and box bounds on the decision vector.
    fun, grad : callables
        Scalar objective ``fun(x) -> float`` and its gradient ``grad(x) -> (n,)``
        (data-fit only -- continuity is handled by the constraints).
    n_g : int
        Number of equality constraints (defect rows).
    g_fun : callable
        ``g_fun(x) -> (n_g,)`` constraint residuals (driven to 0).
    g_jac_vals : callable
        ``g_jac_vals(x) -> (nnz,)`` -- the constraint-Jacobian nonzeros, aligned
        one-to-one with ``(jac_rows, jac_cols)``.
    jac_rows, jac_cols : np.ndarray, shape (nnz,)
        Row/column indices of the (fixed) constraint-Jacobian sparsity pattern.
    options : dict, optional
        twin4build/SciPy-style options; see :func:`_map_options`.

    Returns
    -------
    types.SimpleNamespace
        SciPy-``OptimizeResult``-like (``x``/``fun``/``success``/``nit``/``message``).
    """
    ca = _require_casadi()

    x0 = np.asarray(x0, dtype=np.float64).flatten()
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    n = x0.size
    jac_rows = np.asarray(jac_rows, dtype=np.int64).flatten()
    jac_cols = np.asarray(jac_cols, dtype=np.int64).flatten()

    # Fixed constraint-Jacobian sparsity, and the permutation from our
    # (jac_rows, jac_cols) ordering to CasADi's canonical nonzero storage order.
    jac_sp = ca.Sparsity.triplet(n_g, n, jac_rows.tolist(), jac_cols.tolist())
    sp_rows, sp_cols = jac_sp.get_triplet()
    pos = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(sp_rows, sp_cols))}
    perm = np.array([pos[(int(r), int(c))] for r, c in zip(jac_rows, jac_cols)], dtype=np.int64)

    class _ObjGradCB(ca.Callback):
        def __init__(self, name, opts={}):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        def get_n_in(self):
            return 2

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1) if i == 0 else ca.Sparsity.dense(1, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(1, n)

        def eval(self, arg):
            x = np.asarray(arg[0]).flatten()
            return [ca.DM(np.asarray(grad(x), dtype=np.float64).reshape(1, n))]

    class _ObjCB(ca.Callback):
        def __init__(self, name, opts={}):
            ca.Callback.__init__(self)
            self._g = None
            self.construct(name, opts)

        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(1, 1)

        def has_jacobian(self):
            return True

        def get_jacobian(self, name, inames, onames, opts):
            self._g = _ObjGradCB(name, opts)
            return self._g

        def eval(self, arg):
            return [float(fun(np.asarray(arg[0]).flatten()))]

    class _GJacCB(ca.Callback):
        def __init__(self, name, opts={}):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        def get_n_in(self):
            return 2

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1) if i == 0 else ca.Sparsity.dense(n_g, 1)

        def get_sparsity_out(self, i):
            return jac_sp  # <-- the block-bidiagonal pattern IPOPT exploits

        def eval(self, arg):
            x = np.asarray(arg[0]).flatten()
            vals = np.asarray(g_jac_vals(x), dtype=np.float64).flatten()
            reordered = np.empty_like(vals)
            reordered[perm] = vals
            return [ca.DM(jac_sp, reordered)]

    class _GCB(ca.Callback):
        def __init__(self, name, opts={}):
            ca.Callback.__init__(self)
            self._jac = None
            self.construct(name, opts)

        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(n_g, 1)

        def has_jacobian(self):
            return True

        def get_jacobian(self, name, inames, onames, opts):
            self._jac = _GJacCB(name, opts)
            return self._jac

        # Declare the constraint-Jacobian *sparsity pattern* explicitly.  CasADi
        # keeps this separate from the Jacobian evaluation function above: without
        # it CasADi assumes a dense d(g)/dx (n_g x n) and hands IPOPT that pattern
        # (n_g * n nonzeros) -- defeating the whole point of collocation.  With it,
        # IPOPT allocates only the block-bidiagonal structure and its sparse linear
        # solver factorizes the KKT system in ~linear time in the horizon.
        def has_jac_sparsity(self, oind, iind):
            return True

        def get_jac_sparsity(self, oind, iind, symmetric):
            return jac_sp

        def eval(self, arg):
            x = np.asarray(arg[0]).flatten()
            return [ca.DM(np.asarray(g_fun(x), dtype=np.float64).reshape(n_g, 1))]

    obj_cb = _ObjCB("t4b_collocation_objective")
    g_cb = _GCB("t4b_collocation_constraints")

    X = ca.MX.sym("x", n)
    nlp = {"x": X, "f": obj_cb(X), "g": g_cb(X)}

    ipopt_opts = {"hessian_approximation": "limited-memory", "print_level": print_level}
    ipopt_opts.update(_map_options(options))
    solver_opts = {"ipopt": ipopt_opts}
    if quiet:
        solver_opts["print_time"] = False

    solver = ca.nlpsol("t4b_ipopt_c", "ipopt", nlp, solver_opts)
    sol = solver(x0=x0, lbx=lb, ubx=ub, lbg=np.zeros(n_g), ubg=np.zeros(n_g))

    stats = solver.stats()
    return SimpleNamespace(
        x=np.asarray(sol["x"]).flatten(),
        fun=float(sol["f"]),
        success=bool(stats.get("success", False)),
        nit=stats.get("iter_count", None),
        message=str(stats.get("return_status", "")),
        status=str(stats.get("return_status", "")),
    )
