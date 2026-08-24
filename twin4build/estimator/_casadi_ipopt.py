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

import warnings as _warnings
import numpy as np

from twin4build.utils.logger import LOGGER


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
                     "acceptable_iter", "constr_viol_tol",
                     "acceptable_constr_viol_tol", "acceptable_dual_inf_tol",
                     "acceptable_compl_inf_tol", "acceptable_obj_change_tol",
                     "mu_strategy", "linear_solver", "hessian_approximation"):
            ipopt_opts[key] = options.pop(key)

    # ``options`` may still hold SciPy-only keys (xtol, gtol, ...) that IPOPT
    # does not understand.  Drop them rather than crash the solve -- callers
    # legitimately pass one options dict to several backends -- but say so.
    #
    # Dropping them SILENTLY is how a misspelled or not-yet-supported option
    # becomes an invisible change of experiment: the caller sets it, nothing
    # rejects it, the default applies instead, and the run looks successful.
    # That is exactly how a stale install ignores ``boundary_state_init`` and
    # silently seeds boundary states from data instead of the warm start.
    # Deliberately ``warnings.warn`` and not ``LOGGER.warning``: LOGGER output
    # is suppressed by default (verified -- a LOGGER.config line in the
    # collocation setup produces no output in a normal run), so logging this
    # would leave the failure exactly as invisible as it is today.  A
    # UserWarning surfaces in notebooks and in pytest.
    if options:
        _warnings.warn(
            f"IPOPT backend ignoring {len(options)} unrecognized option(s): "
            f"{', '.join(sorted(map(str, options)))}. These have NO effect -- "
            "the corresponding defaults apply instead. Check the spelling, and "
            "that the installed twin4build supports the option "
            f"({__file__}).",
            UserWarning,
            stacklevel=3,
        )
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
    hess_vals: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    hess_rows: Optional[np.ndarray] = None,
    hess_cols: Optional[np.ndarray] = None,
    early_stopping: Optional[Dict] = None,
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
    hess_vals : callable, optional
        ``hess_vals(x, sigma) -> (nnz_h,)`` or, to build the EXACT Hessian,
        ``hess_vals(x, sigma, lam_g) -> (nnz_h,)`` (the arity is probed) --
        the nonzeros of the
        Hessian of the Lagrangian ``sigma * d2f + sum(lam_g * d2g)``, aligned
        with ``(hess_rows, hess_cols)``.  When given, IPOPT runs with this
        second-order information (e.g. a Gauss-Newton ``J^T W J``) instead of
        the limited-memory BFGS approximation, typically cutting the iteration
        count by an order of magnitude.  The provider may ignore ``lam_g``
        (constraint curvature), as Gauss-Newton does.
    hess_rows, hess_cols : np.ndarray, optional
        Row/column indices of the fixed Hessian sparsity pattern.  Must cover
        only the upper triangle (``row <= col``); the matrix is symmetric.
    early_stopping : dict, optional
        Enable DNN-style early stopping via an IPOPT iteration callback, plus
        best-feasible-iterate checkpointing.  Keys (all optional):

        * ``feas_tol`` (default 1e-2): only iterates with ``max|g| <=
          feas_tol`` may update the checkpoint or count as objective progress.
        * ``patience`` (default 10): stop after this many consecutive
          iterations without progress.  Counting starts once a feasible
          incumbent exists and advances on EVERY iteration -- infeasible
          iterates cannot reset the objective counter (an off-manifold f is
          meaningless), they just fail to show progress.
        * ``min_delta_rel`` (default 1e-3): relative objective decrease (vs the
          best-so-far) that counts as progress.  The deliberately coarse
          default separates the initial descent (per-iteration gains of
          percents) from the plateau creep (noise-level fractions of a
          percent that would otherwise keep resetting the counter for
          hundreds of iterations).
        * ``theta_tol`` (default 1e-4): movement of ``x[:n_theta]`` (inf-norm)
          that counts as progress.
        * ``n_theta`` (default ``n``): length of the leading parameter block
          monitored by the theta-stagnation rule.

        The solve aborts (``User_Requested_Stop``) when the objective rule OR
        the theta rule fires; the returned ``x`` is the best feasible iterate
        seen (also restored when IPOPT ends at max-iter / restoration-failure
        with a worse or infeasible last iterate).

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

    # Optional user-supplied Hessian of the Lagrangian (e.g. Gauss-Newton).
    # CasADi's nlpsol accepts a custom ``hess_lag`` Function with signature
    # (x, p, lam_f, lam_g) -> triu_hess_gamma_x_x; wrapping the Python provider
    # in a Callback on a fixed upper-triangular sparsity lets IPOPT run its
    # full-Newton path instead of limited-memory BFGS.
    hess_cb = None
    if hess_vals is not None:
        # Probe the provider's arity once: (x, sigma) = Gauss-Newton (objective
        # curvature only), (x, sigma, lam_g) = exact Hessian of the Lagrangian.
        # NOTE the probe follows ``__wrapped__``, so a provider wrapped with
        # ``functools.wraps`` is detected correctly -- but a bare decorator
        # (``def w(*a, **kw)``) reports 2 parameters and would be called
        # WITHOUT lam_g, whereupon an exact provider raises inside the CasADi
        # callback and IPOPT bails after a couple of iterations without an
        # obvious error.  Log what was detected so that stays diagnosable.
        try:
            import inspect

            _hess_takes_lam = (
                len(inspect.signature(hess_vals).parameters) >= 3
            )
        except (TypeError, ValueError):  # builtins / C callables
            _hess_takes_lam = False
        LOGGER.config(
            "Hessian provider: %s (arity probe saw %s). Wrap providers with "
            "functools.wraps if you decorate them.",
            "EXACT (receives lam_g)" if _hess_takes_lam
            else "Gauss-Newton (no lam_g)",
            "3+ args" if _hess_takes_lam else "<3 args",
        )
        hess_rows = np.asarray(hess_rows, dtype=np.int64).flatten()
        hess_cols = np.asarray(hess_cols, dtype=np.int64).flatten()
        hess_sp = ca.Sparsity.triplet(n, n, hess_rows.tolist(), hess_cols.tolist())
        h_rows, h_cols = hess_sp.get_triplet()
        h_pos = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(h_rows, h_cols))}
        h_perm = np.array(
            [h_pos[(int(r), int(c))] for r, c in zip(hess_rows, hess_cols)],
            dtype=np.int64,
        )

        class _HessLagCB(ca.Callback):
            def __init__(self, name):
                ca.Callback.__init__(self)
                self.construct(name, {})

            def get_n_in(self):
                return 4

            def get_n_out(self):
                return 1

            def get_name_in(self, i):
                return ["x", "p", "lam_f", "lam_g"][i]

            def get_name_out(self, i):
                return "triu_hess_gamma_x_x"

            def get_sparsity_in(self, i):
                if i == 0:
                    return ca.Sparsity.dense(n, 1)
                if i == 1:
                    return ca.Sparsity(0, 1)  # no NLP parameters
                if i == 2:
                    return ca.Sparsity.dense(1, 1)
                return ca.Sparsity.dense(n_g, 1)

            def get_sparsity_out(self, i):
                return hess_sp

            def eval(self, arg):
                x = np.asarray(arg[0]).flatten()
                sigma = float(arg[2])
                # ``lam_g`` carries the constraint multipliers.  A provider that
                # returns the EXACT Hessian of the Lagrangian needs them for the
                # ``sum(lam_g * d2g)`` term; a Gauss-Newton provider does not.
                # Both are supported -- the arity is probed once, so existing
                # two-argument providers keep working.
                if _hess_takes_lam:
                    lam_g = np.asarray(arg[3]).flatten()
                    vals = np.asarray(
                        hess_vals(x, sigma, lam_g), dtype=np.float64
                    ).flatten()
                else:
                    vals = np.asarray(hess_vals(x, sigma), dtype=np.float64).flatten()
                reordered = np.empty_like(vals)
                reordered[h_perm] = vals
                return [ca.DM(hess_sp, reordered)]

        hess_cb = _HessLagCB("nlp_hess_l")

    # Optional early stopping: an IPOPT iteration callback implementing the
    # patience rules from DNN training, plus a best-feasible-iterate
    # checkpoint ("restore best weights").  IPOPT hands the callback the full
    # primal-dual iterate each iteration; returning nonzero aborts the solve
    # with ``User_Requested_Stop``.
    iter_cb = None
    es = None
    if early_stopping is not None:
        es = {
            "feas_tol": float(early_stopping.get("feas_tol", 1e-2)),
            "patience": int(early_stopping.get("patience", 10)),
            "min_delta_rel": float(early_stopping.get("min_delta_rel", 1e-3)),
            "theta_tol": float(early_stopping.get("theta_tol", 1e-4)),
            "n_theta": int(early_stopping.get("n_theta", n)),
            # state
            "f_best": np.inf, "z_best": None,
            "stall_f": 0, "stall_theta": 0, "stop_reason": None,
        }

        # Seed the checkpoint with the WARM START itself, before IPOPT runs.
        # The iteration callback only ever sees post-``bound_push`` iterates --
        # IPOPT moves every variable strictly inside its bounds before the
        # first callback -- so a warm start sitting ON its bounds (which a
        # converged SLSQP optimum typically does) is never a candidate for
        # "best feasible iterate", and the solve can return something worse
        # than what it was handed.  Recording x0 here makes the warm start a
        # floor: with the restore below, a collocation refinement can improve
        # on its input or leave it alone, but never degrade it.
        if n_g:
            _g0 = np.asarray(g_fun(np.asarray(x0, dtype=np.float64))).flatten()
            _viol0 = float(np.abs(_g0).max()) if _g0.size else 0.0
        else:
            _viol0 = 0.0
        if _viol0 <= es["feas_tol"]:
            es["z_best"] = np.asarray(x0, dtype=np.float64).copy()
            es["f_best"] = float(fun(np.asarray(x0, dtype=np.float64)))
            LOGGER.config(
                "Early stopping: warm start checkpointed as the incumbent "
                "(f=%.6g, max|g|=%.3e) -- the solve cannot return worse.",
                es["f_best"], _viol0,
            )
        else:
            LOGGER.config(
                "Early stopping: warm start is infeasible (max|g|=%.3e > "
                "feas_tol=%.3e); no incumbent until the first feasible iterate.",
                _viol0, es["feas_tol"],
            )

        class _IterCB(ca.Callback):
            def __init__(self, name):
                ca.Callback.__init__(self)
                self.construct(name, {})

            def get_n_in(self):
                return ca.nlpsol_n_out()

            def get_n_out(self):
                return 1

            def get_name_in(self, i):
                return ca.nlpsol_out(i)

            def get_name_out(self, i):
                return "ret"

            def get_sparsity_in(self, i):
                name = ca.nlpsol_out(i)
                if name == "f":
                    return ca.Sparsity.dense(1, 1)
                if name in ("x", "lam_x"):
                    return ca.Sparsity.dense(n, 1)
                if name in ("g", "lam_g"):
                    return ca.Sparsity.dense(n_g, 1)
                return ca.Sparsity(0, 0)

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(1, 1)

            def eval(self, arg):
                x = np.asarray(arg[0]).flatten()
                f = float(arg[1])
                viol = float(np.abs(np.asarray(arg[2])).max()) if n_g else 0.0
                feasible = viol <= es["feas_tol"]
                # Counting starts at the first feasible incumbent: before that
                # (initial infeasibility reduction) the objective is evaluated
                # off the trajectory manifold and is meaningless.
                if es["z_best"] is None:
                    if feasible:
                        es["f_best"] = f
                        es["z_best"] = x.copy()
                    return [0.0]
                nt = es["n_theta"]
                # Progress = a feasible iterate materially better than the
                # incumbent.  Infeasible iterates (restoration excursions, big
                # rejected steps) cannot reset the counter -- if the excursion
                # pays off, the improved feasible landing point resets it.
                improved_f = feasible and f < es["f_best"] - es[
                    "min_delta_rel"
                ] * max(abs(es["f_best"]), 1e-12)
                moved_theta = (
                    float(np.abs(x[:nt] - es["z_best"][:nt]).max()) > es["theta_tol"]
                )
                if feasible and f < es["f_best"]:
                    es["f_best"] = f
                    es["z_best"] = x.copy()
                es["stall_f"] = 0 if improved_f else es["stall_f"] + 1
                es["stall_theta"] = 0 if moved_theta else es["stall_theta"] + 1
                if es["stall_f"] >= es["patience"]:
                    es["stop_reason"] = (
                        f"objective stagnant for {es['stall_f']} iterations"
                    )
                    return [1.0]
                if es["stall_theta"] >= es["patience"]:
                    es["stop_reason"] = (
                        f"theta stagnant for {es['stall_theta']} iterations"
                    )
                    return [1.0]
                return [0.0]

        iter_cb = _IterCB("t4b_early_stop")

    obj_cb = _ObjCB("t4b_collocation_objective")
    g_cb = _GCB("t4b_collocation_constraints")

    X = ca.MX.sym("x", n)
    nlp = {"x": X, "f": obj_cb(X), "g": g_cb(X)}

    ipopt_opts = {
        # With a user Hessian IPOPT runs its exact-Newton path; otherwise
        # fall back to limited-memory BFGS (no second-order info available).
        "hessian_approximation": "exact" if hess_cb is not None else "limited-memory",
        "print_level": print_level,
        # The equality constraints are the DYNAMICS: any violation is a
        # non-physical forcing injected into the trajectory, and the returned
        # "solution" is then not a trajectory of the model at all.  IPOPT's
        # defaults (constr_viol_tol=1e-4, acceptable_constr_viol_tol=1e-2)
        # allow per-step slack that compounds over hundreds of steps, so
        # tighten both; callers can still override via ``options``.
        "constr_viol_tol": 1e-8,
        "acceptable_constr_viol_tol": 1e-8,
    }
    ipopt_opts.update(_map_options(options))
    solver_opts = {"ipopt": ipopt_opts}
    if hess_cb is not None:
        solver_opts["hess_lag"] = hess_cb
    if iter_cb is not None:
        solver_opts["iteration_callback"] = iter_cb
        solver_opts["iteration_callback_step"] = 1
    if quiet:
        solver_opts["print_time"] = False

    solver = ca.nlpsol("t4b_ipopt_c", "ipopt", nlp, solver_opts)
    sol = solver(x0=x0, lbx=lb, ubx=ub, lbg=np.zeros(n_g), ubg=np.zeros(n_g))

    stats = solver.stats()
    return_status = str(stats.get("return_status", ""))
    message = return_status
    x_opt = np.asarray(sol["x"]).flatten()
    f_opt = float(sol["f"])
    success = bool(stats.get("success", False))
    if es is not None:
        if es["stop_reason"]:
            message = f"{return_status} ({es['stop_reason']})"
            success = True  # a deliberate, feasible stop -- not a failure
        # Restore the best feasible iterate when IPOPT's last iterate is worse
        # or infeasible (max-iter, user-stop and restoration exits all return
        # whatever the final iterate happened to be).
        if es["z_best"] is not None:
            sol_viol = (
                float(np.abs(np.asarray(sol["g"])).max()) if n_g else 0.0
            )
            if sol_viol > es["feas_tol"] or es["f_best"] < f_opt:
                x_opt = es["z_best"]
                f_opt = es["f_best"]
                message += " [best feasible iterate restored]"
    return SimpleNamespace(
        x=x_opt,
        fun=f_opt,
        success=success,
        nit=stats.get("iter_count", None),
        message=message,
        status=return_status,
    )
