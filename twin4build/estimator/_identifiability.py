"""Practical identifiability of an estimate from the residual Jacobian.

At the optimum the scaled residual vector ``r(theta)`` and its Jacobian
``J = dr/dtheta`` (``theta`` in the solver's normalized box coordinates,
so one unit is the parameter's whole admissible range) say how well the
data pin every parameter *locally*:

* a column of ``J`` that is (numerically) zero is a parameter the
  residuals do not react to at all -- the solver had no information about
  it and left it wherever it started;
* a singular value of ``J`` that is tiny relative to the largest names a
  direction in parameter space along which the residuals do not change:
  the parameters loading on that singular vector can only be pinned as a
  combination (``V/G_occ`` style trade-offs);
* the Gauss-Newton covariance ``sigma^2 (J^T J)^-1`` gives every parameter a
  standard error and every pair a correlation; a correlation near +-1 is a
  practical trade-off even when no single direction is exactly flat;
* a parameter sitting on a bound is pinned by the bound, not by the data.

This is the classic local (Fisher-information) analysis -- Brun, Reichert
and Kuensch (2001), Raue et al. (2009) for the profile-likelihood view it
approximates.  It is cheap (one forward-mode Jacobian) and catches the
degeneracies an estimator otherwise hides by simply picking a point on the
flat valley.  It cannot see non-local structure (two separate optima); a
multi-start or a profile likelihood is needed for that.

:func:`analyze` is pure NumPy so it can be unit-tested on hand-made
Jacobians; :meth:`Estimator._log_identifiability` feeds it the Jacobian of
the fitted model and logs the findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Relative column norm below which a parameter is "dead" (no residual
#: reacts to it) -- same threshold as the initial Jacobian diagnostic.
DEAD_REL_NORM = 1e-6
#: Relative column norm below which a parameter is "weak".
WEAK_REL_NORM = 1e-3
#: Singular value (relative to the largest) below which the singular
#: direction counts as flat.
FLAT_REL_SINGULAR = 1e-3
#: Loading (|v_i| of a unit singular vector) above which a parameter is
#: reported as part of a flat direction.
FLAT_LOADING = 0.3
#: |correlation| above which a pair is reported as a trade-off.
TRADEOFF_CORR = 0.95
#: Distance from a bound (in normalized units, i.e. fraction of the range)
#: below which a parameter is "at bound".
AT_BOUND_TOL = 1e-3


@dataclass
class IdentifiabilityReport:
    """Findings of :func:`analyze`; every index refers to ``names``."""

    names: List[str]
    #: ``||J[:, i]||`` relative to the largest column.
    rel_col_norm: np.ndarray
    dead: List[int] = field(default_factory=list)
    weak: List[int] = field(default_factory=list)
    at_lower: List[int] = field(default_factory=list)
    at_upper: List[int] = field(default_factory=list)
    #: singular values of the unit-column-scaled Jacobian, descending.
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: flat directions: ``(rel_singular_value, [(index, loading), ...])``
    flat_directions: List[Tuple[float, List[Tuple[int, float]]]] = field(default_factory=list)
    #: pairs ``(i, j, corr)`` with ``|corr| >= TRADEOFF_CORR``.
    tradeoffs: List[Tuple[int, int, float]] = field(default_factory=list)
    #: Gauss-Newton standard error per parameter in normalized units (a
    #: fraction of the admissible range); ``nan`` for dead parameters.
    std_err: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: parameters whose standard error exceeds their whole range.
    unpinned: List[int] = field(default_factory=list)
    condition_number: float = float("nan")
    n_residuals: int = 0

    @property
    def clean(self) -> bool:
        """No dead, flat, traded-off or unpinned parameter."""
        return not (self.dead or self.flat_directions or self.tradeoffs or self.unpinned)

    def lines(self) -> List[str]:
        """Human-readable findings, one per line (empty when clean)."""
        out: List[str] = []
        n = lambda i: self.names[i]  # noqa: E731
        for i in self.dead:
            out.append(f"{n(i)}: no residual reacts to it (|J| rel {self.rel_col_norm[i]:.1e}) -- not identifiable")
        for i in self.weak:
            out.append(f"{n(i)}: weak sensitivity (|J| rel {self.rel_col_norm[i]:.1e})")
        for rel_sv, loads in self.flat_directions:
            if len(loads) == 1:
                out.append(f"flat direction (sigma rel {rel_sv:.1e}): {n(loads[0][0])} barely moves any residual")
                continue
            combo = " ".join(f"{load:+.2f}*{n(i)}" for i, load in loads)
            out.append(f"flat direction (sigma rel {rel_sv:.1e}): only the combination {combo} is determined")
        for i, j, c in self.tradeoffs:
            out.append(f"trade-off {n(i)} <-> {n(j)}: correlation {c:+.3f}")
        for i in self.unpinned:
            out.append(f"{n(i)}: standard error {self.std_err[i]:.2g} x range -- not pinned by the data")
        for i in self.at_lower:
            out.append(f"{n(i)}: at lower bound")
        for i in self.at_upper:
            out.append(f"{n(i)}: at upper bound")
        return out


def analyze(
    jac: np.ndarray,
    residual: Optional[np.ndarray],
    names: Sequence[str],
    x: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
) -> IdentifiabilityReport:
    """Local identifiability analysis of ``jac`` (``n_res x n_theta``).

    ``residual`` (``n_res``) sets the noise level for the standard errors
    (``sigma^2 = r^T r / (n_res - n_theta)``); ``None`` skips them.  ``x``,
    ``lb``, ``ub`` (normalized) flag parameters at a bound.  Uniform
    rescaling of ``jac`` and ``residual`` does not change any finding.
    """
    J = np.asarray(jac, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(f"jac must be 2-D (n_res, n_theta), got shape {J.shape}")
    n_res, p = J.shape
    names = list(names)
    if len(names) != p:
        raise ValueError(f"{len(names)} names for {p} Jacobian columns")
    report = IdentifiabilityReport(names=names, rel_col_norm=np.zeros(p), n_residuals=n_res)
    report.std_err = np.full(p, np.nan)
    if p == 0 or n_res == 0:
        return report

    col_norm = np.linalg.norm(J, axis=0)
    max_norm = float(col_norm.max())
    if max_norm <= 0.0 or not np.isfinite(max_norm):
        report.rel_col_norm = np.zeros(p)
        report.dead = list(range(p))
        return report
    rel = col_norm / max_norm
    report.rel_col_norm = rel
    report.dead = [int(i) for i in np.flatnonzero(rel < DEAD_REL_NORM)]
    report.weak = [int(i) for i in np.flatnonzero((rel >= DEAD_REL_NORM) & (rel < WEAK_REL_NORM))]
    live = np.flatnonzero(rel >= DEAD_REL_NORM)

    if x is not None and lb is not None and ub is not None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        rng = np.where(ub - lb > 0, ub - lb, 1.0)
        report.at_lower = [int(i) for i in np.flatnonzero((x - lb) / rng < AT_BOUND_TOL)]
        report.at_upper = [int(i) for i in np.flatnonzero((ub - x) / rng < AT_BOUND_TOL)]

    if live.size == 0:
        return report

    # -- flat directions: SVD of the unit-column Jacobian (scale-free) ------
    Jl = J[:, live] / col_norm[live]
    try:
        _, s, vt = np.linalg.svd(Jl, full_matrices=False)
    except np.linalg.LinAlgError:
        return report
    report.singular_values = s
    s_max = float(s[0]) if s.size else 0.0
    if s_max > 0:
        s_min = float(s[-1]) if s.size else 0.0
        report.condition_number = s_max / s_min if s_min > 0 else float("inf")
        # A rank deficit (fewer live columns than rows is fine; fewer rows
        # than columns leaves p - n_res exactly flat directions).
        for k in range(s.size):
            rel_sv = float(s[k]) / s_max
            if rel_sv >= FLAT_REL_SINGULAR:
                continue
            v = vt[k]
            loads = [(int(live[i]), float(v[i])) for i in np.flatnonzero(np.abs(v) >= FLAT_LOADING)]
            if not loads:
                # spread thin over many parameters: report the top three
                top = np.argsort(-np.abs(v))[:3]
                loads = [(int(live[i]), float(v[i])) for i in top]
            loads.sort(key=lambda t: -abs(t[1]))
            report.flat_directions.append((rel_sv, loads))

    # -- covariance: correlations and standard errors -----------------------
    JtJ = Jl.T @ Jl
    try:
        cov_unit = np.linalg.pinv(JtJ, rcond=FLAT_REL_SINGULAR**2)
    except np.linalg.LinAlgError:
        return report
    d = np.sqrt(np.clip(np.diag(cov_unit), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov_unit / np.outer(d, d)
    for a in range(live.size):
        for b in range(a + 1, live.size):
            c = corr[a, b]
            if np.isfinite(c) and abs(c) >= TRADEOFF_CORR:
                report.tradeoffs.append((int(live[a]), int(live[b]), float(c)))
    report.tradeoffs.sort(key=lambda t: -abs(t[2]))

    if residual is not None:
        r = np.asarray(residual, dtype=np.float64).reshape(-1)
        dof = max(n_res - int(live.size), 1)
        sigma2 = float(r @ r) / dof
        # cov(theta_live) = sigma^2 (J^T J)^-1 with J in normalized units:
        # undo the unit-column scaling.
        se_unit = np.sqrt(np.clip(np.diag(cov_unit) * sigma2, 0.0, None))
        se = se_unit / col_norm[live]
        report.std_err[live] = se
        report.unpinned = [int(live[i]) for i in np.flatnonzero(se > 1.0)]
    return report
