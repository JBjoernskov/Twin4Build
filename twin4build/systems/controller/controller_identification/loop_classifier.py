"""Data-driven scoring of (sensor, setpoint, actuator) PI loops.

This module implements the pure-NumPy core of the PI-CITS rewire pipeline.
Given an actuator timeseries ``u`` and a candidate ``(sensor, setpoint)``
pair, :func:`score_pair` returns a fit quality score and a data-driven
estimate of the proportional gain ``kp`` (with sign) and integral time
``Ti``, derived from a single multivariate ordinary-least-squares
regression in the *first-difference* domain.

Mathematical rationale
----------------------

A discrete PI controller satisfies::

    u[k] = kp * e[k] + (kp / Ti) * sum_{i<=k}(h * e[i]) + offset

Differencing once removes the cumulative integral and the constant offset::

    Δu[k] = kp * Δe[k] + (kp * h / Ti) * e[k]                          (1)

Equation (1) is a *clean* linear regression of ``Δu`` on the two regressors
``Δe`` and ``e``: the coefficient on ``Δe`` is ``kp`` exactly (no
discretization bias), and the coefficient on ``e`` is ``kp*h/Ti`` from
which ``Ti`` is recovered as ``|kp * h / coef_e|``.

Crucially, this regression is *consistent* under stationarity even when the
true ``Ti`` is unknown.  If we omit the ``e`` regressor we get the simple
``cov(Δu,Δe)/var(Δe)`` slope, which is biased multiplicatively by
``(1 + h/(2*Ti))`` -- still small for ``Ti >> h``, but the joint regression
removes that bias entirely.

The R^2 of the joint fit is the natural classification statistic for
deciding which candidate ``(sensor, setpoint)`` pair is the actual feedback
loop driving ``u``: the right pair has high R^2 (often > 0.5 on clean HVAC
data); wrong pairs score near zero.
"""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Third party imports
import numpy as np


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LoopScore:
    """Result of scoring a single ``(sensor, setpoint, actuator)`` pair.

    Attributes:
        slope: Signed slope ``kp`` from the joint regression on ``Δe``.
            Positive means a direct-acting loop (rising error opens the
            actuator further); negative means reverse-acting.
        kp: ``|slope|``.  Always non-negative.
        Ti: Integral time recovered from the joint coefficient on ``e``.
            ``None`` when the integral coefficient is too small to identify
            (purely-P loop, no significant accumulated error).
        r2: Coefficient of determination of the joint regression, in
            ``[-inf, 1]``.  Used as the classification statistic.
        n_active: Number of samples used after masking saturated and
            non-finite entries.
        reason: Human-readable failure mode when the score is unreliable
            (``"too_few_active_samples"``, ``"constant_setpoint"``,
            ``"degenerate_design"``).  ``None`` on a successful fit.
    """

    slope: float
    kp: float
    Ti: Optional[float]
    r2: float
    n_active: int
    reason: Optional[str] = None


@dataclass
class ActuatorSeeds:
    """Data-driven seeds derived from the actuator timeseries alone.

    These do not depend on any sensor/setpoint pairing; they describe the
    operating envelope of the actuator and are safe to apply even when the
    classifier fails.
    """

    output_min_x0: float
    output_max_x0: float
    default_output_x0: float


@dataclass
class ActuatorBimodalSeeds:
    """GMM-based actuator seeds plus a sample-level on/off mask.

    Decomposes the actuator histogram into two Gaussian components and
    identifies the component whose mean is nearest a saturation bound
    (``0`` or ``1``) as the *off mode*; the other component is the
    *active mode* over which the controller actually modulates.

    The on/off mask returned here is the empirical analog of the
    BandGate's ground-truth activation: it labels every sample as either
    ``parked at default_output`` (off mode) or ``inside the modulating
    range`` (active mode).  Downstream seeders can use it to:

    1. compute ``output_min`` / ``output_max`` from active samples only
       (instead of full-trajectory percentiles which are dragged toward
       the saturation bound by the off-mode mass);
    2. compute ``default_output`` from off samples (or from ``kind`` when
       the off mode is hard against a bound);
    3. seed the gate by classifying which ``onOffSignal`` slot best
       predicts ``on_mask`` (a 1D classification, not a joint nonlinear
       fit);
    4. run the kp/Ti regression on ``on_mask`` only, removing the bias
       that setback samples (where ``Δu ~ 0`` regardless of ``Δe``)
       inject into the slope estimate.

    Attributes:
        output_min_x0, output_max_x0: Active-mode quantiles, clamped to
            ``[0, 1]``.  For ``kind == "always_on"`` (no off mode found)
            falls back to full-trajectory percentiles.
        default_output_x0: Value the actuator parks at when the gate is
            off.  ``1.0`` for ``kind == "damper"``, ``0.0`` for
            ``kind == "reheat"``, the empirical off-mode mean for
            ``"ambiguous"`` / ``"always_on"``.
        kind: One of ``"damper"`` (off mode at the upper bound),
            ``"reheat"`` (off mode at the lower bound),
            ``"always_on"`` (no cluster near a bound -- modulating the
            entire time), ``"always_off"`` (no active cluster found --
            stuck at a bound), or ``"ambiguous"`` (GMM converged but
            neither cluster is near a bound and the spread is small).
        on_mask: Boolean array of shape ``(N,)`` aligned with ``u``.
            ``True`` where the sample's posterior probability of the
            active mode exceeds 0.5 (or everywhere ``True`` for
            ``kind == "always_on"``).
        bimodality: Score in ``[0, ~1]``.  Roughly the normalised
            separation between cluster means relative to the pooled
            standard deviation.  Values above ``~0.5`` indicate the
            two-component decomposition is well-supported by the data.
        off_mean, active_mean: GMM cluster means for the off and active
            components.
        off_weight, active_weight: GMM mixture weights summing to ``1``.
        reason: Failure mode for ``"ambiguous"`` / ``"always_off"``
            outcomes; ``None`` on a confident decomposition.
    """

    output_min_x0: float
    output_max_x0: float
    default_output_x0: float
    kind: str
    on_mask: np.ndarray
    bimodality: float
    off_mean: float
    active_mean: float
    off_weight: float
    active_weight: float
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------


def score_pair(
    u: np.ndarray,
    sp: np.ndarray,
    fb: np.ndarray,
    h: float,
    *,
    n_min: int = 50,
    sat_lo: float = 0.03,
    sat_hi: float = 0.97,
    on_mask: Optional[np.ndarray] = None,
) -> LoopScore:
    """Score one ``(actuator, setpoint, feedback_sensor)`` pair.

    Performs the joint regression ``Δu ~ Δe + e_mid + 1`` on the masked,
    differenced signals and returns a :class:`LoopScore`.

    Args:
        u: Actuator timeseries, shape ``(N,)``.  Assumed to be in
            ``[0, 1]`` (saturating actuator).
        sp: Setpoint timeseries, shape ``(N,)``.  Same units as ``fb``.
        fb: Feedback (measurement) timeseries, shape ``(N,)``.
        h: Sample step in seconds.
        n_min: Minimum number of samples after masking.  Below this the
            score is rejected with reason ``"too_few_active_samples"``.
        sat_lo, sat_hi: Saturation bounds on ``u``; samples with ``u``
            outside ``(sat_lo, sat_hi)`` are excluded from the fit because
            the actuator is at a clipping bound and contributes no
            information about the controller gain.
        on_mask: Optional boolean array of shape ``(N,)`` aligned with
            ``u``.  When provided, samples where ``on_mask`` is ``False``
            are excluded from the regression in addition to the
            saturation filter.  Use the GMM-derived
            :attr:`ActuatorBimodalSeeds.on_mask` to restrict the fit to
            samples where the PI loop is actually active; without this,
            setback / off samples (where ``Δu ~ 0`` regardless of
            ``Δe``) systematically pull the slope estimate toward zero
            and the integrator coefficient toward
            ``e_mid``-correlated noise, biasing ``kp`` low and ``Ti``
            high (often pinning them at the global floor / ceiling
            even when ``r2`` is high -- the regression "explains" the
            data with a degenerate near-rank-deficient fit).

    Returns:
        :class:`LoopScore`.  ``r2`` is set to 0 (and ``reason`` populated)
        when the regression is degenerate.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    sp = np.asarray(sp, dtype=np.float64).reshape(-1)
    fb = np.asarray(fb, dtype=np.float64).reshape(-1)
    n = min(len(u), len(sp), len(fb))
    u, sp, fb = u[:n], sp[:n], fb[:n]

    e = sp - fb
    sample_ok = (
        np.isfinite(u)
        & np.isfinite(e)
        & (u > sat_lo)
        & (u < sat_hi)
    )
    if on_mask is not None:
        on_mask = np.asarray(on_mask, dtype=bool).reshape(-1)
        m = min(on_mask.size, n)
        # Pad with False if the on_mask is shorter than the regressor
        # signals (defensive: the GMM is computed on the actuator series
        # which is length-aligned with ``u``, so padding should never
        # actually trigger in practice).
        if on_mask.size < n:
            padded = np.zeros(n, dtype=bool)
            padded[: on_mask.size] = on_mask
            on_mask = padded
        else:
            on_mask = on_mask[:n]
        sample_ok = sample_ok & on_mask
    # Differences are evaluated on the *original* (non-gappy) signal so each
    # ``du[k] = u[k+1] - u[k]`` is between truly consecutive timesteps.  We
    # then drop diff samples where either endpoint is masked out: this keeps
    # the regression on diffs that are physically meaningful instead of
    # jumping across saturated/missing gaps.
    du = np.diff(u)
    de = np.diff(e)
    e_mid = 0.5 * (e[:-1] + e[1:])
    pair_ok = sample_ok[:-1] & sample_ok[1:]
    n_active = int(pair_ok.sum())
    if n_active < n_min:
        return LoopScore(
            slope=float("nan"),
            kp=float("nan"),
            Ti=None,
            r2=0.0,
            n_active=n_active,
            reason="too_few_active_samples",
        )

    du = du[pair_ok]
    de = de[pair_ok]
    e_mid = e_mid[pair_ok]

    # Reject if the regressor on Δe has no variation (constant setpoint &
    # constant feedback => no information about kp).
    if np.var(de) < 1e-12:
        return LoopScore(
            slope=0.0,
            kp=0.0,
            Ti=None,
            r2=0.0,
            n_active=n_active,
            reason="constant_setpoint",
        )

    X = np.column_stack([de, e_mid, np.ones_like(de)])
    try:
        beta, *_ = np.linalg.lstsq(X, du, rcond=None)
    except np.linalg.LinAlgError:
        return LoopScore(
            slope=0.0,
            kp=0.0,
            Ti=None,
            r2=0.0,
            n_active=n_active,
            reason="degenerate_design",
        )

    slope = float(beta[0])
    coef_e = float(beta[1])

    pred = X @ beta
    ss_tot = float(np.sum((du - du.mean()) ** 2))
    ss_res = float(np.sum((du - pred) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Recover Ti from coef_e = kp * h / Ti.  Guard against tiny coefficients
    # that drive Ti -> infinity (purely-P loop) or sign flips that would
    # produce a negative Ti.
    if abs(slope) < 1e-12 or abs(coef_e) < 1e-12:
        Ti: Optional[float] = None
    else:
        Ti_raw = abs(slope * h / coef_e)
        # Reject pathological estimates (Ti near zero or absurdly large).
        if not np.isfinite(Ti_raw) or Ti_raw <= 0.0:
            Ti = None
        else:
            Ti = Ti_raw

    return LoopScore(
        slope=slope,
        kp=abs(slope),
        Ti=Ti,
        r2=float(r2),
        n_active=n_active,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Actuator-only seed derivation (no pairing required)
# ---------------------------------------------------------------------------


def derive_actuator_seeds(
    u: np.ndarray,
    *,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    cushion: float = 0.02,
) -> ActuatorSeeds:
    """Derive ``output_min``, ``output_max`` and ``default_output`` from ``u``.

    The min/max are robust percentiles (default 1st/99th) padded by
    ``cushion`` to leave the optimizer some wiggle room.  ``default_output``
    is the median actuator value over the *idle* part of the trajectory --
    samples where ``|Δu|`` is in the bottom quartile, i.e. the actuator is
    parked.  This typically resolves to ~0 for reheat valves and ~1 for
    dampers without any kind-specific hard-coding.

    Note on bimodal data
    --------------------
    Highly bimodal trajectories (e.g. a damper that sits at 0 most of the
    time and modulates between [output_min_floor, 1] otherwise) are
    inherently hard to seed for ``output_min`` without overlapping with
    transient/edge samples.  We deliberately keep the simple full-trajectory
    percentile here -- attempting to filter to an "active branch" tended to
    pick up sigmoid bandgate transition samples and pull ``output_min``
    *below* the true saturation floor, leaving the optimizer with a worse
    starting basin.  The 1st-percentile-with-cushion is a conservative seed
    that the optimizer can climb away from when needed.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    finite = np.isfinite(u)
    if not finite.any():
        return ActuatorSeeds(
            output_min_x0=0.0,
            output_max_x0=1.0,
            default_output_x0=0.5,
        )

    u_f = u[finite]
    output_min_x0 = max(0.0, float(np.percentile(u_f, p_lo)) - cushion)
    output_max_x0 = min(1.0, float(np.percentile(u_f, p_hi)) + cushion)

    # Idle: samples where |Δu| is in the bottom quartile.  Falls back to
    # global median when the trajectory is too short to differentiate.
    if len(u_f) >= 4:
        du = np.abs(np.diff(u_f))
        thr = float(np.percentile(du, 25))
        idle = du <= thr  # length len(u_f)-1, aligned with u_f[1:]
        if idle.any():
            default_output_x0 = float(np.median(u_f[1:][idle]))
        else:
            default_output_x0 = float(np.median(u_f))
    else:
        default_output_x0 = float(np.median(u_f))

    # Clamp default_output to [0, 1] for safety.
    default_output_x0 = float(np.clip(default_output_x0, 0.0, 1.0))

    return ActuatorSeeds(
        output_min_x0=output_min_x0,
        output_max_x0=output_max_x0,
        default_output_x0=default_output_x0,
    )


# ---------------------------------------------------------------------------
# Bimodal (GMM-based) actuator decomposition
# ---------------------------------------------------------------------------


def _fit_2component_gmm_1d(
    x: np.ndarray,
    *,
    n_iter: int = 100,
    tol: float = 1e-7,
    var_floor: float = 1e-6,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit a 2-component 1D Gaussian mixture by EM.

    Pure NumPy, no SciPy dependency.  Initialises means at the
    25th / 75th percentile of ``x`` to bias the two clusters apart and
    avoid the degenerate "both clusters collapse to the global mean"
    fixed point that random init can fall into.

    Args:
        x: 1D finite samples, shape ``(N,)``.
        n_iter: Maximum EM iterations.
        tol: Convergence tolerance on the log-likelihood.
        var_floor: Minimum allowed component variance.  Prevents a
            cluster collapsing onto a single sample (which yields
            zero variance and degenerate posteriors).

    Returns:
        ``(mu, var, w, resp)`` where:
            - ``mu``: shape ``(2,)``, cluster means sorted ascending.
            - ``var``: shape ``(2,)``, cluster variances in the same
              order.
            - ``w``: shape ``(2,)``, mixture weights summing to 1.
            - ``resp``: shape ``(N, 2)``, posterior probabilities
              :math:`p(k | x_n)`.
        Returns ``None`` when ``x`` has fewer than 4 samples or zero
        variance (single-mode or degenerate input).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return None
    total_var = float(np.var(x))
    if total_var < var_floor:
        return None

    p25 = float(np.percentile(x, 25))
    p75 = float(np.percentile(x, 75))
    if p25 == p75:  # pathological (e.g. >50% mass at single value)
        p25 = float(np.min(x))
        p75 = float(np.max(x))
        if p25 == p75:
            return None
    mu = np.array([p25, p75], dtype=np.float64)
    var = np.full(2, max(total_var / 4.0, var_floor), dtype=np.float64)
    w = np.array([0.5, 0.5], dtype=np.float64)

    log_lik_old = -np.inf
    resp = np.zeros((n, 2), dtype=np.float64)
    for _ in range(n_iter):
        # E-step in log-space for numerical stability.
        # log N(x | mu_k, var_k) = -0.5 log(2 pi var_k) - 0.5 (x-mu_k)^2 / var_k
        log_p = (
            -0.5 * np.log(2.0 * np.pi * var[None, :])
            - 0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]
        )
        log_w_p = np.log(w[None, :] + 1e-300) + log_p
        log_norm = np.logaddexp(log_w_p[:, 0], log_w_p[:, 1])
        log_resp = log_w_p - log_norm[:, None]
        resp = np.exp(log_resp)

        # M-step.
        n_k = resp.sum(axis=0)
        if np.any(n_k < 1e-6):
            break
        mu = (resp * x[:, None]).sum(axis=0) / n_k
        diff_sq = (x[:, None] - mu[None, :]) ** 2
        var = (resp * diff_sq).sum(axis=0) / n_k
        var = np.maximum(var, var_floor)
        w = n_k / float(n)

        log_lik = float(log_norm.sum())
        if abs(log_lik - log_lik_old) < tol:
            break
        log_lik_old = log_lik

    order = np.argsort(mu)
    return mu[order], var[order], w[order], resp[:, order]


def derive_actuator_seeds_gmm(
    u: np.ndarray,
    *,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    cushion: float = 0.02,
    sat_band: float = 0.15,
    bimodality_min: float = 0.5,
    min_active_weight: float = 0.05,
    min_active_samples: int = 30,
) -> ActuatorBimodalSeeds:
    """Compute actuator seeds from a 2-component GMM decomposition.

    Algorithm (per-CITS, pre-gradient):

    1. Drop non-finite samples.
    2. Fit a 2-component 1D GMM.  The fitter is robust to the typical
       HVAC failure modes (constant signals, all-at-bound) -- it returns
       ``None`` and we fall back to the legacy percentile method.
    3. Sort components by mean and identify the *off mode* as the
       cluster whose mean is nearest a saturation bound:

       - off mean ``< sat_band``                  -> ``kind="reheat"``
         (default = 0, valve / heating command parks closed)
       - off mean ``> 1 - sat_band``              -> ``kind="damper"``
         (default = 1, damper parks open)
       - neither cluster within ``sat_band`` of a bound -> ``kind="always_on"``
         (the actuator modulates without ever parking; gate should be
         permanently active downstream).

    4. Compute ``on_mask`` from the active-mode posterior probability.
    5. Seed ``output_min`` / ``output_max`` from active-mode quantiles
       (or full-trajectory if ``kind="always_on"``).
    6. Pick ``default_output_x0`` from the saturation bound dictated by
       ``kind`` (1.0 / 0.0) for clean cases; fall back to ``off_mean``
       for ``"always_on"`` / ``"ambiguous"``.

    The bimodality threshold guards against false-positive decompositions
    on noisy signals: when the inter-cluster separation is below
    ``bimodality_min`` (in normalised pooled-std units) the result is
    flagged ``ambiguous`` and the seeds revert to whole-trajectory
    percentiles.  That keeps this function as a drop-in upgrade -- a
    CITS that doesn't decompose cleanly gets the same seeds the legacy
    extractor would have produced.

    Args:
        u: Actuator timeseries, shape ``(N,)``, expected in ``[0, 1]``.
        p_lo, p_hi: Quantiles inside the active mode used for the
            ``output_min`` / ``output_max`` seeds (default 1st / 99th).
        cushion: Half-width safety margin clipped onto the percentile
            seeds before clamping to ``[0, 1]``.
        sat_band: Distance from a saturation bound (``0`` or ``1``)
            within which a cluster's mean is treated as "parked".
            Defaults to ``0.15`` so a damper that parks at ``~0.85--1.0``
            still classifies as ``kind="damper"``.
        bimodality_min: Minimum normalised cluster separation for a
            confident decomposition.  Below this threshold ``kind`` is
            ``"ambiguous"`` and seeds revert to global percentiles.
        min_active_weight: Minimum mixture weight on the active mode
            (samples-per-cluster fraction).  Below this the actuator
            is considered ``"always_off"``.
        min_active_samples: Minimum number of samples in the active
            mode (after the posterior threshold).  Below this the
            seeds also revert to percentile fallback.

    Returns:
        :class:`ActuatorBimodalSeeds` with seeds + on/off mask + GMM
        diagnostics.  Always returns a finite, well-formed seeds object
        even on degenerate inputs (degenerate -> ``always_off`` /
        ``ambiguous`` with percentile-based fallback values).
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    n_total = u.size
    finite = np.isfinite(u)
    n_finite = int(finite.sum())

    # Default fallback: full-trajectory percentile seeds + an all-True
    # on_mask (so downstream code treats every sample as active).
    fallback = derive_actuator_seeds(u, p_lo=p_lo, p_hi=p_hi, cushion=cushion)

    def _fallback(reason: str, kind: str = "ambiguous") -> ActuatorBimodalSeeds:
        on_mask = np.ones(n_total, dtype=bool)
        return ActuatorBimodalSeeds(
            output_min_x0=fallback.output_min_x0,
            output_max_x0=fallback.output_max_x0,
            default_output_x0=fallback.default_output_x0,
            kind=kind,
            on_mask=on_mask,
            bimodality=0.0,
            off_mean=float("nan"),
            active_mean=float("nan"),
            off_weight=float("nan"),
            active_weight=float("nan"),
            reason=reason,
        )

    if n_finite < 8:
        return _fallback("too_few_finite_samples", kind="ambiguous")

    fit = _fit_2component_gmm_1d(u[finite])
    if fit is None:
        return _fallback("gmm_fit_failed", kind="ambiguous")

    mu, var, w, resp_finite = fit
    sigma = np.sqrt(var)
    # Bimodality: cluster separation in pooled-std units.  Higher = better.
    pooled_sd = float(0.5 * (sigma[0] + sigma[1]) + 1e-12)
    bimodality = float((mu[1] - mu[0]) / pooled_sd)
    bimodality = max(0.0, bimodality)

    # Identify the off cluster as the one whose mean is nearest a bound.
    dist_to_bound_0 = float(min(mu[0], 1.0 - mu[0]))
    dist_to_bound_1 = float(min(mu[1], 1.0 - mu[1]))
    if dist_to_bound_0 <= dist_to_bound_1:
        off_idx, active_idx = 0, 1
    else:
        off_idx, active_idx = 1, 0
    off_mean = float(mu[off_idx])
    active_mean = float(mu[active_idx])
    off_weight = float(w[off_idx])
    active_weight = float(w[active_idx])

    # Decide kind from the off-mode mean's distance to a bound.
    if off_mean < sat_band:
        kind = "reheat"
        default_out = 0.0
    elif off_mean > 1.0 - sat_band:
        kind = "damper"
        default_out = 1.0
    else:
        # Neither cluster is parked at a bound -- the actuator modulates
        # all the time.  Treat as always-on (no gate needed in principle;
        # gate parameters can stay at neutral defaults).
        kind = "always_on"
        default_out = active_mean  # arbitrary; gate is effectively unused

    # Confidence guards.  If the decomposition is poor (low bimodality,
    # active mode tiny), revert to percentile fallback under
    # kind="ambiguous"/"always_off" so downstream code can detect it.
    if bimodality < bimodality_min:
        return _fallback(
            f"low_bimodality (sep={bimodality:.2f} < {bimodality_min})",
            kind="ambiguous",
        )
    if active_weight < min_active_weight:
        return _fallback(
            f"active_mode_too_small (w={active_weight:.3f} < {min_active_weight})",
            kind="always_off",
        )

    # Build the on/off mask aligned with the *original* (possibly
    # NaN-bearing) ``u`` array.  Samples flagged non-finite get
    # on_mask=False -- they couldn't have been used by the EM and shouldn't
    # be used by downstream regressions either.
    on_mask = np.zeros(n_total, dtype=bool)
    active_post = resp_finite[:, active_idx]
    on_mask[finite] = active_post > 0.5

    n_active_samples = int(on_mask.sum())
    if n_active_samples < min_active_samples:
        return _fallback(
            f"too_few_active_samples ({n_active_samples} < {min_active_samples})",
            kind="always_off",
        )

    # Active-mode percentile seeds: take quantiles within on_mask only.
    u_active = u[on_mask]
    output_min_x0 = max(0.0, float(np.percentile(u_active, p_lo)) - cushion)
    output_max_x0 = min(1.0, float(np.percentile(u_active, p_hi)) + cushion)
    if output_max_x0 <= output_min_x0:
        # Active mode collapsed to a single value -- widen symmetrically
        # so the optimizer has finite slack on both sides.
        mid = 0.5 * (output_min_x0 + output_max_x0)
        output_min_x0 = max(0.0, mid - 0.05)
        output_max_x0 = min(1.0, mid + 0.05)

    return ActuatorBimodalSeeds(
        output_min_x0=output_min_x0,
        output_max_x0=output_max_x0,
        default_output_x0=float(np.clip(default_out, 0.0, 1.0)),
        kind=kind,
        on_mask=on_mask,
        bimodality=bimodality,
        off_mean=off_mean,
        active_mean=active_mean,
        off_weight=off_weight,
        active_weight=active_weight,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Gate-input seed derivation from on_mask
# ---------------------------------------------------------------------------


@dataclass
class GateSeeds:
    """Data-driven seeds for the BandGate parameters.

    Built by ranking every ``onOffSignal`` candidate against the
    ``on_mask`` (the GMM's sample-level binary label of "PI is active")
    via ROC AUC, then deriving ``gamma_gate``, ``gate.threshold`` and
    ``gate.band`` from the winning slot.

    Attributes:
        gamma_gate_x0: Per-slot weight, shape ``(n_slots,)``, summing to
            ``1``.  Built as ``softmax(beta * disc_j)`` where
            ``disc_j = max(auc_j, 1 - auc_j)``; near-one-hot when one
            slot strongly predicts ``on_mask``, uniform when no slot
            is informative.
        gate_threshold_x0: Lower edge of the BandGate's active range
            in *normalised* signal units (the CITS forward pass divides
            each onOffSignal slot by its per-slot ``[oo_min, oo_max]``
            range, so this seed is in ``[0, 1]`` for typical
            schedule signals).
        gate_band_x0: Width of the BandGate's active range in
            normalised units; the upper edge is
            ``threshold + band``.
        auc_per_slot: Per-slot ROC AUC of the slot's normalised value
            predicting ``on_mask = True``.  AUC ``> 0.5`` means high
            signal -> active; ``< 0.5`` means low signal -> active
            (anti-polarity); near ``0.5`` means uninformative.
        winner_slot: Index of the slot with the highest discriminative
            score ``max(auc, 1 - auc)``.
        winner_polarity: ``+1`` if high signal predicts ``on_mask=True``
            (typical schedule); ``-1`` if the opposite (e.g. an
            "is_setback" signal high during the off period).
        confidence: ``"high"`` / ``"medium"`` / ``"low"`` based on the
            discriminative score.  ``"low"`` falls back to neutral
            seeds (uniform gamma_gate, ``threshold=0.5``, ``band=1.0``)
            so the optimizer is free to find a basin without
            data-driven bias.
        reason: Failure mode for ``"low"`` confidence outcomes
            (e.g. ``"no_finite_samples"``); ``None`` otherwise.
    """

    gamma_gate_x0: np.ndarray
    gate_threshold_x0: float
    gate_band_x0: float
    auc_per_slot: np.ndarray
    winner_slot: int
    winner_polarity: int
    confidence: str
    reason: Optional[str] = None


def _roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC AUC of a continuous ``score`` for a binary ``label``.

    Pure-NumPy Mann-Whitney U formulation with average-rank tie
    handling.  Returns ``0.5`` (uninformative) when either class is
    empty or all scores tie.

    Args:
        score: Continuous signal, shape ``(N,)``.  Higher score is
            interpreted as evidence for ``label = True``.
        label: Boolean array, same length.

    Returns:
        AUC in ``[0, 1]``.  ``0.5`` is chance; ``1`` means perfect
        ranking; ``0`` means perfectly inverted ranking (the score
        would be a perfect predictor with sign flipped).
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label, dtype=bool).ravel()
    n = score.size
    if n == 0 or label.size != n:
        return 0.5
    finite = np.isfinite(score)
    if finite.sum() == 0:
        return 0.5
    score = score[finite]
    label = label[finite]
    n = score.size
    n_pos = int(label.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Average rank within tie groups (so the AUC is invariant to tie
    # breaking and matches the classic Mann-Whitney definition).
    score_sorted = score[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and score_sorted[j] == score_sorted[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j - 1]])
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = float(ranks[label].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def derive_gate_seeds_from_on_mask(
    on_mask: np.ndarray,
    on_off_signals: list,
    on_off_norm_min: np.ndarray,
    on_off_norm_max: np.ndarray,
    *,
    auc_high: float = 0.85,
    auc_low: float = 0.65,
    sharpness_beta: float = 8.0,
    band_lb: float = 0.05,
    band_ub: float = 5.0,
    threshold_lb: float = -1.0,
    threshold_ub: float = 1.5,
    margin: float = 0.15,
    neutral_threshold: float = 0.5,
    neutral_band: float = 1.0,
) -> GateSeeds:
    """Rank ``onOffSignal`` slots against ``on_mask`` and derive gate seeds.

    Pipeline per CITS:

    1. Normalise every slot's raw signal by its per-slot
       ``[oo_min, oo_max]`` (the same normalisation the CITS forward
       pass uses) so AUCs and threshold seeds are unit-free.
    2. Compute ROC AUC of each normalised slot vs ``on_mask``.  AUC ~ 1
       means high signal -> active; AUC ~ 0 means low signal -> active
       (anti-polarity); AUC ~ 0.5 means uninformative.
    3. Per-slot discriminative score ``disc_j = max(auc_j, 1 - auc_j)``.
       The winner is the slot with the highest ``disc_j``.
    4. ``gamma_gate_x0 = softmax(beta * disc)``.  With
       ``beta=sharpness_beta`` (default 8) this is near one-hot when
       one slot dominates, soft when several slots tie.
    5. On the winning slot, compute the active-range quantiles
       ``[q_lo, q_hi]`` of the *normalised* signal restricted to
       ``on_mask = True``.  These bracket the values the signal takes
       while the controller is running.
    6. ``gate_threshold_x0 = q_lo - margin`` and
       ``gate_band_x0 = (q_hi - q_lo) + 2 * margin`` so the active
       band brackets the on-state values with a small safety margin.
       For the typical normalised binary schedule (off=0, on=1) this
       resolves to roughly ``threshold = 0.85, band = 0.30``,
       matching the bldg1 setback/occupied dynamic.
    7. If the winner has anti-polarity (AUC < 0.5), the active samples
       cluster *low* and the threshold/band seeds bracket the low
       range -- still in the same forward-pass frame, no polarity
       hack needed because the BandGate is symmetric.

    Args:
        on_mask: Sample-level boolean array, ``True`` where the
            controller is active.  Output of
            :func:`derive_actuator_seeds_gmm`.  Length must match each
            ``on_off_signals`` element.
        on_off_signals: List of length ``n_slots`` of raw 1D signal
            arrays (in physical units; they will be normalised here
            using the supplied ``on_off_norm_min/max``).  ``None``
            entries are tolerated and treated as missing slots.
        on_off_norm_min, on_off_norm_max: Per-slot normalisation
            bounds, length ``n_slots``.  Must be the same values
            written onto the CITS by
            :func:`_populate_on_off_signal_norm_bounds`, so that the
            seeds are in the same frame the forward pass uses.
        auc_high, auc_low: Confidence ladder cutoffs on the *winner's*
            discriminative score ``max(auc, 1 - auc)``.
        sharpness_beta: Softmax temperature for ``gamma_gate``.
            Higher = more peaked.  ``8`` makes a 0.95-vs-0.5 win
            essentially one-hot.
        band_lb, band_ub: Lower / upper clamp on the band seed.  The
            lower clamp is set just above the BandGate's
            gradient-vanishing limit so a multi-level setpoint
            schedule with closely-spaced active and idle clusters
            (e.g. 20 °C active vs 22 °C idle on a 14 K-wide
            normalised range -- gap ~ 0.14) can be bracketed without
            spilling into the neighbouring cluster.
        threshold_lb, threshold_ub: Lower / upper clamp on the
            threshold seed.  The threshold can dip slightly below
            ``0`` or above ``1`` because the BandGate is differentiable
            outside ``[0, 1]`` -- this lets the seed bracket
            saturated-binary signals (where on=1, off=0) cleanly.
        margin: Extra width added on both sides of the active-range
            quantiles when computing the band; padding for noise.
        neutral_threshold, neutral_band: Fallback values used when the
            decomposition has no informative slot
            (``confidence="low"``).

    Returns:
        :class:`GateSeeds`.  Always returns a finite, well-formed
        seeds object even on degenerate inputs.
    """
    n_slots = int(np.asarray(on_off_norm_min).size)
    on_mask = np.asarray(on_mask, dtype=bool).ravel()

    def _neutral(reason: str) -> GateSeeds:
        if n_slots > 0:
            gamma = np.full(n_slots, 1.0 / n_slots, dtype=np.float64)
            auc = np.full(n_slots, 0.5, dtype=np.float64)
        else:
            gamma = np.zeros(0, dtype=np.float64)
            auc = np.zeros(0, dtype=np.float64)
        return GateSeeds(
            gamma_gate_x0=gamma,
            gate_threshold_x0=float(neutral_threshold),
            gate_band_x0=float(neutral_band),
            auc_per_slot=auc,
            winner_slot=0,
            winner_polarity=1,
            confidence="low",
            reason=reason,
        )

    if n_slots <= 0 or on_mask.size == 0:
        return _neutral("no_slots_or_empty_mask")
    n_pos = int(on_mask.sum())
    n_neg = int((~on_mask).sum())
    if n_pos < 5 or n_neg < 5:
        return _neutral(
            f"degenerate_on_mask (n_pos={n_pos}, n_neg={n_neg})"
        )

    oo_min = np.asarray(on_off_norm_min, dtype=np.float64).ravel()
    oo_max = np.asarray(on_off_norm_max, dtype=np.float64).ravel()
    if oo_min.size != n_slots or oo_max.size != n_slots:
        return _neutral("norm_bounds_size_mismatch")
    span = oo_max - oo_min
    span = np.where(np.abs(span) < 1e-12, 1.0, span)

    auc_per_slot = np.full(n_slots, 0.5, dtype=np.float64)
    norm_signals: list = [None] * n_slots
    for j in range(n_slots):
        if j >= len(on_off_signals):
            continue
        sig = on_off_signals[j]
        if sig is None:
            continue
        sig = np.asarray(sig, dtype=np.float64).ravel()
        n = min(sig.size, on_mask.size)
        if n < 10:
            continue
        sig = sig[:n]
        s_norm = (sig - oo_min[j]) / span[j]
        norm_signals[j] = s_norm
        auc_per_slot[j] = _roc_auc(s_norm, on_mask[:n])

    disc = np.maximum(auc_per_slot, 1.0 - auc_per_slot)
    winner = int(np.argmax(disc))
    winner_disc = float(disc[winner])

    # Confidence ladder.
    if winner_disc >= auc_high:
        confidence = "high"
    elif winner_disc >= auc_low:
        confidence = "medium"
    else:
        confidence = "low"

    # We compute the quantile-based threshold/band even when the
    # confidence ladder labels the winner as ``"low"`` (winner_disc <
    # auc_low, default 0.65).  Rationale: even at AUC ~ 0.55-0.65 the
    # winner slot's active-range quantiles still cluster the on_mask
    # samples around a meaningful range (e.g. AVRM107A on bldg1: the
    # ``Air_Temp_Setpoint`` schedule jumps to 20 °C only when reheat
    # is needed, which gives best_disc=0.64 -- borderline by the AUC
    # ladder, but the on_mask=True quantiles still pin the gate band
    # tightly around 20 °C).  Falling back to a neutral
    # ``[T_lo=0.5, band=1.0]`` band in this borderline case throws
    # away that information and makes the simulator's gate
    # effectively transparent, so the controller modulates whenever
    # any setpoint deviation appears, not just during the
    # schedule-active windows.  At truly random AUC (~0.5) the active
    # samples are spread across the full normalised range, the
    # quantile ``[q_lo, q_hi]`` widens to ~``[0, 1]`` and the
    # threshold/band clamps onto the neutral defaults anyway, so this
    # branch degrades gracefully.
    if norm_signals[winner] is None:
        out = _neutral(
            f"winner_slot_signal_missing (best disc={winner_disc:.3f})"
        )
        out.auc_per_slot = auc_per_slot
        out.winner_slot = winner
        return out

    winner_auc = float(auc_per_slot[winner])
    winner_polarity = 1 if winner_auc >= 0.5 else -1

    # gamma_gate as softmax over disc.  Subtract max for numerical
    # stability; with sharpness_beta=8 a 0.95 vs 0.5 split is
    # essentially one-hot.
    logits = sharpness_beta * (disc - disc.max())
    gamma = np.exp(logits)
    gamma = gamma / gamma.sum()

    # Active-range quantiles on the winner's normalised signal.  Use
    # the on_mask alignment (which has already been re-trimmed in the
    # AUC step), so re-trim here.
    s_w = norm_signals[winner]
    n = min(s_w.size, on_mask.size)
    s_w = s_w[:n]
    mask_active = on_mask[:n]

    s_active = s_w[mask_active]
    s_idle = s_w[~mask_active]
    s_active = s_active[np.isfinite(s_active)]
    s_idle = s_idle[np.isfinite(s_idle)]

    if s_active.size < 5 or s_idle.size < 5:
        out = _neutral("active_or_idle_subset_too_small")
        out.auc_per_slot = auc_per_slot
        out.winner_slot = winner
        out.winner_polarity = winner_polarity
        return out

    # Adaptive quantile bracket.  Narrower brackets for borderline
    # AUC reject ``on_mask`` label noise.
    #
    # Why this matters: at ``winner_disc=0.64`` (AVRM107A on bldg1),
    # only ~64 % of the on_mask=True samples truly match the active
    # schedule value (e.g. ``Air_Temp_Setpoint=20 °C`` -- norm 0.32
    # in the slot's [15.556, 29.444] frame).  The remaining ~36 %
    # are mis-labelled by the actuator-GMM's edge cases (e.g. a brief
    # ramp-up where the valve is just opening but the schedule is
    # still in its inactive 22 °C state -- norm 0.46).  A
    # ``[10 %, 90 %]`` bracket on this noisy active subset reaches
    # past the dominant mode into the contaminating cluster, so
    # ``q_hi`` lands at ~0.46 instead of ~0.34 and the gate band
    # ends up *covering both* 20 °C and 22 °C -- the simulator then
    # treats 22 °C (which the user has confirmed is the OFF state)
    # as "active".  Tightening to ``[35 %, 65 %]`` at low AUC focuses
    # on the mode of the active distribution and excludes the
    # ~17.5 % tails on each side, which is enough to drop the
    # 22 °C mass.  At ``winner_disc >= auc_high`` (~0.85) the slot
    # is informative enough that we trust the broad bracket and
    # capture the genuine active range.
    if winner_disc >= auc_high:
        q_low_pct, q_high_pct = 0.10, 0.90
    elif winner_disc >= auc_low:
        q_low_pct, q_high_pct = 0.20, 0.80
    else:
        q_low_pct, q_high_pct = 0.35, 0.65
    q_lo = float(np.quantile(s_active, q_low_pct))
    q_hi = float(np.quantile(s_active, q_high_pct))
    if q_hi <= q_lo:
        # Active samples concentrated at one value (binary schedule).
        # Widen by margin on each side so the BandGate has finite
        # gradient and the optimizer can still polish.
        mid = q_lo
        q_lo = mid - 0.5 * margin
        q_hi = mid + 0.5 * margin

    threshold_x0 = q_lo - margin
    band_x0 = (q_hi - q_lo) + 2.0 * margin

    # Idle-aware band tightening.  A fixed ``margin`` (default 0.15)
    # plus a 0.2 ``band_lb`` floor produces a gate band ~ 0.3 wide
    # in normalised units, which is fine when the active and the
    # nearest idle cluster sit > 0.3 apart but disastrous when a
    # multi-level setpoint schedule keeps them ~ 0.14 apart -- e.g.
    # AVRM107A on bldg1 has ``Air_Temp_Setpoint`` toggling between
    # 20 °C (norm 0.32, on_mask=True mode) and 22 °C (norm 0.46,
    # on_mask=False), so a 0.3-wide band centred on 0.32 stretches
    # to ~ 0.47 and the gate fires *also* at 22 °C, which the user
    # has confirmed is the OFF state.  We compute the upper edge of
    # the active cluster (90th percentile, robust to label noise),
    # the lower edge of the *closest* idle cluster *above* the
    # active mode (10th percentile of idle samples that exceed the
    # active 90th percentile), and clamp the gate's upper edge to
    # half-way between them.  Same construction on the lower side
    # (closest idle cluster *below* the active mode, capped at
    # half-way).  When the schedule is genuinely binary (idle far
    # from active) the half-way clamp lands well outside the
    # quantile-derived band and is a no-op; when it's multi-level
    # the clamp pulls the band in tight enough to land between the
    # two clusters.
    upper_edge = threshold_x0 + band_x0
    lower_edge = threshold_x0
    a_med = float(np.quantile(s_active, 0.50))
    a_hi_robust = float(np.quantile(s_active, 0.90))
    a_lo_robust = float(np.quantile(s_active, 0.10))
    idle_above = s_idle[s_idle > a_hi_robust]
    idle_below = s_idle[s_idle < a_lo_robust]
    if idle_above.size >= 5:
        idle_upper_neighbour = float(np.quantile(idle_above, 0.10))
        max_upper = a_med + 0.5 * (idle_upper_neighbour - a_med)
        if max_upper < upper_edge:
            upper_edge = max_upper
    if idle_below.size >= 5:
        idle_lower_neighbour = float(np.quantile(idle_below, 0.90))
        min_lower = a_med - 0.5 * (a_med - idle_lower_neighbour)
        if min_lower > lower_edge:
            lower_edge = min_lower
    if upper_edge > lower_edge:
        threshold_x0 = lower_edge
        band_x0 = upper_edge - lower_edge

    threshold_x0 = float(np.clip(threshold_x0, threshold_lb, threshold_ub))
    band_x0 = float(np.clip(band_x0, band_lb, band_ub))

    # Tag the seeds with a separate ``reason`` for borderline AUC so
    # downstream logs can flag a "we kept this, but the slot is only
    # mildly informative" outcome without obscuring the kp/Ti
    # regression mask intersection (which still uses the original
    # ``confidence != "low"`` gate).
    reason = None
    if confidence == "low":
        reason = (
            f"borderline_slot_kept_for_gate_seeds "
            f"(best disc={winner_disc:.3f})"
        )

    return GateSeeds(
        gamma_gate_x0=gamma,
        gate_threshold_x0=threshold_x0,
        gate_band_x0=band_x0,
        auc_per_slot=auc_per_slot,
        winner_slot=winner,
        winner_polarity=winner_polarity,
        confidence=confidence,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Confidence labelling
# ---------------------------------------------------------------------------


def confidence_label(
    r2: float,
    n_active: int,
    *,
    r2_high: float = 0.5,
    r2_low: float = 0.2,
    n_min: int = 50,
) -> str:
    """Map an ``(r2, n_active)`` pair to a confidence tier.

    Returns one of ``"high"``, ``"medium"``, ``"low"``, ``"failed"``.
    """
    if not np.isfinite(r2) or n_active < n_min:
        return "failed"
    if r2 >= r2_high:
        return "high"
    if r2 >= r2_low:
        return "medium"
    return "low"
