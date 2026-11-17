"""Open-loop fits of identified control loops to their actuator's measured command.

How well does a candidate loop explain the command?  The loop is run open on
the measured signals -- the :class:`PIDControllerSystem` law (positional,
conditional integration, ``Td = 0``) on the measured feedback about the
setpoint -- with its gains searched on a grid, and scored on every sample:
where the loop's gate is off the actuator sits at its park value, where it
is on the command is the loop's output.  The rewire ranks a controller's
candidate (feedback, setpoint) pairs by this fit, and asks whether a second
loop joined by max (a CO2 loop over a temperature loop) explains the command
better than the first alone.

Scoring every sample, not only those where the actuator modulates, is the
point: a radiator valve that stays shut all night while the room sits 1.5 K
below a 22 C setpoint says the setpoint is not 22 C, and only the shut
samples say it.

* :func:`pi_outputs`: the loop's command over a series, for many gains at once.
* :func:`fit_pi`: the best gains and direction for one (feedback, setpoint) pair.
* :func:`fit_max_loop`: the best second loop on another feedback, about a
  constant setpoint, joined to a first loop's output by max.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Proportional gains as multiples of one over the error's spread: a gain of
#: ``kappa / spread`` moves the command by ``kappa`` for a typical error.
KAPPA_GRID = np.logspace(-2.0, 2.5, 19)
#: Integral times [s]; ``inf`` is the proportional loop.
TI_GRID = np.r_[np.logspace(np.log10(300.0), 7.0, 12), np.inf]
#: Constant setpoints of a max loop, as quantiles of its feedback.
SETPOINT_QUANTILES = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98])


@dataclass
class LoopFit:
    """The best open-loop fit of a loop to its actuator's command.

    ``rmse`` over the scored samples, ``r2`` the share of the command's
    variance it explains, ``kp`` per unit of error, ``Ti`` in seconds
    (``inf``: proportional), ``is_reverse`` the direction (``True``: the
    command rises when the feedback is below the setpoint), ``setpoint`` the
    constant setpoint of a max loop (``None`` for a pair's own setpoint),
    ``command`` the loop's command before the gate (a max loop's: the larger
    of its own and the first loop's), ``output`` the fitted command with the
    gate, ``n`` the scored samples."""

    rmse: float
    r2: float
    kp: float
    Ti: float
    is_reverse: bool
    n: int
    setpoint: Optional[float] = None
    command: Optional[np.ndarray] = field(default=None, repr=False)
    output: Optional[np.ndarray] = field(default=None, repr=False)
    reason: Optional[str] = None


def _filled(x: np.ndarray) -> np.ndarray:
    """``x`` with gaps held at the last value (the first at the first value)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    idx = np.where(finite, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    x[: int(np.argmax(finite))] = x[int(np.argmax(finite))]
    return x


def pi_outputs(e: np.ndarray, kp: np.ndarray, Ti: np.ndarray, h: float, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """The command of the positional PI with conditional integration
    (:class:`PIDControllerSystem` with ``Td = 0``) over the error ``e``
    ``(n_t,)``, for every gain pair ``(kp[j], Ti[j])``: ``(n_t, n_gains)``.
    The integral starts at zero; ``Ti = inf`` is the proportional loop."""
    kp = np.asarray(kp, dtype=np.float64).reshape(-1)
    Ti = np.asarray(Ti, dtype=np.float64).reshape(-1)
    ki = np.where(np.isfinite(Ti), kp * h / np.where(np.isfinite(Ti), Ti, 1.0), 0.0)
    integral = np.zeros(kp.size)
    out = np.empty((e.size, kp.size))
    for t in range(e.size):
        p = kp * e[t]
        integral = np.minimum(
            np.maximum(integral + ki * e[t], np.minimum(integral, lower - p)),
            np.maximum(integral, upper - p),
        )
        out[t] = np.clip(p + integral, lower, upper)
    return out


def _score(u, predicted, scored):
    residual = predicted[scored] - u[scored, None]
    return np.sqrt(np.mean(residual**2, axis=0))


def _predict(outputs, gate, park, base):
    if base is not None:
        outputs = np.maximum(outputs, base[:, None])
    if gate is None:
        return outputs
    return np.where(gate[:, None], outputs, park)


def _spread(x: np.ndarray) -> float:
    """Half the 5-95 % range: a typical error, also of a peaky signal (a
    CO2 that sits at its baseline most of the day)."""
    q95, q5 = np.percentile(x, [95, 5])
    spread = float(q95 - q5) / 2.0
    return spread if spread > 1e-9 else float(np.std(x)) or 1.0


def _search(u, errors, h, *, gate, park, lower, upper, base, scored, spread):
    """The best ``(rmse, column)`` over the gain grid and the errors'
    columns: ``errors`` is ``(n_t, n_err)`` (one column per direction and
    setpoint), every column run with every gain pair."""
    kappa, Ti = np.meshgrid(KAPPA_GRID, TI_GRID, indexing="ij")
    kappa, Ti = kappa.ravel(), Ti.ravel()
    best = (np.inf, None)
    for k in range(errors.shape[1]):
        kp = kappa / spread
        rmse = _score(u, _predict(pi_outputs(errors[:, k], kp, Ti, h, lower, upper), gate, park, base), scored)
        j = int(np.argmin(rmse))
        if rmse[j] < best[0]:
            best = (float(rmse[j]), (k, float(kp[j]), float(Ti[j])))
    if best[1] is None:
        return best
    # refine the gains around the best on a finer grid
    k, kp0, Ti0 = best[1]
    kp_fine = kp0 * np.logspace(-0.3, 0.3, 7)
    Ti_fine = Ti0 * np.logspace(-0.5, 0.5, 7) if np.isfinite(Ti0) else np.array([np.inf, 3e7, 1e7])
    kp_f, Ti_f = (a.ravel() for a in np.meshgrid(kp_fine, Ti_fine, indexing="ij"))
    rmse = _score(u, _predict(pi_outputs(errors[:, k], kp_f, Ti_f, h, lower, upper), gate, park, base), scored)
    j = int(np.argmin(rmse))
    if rmse[j] < best[0]:
        best = (float(rmse[j]), (k, float(kp_f[j]), float(Ti_f[j])))
    return best


def _result(u, errors, h, best, *, gate, park, lower, upper, base, scored, directions, setpoints=None) -> LoopFit:
    rmse, (k, kp, Ti) = best
    command = pi_outputs(errors[:, k], np.array([kp]), np.array([Ti]), h, lower, upper)
    if base is not None:
        command = np.maximum(command, base[:, None])
    output = _predict(command, gate, park, None)[:, 0]
    var = float(np.var(u[scored]))
    return LoopFit(
        rmse=rmse,
        r2=1.0 - rmse**2 / var if var > 1e-12 else 0.0,
        kp=kp,
        Ti=Ti,
        is_reverse=bool(directions[k]),
        n=int(scored.sum()),
        setpoint=None if setpoints is None else float(setpoints[k]),
        command=command[:, 0],
        output=output,
    )


def _scored(u, *signals, n_min):
    scored = np.isfinite(u)
    for s in signals:
        scored &= np.isfinite(s)
    return scored if scored.sum() >= n_min else None


def fit_pi(
    u: np.ndarray,
    setpoint: np.ndarray,
    feedback: np.ndarray,
    h: float,
    *,
    gate: Optional[np.ndarray] = None,
    park: float = 0.0,
    lower: float = 0.0,
    upper: float = 1.0,
    n_min: int = 50,
) -> LoopFit:
    """The PI on ``setpoint - feedback`` (reverse acting) or ``feedback -
    setpoint`` (direct) that best explains the command ``u``: gains and
    direction searched, ``gate`` (bool per sample; ``None``: always on)
    holding the command at ``park`` where it is off."""
    n = min(len(u), len(setpoint), len(feedback))
    u = np.asarray(u, dtype=np.float64)[:n]
    raw_sp, raw_fb = np.asarray(setpoint, dtype=np.float64)[:n], np.asarray(feedback, dtype=np.float64)[:n]
    scored = _scored(u, raw_sp, raw_fb, n_min=n_min)
    if scored is None:
        return LoopFit(np.inf, 0.0, np.nan, np.nan, True, 0, reason="too_few_samples")
    gate = None if gate is None else np.asarray(gate, dtype=bool)[:n]
    e = _filled(raw_sp) - _filled(raw_fb)
    errors = np.stack([e, -e], axis=1)
    directions = (True, False)
    kw = dict(gate=gate, park=park, lower=lower, upper=upper, base=None, scored=scored)
    best = _search(u, errors, h, spread=_spread(e[scored]), **kw)
    return _result(u, errors, h, best, directions=directions, **kw)


def fit_max_loop(
    u: np.ndarray,
    feedback: np.ndarray,
    base: np.ndarray,
    h: float,
    *,
    gate: Optional[np.ndarray] = None,
    park: float = 0.0,
    lower: float = 0.0,
    upper: float = 1.0,
    n_min: int = 50,
) -> LoopFit:
    """The PI on ``feedback`` about a constant setpoint that, joined by max
    to the command ``base`` of a first loop (before its gate: the first
    loop's :attr:`LoopFit.command`), best explains
    the command ``u``.  The setpoint is searched among quantiles of the
    feedback where the gate is on, with the gains and the direction (a CO2
    loop opening a damper above a level is direct acting)."""
    n = min(len(u), len(feedback), len(base))
    u = np.asarray(u, dtype=np.float64)[:n]
    raw = np.asarray(feedback, dtype=np.float64)[:n]
    scored = _scored(u, raw, n_min=n_min)
    if scored is None:
        return LoopFit(np.inf, 0.0, np.nan, np.nan, False, 0, reason="too_few_samples")
    y = _filled(raw)
    gate = None if gate is None else np.asarray(gate, dtype=bool)[:n]
    on = scored if gate is None else (scored & gate)
    levels = np.quantile(y[on if on.sum() >= n_min else scored], SETPOINT_QUANTILES)
    errors = np.concatenate([levels[None, :] - y[:, None], y[:, None] - levels[None, :]], axis=1)
    directions = (True,) * levels.size + (False,) * levels.size
    setpoints = np.r_[levels, levels]
    kw = dict(gate=gate, park=park, lower=lower, upper=upper, base=np.asarray(base, dtype=np.float64)[:n], scored=scored)
    spread = _spread(y[scored])
    best = _search(u, errors, h, spread=spread, **kw)
    if best[1] is None:
        return LoopFit(np.inf, 0.0, np.nan, np.nan, False, 0, reason="no_fit")
    # the setpoint between the quantiles next to the best one
    k = best[1][0]
    side, q = divmod(k, levels.size)
    fine = np.linspace(levels[max(q - 1, 0)], levels[min(q + 1, levels.size - 1)], 9)
    sign = 1.0 if side == 0 else -1.0
    fine_errors = sign * (fine[None, :] - y[:, None])
    refined = _search(u, fine_errors, h, spread=spread, **kw)
    if refined[1] is not None and refined[0] < best[0]:
        return _result(u, fine_errors, h, refined, directions=(directions[k],) * fine.size, setpoints=fine, **kw)
    return _result(u, errors, h, best, directions=directions, setpoints=setpoints, **kw)
