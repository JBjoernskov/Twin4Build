"""Differentiable clamp with selectable forward/backward semantics.

This module provides a single reusable bounded-output function that
replaces hard ``torch.clamp`` in differentiable simulators with a
gradient-friendly smooth alternative.  Two modes are offered:

* ``"smooth"`` -- power-law (or exponential / hyperbolic / sqrt)
  asymptotic curves around each bound.  Forward output approaches but
  never reaches the bound, leaving a *faithfulness floor* of size
  ``curve_start``; backward is non-zero everywhere, so the optimizer
  can recover from any initial point including deep windup.

* ``"hard"`` -- bit-exact ``torch.clamp``.  Forward hits the bound
  exactly; backward is the true subgradient (1.0 in the interior, 0.0
  at/past the bound).  Removes the smooth-mode faithfulness floor and
  the associated parameter bias on samples whose measurements sit at
  a bound, but cannot lift parameters out of saturation on its own.

The recommended workflow is **two-stage estimation**: run with the
default ``"smooth"`` mode for exploration / cold-start convergence,
then warm-start a refinement run in ``"hard"`` mode to remove the
smooth-mode bias.  See :func:`twin4build.estimator.refinement` /
``estimate_with_refinement`` for the helper that automates this.

The mode is read from a process-global flag at every call site, so a
single context manager (:func:`saturation_mode`) flips every PID,
SAT-compensated rule, occupancy, etc. clamp uniformly across the
model.  Call sites that should *always* stay smooth (gate / switch
sub-systems where threshold estimation requires non-zero gradient
everywhere) opt out by passing ``mode="smooth"`` explicitly.

The smooth branch is safe for reverse-mode AD (``jacrev``): the
``torch.clamp`` inside ``_curve_function`` prevents ``torch.pow`` from
receiving a negative base in non-selected ``torch.where`` branches,
avoiding the ``0 * NaN = NaN`` issue.
"""

from contextlib import contextmanager
from typing import Optional

import torch

_VALID_MODES = ("smooth", "hard")
_GLOBAL_SATURATION_MODE: str = "smooth"


def get_saturation_mode() -> str:
    """Return the current process-global saturation mode (``"smooth"`` or ``"hard"``)."""
    return _GLOBAL_SATURATION_MODE


def set_saturation_mode(mode: str) -> None:
    """Set the process-global saturation mode.

    Affects every :func:`clamp` call that does not pass an explicit
    ``mode`` keyword.  Use the :func:`saturation_mode` context manager
    for scoped overrides; this setter is for advanced cases (e.g.
    persistent overrides from a configuration file).
    """
    global _GLOBAL_SATURATION_MODE
    if mode not in _VALID_MODES:
        raise ValueError(
            f"saturation mode must be one of {_VALID_MODES!r}, got {mode!r}"
        )
    _GLOBAL_SATURATION_MODE = mode


@contextmanager
def saturation_mode(mode: str):
    """Context manager that scopes a saturation-mode override.

    .. code-block:: python

        with saturation_mode("hard"):
            estimator.estimate(...)   # bias-correction refinement pass

    On exit, the previous global mode is restored, including when the
    block exits via exception.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"saturation mode must be one of {_VALID_MODES!r}, got {mode!r}"
        )
    global _GLOBAL_SATURATION_MODE
    previous = _GLOBAL_SATURATION_MODE
    _GLOBAL_SATURATION_MODE = mode
    try:
        yield
    finally:
        _GLOBAL_SATURATION_MODE = previous


def clamp(
    u: torch.Tensor,
    lower: float = 0.0,
    upper: float = 1.0,
    eps: float = 0.0,
    curve_start: float = 0.1,
    steepness: float = 1.0,
    curve_type: str = "power",
    power_exp: float = 0.5,
    mode: Optional[str] = None,
) -> torch.Tensor:
    r"""Differentiable clamp into ``[lower + eps, upper - eps]``.

    Args:
        u: Input tensor (any shape).
        lower: Hard lower bound.
        upper: Hard upper bound.
        eps: Shrink the effective range by ``eps`` on each side.
        curve_start: Width of the curved transition region on each side
            (smooth mode only -- ignored when ``mode == "hard"``).
        steepness: Gain applied inside the curve function (smooth mode
            only).  Higher values make the asymptote approach the bound
            faster.
        curve_type: ``"power"`` | ``"exponential"`` | ``"hyperbolic"`` |
            ``"sqrt"`` (smooth mode only).
        power_exp: Exponent used when *curve_type* is ``"power"``
            (smaller = slower gradient decay near the bound).
        mode: ``"smooth"`` or ``"hard"``; if ``None`` (default), the
            current process-global mode is used (see
            :func:`saturation_mode`).  Pass an explicit value to opt
            out of the global toggle -- e.g. ``mode="smooth"`` for
            gate / switch sub-systems whose threshold parameters
            require non-zero gradient even at convergence.

    Returns:
        Tensor with the same shape as ``u``, values in
        ``[lower + eps, upper - eps]``.
    """
    effective_mode = mode if mode is not None else _GLOBAL_SATURATION_MODE
    if effective_mode == "hard":
        return torch.clamp(u, min=lower + eps, max=upper - eps)
    if effective_mode != "smooth":
        raise ValueError(
            f"saturation mode must be one of {_VALID_MODES!r}, got {effective_mode!r}"
        )

    effective_min = lower + eps
    effective_max = upper - eps
    lower_curve_point = effective_min + curve_start
    upper_curve_point = effective_max - curve_start

    def _curve(x: torch.Tensor) -> torch.Tensor:
        safe_x = torch.clamp(x, min=0.0)
        scaled_x = steepness * safe_x / curve_start
        if curve_type == "exponential":
            return 1 - torch.exp(-scaled_x)
        elif curve_type == "hyperbolic":
            return scaled_x / (1 + scaled_x)
        elif curve_type == "sqrt":
            safe_base = torch.clamp(1 + scaled_x, min=1e-10)
            return 1 - 1 / torch.sqrt(safe_base)
        else:  # "power" (default)
            safe_base = torch.clamp(1 + scaled_x, min=1e-10)
            return 1 - 1 / torch.pow(safe_base, power_exp)

    return torch.where(
        u < lower_curve_point,
        effective_min + curve_start * (1 - _curve(lower_curve_point - u)),
        torch.where(
            u > upper_curve_point,
            effective_max - curve_start * (1 - _curve(u - upper_curve_point)),
            u,
        ),
    )


# Backward-compatibility aliases.  The old function names are retained
# so that external code (and any not-yet-migrated call sites in this
# repo) keeps working.  Both delegate to :func:`clamp` and always force
# ``mode="smooth"`` to preserve their historical behavior -- callers
# that adopted the old API explicitly wanted the smooth function.
def smooth_saturation(
    u: torch.Tensor,
    lower: float = 0.0,
    upper: float = 1.0,
    eps: float = 0.0,
    curve_start: float = 0.01,
    steepness: float = 1.0,
    curve_type: str = "power",
    power_exp: float = 0.5,
) -> torch.Tensor:
    """Deprecated alias for ``clamp(..., mode="smooth")``.

    Kept for backward compatibility.  New code should call :func:`clamp`
    directly and either rely on the global mode or pass ``mode=...``
    explicitly.
    """
    return clamp(
        u,
        lower=lower,
        upper=upper,
        eps=eps,
        curve_start=curve_start,
        steepness=steepness,
        curve_type=curve_type,
        power_exp=power_exp,
        mode="smooth",
    )


def hardclamp_smooth_grad(
    u: torch.Tensor,
    lower: float = 0.0,
    upper: float = 1.0,
    eps: float = 0.0,
    curve_start: float = 0.01,
    steepness: float = 1.0,
    curve_type: str = "power",
    power_exp: float = 0.5,
) -> torch.Tensor:
    """Deprecated -- use ``clamp(..., mode="hard")`` after a smooth
    warm-start instead.

    The old "hard forward, smooth backward via detach trick" approach
    is superseded by the explicit two-stage workflow: smooth-mode
    estimation followed by hard-mode refinement.  Pure
    :func:`torch.clamp` (i.e. ``mode="hard"``) gives the same exact-
    forward behavior with the *true* subgradient backward, eliminating
    the phantom-gradient bias that the detach trick re-introduced.
    """
    soft = clamp(
        u,
        lower=lower,
        upper=upper,
        eps=eps,
        curve_start=curve_start,
        steepness=steepness,
        curve_type=curve_type,
        power_exp=power_exp,
        mode="smooth",
    )
    hard = torch.clamp(u, min=lower + eps, max=upper - eps)
    return hard.detach() + (soft - soft.detach())
