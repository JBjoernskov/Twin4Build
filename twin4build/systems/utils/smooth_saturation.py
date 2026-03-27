"""Unified smooth saturation function for differentiable bounding.

This module provides a single reusable function that replaces hard ``torch.clamp``,
``torch.sigmoid``, and ``softplus`` operations with a smooth, fully differentiable
alternative based on power-law (or exponential/hyperbolic) asymptotic curves.

The function is safe for reverse-mode AD (``jacrev``): the ``torch.clamp`` inside
``_curve_function`` prevents ``torch.pow`` from receiving a negative base in
non-selected ``torch.where`` branches, avoiding the ``0 * NaN = NaN`` issue.
"""

import torch


def smooth_saturation(
    u: torch.Tensor,
    lower: float = 0.0,
    upper: float = 1.0,
    eps: float = 0.0,
    curve_start: float = 0.1,
    steepness: float = 1.0,
    curve_type: str = "power",
    power_exp: float = 0.5,
) -> torch.Tensor:
    r"""Smoothly saturate ``u`` into the interval ``[lower + eps, upper - eps]``.

    Three regions:

    * **Linear pass-through** for values well inside the bounds.
    * **Lower curve** that asymptotically approaches ``lower + eps``.
    * **Upper curve** that asymptotically approaches ``upper - eps``.

    The ``curve_start`` parameter controls the width (in output units) of the
    transition zone between the linear region and each bound.

    Args:
        u: Input tensor (any shape).
        lower: Hard lower bound.
        upper: Hard upper bound.
        eps: Shrink the effective range by ``eps`` on each side.
        curve_start: Width of the curved transition region on each side
            (in the same units as ``u``).
        steepness: Gain applied inside the curve function.  Higher values
            make the asymptote approach the bound faster.
        curve_type: ``"power"`` | ``"exponential"`` | ``"hyperbolic"`` | ``"sqrt"``.
        power_exp: Exponent used when *curve_type* is ``"power"``
            (smaller = slower gradient decay near the bound).

    Returns:
        Tensor with the same shape as ``u``, values in
        ``[lower + eps, upper - eps]``.
    """
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
