# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import REQUIRED, Input, Output, Parameter, Setting


def _increasing(X) -> tuple:
    """The x of the points as a tuple of floats; they must increase."""
    points = tuple(float(v) for v in torch.as_tensor(X, dtype=torch.float64).reshape(-1).tolist())
    if len(points) < 2:
        raise ValueError(f"PiecewiseLinearSystem: X needs at least two points, got {list(points)}")
    if any(b <= a for a, b in zip(points, points[1:])):
        raise ValueError(f"PiecewiseLinearSystem: X must increase, got {list(points)}")
    return points


class PiecewiseLinearSystem(core.System):
    """``y(x)`` through the points ``(X, Y)``: linear between two points,
    constant beyond the first and the last.

    ``X`` is a setting: the points' positions are fixed.  ``Y`` is a
    parameter of one value per point, so it can be estimated or optimized
    (the supply temperatures of an outdoor temperature compensation curve).
    Its range is the run's: ``tb.Variable(curve.Y, lb=12.0, ub=24.0)``.

    Example::

        curve = tb.PiecewiseLinearSystem(X=[-12, 5, 20], Y=[60, 50, 20], id="curve")
    """

    REMOVED_ATTRIBUTES = {
        "Y_bounds": "give the range with the run's variable: tb.Variable(curve.Y, lb=..., ub=...)",
    }

    X = Setting(REQUIRED, convert=_increasing, doc="The x of the points, increasing")
    x = Input(doc="The value to look up")
    y = Output(doc="y(x)")
    Y = Parameter(REQUIRED, size=X, doc="The y of the points, one per x")

    @staticmethod
    def interpolate(x: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """``y(x)`` through the points ``(X, Y)``, ``X`` increasing; constant
        beyond the first and the last point.  ``Y`` is one row of points
        ``(n,)``, or one row per copy ``(m, n)`` for an ``x`` of ``m`` rows
        or a multiple of them (every simulation of a copy reads its row).
        Differentiable in ``Y`` and in ``x``."""
        x = torch.as_tensor(x)
        shape = x.shape
        x = x.reshape(-1)
        X = torch.as_tensor(X).to(dtype=x.dtype, device=x.device)
        Y = torch.as_tensor(Y).to(dtype=x.dtype, device=x.device).reshape(-1, X.numel())
        if Y.shape[0] != x.shape[0]:
            Y = Y.repeat(x.shape[0] // Y.shape[0], 1)
        slope = (Y[:, 1:] - Y[:, :-1]) / (X[1:] - X[:-1])
        intercept = Y[:, :-1] - slope * X[:-1]
        segment = torch.clamp(torch.searchsorted(X.contiguous(), x.contiguous()) - 1, 0, X.numel() - 2).unsqueeze(-1)
        y = torch.gather(slope, 1, segment).squeeze(-1) * x + torch.gather(intercept, 1, segment).squeeze(-1)
        y = torch.where(x <= X[0], Y[:, 0], torch.where(x >= X[-1], Y[:, -1], y))
        return y.reshape(shape)

    def forward(self, x, inputs, p, step_size):
        """Stateless: the curve at the input, each copy on its own points."""
        u = inputs["x"]
        return x, {"y": self.interpolate(u, torch.as_tensor(self.X, dtype=u.dtype, device=u.device), p["Y"])}
