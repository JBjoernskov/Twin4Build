# Standard library imports
import datetime
from typing import Dict, List, Optional, Tuple

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class PiecewiseLinearSystem(core.System):
    """A system implementing piecewise linear interpolation functionality.

    This class provides core functionality for systems that need to perform piecewise
    linear interpolation between data points. It supports both direct point-to-point
    interpolation and fitting of piecewise linear functions to data.

    Args:
        X: X coordinates
        Y: Y coordinates
        **kwargs: Additional keyword arguments

    Note:
        When X and Y are provided during initialization, the system automatically
        calculates the piecewise linear coefficients.
    """

    def __init__(
        self,
        X: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        Y_bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> None:
        """Initialize the piecewise linear system.

        Args:
            X: X coordinates. Defaults to None.
            Y: Y coordinates. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)

        self.input = {
            "x": tps.Scalar(),
        }
        self.output = {
            "y": tps.Scalar(),
        }

        # The X coordinates are structural data (sorted, fixed); the Y
        # coordinates are a ``tps.Parameter`` so they can be estimated or
        # optimized (e.g. the supply-temperature points of an outdoor
        # temperature compensation curve).  ``Y_bounds`` makes them
        # estimable through ``get_estimable_parameters``.
        self._X = None
        self._Y = None
        if X is not None and Y is not None:
            X = torch.as_tensor(X, dtype=tps.float_dtype())
            Y = torch.as_tensor(Y, dtype=tps.float_dtype())
            order = torch.argsort(X)
            self._X = X[order].detach().clone()
            Y = Y[order].detach().clone()
            if Y_bounds:
                lo, hi = float(Y_bounds[0]), float(Y_bounds[1])
            else:  # a generous range around the points, for normalization only
                span = max(1.0, float(Y.max() - Y.min()))
                lo, hi = float(Y.min()) - span, float(Y.max()) + span
            self._Y = tps.Parameter(
                Y, min_value=torch.full_like(Y, lo), max_value=torch.full_like(Y, hi), requires_grad=False
            )
        self.parameter = {"Y": {"lb": float(Y_bounds[0]), "ub": float(Y_bounds[1])}} if Y_bounds else {}
        self._config = {"parameters": ["Y"]}

    @property
    def config(self) -> Dict[str, List[str]]:
        """Get the configuration parameters.

        Returns:
            Dict[str, List[str]]: Dictionary containing configuration parameter names.
        """
        return self._config

    @property
    def X(self) -> Optional[torch.Tensor]:
        """
        Get the X coordinates of the interpolation points.
        """
        return self._X

    @X.setter
    def X(self, value: Optional[torch.Tensor]) -> None:
        """
        Set the X coordinates of the interpolation points.
        """
        self._X = value

    @property
    def Y(self) -> Optional[tps.Parameter]:
        """
        Get the Y coordinates of the interpolation points.
        """
        return self._Y

    @Y.setter
    def Y(self, value: Optional[torch.Tensor]) -> None:
        """
        Set the Y coordinates of the interpolation points.
        """
        self._Y = value

    @staticmethod
    def interpolate(x: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear ``y(x)`` through the points ``(X, Y)``, ``X``
        sorted; constant beyond the first and last point.  Differentiable
        in ``Y`` (and in ``x``), which is what estimation and optimization
        of the points need."""
        x = torch.as_tensor(x)
        shape = x.shape
        x = x.reshape(-1)
        X = X.to(dtype=x.dtype, device=x.device)
        Y = Y.to(dtype=x.dtype, device=x.device).reshape(-1)
        slope = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
        intercept = Y[:-1] - slope * X[:-1]
        segment = torch.clamp(torch.searchsorted(X.contiguous(), x) - 1, 0, slope.numel() - 1)
        y = slope[segment] * x + intercept[segment]
        y = torch.where(x <= X[0], Y[0].expand_as(x), torch.where(x >= X[-1], Y[-1].expand_as(x), y))
        return y.reshape(shape)

    def _points_Y(self) -> torch.Tensor:
        """The current Y points as a tensor: a ``tps.Parameter`` when the
        table was given at construction, a plain tensor when a subclass
        sets the table itself (the schedule resolves its points per step)."""
        return self._Y.get() if hasattr(self._Y, "get") else self._Y

    def _get_a_b_vectors(self) -> None:
        """Kept for subclasses that set ``_X`` / ``_Y`` directly and call
        this to refresh the table: the interpolation reads the points as
        they are, so there is nothing to derive."""
        return None

    def _get_Y(self, X: torch.Tensor) -> torch.Tensor:
        """Interpolated Y at ``X`` with the component's current points."""
        return self.interpolate(X, self._X, self._points_Y())

    PARAM_NAMES = ("Y",)  # the X coordinates are structural, the Y points a parameter

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step piecewise-linear interpolation (functorch-safe,
        stateless).  The interpolation table is fixed (structural) data, so
        :meth:`_get_Y` is a pure function of the input."""
        Y = params["Y"] if "Y" in params else self._points_Y()
        return x, {"y": self.interpolate(inputs["x"], self._X, Y)}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Perform a single interpolation step.

        Thin port-I/O wrapper delegating the math to :meth:`forward`.

        Args:
            second_time (Optional[float], optional): Current simulation time in seconds.
                Defaults to None.
            date_time (Optional[datetime.datetime], optional): Current simulation date_time.
                Defaults to None.
            step_size (Optional[float], optional): Time step size in seconds.
                Defaults to None.
        """
        inputs = {"x": self.input["x"].get()}
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["y"]._set(outs["y"], i_t=step_index)
