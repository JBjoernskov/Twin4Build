# Standard library imports
import datetime
from typing import Dict, List, Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.types import _convert_to_1D_scalar_tensor


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

        # Store attributes as private variables
        self._X = X
        self._Y = Y
        self._XY = None
        self._a_vec = None
        self._b_vec = None

        if X is not None and Y is not None:
            # Stack X and Y coordinates
            self._XY = torch.stack([X, Y]).T

            # Sort by X coordinates to ensure proper ordering for searchsorted
            sorted_indices = torch.argsort(self._XY[:, 0])
            self._XY = self._XY[sorted_indices]

            # Update X and Y to reflect sorted order
            self._X = self._XY[:, 0]
            self._Y = self._XY[:, 1]

            self._get_a_b_vectors()
        self._config = {"parameters": []}

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
    def Y(self) -> Optional[torch.Tensor]:
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

    def _get_a_b_vectors(self) -> None:
        """Calculate slope and intercept vectors for all linear segments.

        For each segment between consecutive points, calculates:
        - Slope (a): (y2-y1)/(x2-x1)
        - Intercept (b): y1 - a*x1
        """
        self._a_vec = (self._XY[1:, 1] - self._XY[0:-1, 1]) / (
            self._XY[1:, 0] - self._XY[0:-1, 0]
        )
        self._b_vec = self._XY[0:-1, 1] - self._a_vec * self._XY[0:-1, 0]

    def _get_Y(self, X: torch.Tensor) -> torch.Tensor:
        """Get interpolated Y value for given X.

        Performs piecewise linear interpolation:
        - If X is below range, returns first Y value
        - If X is above range, returns last Y value
        - Otherwise finds appropriate segment and calculates Y = ax + b

        Args:
            X (torch.Tensor): X values to interpolate at, shape (batch_size,).

        Returns:
            torch.Tensor: Interpolated Y values, shape (batch_size,).
        """

        # if X <= self._XY[0, 0].item():
        #     Y = self._XY[0, 1].item()
        # elif X >= self._XY[-1, 0].item():
        #     Y = self._XY[-1, 1].item()
        # else:
        #     cond = X < self._XY[:, 0]
        #     idx = torch.where(cond)[0][0].item() - 1
        #     a = self._a_vec[idx].item()
        #     b = self._b_vec[idx].item()
        #     Y = a * X + b

        # Convert X to tensor if it's a scalar (float or int) using the safe converter
        X = _convert_to_1D_scalar_tensor(X)

        # Use searchsorted to find the segment index for each X value
        # searchsorted returns indices where X would be inserted to maintain sorted order
        indices = torch.searchsorted(self._XY[:, 0].contiguous(), X)

        # Clamp indices to valid segment range [0, len(a_vec)-1]
        # indices is where X would be inserted, so segment_idx = indices - 1
        segment_idx = torch.clamp(indices - 1, 0, len(self._a_vec) - 1)

        # Get the slope and intercept for each segment
        a = self._a_vec[segment_idx]  # shape: (batch_size,)
        b = self._b_vec[segment_idx]  # shape: (batch_size,)

        # Calculate interpolated values
        Y_interp = a * X + b

        # Handle boundary conditions using torch.where
        # If X <= first X value, use first Y value
        # If X >= last X value, use last Y value
        # Otherwise use interpolated value
        Y = torch.where(
            X <= self._XY[0, 0],
            self._XY[0, 1].expand_as(X),
            torch.where(X >= self._XY[-1, 0], self._XY[-1, 1].expand_as(X), Y_interp),
        )

        return Y

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Perform a single interpolation step using new implementation.

        Args:
            second_time (Optional[float], optional): Current simulation time in seconds.
                Defaults to None.
            date_time (Optional[datetime.datetime], optional): Current simulation date_time.
                Defaults to None.
            step_size (Optional[float], optional): Time step size in seconds.
                Defaults to None.
        """
        X = self.input["x"].get()
        self.output["y"].set(self._get_Y(X), step_index)
