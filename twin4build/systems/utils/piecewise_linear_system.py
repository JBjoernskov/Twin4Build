# Standard library imports
import datetime
from typing import Dict, List, Optional

# Third party imports
import numpy as np

# Local application imports
import twin4build.core as core


class PiecewiseLinearSystem(core.System):
    """A system implementing piecewise linear interpolation functionality.

    This class provides core functionality for systems that need to perform piecewise
    linear interpolation between data points. It supports both direct point-to-point
    interpolation and fitting of piecewise linear functions to data.

    Note:
        When X and Y are provided during initialization, the system automatically
        calculates the piecewise linear coefficients.
    """

    def __init__(
        self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None, **kwargs
    ) -> None:
        """Initialize the piecewise linear system.

        Args:
            X (Optional[np.ndarray], optional): X coordinates. Defaults to None.
            Y (Optional[np.ndarray], optional): Y coordinates. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)

        # Store attributes as private variables
        self._X = X
        self._Y = Y
        self._XY = None
        self._a_vec = None
        self._b_vec = None

        if X is not None and Y is not None:
            self._XY = np.array([X, Y]).transpose()
            self.get_a_b_vectors()
        self._config = {"parameters": []}

    @property
    def config(self) -> Dict[str, List[str]]:
        """Get the configuration parameters.

        Returns:
            Dict[str, List[str]]: Dictionary containing configuration parameter names.
        """
        return self._config

    @property
    def X(self) -> Optional[np.ndarray]:
        """
        Get the X coordinates of the interpolation points.
        """
        return self._X

    @X.setter
    def X(self, value: Optional[np.ndarray]) -> None:
        """
        Set the X coordinates of the interpolation points.
        """
        self._X = value

    @property
    def Y(self) -> Optional[np.ndarray]:
        """
        Get the Y coordinates of the interpolation points.
        """
        return self._Y

    @Y.setter
    def Y(self, value: Optional[np.ndarray]) -> None:
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
        self._a_vec = (self.XY[1:, 1] - self.XY[0:-1, 1]) / (
            self.XY[1:, 0] - self.XY[0:-1, 0]
        )
        self._b_vec = self.XY[0:-1, 1] - self.a_vec * self.XY[0:-1, 0]

    def _get_Y(self, X: float) -> float:
        """Get interpolated Y value for given X.

        Performs piecewise linear interpolation:
        - If X is below range, returns first Y value
        - If X is above range, returns last Y value
        - Otherwise finds appropriate segment and calculates Y = ax + b

        Args:
            X (float): X value to interpolate at.

        Returns:
            float: Interpolated Y value.
        """
        if X <= self.XY[0, 0]:
            Y = self.XY[0, 1]
        elif X >= self.XY[-1, 0]:
            Y = self.XY[-1, 1]
        else:
            cond = X < self.XY[:, 0]
            idx = np.where(cond)[0][0] - 1
            a = self.a_vec[idx]
            b = self.b_vec[idx]
            Y = a * X + b
        return Y

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """Perform a single interpolation step using new implementation.

        Args:
            secondTime (Optional[float], optional): Current simulation time in seconds.
                Defaults to None.
            dateTime (Optional[datetime.datetime], optional): Current simulation datetime.
                Defaults to None.
            stepSize (Optional[float], optional): Time step size in seconds.
                Defaults to None.
        """
        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        self.output[key].set(self._get_Y(X), stepIndex)
