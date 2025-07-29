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

    Attributes:
        X (np.ndarray, optional): X coordinates of the interpolation points.
        Y (np.ndarray, optional): Y coordinates of the interpolation points.
        XY (np.ndarray): Combined array of X,Y coordinates as [[x1,y1], [x2,y2], ...].
        a_vec (np.ndarray): Slope coefficients for each linear segment.
        b_vec (np.ndarray): Intercept coefficients for each linear segment.
        _config (Dict[str, List[str]]): Configuration parameters.

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

        self.X = X
        self.Y = Y

        if X is not None and Y is not None:
            self.XY = np.array([X, Y]).transpose()
            self.get_a_b_vectors()
        self._config = {"parameters": []}

    @property
    def config(self) -> Dict[str, List[str]]:
        """Get the configuration parameters.

        Returns:
            Dict[str, List[str]]: Dictionary containing configuration parameter names.
        """
        return self._config

    def get_a_b_vectors(self) -> None:
        """Calculate slope and intercept vectors for all linear segments.

        For each segment between consecutive points, calculates:
        - Slope (a): (y2-y1)/(x2-x1)
        - Intercept (b): y1 - a*x1
        """
        self.a_vec = (self.XY[1:, 1] - self.XY[0:-1, 1]) / (
            self.XY[1:, 0] - self.XY[0:-1, 0]
        )
        self.b_vec = self.XY[0:-1, 1] - self.a_vec * self.XY[0:-1, 0]

    def get_Y(self, X: float) -> float:
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
        secondTime: Optional[float] = None,
        dateTime: Optional[datetime.datetime] = None,
        stepSize: Optional[float] = None,
        stepIndex: Optional[int] = None,
        simulator: Optional[core.Simulator] = None,
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
        self.output[key].set(self.get_Y(X), stepIndex)
