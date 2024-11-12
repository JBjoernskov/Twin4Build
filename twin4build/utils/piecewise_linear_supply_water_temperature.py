from twin4build.saref4syst.system import System
from pwlf import PiecewiseLinFit
from typing import Optional, Dict, Any, Union, List
import datetime
import pandas as pd
import numpy as np

class PiecewiseLinearSupplyWaterTemperatureSystem(System):
    """A system for modeling supply water temperature using piecewise linear functions.
    
    This class implements a temperature control system that uses different piecewise
    linear models for normal operation and boost periods. It supports calibration
    with input-output data and provides time-dependent temperature predictions.

    Attributes:
        model (Dict[str, PiecewiseLinFit]): Dictionary containing fitted models for
            "normal" and "boost" operation modes.
        _config (Dict[str, List[str]]): Configuration parameters.

    Key Components:
        - Separate models for normal and boost periods
        - Time-dependent model selection
        - Piecewise linear fitting for each operation mode
        - Configurable number of line segments per mode
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the supply water temperature system.

        Args:
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._config = {"parameters": []}

    @property
    def config(self) -> Dict[str, List[str]]:
        """Get the configuration parameters.

        Returns:
            Dict[str, List[str]]: Dictionary containing configuration parameter names.
        """
        return self._config

    def cache(self, startTime: Optional[datetime.datetime] = None,
             endTime: Optional[datetime.datetime] = None,
             stepSize: Optional[float] = None) -> None:
        """Cache method (placeholder for interface compatibility).

        Args:
            startTime (Optional[datetime.datetime], optional): Start time. Defaults to None.
            endTime (Optional[datetime.datetime], optional): End time. Defaults to None.
            stepSize (Optional[float], optional): Time step size. Defaults to None.
        """
        pass

    def initialize(self, startTime: Optional[datetime.datetime] = None,
                  endTime: Optional[datetime.datetime] = None,
                  stepSize: Optional[float] = None,
                  model: Optional[Any] = None) -> None:
        """Initialize method (placeholder for interface compatibility).

        Args:
            startTime (Optional[datetime.datetime], optional): Start time. Defaults to None.
            endTime (Optional[datetime.datetime], optional): End time. Defaults to None.
            stepSize (Optional[float], optional): Time step size. Defaults to None.
            model (Optional[Any], optional): Model object. Defaults to None.
        """
        pass

    def do_step(self, secondTime: Optional[float] = None,
                dateTime: Optional[datetime.datetime] = None,
                stepSize: Optional[float] = None) -> None:
        """Perform a single prediction step using the calibrated model.

        Uses different models based on the time of day:
        - Boost model: Between 5:00 and 7:00
        - Normal model: All other times

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
        if dateTime.hour >= 5 and dateTime.hour <= 7:
            self.output[key].set(self.model["boost"].predict(X)[0])
        else:
            self.output[key].set(self.model["normal"].predict(X)[0])

    def calibrate(self, input: Dict[str, pd.DataFrame], 
                 output: Dict[str, np.ndarray],
                 n_line_segments: Dict[str, int]) -> None:
        """Calibrate piecewise linear models for each operation mode.

        Fits separate models for normal and boost operation using the provided
        input-output data and specified number of line segments.

        Args:
            input (Dict[str, pd.DataFrame]): Input data for each operation mode.
            output (Dict[str, np.ndarray]): Target output data for each mode.
            n_line_segments (Dict[str, int]): Number of line segments for each mode.
        """
        self.model = {}
        for key in input.keys():
            X = input[key].iloc[:,0]
            self.model[key] = PiecewiseLinFit(X, output[key])
            self.model[key].fit(n_line_segments[key])
