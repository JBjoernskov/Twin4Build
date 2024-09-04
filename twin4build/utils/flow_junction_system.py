"""
Flow Junction System Module

This module defines a FlowJunctionSystem class which models the behavior of a flow junction
in a heating, ventilation, and air conditioning (HVAC) system.

Classes:
    FlowJunctionSystem: Represents a flow junction system with configurable operation modes.
"""

import twin4build.saref4syst.system as system
from numpy import NaN

class FlowJunctionSystem(system.System):
    """
    A class representing a flow junction system in an HVAC setup.

    This class models the behavior of a flow junction, which can operate in different modes
    (e.g., supply or exhaust) and calculates output flow rates and temperatures based on inputs.

    Attributes:
        operationMode (str): The operation mode of the flow junction (e.g., "return").
        _config (dict): Configuration dictionary for the system.

    Args:
        operationMode (str, optional): The operation mode of the flow junction. Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent System class.
    """

    def __init__(self, operationMode=None, **kwargs):
        """
        Initialize the FlowJunctionSystem.

        Args:
            operationMode (str, optional): The operation mode of the flow junction. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent System class.
        """
        super().__init__(**kwargs)
        self.operationMode = operationMode
        self._config = {"parameters": []}

    @property
    def config(self):
        """
        Get the configuration of the FlowJunctionSystem.

        Returns:
            dict: The configuration dictionary.
        """
        return self._config

    def cache(self, startTime=None, endTime=None, stepSize=None):
        """
        Cache method (placeholder).

        Args:
            startTime: The start time for caching.
            endTime: The end time for caching.
            stepSize: The step size for caching.
        """
        pass

    def initialize(self, startTime=None, endTime=None, stepSize=None, model=None):
        """
        Initialize the FlowJunctionSystem (placeholder).

        Args:
            startTime: The start time for initialization.
            endTime: The end time for initialization.
            stepSize: The step size for initialization.
            model: The model object, if any.
        """
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        """
        Perform a single step of the simulation.

        This method calculates the output flow rate and temperature based on the input flow rates
        and temperatures. The behavior depends on the operation mode of the system.

        Args:
            secondTime: The current time in seconds.
            dateTime: The current date and time.
            stepSize: The size of the time step.

        Note:
            - The output flow rate is the sum of all input flow rates.
            - If the operation mode is "return" and the output flow rate is non-zero,
              the output temperature is calculated as a weighted average of input temperatures.
            - If the conditions are not met, the output temperature is set to NaN.
        """
        self.output["flowRate"] = sum(v for k, v in self.input.items() if "flowRate" in k)
        
        if self.output["flowRate"] != 0 and self.operationMode == "return":
            self.output["flowTemperatureOut"] = sum(
                v * self.input[k.replace("flowRate", "flowTemperatureIn")]
                for k, v in self.input.items() if "flowRate" in k
            ) / self.output["flowRate"]
        else:
            self.output["flowTemperatureOut"] = NaN