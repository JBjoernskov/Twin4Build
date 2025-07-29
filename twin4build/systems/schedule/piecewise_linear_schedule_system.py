# Standard library imports
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np

# Local application imports
import twin4build.core as core
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
from twin4build.translator.translator import Node, SignaturePattern


def get_signature_pattern() -> SignaturePattern:
    """Create a signature pattern for PiecewiseLinearScheduleSystem.

    Returns:
        SignaturePattern: Pattern matching Schedule core class with priority 0.
    """
    node0 = Node(cls=(core.namespace.S4BLDG.Schedule,))
    sp = SignaturePattern(
        semantic_model_=core.ontologies,
        ownedBy="PiecewiseLinearScheduleSystem",
        priority=0,
    )
    sp.add_modeled_node(node0)
    return sp


class PiecewiseLinearScheduleSystem(PiecewiseLinearSystem, ScheduleSystem):
    """A schedule system using piecewise linear interpolation.

    This class combines functionality from PiecewiseLinearSystem and ScheduleSystem
    to create a scheduling system that interpolates between schedule points using
    piecewise linear functions. It supports different schedules for weekdays,
    weekends, and individual days of the week.

    Attributes:
        sp (List[SignaturePattern]): List of signature patterns for component matching.
        input (Dict[str, Any]): Input values for interpolation.
        output (Dict[str, Any]): Output values after interpolation.
        _config (Dict[str, List[str]]): Configuration parameters for the schedule system.

    Key Components:
        - Supports multiple schedule types (weekday, weekend, individual days)
        - Uses piecewise linear interpolation between schedule points
        - Configurable noise addition
        - Real-time schedule value calculation
    """

    sp = [get_signature_pattern()]

    def __init__(self, **kwargs) -> None:
        """Initialize the piecewise linear schedule system.

        Args:
            **kwargs: Keyword arguments passed to parent classes.
                Supported parameters include:
                - weekDayRulesetDict: Schedule rules for weekdays
                - weekendRulesetDict: Schedule rules for weekends
                - mondayRulesetDict: Schedule rules for Monday
                - tuesdayRulesetDict: Schedule rules for Tuesday
                - wednesdayRulesetDict: Schedule rules for Wednesday
                - thursdayRulesetDict: Schedule rules for Thursday
                - fridayRulesetDict: Schedule rules for Friday
                - saturdayRulesetDict: Schedule rules for Saturday
                - sundayRulesetDict: Schedule rules for Sunday
                - add_noise: Whether to add noise to schedule values
        """
        super().__init__(**kwargs)
        self.input = {}
        self.output = {}
        self._config = {
            "parameters": [
                "weekDayRulesetDict",
                "weekendRulesetDict",
                "mondayRulesetDict",
                "tuesdayRulesetDict",
                "wednesdayRulesetDict",
                "thursdayRulesetDict",
                "fridayRulesetDict",
                "saturdayRulesetDict",
                "sundayRulesetDict",
                "add_noise",
            ]
        }

    @property
    def config(self) -> Dict[str, List[str]]:
        """Get the configuration parameters.

        Returns:
            Dict[str, List[str]]: Dictionary containing configuration parameter names.
        """
        return self._config

    def do_step(
        self,
        secondTime: Optional[float] = None,
        dateTime: Optional[datetime.datetime] = None,
        stepSize: Optional[float] = None,
        stepIndex: Optional[int] = None,
    ) -> None:
        """Execute one time step of the schedule system.

        Gets the schedule value for the current time, updates the interpolation
        points, and calculates the output value using piecewise linear interpolation.

        Args:
            secondTime (Optional[float], optional): Current simulation time in seconds.
                Defaults to None.
            dateTime (Optional[datetime.datetime], optional): Current simulation datetime.
                Defaults to None.
            stepSize (Optional[float], optional): Time step size in seconds.
                Defaults to None.
        """
        schedule_value = self.get_schedule_value(dateTime)
        self.XY = np.array([schedule_value["X"], schedule_value["Y"]]).transpose()
        self.get_a_b_vectors()

        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        self.output[key].set(self.get_Y(X), stepIndex)
