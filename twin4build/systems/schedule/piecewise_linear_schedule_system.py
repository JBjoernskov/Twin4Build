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


class PiecewiseLinearScheduleSystem(PiecewiseLinearSystem, ScheduleSystem):
    """A schedule system using piecewise linear interpolation.

    This class combines functionality from PiecewiseLinearSystem and ScheduleSystem
    to create a scheduling system that interpolates between schedule points using
    piecewise linear functions. It supports different schedules for weekdays,
    weekends, and individual days of the week.

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


    Key Components:
        - Supports multiple schedule types (weekday, weekend, individual days)
        - Uses piecewise linear interpolation between schedule points
        - Configurable noise addition
        - Real-time schedule value calculation
    """
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

        # Define inputs and outputs as private variables
        self._input = {}
        self._output = {}
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

    @property
    def input(self) -> dict:
        """
        Get the input ports of the piecewise linear schedule system.

        Returns:
            dict: Dictionary containing input ports (configured dynamically)
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the piecewise linear schedule system.

        Returns:
            dict: Dictionary containing output ports (configured dynamically)
        """
        return self._output

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Execute one time step of the schedule system.

        Gets the schedule value for the current time, updates the interpolation
        points, and calculates the output value using piecewise linear interpolation.

        Args:
            second_time: Current simulation time in seconds.
            date_time: Current simulation date_time.
            step_size: Time step size in seconds.
            step_index: Current simulation step index.
        """
        schedule_value = self.get_schedule_value(date_time)
        self.XY = np.array([schedule_value["X"], schedule_value["Y"]]).transpose()
        self.get_a_b_vectors()

        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        self.output[key].set(self.get_Y(X), step_index)


def saref_signature_pattern() -> SignaturePattern:
    """
    Get the SAREF signature pattern of the piecewise linear schedule component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the piecewise linear schedule component.
    """
    node0 = Node(cls=(core.namespace.S4BLDG.Schedule,))
    sp = SignaturePattern(
        semantic_model_=core.ontologies,
        id="piecewise_linear_schedule_signature_pattern",
    )
    sp.add_modeled_node(node0)
    return sp


def brick_signature_pattern() -> SignaturePattern:
    """
    Get the BRICK signature pattern of the piecewise linear schedule component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the piecewise linear schedule component.
    """
    node0 = Node(cls=core.namespace.BRICK.Schedule)
    sp = SignaturePattern(
        semantic_model_=core.ontologies,
        id="piecewise_linear_schedule_signature_pattern_brick",
    )
    sp.add_modeled_node(node0)
    return sp


PiecewiseLinearScheduleSystem.add_signature_pattern(brick_signature_pattern())
PiecewiseLinearScheduleSystem.add_signature_pattern(saref_signature_pattern())
