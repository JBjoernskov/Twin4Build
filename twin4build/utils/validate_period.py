# Standard library imports
import datetime
from typing import List, Tuple, Union


def validate_period(
    start_time: Union[datetime.datetime, List[datetime.datetime]],
    end_time: Union[datetime.datetime, List[datetime.datetime]],
    step_size: Union[int, List[int]],
) -> Tuple[List[datetime.datetime], List[datetime.datetime], List[int]]:

    # Check if start_time, end_time, and step_size are lists or datetime.datetime objects and if they are, check if they are the same length
    assert isinstance(
        start_time, (list, datetime.datetime)
    ), "start_time must be a list of datetime.datetime objects or a single datetime.datetime object"
    assert isinstance(
        end_time, (list, datetime.datetime)
    ), "end_time must be a list of datetime.datetime objects or a single datetime.datetime object"
    assert isinstance(
        step_size, (list, int)
    ), "step_size must be a list of integers or a single integer"
    if (
        isinstance(start_time, list)
        or isinstance(end_time, list)
        or isinstance(step_size, list)
    ):
        assert isinstance(start_time, list) and isinstance(
            end_time, list
        ), "if start_time or end_time are lists, they must both be lists"
        if isinstance(step_size, int):
            step_size = [step_size] * len(start_time)
        assert (
            len(start_time) == len(end_time) == len(step_size)
        ), "start_time, end_time, and step_size must be the same length"
    else:
        assert isinstance(
            start_time, datetime.datetime
        ), "start_time must be a datetime.datetime object or list of datetime.datetime objects"
        assert isinstance(
            end_time, datetime.datetime
        ), "end_time must be a datetime.datetime object or list of datetime.datetime objects"
        assert isinstance(
            step_size, int
        ), "step_size must be an integer or list of integers"
        start_time = [start_time]
        end_time = [end_time]
        step_size = [step_size]

    # Semantic checks: each (start, end, step) triple must describe a
    # forward-flowing, strictly positive-stepped period.  Catching this
    # here means every caller (Simulator, Estimator, Optimizer, ...) gets
    # the same error message instead of failing later with a confusing
    # secondary error from ``get_simulation_timesteps`` or downstream
    # tensor sizing.
    for i, (s, e, dt) in enumerate(zip(start_time, end_time, step_size)):
        if e <= s:
            raise ValueError(
                f"Invalid period at index {i}: end_time ({e!r}) must be "
                f"strictly after start_time ({s!r})."
            )
        if dt <= 0:
            raise ValueError(
                f"Invalid step_size at index {i}: step_size must be a "
                f"positive integer (got {dt!r})."
            )

    return start_time, end_time, step_size
