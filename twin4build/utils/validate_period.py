import datetime
from typing import Union, List, Tuple

def validate_period(start_time: Union[datetime.datetime, List[datetime.datetime]], end_time: Union[datetime.datetime, List[datetime.datetime]], step_size: Union[int, List[int]]) -> Tuple[List[datetime.datetime], List[datetime.datetime], List[int]]:

    # Check if start_time, end_time, and step_size are lists or datetime.datetime objects and if they are, check if they are the same length
    assert isinstance(start_time, (list, datetime.datetime)), "start_time must be a list of datetime.datetime objects or a single datetime.datetime object"
    assert isinstance(end_time, (list, datetime.datetime)), "end_time must be a list of datetime.datetime objects or a single datetime.datetime object"
    assert isinstance(step_size, (list, int)), "step_size must be a list of integers or a single integer"
    if isinstance(start_time, list) or isinstance(end_time, list) or isinstance(step_size, list):
        assert isinstance(start_time, list) and isinstance(end_time, list), "if start_time or end_time are lists, they must both be lists"
        if isinstance(step_size, int):
            step_size = [step_size] * len(start_time)
        assert len(start_time) == len(end_time) == len(step_size), "start_time, end_time, and step_size must be the same length"
    else:
        assert isinstance(start_time, datetime.datetime), "start_time must be a datetime.datetime object or list of datetime.datetime objects"
        assert isinstance(end_time, datetime.datetime), "end_time must be a datetime.datetime object or list of datetime.datetime objects"
        assert isinstance(step_size, int), "step_size must be an integer or list of integers"
        start_time = [start_time]
        end_time = [end_time]
        step_size = [step_size]

    return start_time, end_time, step_size