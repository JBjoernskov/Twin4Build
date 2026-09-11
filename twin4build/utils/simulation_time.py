"""Shared simulation-timestep construction."""

from __future__ import annotations

import datetime
import math
from typing import List, Tuple, Union

import numpy as np


def get_simulation_timesteps(
    start_time: Union[List[datetime.datetime], datetime.datetime],
    end_time: Union[List[datetime.datetime], datetime.datetime],
    step_size: Union[List[int], int],
) -> Tuple[np.ndarray, np.ndarray, int, List[int]]:
    """Generate second-based and datetime-based simulation timesteps."""
    if isinstance(start_time, datetime.datetime):
        start_time = [start_time]
    if isinstance(end_time, datetime.datetime):
        end_time = [end_time]
    if isinstance(step_size, int):
        step_size = [step_size]
    second_time_steps = []
    date_time_steps = []
    n_timesteps = []
    for start_time_, end_time_, step_size_ in zip(start_time, end_time, step_size):
        n_steps = math.floor((end_time_ - start_time_).total_seconds() / step_size_)
        second_time_steps.append([i * step_size_ for i in range(n_steps)])
        date_time_steps.append(
            [
                start_time_ + datetime.timedelta(seconds=i * step_size_)
                for i in range(n_steps)
            ]
        )
        n_timesteps.append(n_steps)
    max_timesteps = max(len(time_steps) for time_steps in second_time_steps)
    second_time_steps = [
        time_steps + [np.nan] * (max_timesteps - len(time_steps))
        for time_steps in second_time_steps
    ]
    date_time_steps = [
        time_steps + [np.nan] * (max_timesteps - len(time_steps))
        for time_steps in date_time_steps
    ]
    return (
        np.array(second_time_steps),
        np.array(date_time_steps),
        max_timesteps,
        n_timesteps,
    )
