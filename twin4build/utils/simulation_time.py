"""Shared simulation-timestep construction."""

from __future__ import annotations

import datetime
import math
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def _elapsed_seconds(start: datetime.datetime, end: datetime.datetime) -> float:
    """Seconds from ``start`` to ``end`` in absolute time.  Two aware
    datetimes with one ``tzinfo`` subtract as wall-clock times in Python, an
    hour off across a DST change (#243); in UTC they do not."""
    if start.tzinfo is not None and end.tzinfo is not None:
        return (end.astimezone(datetime.timezone.utc) - start.astimezone(datetime.timezone.utc)).total_seconds()
    return (end - start).total_seconds()


def _steps(start: datetime.datetime, n_steps: int, step_size: float) -> List[datetime.datetime]:
    """``n_steps`` times ``step_size`` seconds of absolute time apart from
    ``start``, in ``start``'s time zone (the local time a DST change moves);
    one vectorised conversion, not one ``astimezone`` per step."""
    if start.tzinfo is None:
        return [start + datetime.timedelta(seconds=i * step_size) for i in range(n_steps)]
    utc = pd.date_range(start.astimezone(datetime.timezone.utc), periods=n_steps, freq=pd.Timedelta(seconds=step_size))
    return list(utc.tz_convert(start.tzinfo).to_pydatetime())


def get_simulation_timesteps(
    start_time: Union[List[datetime.datetime], datetime.datetime],
    end_time: Union[List[datetime.datetime], datetime.datetime],
    step_size: Union[List[int], int],
) -> Tuple[np.ndarray, np.ndarray, int, List[int]]:
    """Generate second-based and datetime-based simulation timesteps.

    The steps are ``step_size`` seconds of absolute time apart, and a period
    holds as many as fit in its real duration: across a DST change a day is
    23 or 25 hours long, as long as the data loaded for it.  The datetimes
    are local times in the start's time zone."""
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
        n_steps = math.floor(_elapsed_seconds(start_time_, end_time_) / step_size_)
        second_time_steps.append([i * step_size_ for i in range(n_steps)])
        date_time_steps.append(_steps(start_time_, n_steps, step_size_))
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
