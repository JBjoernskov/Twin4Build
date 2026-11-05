"""Readers: where a data source's raw series comes from (ADR 0001, decision 6).

A data source System takes one ``source`` setting holding a reader.  The
reader's class says where the data comes from -- a CSV or Excel file
(:class:`CsvSource`), a DataFrame (:class:`DataFrameSource`) or the
time-series database (:class:`DatabaseSource`) -- and the options about
reading live on it: the resampling rule, the gap limit, and the fill for the
steps the data leave without a value.

:meth:`Source.read` turns the run's windows into one row of values per
period on the run's grid, the way every data source has read its data so far:
one window per period, the declared fill applied to the period's own steps,
NaN where the data give nothing (an error unless the data source allows gaps).
"""

from __future__ import annotations

# Standard library imports
import copy
import datetime
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
from twin4build.utils.data_loaders.load import (
    DEFAULT_DT_LIMIT,
    RESAMPLE_METHODS,
    contiguous_spans,
    load_from_database,
    load_from_spreadsheet,
    sample_from_df,
)
from twin4build.utils.get_main_dir import get_main_dir

#: How the steps the data leave without a value are filled (``None``: they
#: stay NaN).  ``linear``: straight lines inside the period's readings;
#: ``interpolate``: the same, the nearest reading held at the edges and the
#: nearest one outside the period where it has none; ``constant``: the last
#: value held; ``copy_week``: the same steps of an earlier week.
FILLS = ("linear", "interpolate", "constant", "copy_week")
#: How many weeks back ``copy_week`` looks.
COPY_WEEKS = 16
#: How far ``interpolate`` / ``constant`` look outside a period without data.
HOLD_WEEKS = 104


@dataclass
class Reading:
    """What :meth:`Source.read` gives for a run: ``values``, ``(n_s, n_t)``
    floats on the run's grid (a period's steps past its own end hold its last
    value); ``filled``, ``(n_s, n_t)`` booleans, the steps the declared fill
    supplied; ``raw``, one series per period as read (NaN where the data give
    nothing); and ``n_t``, the number of steps of the longest period."""

    values: np.ndarray
    filled: np.ndarray
    raw: List[pd.Series]
    n_t: int


class Source:
    """Base class of the readers.

    Args:
        resample_method: How readings are put on the step grid: one of
            ``RESAMPLE_METHODS``.
        dt_limit: The longest gap [s] the resampling bridges; a longer one is
            NaN.
        fill: How the steps the data leave without a value are filled: one of
            :data:`FILLS`, or ``None``.
        cache: Whether the loaders keep the series they read on disk.
    """

    TYPE = ""

    def __init__(self, *, resample_method: str = "linear", dt_limit=DEFAULT_DT_LIMIT,
                 fill: Optional[str] = None, cache: bool = True):
        if resample_method not in RESAMPLE_METHODS:
            raise ValueError(f"resample_method must be one of {RESAMPLE_METHODS}, got {resample_method!r}")
        if fill is not None and fill not in FILLS:
            raise ValueError(f"fill must be one of {FILLS} or None, got {fill!r}")
        self.resample_method = resample_method
        self.dt_limit = dt_limit
        self.fill = fill
        self.cache = cache
        self._last = None  # (windows, transformation, allow_missing) -> Reading

    # -- what a reader says about itself ---------------------------------------

    def describe(self) -> str:
        """A short name of the source, for messages."""
        return type(self).__name__

    def to_config(self) -> Dict[str, Any]:
        """The reader as a dict a saved model holds."""
        return {
            "type": self.TYPE,
            "resample_method": self.resample_method,
            "dt_limit": self.dt_limit,
            "fill": self.fill,
            **self._own_config(),
        }

    def _own_config(self) -> Dict[str, Any]:
        return {}

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        new._last = None
        memo[id(self)] = new
        return new

    # -- reading -----------------------------------------------------------------

    def window(self, start_time, end_time, step_size, fetch_span=None) -> pd.Series:
        """The values of one window on its step grid (NaN where the data give
        nothing). Implemented by each reader."""
        raise NotImplementedError

    def read(self, start_time: Sequence, end_time: Sequence, step_size: Sequence, *,
             transformation: Optional[Callable] = None, allow_missing: bool = False,
             owner: str = "") -> Reading:
        """One row of values per period on the run's grid.

        Args:
            start_time, end_time, step_size: The run's periods.
            transformation: Applied to every value read.
            allow_missing: A step left without a value is 0 instead of an error
                (the series kept in :attr:`Reading.raw` keeps the NaN).
            owner: The data source's id, for messages.
        """
        windows = tuple(zip(start_time, end_time, step_size))
        key = (windows, id(transformation), bool(allow_missing))
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        from twin4build.utils.simulation_time import get_simulation_timesteps

        spans = contiguous_spans(start_time, end_time)
        raw = []
        for start, end, step in windows:
            span = next((a, b) for a, b in spans if a <= start and end <= b)
            series = self.window(start, end, step, fetch_span=span)
            if transformation is not None:
                series = series.apply(transformation)
            raw.append(series)

        _, _, n_t, period_steps = get_simulation_timesteps(start_time, end_time, step_size)
        values = np.full((len(raw), n_t), np.nan)
        for b, series in enumerate(raw):
            size = len(series.index)
            if size == 0 and (self.fill is not None or allow_missing):
                continue
            if size == 0:
                raise ValueError(
                    f'No data for "{owner}" ({self.describe()}) in the window '
                    f"{start_time[b]} -> {end_time[b]}. Check the timeseries id / table and that the point "
                    "holds numeric values in that period."
                )
            n = min(size, n_t)
            values[b, :n] = np.asarray(series.values, dtype=float)[:n]

        # A period's own steps are its first ``period_steps[p]``; the rest pad
        # a shorter period to the longest one (stepped together, never kept).
        own = np.zeros(values.shape, dtype=bool)
        for b, n_own in enumerate(period_steps):
            own[b, : int(n_own)] = True
        before = np.isnan(values) & own
        if self.fill is not None and before.any():
            filled_values = self._fill(np.where(own, values, np.nan), start_time, end_time, step_size, transformation)
            values = np.where(own, filled_values, values)
        filled = before & ~np.isnan(values)
        for b, n_own in enumerate(period_steps):
            n_own = int(n_own)
            if 0 < n_own < values.shape[1]:
                values[b, n_own:] = values[b, n_own - 1]
        if filled.any():
            from twin4build.utils.logger import LOGGER

            LOGGER.warning(
                "%s: %d step(s) without data filled by the declared fill %r (periods %s); "
                "the measurements they reach are not scored",
                owner, int(filled.sum()), self.fill, np.flatnonzero(filled.any(axis=1)).tolist(),
            )

        nan_mask = np.isnan(values) & own
        values = np.where(np.isnan(values) & ~own, 0.0, values)
        if nan_mask.any() and allow_missing:
            values = np.where(nan_mask, 0.0, values)
            nan_mask = np.zeros_like(nan_mask)
        if nan_mask.any():
            nan_count = int(nan_mask.sum())
            rows = np.where(nan_mask.any(axis=1))[0]
            first = [
                str(pd.Timestamp(start_time[b]) + pd.Timedelta(seconds=float(np.flatnonzero(nan_mask[b])[0]) * float(step_size[b])))
                for b in rows[:3]
            ]
            declared = f" (the declared fill {self.fill!r} left these)" if self.fill else ""
            raise AssertionError(
                f"Values contain {nan_count} NaN(s) ({nan_count / values.size * 100:.1f}%) in "
                f"'{owner}' ({self.describe()}, batch indices: {rows.tolist()}, first missing steps: {first}). "
                f"No reading within dt_limit={self.dt_limit} s, or beyond the first or last reading; "
                f"a source with such gaps must declare a fill (one of {FILLS}){declared}."
            )
        reading = Reading(values=values, filled=filled, raw=raw, n_t=n_t)
        self._last = (key, reading)
        return reading

    def _fill(self, values, start_time, end_time, step_size, transformation) -> np.ndarray:
        """``values`` (``(n_s, n_t)``, NaN where the data give no value) with
        :attr:`fill` applied to the NaN steps only."""
        out = values.copy()
        for b in range(out.shape[0]):
            row = out[b]
            missing = np.isnan(row)
            if not missing.any():
                continue
            if self.fill == "linear":
                ok = np.flatnonzero(~missing)
                if len(ok) >= 2:
                    steps = np.arange(len(row))
                    inside = missing & (steps > ok[0]) & (steps < ok[-1])
                    row[inside] = np.interp(np.flatnonzero(inside), ok, row[ok])
            elif self.fill == "interpolate":
                ok = np.flatnonzero(~missing)
                if len(ok):
                    row[missing] = np.interp(np.flatnonzero(missing), ok, row[ok])
                else:
                    value = self._nearest_outside(start_time[b], end_time[b], step_size[b], transformation)
                    if value is not None:
                        row[:] = value
            elif self.fill == "constant":
                ok = np.flatnonzero(~missing)
                if len(ok):
                    idx = np.maximum.accumulate(np.where(~missing, np.arange(len(row)), -1))
                    idx[idx < 0] = ok[0]
                    row[missing] = row[idx[missing]]
                else:
                    value = self._nearest_outside(start_time[b], end_time[b], step_size[b], transformation)
                    if value is not None:
                        row[:] = value
            elif self.fill == "copy_week":
                week = datetime.timedelta(days=7)
                for k in range(1, COPY_WEEKS + 1):
                    missing = np.isnan(row)
                    if not missing.any():
                        break
                    earlier = self._load_window(start_time[b] - k * week, end_time[b] - k * week, step_size[b], transformation)
                    if earlier is None:
                        continue
                    m = min(len(earlier), len(row))
                    take = missing[:m] & ~np.isnan(earlier[:m])
                    row[:m][take] = earlier[:m][take]
            out[b] = row
        return out

    def _nearest_outside(self, start_time, end_time, step_size, transformation) -> Optional[float]:
        """The last value before ``start_time``, else the first after
        ``end_time``, within :data:`HOLD_WEEKS` weeks."""
        reach = datetime.timedelta(days=7 * HOLD_WEEKS)
        earlier = self._load_window(start_time - reach, start_time, step_size, transformation)
        if earlier is not None and np.isfinite(earlier).any():
            return float(earlier[np.flatnonzero(np.isfinite(earlier))[-1]])
        later = self._load_window(end_time, end_time + reach, step_size, transformation)
        if later is not None and np.isfinite(later).any():
            return float(later[np.flatnonzero(np.isfinite(later))[0]])
        return None

    def _load_window(self, start_time, end_time, step_size, transformation):
        """The values of another window, or ``None`` when it has none."""
        try:
            series = self.window(start_time, end_time, step_size)
        except Exception:  # noqa: BLE001 -- no data in that window
            return None
        if len(series.index) == 0:
            return None
        if transformation is not None:
            series = series.apply(transformation)
        return np.asarray(series.values, dtype=float)


class CsvSource(Source):
    """A CSV or Excel file.

    Args:
        path: The file. A relative path is read from the directory of the
            main script.
        date_column: The column (name or index) of the timestamps.
        value_column: The column (name or index) of the values.
        **options: :class:`Source`'s reading options.
    """

    TYPE = "CsvSource"

    def __init__(self, path: str, date_column=0, value_column=1, **options):
        super().__init__(**options)
        self.path = path
        self.date_column = date_column
        self.value_column = value_column

    @property
    def resolved_path(self) -> str:
        if os.path.isabs(self.path):
            return self.path
        return os.path.join(get_main_dir(), self.path)

    def describe(self) -> str:
        return f"file {self.path!r}"

    def _own_config(self):
        return {"path": self.path, "date_column": self.date_column, "value_column": self.value_column}

    def window(self, start_time, end_time, step_size, fetch_span=None) -> pd.Series:
        return load_from_spreadsheet(
            self.resolved_path, self.date_column, self.value_column, step_size=step_size,
            start_time=start_time, end_time=end_time, cache_root=get_main_dir(), cache=self.cache,
            resample_method=self.resample_method, dt_limit=self.dt_limit,
        )


class DataFrameSource(Source):
    """A pandas Series, or a DataFrame whose first column holds the values,
    indexed by timestamps. A saved model keeps the reader's options, not the
    data: set ``df`` again after loading.

    Args:
        df: The data.
        **options: :class:`Source`'s reading options.
    """

    TYPE = "DataFrameSource"

    def __init__(self, df=None, **options):
        super().__init__(**options)
        self.df = df

    def describe(self) -> str:
        return "a DataFrame"

    def window(self, start_time, end_time, step_size, fetch_span=None) -> pd.Series:
        if self.df is None:
            raise ValueError("DataFrameSource: no DataFrame (a saved model does not keep one; set it again)")
        frame = self.df.to_frame() if isinstance(self.df, pd.Series) else self.df.copy()
        frame = frame.reset_index()
        sampled = sample_from_df(
            frame, date_column=0, value_column=1, step_size=step_size, start_time=start_time,
            end_time=end_time, resample_method=self.resample_method, dt_limit=self.dt_limit,
        )
        return sampled[sampled.columns[0]]


class DatabaseSource(Source):
    """A point of the time-series database.

    Args:
        uuid: The point's id.
        dbconfig: The connection: the keyword arguments of
            ``load_from_database`` (host, port, name, user, password, table).
            Often set later for the whole model (``Model.set_dbconfigs``);
            never saved with the model.
        **options: :class:`Source`'s reading options.
    """

    TYPE = "DatabaseSource"

    def __init__(self, uuid: str, dbconfig: Optional[dict] = None, **options):
        super().__init__(**options)
        self.uuid = uuid
        self.dbconfig = dbconfig

    def describe(self) -> str:
        return f"uuid {self.uuid!r}"

    def _own_config(self):
        return {"uuid": self.uuid}

    def window(self, start_time, end_time, step_size, fetch_span=None) -> pd.Series:
        if self.dbconfig is None:
            raise ValueError(f"{self.describe()}: no database configuration (set_dbconfig)")
        return load_from_database(
            step_size=step_size, start_time=start_time, end_time=end_time, cache_root=get_main_dir(),
            cache=self.cache, sensor_id=self.uuid, fetch_span=fetch_span,
            resample_method=self.resample_method, dt_limit=self.dt_limit, **self.dbconfig,
        )


_READERS = {cls.TYPE: cls for cls in (CsvSource, DataFrameSource, DatabaseSource)}


def reader_from_keywords(df=None, filename=None, date_column=0, value_column=1, uuid=None, dbconfig=None,
                         use_spreadsheet=False, use_database=False, **options):
    """The reader that the keywords of a data source written the old way stand
    for (``df`` first, then a file, then the database), as the old
    ``TimeSeriesInputSystem`` read them.  Transitional: goes when the
    schedule, the outdoor environment and the sensor take a ``source``."""
    if df is not None:
        return DataFrameSource(df, **options)
    if use_spreadsheet:
        return CsvSource(filename, date_column, value_column, **options)
    if use_database:
        return DatabaseSource(uuid, dbconfig, **options)
    raise ValueError("No data source: give a DataFrame, a file or a database uuid")


def as_source(value):
    """A reader from a reader, from the dict a saved model holds, or from a
    string: a database point's id (what a translator pattern gives, with
    ``sp.add_parameter("source", ...)``)."""
    if value is None or isinstance(value, Source):
        return value
    if isinstance(value, str):
        return DatabaseSource(value)
    if isinstance(value, dict):
        options = dict(value)
        cls = _READERS.get(options.pop("type", None))
        if cls is None:
            raise ValueError(f"Not a reader: {value!r}")
        return cls(**options)
    raise TypeError(
        f"A data source's source is a reader (CsvSource, DataFrameSource, DatabaseSource) or a database "
        f"point's id, got {value!r}"
    )
