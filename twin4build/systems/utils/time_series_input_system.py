from __future__ import annotations

# Standard library imports
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.data_loaders.load import (
    contiguous_spans,
    DEFAULT_DT_LIMIT,
    RESAMPLE_METHODS,
    load_from_database,
    load_from_spreadsheet,
    sample_from_df,
)
from twin4build.utils.get_main_dir import get_main_dir
from twin4build.utils.logger import LOGGER
from twin4build.utils.deprecation import deprecate_args, deprecate_name


class TimeSeriesInputSystem(core.System):
    """A system for reading and processing time series data from files or DataFrames.

    This component provides functionality to handle time series data inputs, either from
    CSV files or pandas DataFrames. It supports automatic file path resolution and
    caching of processed data for improved performance.

    Args:
        df: Input dataframe containing time series data. Must have date_time index and value column.
        filename: Path to the CSV file. Can be absolute or relative to cache_root. If relative, will try both current directory and cache_root.
        datecolumn: Index of the date column (0-based). Defaults to 0.
        valuecolumn: Index of the value column (0-based). Defaults to 1.
        use_spreadsheet: Whether to use a spreadsheet for input. Defaults to False.
        use_database: Whether to use a database for input. Defaults to False.
        uuid: UUID for database operations.
        dbconfig: Database configuration parameters.
        cache: Whether to cache loaded/resampled data on disk for faster
            re-initialization. Defaults to True.
        transformation: Optional function applied to the loaded values
            (e.g. unit conversion). Defaults to None.
        resample_method: How the readings become step values: ``"linear"``
            (the value at the step), ``"constant"`` (the last reading at or
            before it) or ``"mean"`` (the mean over the step).  Defaults to
            ``"linear"``.
        dt_limit: The longest stretch between two readings [s] a step value
            may bridge (``None``: any gap inside the readings).  Nothing beyond
            the first or last reading is ever filled.  Defaults to six hours.
        fill: What fills the steps the readings do not give, for an input that
            must have a value at every step: ``None`` (default: such a step is
            an error), ``"linear"`` (straight lines across the gaps, never
            beyond the readings), ``"interpolate"`` (straight lines across the
            gaps, the nearest reading held beyond them; a period without a
            reading as ``"constant"``), ``"constant"`` (the last value forward, the
            first backward; a period without a reading holds the nearest one
            outside it, up to two years away) or ``"copy_week"`` (the same step one week earlier,
            then two, up to sixteen).  A fill is a decision about the data: the
            steps it filled are logged and kept in :attr:`filled`, and the
            estimator leaves every measurement they reach unscored.
        **kwargs: Additional keyword arguments
    """

    #: Deprecated keyword names and the ones that replaced them (see
    #: ``System.DEPRECATED_KEYWORDS``).
    DEPRECATED_KEYWORDS = {
        "datecolumn": "date_column",
        "valuecolumn": "value_column",
    }
    #: The fills an input may declare (``None``: no fill).
    FILLS = ("linear", "interpolate", "constant", "copy_week")
    #: How many weeks back ``"copy_week"`` looks.
    COPY_WEEKS = 16
    #: How far ``"constant"`` looks for a value to hold over a period without one.
    HOLD_WEEKS = 104

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        filename: Optional[str] = None,
        date_column: int = 0,
        value_column: int = 1,
        source: Optional[str] = None,
        use_spreadsheet: bool = False,
        use_database: bool = False,
        uuid: Optional[str] = None,
        dbconfig: Optional[Dict[str, Any]] = None,
        cache: Optional[bool] = True,
        transformation: Optional[callable] = None,
        resample_method: str = "linear",
        dt_limit: Optional[float] = DEFAULT_DT_LIMIT,
        fill: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the TimeSeriesInputSystem.

        Args:
            df: Input dataframe containing time series data. Must have date_time index and value column.
            filename: Path to the CSV file. Can be absolute or relative to cache_root. If relative, will try both current directory and cache_root.
            date_column: Index of the date column (0-based). Defaults to 0.
            value_column: Index of the value column (0-based). Defaults to 1.
            source: Preferred data-source selector: ``"spreadsheet"``, ``"database"``,
                ``"df"``, or ``None`` for auto-detect.
            use_spreadsheet / use_database: Legacy boolean source flags (deprecated;
                use ``source=``; removed in 2.1).
            uuid: UUID for database operations.
            dbconfig: Database configuration parameters.
            cache: Whether to cache loaded/resampled data on disk for faster
                re-initialization. Defaults to True.
            transformation: Optional function applied to the loaded values
                (e.g. unit conversion). Defaults to None.
            **kwargs: Additional keyword arguments passed to parent System class.

        Raises:
            AssertionError: If neither df nor filename is provided.
            ValueError: If the specified file cannot be found in any of the search paths.
        """

        for legacy_key, new_key in (
            ("useSpreadsheet", "use_spreadsheet"),
            ("useDatabase", "use_database"),
        ):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` has been removed. Use `{new_key}` instead."
                )

        legacy = deprecate_args(
            list(TimeSeriesInputSystem.DEPRECATED_KEYWORDS),
            list(TimeSeriesInputSystem.DEPRECATED_KEYWORDS.values()),
            [None] * len(TimeSeriesInputSystem.DEPRECATED_KEYWORDS),
            kwargs,
        )
        date_column = legacy.get("date_column", date_column)
        value_column = legacy.get("value_column", value_column)

        if source is not None:
            source = str(source).lower()
            assert source in {
                "spreadsheet",
                "database",
                "df",
            }, f"source must be 'spreadsheet', 'database', or 'df', got {source!r}"
            use_spreadsheet = source == "spreadsheet"
            use_database = source == "database"
        elif use_spreadsheet or use_database:
            deprecate_name(
                "use_spreadsheet/use_database",
                "source='spreadsheet'|'database'|'df'",
            )

        assert (
            use_spreadsheet == False or use_database == False
        ), "use_spreadsheet and use_database cannot both be True."
        super().__init__(**kwargs)
        assert (
            df is not None or filename is not None or uuid is not None
        ), f'Either "df" or "filename" or "uuid" must be provided as argument.'

        self._df_init = df
        self.df = []
        self._use_spreadsheet = use_spreadsheet
        self._use_database = use_database
        self._filename = filename
        self._date_column = date_column
        self._value_column = value_column
        self._uuid = uuid
        self._dbconfig = dbconfig
        self._cached_initialize_arguments = []
        self._cache_root = get_main_dir()
        self._cache = cache
        self._transformation = transformation
        assert resample_method in RESAMPLE_METHODS, f"resample_method must be one of {RESAMPLE_METHODS}, got {resample_method!r}"
        assert fill is None or fill in self.FILLS, f"fill must be None or one of {self.FILLS}, got {fill!r}"
        self.resample_method = resample_method
        self.dt_limit = dt_limit
        self.fill = fill
        #: Per period and step, the steps a declared fill gave a value (``(n_s, n_t)`` bool), after initialize.
        self.filled = None

        # Define inputs and outputs as private variables
        self._input = {}
        self._output = {"value": tps.Scalar(is_leaf=True)}

        if filename is not None:
            if os.path.isfile(filename):  # Absolute or relative was provided
                self._filename = filename
            else:  # Check if relative path to root was provided
                filename = filename.lstrip("/\\")
                filename_ = os.path.join(self._cache_root, filename)
                if os.path.isfile(filename_) == False:
                    raise (
                        ValueError(
                            f'Neither one of the following filenames exist: \n"{filename}"\n{filename_}'
                        )
                    )
                self._filename = filename_

        self._config = {
            "parameters": [],
            "spreadsheet": ["filename", "date_column", "value_column"],
            "database": ["uuid", "dbconfig"],
        }

    @property
    def config(self):
        """
        Get the configuration of the TimeSeriesInputSystem.

        Returns:
            dict: The configuration dictionary.
        """
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the time series input system.

        Returns:
            dict: Dictionary containing input ports (empty for leaf systems)
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the time series input system.

        Returns:
            dict: Dictionary containing output ports:
                - "value": Time series values [units depend on data]
        """
        return self._output

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """
        Get the processed input data containing time series values.
        """
        return self._df

    @df.setter
    def df(self, value: Optional[pd.DataFrame]) -> None:
        """
        Set the processed input data containing time series values.
        """
        self._df = value

    @property
    def filename(self) -> Optional[str]:
        """
        Get the path to the input CSV file (absolute or relative to root).
        """
        return self._filename

    @filename.setter
    def filename(self, value: Optional[str]) -> None:
        """
        Set the path to the input CSV file (absolute or relative to root).
        """
        self._filename = value

    @property
    def date_column(self) -> int:
        """
        Get the index of the date/time column (0-based).
        """
        return self._date_column

    @date_column.setter
    def date_column(self, value: int) -> None:
        """
        Set the index of the date/time column (0-based).
        """
        self._date_column = value

    @property
    def value_column(self) -> int:
        """
        Get the index of the value column (0-based).
        """
        return self._value_column

    @value_column.setter
    def value_column(self, value: int) -> None:
        """
        Set the index of the value column (0-based).
        """
        self._value_column = value

    @property
    def use_spreadsheet(self) -> bool:
        """
        Get whether to use a spreadsheet for input.
        """
        return self._use_spreadsheet

    @use_spreadsheet.setter
    def use_spreadsheet(self, value: bool) -> None:
        """
        Set whether to use a spreadsheet for input.
        """
        self._use_spreadsheet = value

    @property
    def use_database(self) -> bool:
        """
        Get whether to use a database for input.
        """
        return self._use_database

    @use_database.setter
    def use_database(self, value: bool) -> None:
        """
        Set whether to use a database for input.
        """
        self._use_database = value

    @property
    def uuid(self) -> Optional[str]:
        """
        Get the UUID for database operations.
        """
        return self._uuid

    @uuid.setter
    def uuid(self, value: Optional[str]) -> None:
        """
        Set the UUID for database operations.
        """
        self._uuid = value

    @property
    def dbconfig(self) -> Optional[Dict[str, Any]]:
        """
        Get the database configuration parameters.
        """
        return self._dbconfig

    @dbconfig.setter
    def dbconfig(self, value: Optional[Dict[str, Any]]) -> None:
        """
        Set the database configuration parameters.
        """
        self._dbconfig = value

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
    ) -> None:
        """
        Initialize the TimeSeriesInputSystem.

        Args:
            start_time (datetime.datetime): Start time for the simulation.
            end_time (datetime.datetime): End time for the simulation.
            step_size (int): Step size for the simulation.
        """
        #
        if len(self._cached_initialize_arguments) > 0 and len(
            self._cached_initialize_arguments
        ) == len(
            start_time
        ):  # Only check first element of tuple for length as all elements are the same length
            is_cached = all(
                start_time_ == c[0] and end_time_ == c[1] and step_size_ == c[2]
                for start_time_, end_time_, step_size_, c in zip(
                    start_time, end_time, step_size, self._cached_initialize_arguments
                )
            )
        else:
            is_cached = False

        # Fast path: the window is unchanged AND the (fixed) value tensor is
        # already built -> skip the expensive rebuild.  ``initialize`` is
        # otherwise re-run on every ``model.initialize`` (i.e. every estimator
        # objective / gradient / constraint evaluation), and even on a data
        # cache-hit it would rebuild the value array and rescan for NaNs --
        # ~0.05 s each, thousands of times, the dominant estimation cost.  The
        # time-series "state" is just the data, which is constant for a fixed
        # window, so skipping that rebuild is safe.  The output PORT must still
        # be re-initialized: this may run inside a functorch grad transform
        # (jacrev of the estimation objective), and ``do_step`` writes the port
        # tensor in place -- mutating a tensor captured from OUTSIDE the
        # transform is an error, so the (cheap) port init recreates it inside.
        if (
            is_cached
            and getattr(self, "values", None) is not None
            and getattr(self, "batch_size", None) == len(start_time)
        ):
            self.output["value"].initialize(
                n_t=self.n_timesteps,
                n_s=self.batch_size,
                n_c=1,
                values=self.values,
            )
            return

        if is_cached == False:
            self.df = []
            self._cached_initialize_arguments = []
            # Windows that lie together are fetched together (one query per
            # span into the loader's raw store, not one per window).
            spans = contiguous_spans(start_time, end_time)
            for start_time_, end_time_, step_size_ in zip(
                start_time, end_time, step_size
            ):
                # if (start_time_, end_time_, step_size_) not in self._cached_initialize_arguments:

                if self._df_init is None:
                    if self.use_spreadsheet:
                        df = load_from_spreadsheet(
                            self.filename,
                            self._date_column,
                            self._value_column,
                            step_size=step_size_,
                            start_time=start_time_,
                            end_time=end_time_,
                            cache_root=self._cache_root,
                            cache=self._cache,
                            resample_method=self.resample_method,
                            dt_limit=self.dt_limit,
                        )
                    elif self.use_database:
                        df = load_from_database(
                            step_size=step_size_,
                            start_time=start_time_,
                            end_time=end_time_,
                            cache_root=self._cache_root,
                            cache=self._cache,
                            sensor_id=self.uuid,
                            fetch_span=next(
                                (a, b) for a, b in spans if a <= start_time_ and end_time_ <= b
                            ),
                            resample_method=self.resample_method,
                            dt_limit=self.dt_limit,
                            **self.dbconfig,
                        )
                else:
                    df_ = self._df_init.copy()
                    df_.reset_index(inplace=True)
                    df = sample_from_df(
                        df_,
                        date_column=0,
                        value_column=1,
                        step_size=step_size_,
                        start_time=start_time_,
                        end_time=end_time_,
                        resample_method=self.resample_method,
                        dt_limit=self.dt_limit,
                    )
                    valuename = df.columns[
                        0
                    ]  # The value column is the first column as we set index to datecolumn
                    df = df[valuename]

                self._cached_initialize_arguments.append(
                    (start_time_, end_time_, step_size_)
                )
                if self._transformation is not None:
                    df = df.apply(self._transformation)
                self.df.append(df)

        _, _, max_timesteps, period_steps = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        # NaN where the data give no value: never a repeated value (the gaps are
        # filled below only by a declared ``fill``, or left to the scoring)
        values = np.full((len(self.df), max_timesteps), np.nan)
        for batch_index, df in enumerate(self.df):
            size = len(df.index)
            if size == 0 and (self.fill is not None or getattr(self, "allow_missing", False)):
                continue  # all NaN: filled below, or unscored
            if size == 0:
                # Surface the offending sensor instead of the opaque numpy
                # "could not broadcast input array from shape (0,0)" error
                # that used to escape from the assignment below.
                source = (
                    f"uuid={self.uuid!r}"
                    if self.uuid is not None
                    else f"filename={self.filename!r}"
                    if self.filename is not None
                    else "df"
                )
                raise ValueError(
                    f'No data for "{self.id}" ({source}) in the window '
                    f"{start_time[batch_index]} -> {end_time[batch_index]}. "
                    "Check the timeseries id / table and that the point "
                    "holds numeric values in that period."
                )
            n = min(size, max_timesteps)
            values[batch_index, :n] = np.asarray(df.values, dtype=float)[:n]

        # a period's own steps are its first ``period_steps[p]``; the rest pad a
        # shorter period to the longest one (stepped together, never kept)
        own = np.zeros(values.shape, dtype=bool)
        for b, n_own in enumerate(period_steps):
            own[b, : int(n_own)] = True
        before = np.isnan(values) & own
        if self.fill is not None and before.any():
            values = np.where(own, self._fill(np.where(own, values, np.nan), start_time, end_time, step_size), values)
        self.filled = before & ~np.isnan(values)
        for b, n_own in enumerate(period_steps):
            n_own = int(n_own)
            if 0 < n_own < values.shape[1]:
                # padding, not data: the period's last value (finite or not)
                values[b, n_own:] = values[b, n_own - 1]
        if self.filled.any():
            LOGGER.warning(
                "%s: %d step(s) without data filled by the declared fill %r (periods %s); "
                "the measurements they reach are not scored",
                self.id, int(self.filled.sum()), self.fill,
                np.flatnonzero(self.filled.any(axis=1)).tolist(),
            )

        nan_mask = np.isnan(values) & own
        values = np.where(np.isnan(values) & ~own, 0.0, values)  # padding past a period without a last value
        if nan_mask.any() and getattr(self, "allow_missing", False):
            # A scoring sensor may carry gaps (samples the objective must not
            # score, see ``_raw_residuals_from_meas``): the frame keeps the
            # NaNs, the port value must stay finite.
            values = np.where(nan_mask, 0.0, values)
            nan_mask = np.zeros_like(nan_mask)
        if nan_mask.any():
            nan_count = int(nan_mask.sum())
            nan_pct = nan_count / values.size * 100
            batch_indices = np.where(nan_mask.any(axis=1))[0]
            first = [
                str(pd.Timestamp(start_time[b]) + pd.Timedelta(seconds=float(np.flatnonzero(nan_mask[b])[0]) * float(step_size[b])))
                for b in batch_indices[:3]
            ]
            declared = f" (the declared fill {self.fill!r} left these)" if self.fill else ""
            raise AssertionError(
                f"Values contain {nan_count} NaN(s) ({nan_pct:.1f}%) in "
                f"TimeSeriesInput '{self.id}' (uuid: {self.uuid}, file: {self.filename}, "
                f"batch indices: {batch_indices.tolist()}, first missing steps: {first}). "
                f"No reading within dt_limit={self.dt_limit} s, or beyond the first or last reading; "
                f"an input with such gaps must declare a fill (one of {self.FILLS}){declared}."
            )

        self.n_timesteps = max_timesteps
        self.batch_size = len(start_time)
        # Convert values from (n_s, n_t) to time-first (n_t, n_s, n_c) where n_c=1
        # First transpose to (n_t, n_s), then unsqueeze to (n_t, n_s, 1)
        self.values = torch.tensor(values, dtype=tps.float_dtype()).T.unsqueeze(
            -1
        )  # (n_t, n_s, 1)
        self.output["value"].initialize(
            n_t=max_timesteps,
            n_s=len(start_time),
            n_c=1,
            values=self.values,
        )

    def _fill(self, values: np.ndarray, start_time, end_time, step_size) -> np.ndarray:
        """``values`` (``(n_s, n_t)``, NaN where the data give no value) with the
        declared :attr:`fill` applied to the NaN steps only."""
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
                # straight lines across the gaps, the nearest reading held at the edges, and the nearest one outside
                # the period where it has none
                ok = np.flatnonzero(~missing)
                if len(ok):
                    row[missing] = np.interp(np.flatnonzero(missing), ok, row[ok])
                else:
                    value = self._nearest_outside(start_time[b], end_time[b], step_size[b])
                    if value is not None:
                        row[:] = value
            elif self.fill == "constant":
                ok = np.flatnonzero(~missing)
                if len(ok):
                    idx = np.maximum.accumulate(np.where(~missing, np.arange(len(row)), -1))
                    idx[idx < 0] = ok[0]  # before the first value: the first value
                    row[missing] = row[idx[missing]]
                else:
                    # a period without a reading: the nearest one outside it
                    value = self._nearest_outside(start_time[b], end_time[b], step_size[b])
                    if value is not None:
                        row[:] = value
            elif self.fill == "copy_week":
                week = datetime.timedelta(days=7)
                for k in range(1, self.COPY_WEEKS + 1):
                    missing = np.isnan(row)
                    if not missing.any():
                        break
                    earlier = self._load_window(start_time[b] - k * week, end_time[b] - k * week, step_size[b])
                    if earlier is None:
                        continue
                    m = min(len(earlier), len(row))
                    take = missing[:m] & ~np.isnan(earlier[:m])
                    row[:m][take] = earlier[:m][take]
            out[b] = row
        return out

    def _nearest_outside(self, start_time, end_time, step_size) -> Optional[float]:
        """The last value before ``start_time``, else the first after
        ``end_time``, within :attr:`HOLD_WEEKS` weeks; ``None`` when there is none."""
        reach = datetime.timedelta(days=7 * self.HOLD_WEEKS)
        earlier = self._load_window(start_time - reach, start_time, step_size)
        if earlier is not None and np.isfinite(earlier).any():
            return float(earlier[np.flatnonzero(np.isfinite(earlier))[-1]])
        later = self._load_window(end_time, end_time + reach, step_size)
        if later is not None and np.isfinite(later).any():
            return float(later[np.flatnonzero(np.isfinite(later))[0]])
        return None

    def _load_window(self, start_time, end_time, step_size):
        """This input's values over another window (the same source, rule and
        limit, the transformation applied), or ``None`` when it has none."""
        try:
            if self._df_init is not None:
                df_ = self._df_init.copy()
                df_.reset_index(inplace=True)
                df = sample_from_df(df_, date_column=0, value_column=1, step_size=step_size, start_time=start_time,
                                    end_time=end_time, resample_method=self.resample_method, dt_limit=self.dt_limit)
                df = df[df.columns[0]]
            elif self.use_spreadsheet:
                df = load_from_spreadsheet(self.filename, self._date_column, self._value_column, step_size=step_size,
                                           start_time=start_time, end_time=end_time, cache_root=self._cache_root,
                                           cache=self._cache, resample_method=self.resample_method, dt_limit=self.dt_limit)
            elif self.use_database:
                df = load_from_database(step_size=step_size, start_time=start_time, end_time=end_time,
                                        cache_root=self._cache_root, cache=self._cache, sensor_id=self.uuid,
                                        resample_method=self.resample_method, dt_limit=self.dt_limit, **self.dbconfig)
            else:
                return None
        except Exception:  # noqa: BLE001 -- no data in that window
            return None
        if len(df.index) == 0:
            return None
        if self._transformation is not None:
            df = df.apply(self._transformation)
        return np.asarray(df.values, dtype=float)

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
        simulator: Optional[core.Simulator] = None,
    ) -> None:
        """
        Perform a single timestep for the TimeSeriesInputSystem.

        Args:
            second_time (int, optional): Current simulation time in seconds.
            date_time (date_time, optional): Current simulation time as a date_time object.
            step_size (int, optional): Step size for the simulation.
        """
        self.output["value"]._set(i_t=step_index)
