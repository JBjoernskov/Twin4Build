"""A measurement point: a sensor's recorded series and what it reads in the
model (ADR 0001, decision 6)."""

# Standard library imports
import datetime
from typing import Any, Callable, Dict, List, Optional, Union

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Setting
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.utils.callable_ref import as_callable
from twin4build.utils.data_loaders.sources import DatabaseSource, DataFrameSource, as_source
from twin4build.utils.logger import autoreset_print


# Properties of spaces


# Properties of spaces


# -----------------------------------------------------------------------------
# Why the zone / AHU air-temperature sensors are expressed as *two* patterns each
# -----------------------------------------------------------------------------
#
# We want the virtual sensor (connected to ``BuildingSpace.indoorTemperature`` /
# ``AHU.supplyAirTemperature``) to appear in the simulation model whether or not
# the BRICK graph actually carries a Brick-reference timeseries UUID.  The first,
# natural encoding was a single pattern with the external-ref / timeseries-id
# triples wrapped in :class:`OptionalRule`.  That encoding is **broken** for a
# very specific reason:
#
# 1. OptionalRule triples are eligible to be matched on a *disconnected* subgraph,
#    separate from the (sensor, vav, room) subgraph.
# 2. In the translator's ``_try_merge_with_incomplete`` disconnected-merge
#    branch, any sub-group whose modeled_node slots are *not* filled is
#    classified as a "shared resource" and placed in ``groups_to_preserve``
#    — i.e. kept for reuse across every subsequent match of the pattern.
# 3. Because the original pattern only declared ``sensor`` as a modeled node,
#    the ``(externalref, timeseries_id)`` sub-group is *always* a shared
#    resource.  The translator therefore picks *one* arbitrary
#    ``(blank_node, uuid_literal)`` pair (the first one it enumerates) and
#    rebinds *every* Zone_Air_Temperature_Sensor in the building to it.
#
# The fix is to split the single pattern into two mutually-exclusive variants:
#
# * ``*_with_ref_pattern`` — requires the external-ref chain via :class:`StepRule`
#   triples and additionally declares ``externalref`` as a modeled node so the
#   (externalref, timeseries_id) sub-group can never be re-used.  Models
#   ``{sensor, externalref}`` (two modeled nodes).
# * ``*_virtual_pattern``  — omits the external-ref chain entirely.  Models
#   ``{sensor}`` (one modeled node).
#
# The translator's MILP objective is
# ``component_selection_cost - semantic_instance_benefit * n_modeled_nodes``,
# and the mutual-exclusion constraint is keyed on each ``modeled_node`` /
# ``sm_node`` pair.  Consequently:
#
# * If a sensor has a Brick timeseries reference, *both* patterns match on the
#   same ``sensor`` modeled node, but the with-ref variant has two modeled
#   nodes (cheaper in the minimisation) and wins.
# * If a sensor has *no* Brick timeseries reference, only the virtual variant
#   matches and is selected.
#
# Net effect: the UUID is preserved when available and the virtual sensor is
# preserved when no UUID exists — without reintroducing the shared-resource
# cross-binding bug.
# -----------------------------------------------------------------------------



def _standard_deviation(value) -> Optional[float]:
    """A measurement's standard deviation: positive, or ``None``."""
    if value is None:
        return None
    value = float(value)
    if not value > 0:
        raise ValueError(f"SensorSystem: measurement_sd must be positive, got {value}.")
    return value


_READER = "give the sensor a reader: source=CsvSource(path, date_column=..., value_column=...), DataFrameSource(df) or DatabaseSource(uuid)"


@autoreset_print
class SensorSystem(core.System):
    """A measurement point: what a sensor recorded, and what it reads in the
    model (ADR 0001, decision 6).

    ::

        tb.SensorSystem(id="room_temperature", source=CsvSource("room.csv", value_column=2))

    Two outputs.  ``measuredData`` is the recorded series, read once at run
    start from ``source`` (:meth:`series`).  ``measuredValue`` is what the
    sensor reads: the value wired to its input, or its recorded series while
    nothing is wired (``measuredValue = Input(default=measuredData)``).  A
    sensor that measures a model output is wired; the estimator scores its
    ``measuredValue`` against ``measuredData``.  A sensor that drives the
    model is not.  A System that needs the measurement itself connects to
    ``measuredData``, so it never closes a loop through the model.

    Args:
        source: The reader of the recorded series:
            :class:`~twin4build.utils.data_loaders.sources.CsvSource`,
            :class:`~twin4build.utils.data_loaders.sources.DataFrameSource` or
            :class:`~twin4build.utils.data_loaders.sources.DatabaseSource`,
            with its reading options (resampling, gap limit, fill).  A string
            is a database point's id.  ``None``: the sensor has no data (and
            must then be wired).
        uuid: The id of the point the sensor stands for, named in
            measurement results; without a ``source`` it is the database point
            the sensor reads.
        transformation: Applied to every value read (a unit conversion); a
            saved model holds its ``module:qualname``.
        allow_missing: Whether a step without a reading is an unscored sample
            rather than an error.  ``None`` (default): yes for a sensor that
            measures a model output, no for one that drives the model, which
            gives its reader a ``fill`` instead.
        scoring_mask: Which samples an estimation scores, see
            :attr:`scoring_mask`.
        measurement_sd: The standard deviation the sensor is scored with, in
            the unit of its readings; used by
            :meth:`~twin4build.estimator.estimator.Estimator.estimate` for an
            entry of ``measurements`` that names the sensor without one.

    After a run, :attr:`time_series_input` holds the reading: its ``values``,
    the per-period series ``df`` as read and the ``filled`` steps.
    """

    #: The 1.x keywords and what replaced them.
    REMOVED_ATTRIBUTES = {
        **{name: _READER for name in ("filename", "datecolumn", "valuecolumn", "use_spreadsheet", "use_database", "use_df")},
        "df": "give the sensor a reader, source=DataFrameSource(df), or call set_series(series)",
        "dbconfig": "give the sensor a reader, source=DatabaseSource(uuid, dbconfig), or call set_dbconfig(dbconfig)",
        **{
            name: "give the reader its reading options, CsvSource(..., resample_method=..., dt_limit=..., fill=...)"
            for name in ("resample_method", "dt_limit", "fill")
        },
    }

    #: The name 1.x saved the transformation under.
    DEPRECATED_KEYWORDS = {"transformation_ref": "transformation"}

    source = Setting(None, convert=as_source, doc="The reader of the recorded series")
    uuid = Setting(None, doc="The id of the point the sensor stands for")
    transformation = Setting(None, convert=as_callable, doc="Applied to every value read")
    allow_missing = Setting(None, doc="Whether a step without a reading is an unscored sample")
    scoring_mask = Setting(None, save=False, doc="Which samples an estimation scores")
    measurement_sd = Setting(None, convert=_standard_deviation, save=False, doc="The standard deviation it is scored with")
    measuredData = Output(doc="The recorded series")
    measuredValue = Input(default=measuredData, doc="The value the sensor reads; unwired, its recorded series")
    measuredValue = Output(doc="What the sensor reads")

    @property
    def reader(self):
        """The reader the recorded series comes from: ``source``, or the
        database point ``uuid`` names; ``None`` for a sensor without data."""
        if self.source is not None:
            return self.source
        if self.uuid is not None:
            return DatabaseSource(self.uuid)
        return None

    @property
    def has_data(self) -> bool:
        """Whether the sensor holds a recorded series (``measuredData``)."""
        return self.reader is not None

    @property
    def is_leaf(self) -> bool:
        """Whether nothing is wired to the sensor's input: it then reads its
        recorded series (True), otherwise the value wired to it (False)."""
        return len(self.connects_at) == 0

    @property
    def time_series_input(self) -> Optional[TimeSeriesInputSystem]:
        """The reading of the last run start (``values``, ``df``,
        ``filled``); ``None`` before it, and for a sensor without data."""
        return self.__dict__.get("_reading")

    def set_series(self, series: Union[pd.Series, pd.DataFrame], uuid: Optional[str] = None) -> None:
        """Read an in-memory series from now on (a ``DataFrameSource``).

        Args:
            series: The readings, a ``pandas.Series`` with a
                ``DatetimeIndex`` or a ``pandas.DataFrame`` with a
                ``DatetimeIndex`` and one column.
            uuid: The sensor's identity.  ``None`` (default) keeps the
                current one.

        Raises:
            TypeError: If ``series`` is neither a Series nor a DataFrame, or
                its index is not a ``DatetimeIndex``.
            ValueError: If a DataFrame has another number of columns than one.
        """
        if isinstance(series, pd.Series):
            frame = series.to_frame(name="value" if series.name is None else series.name)
        elif isinstance(series, pd.DataFrame):
            if series.shape[1] != 1:
                raise ValueError(
                    f"SensorSystem '{self.id}': set_series takes a DataFrame with one column, got {series.shape[1]}."
                )
            frame = series
        else:
            raise TypeError(
                f"SensorSystem '{self.id}': set_series takes a pandas Series or a one-column DataFrame, "
                f"got {type(series).__name__}."
            )
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(
                f"SensorSystem '{self.id}': the series must be indexed by time (a DatetimeIndex), "
                f"got {type(frame.index).__name__}."
            )
        options = {} if self.source is None else {
            "resample_method": self.source.resample_method, "dt_limit": self.source.dt_limit, "fill": self.source.fill,
        }
        self.source = DataFrameSource(frame, **options)
        if uuid is not None:
            self.uuid = uuid

    def set_dbconfig(self, dbconfig: Optional[Dict[str, Any]]) -> None:
        """The database connection of the sensor's database reader (the hook
        :meth:`SimulationModel.set_dbconfigs` uses); a sensor that reads the
        point its ``uuid`` names gets a ``DatabaseSource`` with it."""
        if self.source is None and self.uuid is not None:
            self.source = DatabaseSource(self.uuid, dbconfig)
        elif hasattr(self.source, "dbconfig"):
            self.source.dbconfig = dbconfig

    def set_transformation(self, fn: Optional[Callable]) -> None:
        """The function applied to every value read (the hook
        :meth:`SimulationModel.set_transformations` uses); ``None`` removes
        it."""
        self.transformation = fn

    def series(self, timestamps, step_size):
        """The recorded series on the run's grid, from the reader."""
        drives = "measuredValue" in self.output_defaults()  # nothing wired: the series drives the model
        reader = self.reader
        if reader is None:
            if drives:
                raise ValueError(
                    f"SensorSystem '{self.id}': nothing is wired to measuredValue and the sensor has no "
                    f"recorded series (source=...)"
                )
            self.__dict__["_reading"] = None
            return {"measuredData": np.zeros(timestamps.shape)}
        if not drives and reader.fill is not None:
            raise ValueError(
                f"SensorSystem '{self.id}' ({self.uuid}) measures a model output: its gaps are unscored "
                f"samples, a fill would invent measurements to score (fill={reader.fill!r})."
            )
        allow_missing = self.allow_missing
        if allow_missing is None:
            allow_missing = not drives
        elif allow_missing and drives:
            raise ValueError(
                f"SensorSystem '{self.id}' ({self.uuid}) drives the model: its gaps cannot be unscored "
                f"(allow_missing); give its reader a fill instead (fill=...)."
            )
        reading = self.__dict__.get("_reading")
        if reading is None or reading.source is not reader or reading.transformation is not self.transformation:
            reading = TimeSeriesInputSystem(
                id=f"time series input - {self.id}", source=reader, transformation=self.transformation
            )
            self.__dict__["_reading"] = reading
        reading.allow_missing = bool(allow_missing)
        return {"measuredData": reading.series(timestamps, step_size)["value"]}

    def forward(self, x, inputs, p, step_size):
        """What the sensor reads: its input, which is its recorded series
        while nothing is wired."""
        return x, {"measuredValue": inputs["measuredValue"]}

    def get_physical_readings(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[float],
    ) -> List[pd.Series]:
        """The recorded series of each period, as read (NaN where the data
        give nothing).

        Raises:
            ValueError: If the sensor has no recorded series.
        """
        if not self.has_data:
            raise ValueError(f"SensorSystem '{self.id}' has no recorded series (source=...)")
        self.initialize(start_time, end_time, step_size)
        return self.time_series_input.df
