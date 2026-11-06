"""A series read from a file, a DataFrame or the database (a data source)."""

from __future__ import annotations

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.saref4syst.declarations import Output, Setting
from twin4build.utils.data_loaders.sources import COPY_WEEKS, FILLS, HOLD_WEEKS, as_source


class TimeSeriesInputSystem(core.System):
    """A series read from a file, a DataFrame or the database: a data source
    (ADR 0001, decision 6).

    Its one output, ``value``, is the series on the run's grid, read once at
    run start (:meth:`series`).  Where the data come from, and how they are
    read, is the reader in ``source``::

        tb.TimeSeriesInputSystem(id="supply", source=CsvSource("supply.csv", fill="constant"))

    Args:
        source: The reader: :class:`~twin4build.utils.data_loaders.sources.CsvSource`,
            :class:`~twin4build.utils.data_loaders.sources.DataFrameSource` or
            :class:`~twin4build.utils.data_loaders.sources.DatabaseSource`, with
            its reading options (resampling, gap limit, fill).  A fill is a
            decision about the data: the steps it filled are logged and kept
            in :attr:`filled`, and the estimator leaves every measurement they
            reach unscored.
        transformation: Applied to every value read (a unit conversion, ...).
            Not saved with the model.

    After a run, :attr:`df` holds one series per period as read (NaN where the
    data give nothing), :attr:`values` the output's values ``(n_t, n_s, 1)``
    and :attr:`filled` the ``(n_s, n_t)`` steps the fill supplied.
    """

    FILLS = FILLS
    COPY_WEEKS = COPY_WEEKS
    HOLD_WEEKS = HOLD_WEEKS

    source = Setting(convert=as_source, doc="The reader the values come from")
    transformation = Setting(None, save=False, doc="Applied to every value read")
    value = Output(doc="The series")

    #: A sensor whose samples are scored sets this: a step without a value is
    #: 0 in the output (the per-period series keep the NaN) instead of an error.
    allow_missing = False

    def set_dbconfig(self, dbconfig) -> None:
        """The database connection of a database reader."""
        if hasattr(self.source, "dbconfig"):
            self.source.dbconfig = dbconfig

    def series(self, timestamps, step_size):
        """The values on the run's grid, from the reader."""
        start_time, end_time, step_sizes = self.periods(timestamps, step_size)
        reading = self.source.read(
            start_time, end_time, step_sizes,
            transformation=self.transformation, allow_missing=self.allow_missing, owner=self.id,
        )
        self.df = reading.raw
        self.filled = reading.filled
        self.n_timesteps = reading.n_t
        self.batch_size = len(start_time)
        self.values = torch.tensor(reading.values, dtype=tps.float_dtype()).T.unsqueeze(-1)
        return {"value": reading.values}
