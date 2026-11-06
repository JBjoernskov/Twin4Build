"""The weather at the site: a data source (ADR 0001, decision 6)."""

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Output, Parameter, Setting
from twin4build.utils.callable_ref import as_callable
from twin4build.utils.data_loaders.sources import DataFrameSource, as_source


class OutdoorEnvironmentSystem(core.System):
    r"""The weather at the site: outdoor temperature, global irradiation and
    outdoor CO2, each read from its own reader once at run start.

    ::

        tb.OutdoorEnvironmentSystem(
            id="outdoor",
            outdoorTemperature_source=CsvSource("weather.csv", value_column=1),
            globalIrradiation_source=CsvSource("weather.csv", value_column=2),
        )

    Without a CO2 reader ``outdoorCo2Concentration`` is the constant
    ``co2_default`` (400 ppm).  :meth:`from_dataframe` builds the readers
    from one DataFrame with a column per output.

    Args:
        outdoorTemperature_source: The reader of the outdoor temperature. A
            string is a database point's id, which is what a translator
            pattern gives; likewise for the other readers.
        globalIrradiation_source: The reader of the global irradiation.
        outdoorCo2Concentration_source: The reader of the outdoor CO2.
        co2_default: The outdoor CO2 [ppm] without a CO2 reader.
        a: The gain of the correction ``a * x + b`` of the outdoor
            temperature, applied when ``apply_correction`` is set.
        b: The offset of that correction [degC].
        apply_correction: Apply the correction.
        outdoorTemperature_transformation: A function applied to every
            temperature read (a unit conversion); a saved model holds its
            ``module:qualname``.  :meth:`set_transformation` and
            :meth:`set_transformation_by_type` are the hooks
            :meth:`Model.set_transformations` uses.
        globalIrradiation_transformation: The same for the irradiation
            (clipping a sensor's spikes).
    """

    #: The names 1.x saved the transformations under.
    DEPRECATED_KEYWORDS = {
        "transformation_ref_outdoorTemperature": "outdoorTemperature_transformation",
        "transformation_ref_globalIrradiation": "globalIrradiation_transformation",
    }

    #: The 1.x keywords and what replaced them.
    REMOVED_ATTRIBUTES = {
        **{
            f"{kind}_{feed}": f"give the feed a reader, {feed}_source=CsvSource(...) / DatabaseSource(uuid)"
            for feed in ("outdoorTemperature", "globalIrradiation", "outdoorCo2Concentration")
            for kind in ("filename", "datecolumn", "valuecolumn", "uuid", "dbconfig", "_uuid", "_filename", "_dbconfig")
        },
        **{name: "give each feed a reader (<feed>_source=...); OutdoorEnvironmentSystem.from_dataframe(df)"
           for name in ("df", "use_df", "use_spreadsheet", "use_database")},
        **{f"_transformation_{feed}": f"set {feed}_transformation (or set_transformation)"
           for feed in ("outdoorTemperature", "globalIrradiation")},
    }

    FEEDS = ("outdoorTemperature", "globalIrradiation", "outdoorCo2Concentration")

    outdoorTemperature_source = Setting(None, convert=as_source, doc="The reader of the outdoor temperature")
    globalIrradiation_source = Setting(None, convert=as_source, doc="The reader of the global irradiation")
    outdoorCo2Concentration_source = Setting(None, convert=as_source, doc="The reader of the outdoor CO2")
    co2_default = Setting(400.0, doc="The outdoor CO2 [ppm] without a CO2 reader")
    apply_correction = Setting(False, doc="Apply the linear correction a * x + b to the outdoor temperature")
    outdoorTemperature_transformation = Setting(None, convert=as_callable, doc="Applied to every temperature read")
    globalIrradiation_transformation = Setting(None, convert=as_callable, doc="Applied to every irradiation read")
    a = Parameter(1.0, doc="Gain of the outdoor-temperature correction")
    b = Parameter(0.0, doc="Offset of the outdoor-temperature correction [degC]")
    outdoorTemperature = Output(doc="Outdoor temperature [degC]")
    globalIrradiation = Output(doc="Global irradiation [W/m2]")
    outdoorCo2Concentration = Output(doc="Outdoor CO2 concentration [ppm]")

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        """The readers of the outputs from one DataFrame with a column per
        output, indexed by timestamps (a column it lacks gets no reader)."""
        readers = {
            f"{feed}_source": DataFrameSource(df[[feed]]) for feed in cls.FEEDS if feed in df.columns
        }
        return cls(**{**readers, **kwargs})

    def set_dbconfig(self, dbconfig) -> None:
        """The database connection of every database reader."""
        for feed in self.FEEDS:
            reader = getattr(self, f"{feed}_source")
            if hasattr(reader, "dbconfig"):
                reader.dbconfig = dbconfig

    def set_transformation(self, fn) -> None:
        """The function applied to every outdoor-temperature value read (the
        hook :meth:`Model.set_transformations` uses); ``None`` removes it."""
        self.outdoorTemperature_transformation = fn

    def set_transformation_by_type(self, semantic_type, fn) -> None:
        """Route a per-Brick-class transformation to its feed: temperature
        sensors to the outdoor temperature, solar irradiance / radiance
        sensors to the global irradiation; other classes are ignored."""
        uris = {str(semantic_type.uri)} | {str(c.uri) for c in getattr(semantic_type, "super_classes", [])}
        brick = core.namespace.BRICK
        if str(brick.Temperature_Sensor) in uris:
            self.outdoorTemperature_transformation = fn
        elif uris & {
            str(brick.Solar_Irradiance_Sensor),
            str(brick.Solar_Radiance_Sensor),
            str(brick.Global_Solar_Irradiation_Sensor),
        }:
            self.globalIrradiation_transformation = fn

    def series(self, timestamps, step_size):
        """Every output on the run's grid, from its reader."""
        start_time, end_time, step_sizes = self.periods(timestamps, step_size)
        values = {}
        for feed in self.FEEDS:
            reader = getattr(self, f"{feed}_source")
            if reader is None:
                if feed == "outdoorCo2Concentration":
                    values[feed] = np.full(timestamps.shape, float(self.co2_default))
                    continue
                raise ValueError(f"OutdoorEnvironmentSystem '{self.id}': no reader for {feed} ({feed}_source)")
            values[feed] = reader.read(
                start_time, end_time, step_sizes,
                transformation=getattr(self, f"{feed}_transformation", None),
                owner=f"{self.id}.{feed}",
            ).values
        if self.apply_correction:
            # the correction's parameters as they are at run start
            a = self.a.get().reshape(-1)[0]
            b = self.b.get().reshape(-1)[0]
            values["outdoorTemperature"] = torch.as_tensor(values["outdoorTemperature"], dtype=a.dtype) * a + b
        return values
