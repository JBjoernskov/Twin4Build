# Standard library imports
import datetime
import warnings
import importlib
from typing import Any, Callable, Dict, List, Optional, Union

# Third party imports
import pandas as pd

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.translator.translator import (
    ModeledNode,
    Node,
    OptionalRule,
    PathRule,
    SignaturePattern,
    StepRule,
)
from twin4build.utils.logger import LOGGER, autoreset_print


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


@autoreset_print
class SensorSystem(core.System):
    """A system representing a physical or virtual sensor in the building.

    This class implements sensor functionality, supporting both physical sensors
    (reading from time series data) and virtual sensors (computing values from
    other inputs). It integrates with TimeSeriesInputSystem for data handling.

    Args:
        filename: Path to sensor readings file.
            Defaults to None.
        df: DataFrame containing readings.
            Defaults to None.
        uuid: UUID identifying the time series in the database.
            Defaults to None.
        dbconfig: Configuration of the database to read sensor values from.
            Defaults to None.
        datecolumn: Column index containing date/time information.
            Defaults to 0.
        valuecolumn: Column index containing sensor values.
            Defaults to 1.
        use_spreadsheet: Whether to use a spreadsheet for input.
            Defaults to False.
        use_database: Whether to use a database for input.
            Defaults to False.
        use_df: Whether to use the provided DataFrame for input.
            Defaults to False.
        transformation: Optional function to transform the value.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to parent class.

    Note:
        A sensor must either have connections to other systems (virtual sensor) or
        have data input through filename/df/database (physical sensor). Flags are
        auto-detected if only one data source is provided.
    """


    def __init__(
        self,
        filename: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        uuid: Optional[str] = None,
        dbconfig: Optional[Dict[str, Any]] = None,
        datecolumn: int = 0,
        valuecolumn: int = 1,
        use_spreadsheet: bool = False,
        use_database: bool = False,
        use_df: bool = False,
        transformation: Optional[callable] = None,
        transformation_ref: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the sensor system.

        Args:
            filename: Path to sensor readings file.
                Defaults to None.
            df: DataFrame containing readings.
                Defaults to None.
            datecolumn: Column index containing date/time information.
                Defaults to 0.
            valuecolumn: Column index containing sensor values.
                Defaults to 1.
            use_spreadsheet: Whether to use a spreadsheet for input.
                Defaults to False.
            use_database: Whether to use a database for input.
                Defaults to False.
            use_df: Whether to use the provided DataFrame for input.
                Defaults to False.
            transformation: Optional function to transform the value.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.

        Note:
            Either filename/df must be provided for physical sensors, or
            the sensor must have connections defined for virtual sensors.
            Flags are auto-detected if only one data source is provided.
        """
        for legacy_key, new_key in (
            ("useSpreadsheet", "use_spreadsheet"),
            ("useDatabase", "use_database"),
            ("usedf", "use_df"),
        ):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` has been removed. Use `{new_key}` instead."
                )

        # Count how many data sources are provided
        has_df = df is not None
        has_filename = filename is not None
        has_database = dbconfig is not None or uuid is not None
        n_sources = sum([has_df, has_filename, has_database])
        n_flags = sum([use_spreadsheet, use_database, use_df])

        # If multiple sources provided, user must explicitly set a flag
        assert not (n_sources > 1 and n_flags == 0), (
            "Multiple data sources provided (df, filename, database). "
            "You must explicitly set one of use_df=True, use_spreadsheet=True, or use_database=True "
            "to specify which source to use."
        )

        # Auto-detect data source if no flags are explicitly set
        if not use_spreadsheet and not use_database and not use_df:
            if has_df:
                use_df = True
            elif has_filename:
                use_spreadsheet = True
            elif has_database:
                use_database = True

        assert (
            sum([use_spreadsheet, use_database, use_df]) <= 1
        ), "Only one of use_spreadsheet, use_database, or use_df can be True."
        super().__init__(**kwargs)

        # Define inputs and outputs as private variables
        self._input = {"measuredValue": tps.Scalar()}
        self._output = {
            "measuredValue": tps.Scalar(0)
        }  # TODO: Not necessary to be a leaf scalar, if the sensor has inputs. Need to implement check in initialize()

        # Store attributes as private variables
        self._use_spreadsheet = use_spreadsheet
        self._use_database = use_database
        self._use_df = use_df
        self._filename = filename
        self._df = df
        self._datecolumn = datecolumn
        self._valuecolumn = valuecolumn
        self._uuid = uuid
        self._dbconfig = dbconfig
        self._is_leaf = None
        self._time_series_input = None
        self._transformation = transformation
        if transformation is None and transformation_ref:
            # A serialized model carries the transformation by import path.
            self.transformation_ref = transformation_ref

        self._config = {
            "parameters": ["use_spreadsheet", "use_database", "use_df", "transformation_ref"],
            "spreadsheet": ["filename", "datecolumn", "valuecolumn"],
            "database": ["uuid", "dbconfig"],
        }

    @property
    def config(self):
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the sensor system.

        Returns:
            dict: Dictionary containing input ports:
                - "measuredValue": Measured value input for virtual sensors
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the sensor system.

        Returns:
            dict: Dictionary containing output ports:
                - "measuredValue": Measured value output [units depend on sensor type]
        """
        return self._output

    @property
    def filename(self) -> Optional[str]:
        """
        Get the path to sensor readings file.
        """
        return self._filename

    @filename.setter
    def filename(self, value: Optional[str]) -> None:
        """
        Set the path to sensor readings file.
        Automatically sets use_spreadsheet=True if a value is provided.
        """
        self._filename = value
        if value is not None:
            self._use_spreadsheet = True
            self._use_database = False
            self._use_df = False

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """
        Get the direct DataFrame input of sensor readings.
        """
        return self._df

    @df.setter
    def df(self, value: Optional[pd.DataFrame]) -> None:
        """
        Set the direct DataFrame input of sensor readings.
        Automatically sets use_df=True if a value is provided.
        """
        self._df = value
        if value is not None:
            self._use_df = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def datecolumn(self) -> int:
        """
        Get the column index for date_time values.
        """
        return self._datecolumn

    @datecolumn.setter
    def datecolumn(self, value: int) -> None:
        """
        Set the column index for date_time values.
        """
        self._datecolumn = value

    @property
    def valuecolumn(self) -> int:
        """
        Get the column index for sensor readings.
        """
        return self._valuecolumn

    @valuecolumn.setter
    def valuecolumn(self, value: int) -> None:
        """
        Set the column index for sensor readings.
        """
        self._valuecolumn = value

    @property
    def is_leaf(self) -> bool:
        """
        Get whether the sensor reads from file/DataFrame (True) or is virtual (False).
        """
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, value: bool) -> None:
        """
        Set whether the sensor reads from file/DataFrame (True) or is virtual (False).
        """
        self._is_leaf = value

    @property
    def time_series_input(self) -> Optional[TimeSeriesInputSystem]:
        """
        Get the data handling system for physical sensors.
        """
        return self._time_series_input

    @time_series_input.setter
    def time_series_input(self, value: Optional[TimeSeriesInputSystem]) -> None:
        """
        Set the data handling system for physical sensors.
        """
        self._time_series_input = value

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
    def use_df(self) -> bool:
        """
        Get whether to use a DataFrame for input.
        """
        return self._use_df

    @use_df.setter
    def use_df(self, value: bool) -> None:
        """
        Set whether to use a DataFrame for input.
        """
        self._use_df = value

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
        Automatically sets use_database=True if a value is provided.
        """
        self._uuid = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

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
        Automatically sets use_database=True if a value is provided.
        """
        self._dbconfig = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    def set_dbconfig(self, dbconfig: Optional[Dict[str, Any]]) -> None:
        """Set the database configuration on this sensor.

        Functional sibling of the ``dbconfig`` property setter, exposed
        explicitly so model-level helpers (e.g.
        :meth:`SimulationModel.set_dbconfigs`) can dispatch via duck-typed
        method lookup instead of touching the ``dbconfig`` property.
        """
        self.dbconfig = dbconfig

    @property
    def transformation(self) -> Optional[Callable]:
        """Unit-conversion callable applied to loaded timeseries before they
        are emitted on the sensor's ``measuredValue`` output.  ``None`` means
        no conversion (raw values pass through)."""
        return self._transformation

    @transformation.setter
    def transformation(self, fn: Optional[Callable]) -> None:
        self._transformation = fn

    @property
    def transformation_ref(self) -> Optional[str]:
        """The transformation as an import path ``module:qualname`` -- the
        form that survives ``Model.serialize()`` / ``Model.load(filename=...)``
        (a callable is not a literal).  ``None`` when there is no
        transformation, or when it cannot be named (a lambda or a closure):
        such a model reloads without it, with a warning at serialize time."""
        fn = self._transformation
        if fn is None:
            return None
        qualname = getattr(fn, "__qualname__", "")
        if not qualname or "<" in qualname or fn.__module__ is None:
            warnings.warn(
                f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: the transformation "
                f"{fn!r} is not importable by name and will not survive serialization; "
                "use a module-level function.",
                stacklevel=2,
            )
            return None
        return f"{fn.__module__}:{qualname}"

    @transformation_ref.setter
    def transformation_ref(self, ref: Optional[str]) -> None:
        if not ref:
            return
        module_name, _, qualname = ref.partition(":")
        obj = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        self._transformation = obj

    def set_transformation(self, fn: Optional[Callable]) -> None:
        """Set the unit-conversion callable applied to loaded timeseries.

        Companion to :meth:`SimulationModel.set_transformations` (plural):
        the bulk model-level setter dispatches a per-component call here
        for every match.  Idempotent; subsequent calls overwrite.
        """
        self._transformation = fn

    def validate(self, p) -> tuple[bool, bool, bool, bool]:
        """Validate the sensor system configuration.

        Checks if the sensor has proper inputs for different operational modes.

        Args:
            p: Logging function for validation messages.

        Returns:
            tuple[bool, bool, bool, bool]: Validation status for:
                - Simulator
                - Estimator
                - Evaluator
                - Monitor
        """
        validated_for_simulator = True
        validated_for_estimator = True
        validated_for_optimizer = True

        if (
            len(self.connects_at) == 0
            and self.filename is None
            and self.df is None
            and self.uuid is None
        ):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename or df or uuid must be provided to enable use of Simulator, Estimator, and Optimizer."
            p(message, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif (
            len(self.connects_at) > 0
            and self.filename is None
            and self.df is None
            and self.uuid is None
        ):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename or df or uuid must be provided to enable use of Estimator."
            p(message, status="WARNING")
            validated_for_estimator = False

        self.is_leaf = len(self.connects_at) == 0  # No inputs -> leaf scalar
        self.output["measuredValue"].is_leaf = self.is_leaf

        return (
            validated_for_simulator,
            validated_for_estimator,
            validated_for_optimizer,
        )

    def validate_connections(self, p) -> bool:
        validated = True
        if (
            self.is_leaf
            and self.use_spreadsheet == False
            and self.use_database == False
            and self.use_df == False
        ):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Missing connections for the following input(s) to enable use of Simulator, Estimator, and Optimizer:"
            p(message, status="[WARNING]")
            p.add_level()
            p("measuredValue")
            p.remove_level()
            validated = False
        return validated

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[float],
    ) -> None:
        """Initialize the sensor system.

        Sets up the physical or virtual sensor system and initializes the step instance.

        Args:
            start_time (Optional[datetime.datetime]): Start time for the simulation.
            end_time (Optional[datetime.datetime]): End time for the simulation.
            step_size (Optional[float]): Time step size in seconds.
            model (Optional[Any]): Model object (not used in this class).
        """

        self.validate(LOGGER)
        self.validate_connections(LOGGER)

        if self.use_spreadsheet or self.use_database or self.use_df:
            if self.use_df:
                if self.df is None:
                    raise ValueError("df must be provided when use_df=True.")
            self.time_series_input = TimeSeriesInputSystem(
                id=f"time series input - {self.id}",
                df=self.df,
                filename=self.filename,
                date_column=self.datecolumn,
                value_column=self.valuecolumn,
                use_spreadsheet=self.use_spreadsheet,
                use_database=self.use_database,
                uuid=self.uuid,
                dbconfig=self.dbconfig,
                transformation=self._transformation,
            )
            self.time_series_input.initialize(
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
            )

        else:
            self.time_series_input = None

        assert (
            len(self.connects_at) == 0 and self.time_series_input is None
        ) == False, f'Sensor object "{self.id}" has no inputs and and holds no data.'

        if self.is_leaf:
            # The batch initialization args are calculated in the TimeSeriesInputSystem.initialize() method.
            # They are stored in the physicalSystem object and reused here.
            self.output["measuredValue"].initialize(
                n_t=self.time_series_input.n_timesteps,
                n_s=self.time_series_input.batch_size,
                n_c=1,
                values=self.time_series_input.values,
            )
        else:
            _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
                start_time, end_time, step_size
            )
            batch_size = len(start_time)
            self.input["measuredValue"].initialize(
                n_t=max_timesteps,
                n_s=batch_size,
            )
            self.output["measuredValue"].initialize(
                n_t=max_timesteps,
                n_s=batch_size,
            )

    def do_step(
        self,
        second_time: Optional[float] = None,
        date_time: Optional[datetime.datetime] = None,
        step_size: Optional[float] = None,
        step_index: Optional[int] = None,
    ) -> None:
        """Execute one time step of the sensor system.

        Updates sensor outputs based on either physical readings or virtual calculations.

        Args:
            second_time (Optional[float]): Current simulation time in seconds.
            date_time (Optional[datetime.datetime]): Current simulation date_time.
            step_size (Optional[float]): Time step size in seconds.
        """
        if self.is_leaf:
            self.output["measuredValue"]._set(i_t=step_index)
        else:
            self.output["measuredValue"]._set(
                self.input["measuredValue"].get(), step_index
            )

    def get_physical_readings(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[float],
    ) -> pd.DataFrame:
        """Retrieve physical sensor readings for a specified time period.

        Args:
            start_time (Optional[datetime.datetime]): Start time for readings.
            end_time (Optional[datetime.datetime]): End time for readings.
            step_size (Optional[float]): Time step size in seconds.

        Returns:
            pd.DataFrame: DataFrame containing sensor readings.

        Raises:
            AssertionError: If called on a virtual sensor (no physical readings available).
        """
        self.initialize(start_time, end_time, step_size)
        assert (
            self.time_series_input is not None
        ), f'Cannot return physical readings for Sensor with id "{self.id}" as time_series_input is None.\nEither this sensor has not been intialized or the arguments filename/df/dbconfig were not provided when the object was initialized or the sensor is virtual and has no time_series_input.'
        self.time_series_input.initialize(start_time, end_time, step_size)
        return self.time_series_input.df
