# Standard library imports
import datetime
import os
import warnings
from typing import List, Optional

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.translator.translator import (
    Exact,
    Node,
    Optional_,
    SignaturePattern,
    SinglePath,
)
from twin4build.utils.data_loaders.load import load_from_database, load_from_spreadsheet
from twin4build.utils.deprecation import deprecate_args
from twin4build.utils.get_main_dir import get_main_dir


class OutdoorEnvironmentSystem(core.System, nn.Module):
    """An outdoor environment system model that provides weather data for building simulations.

    This model represents the outdoor environment by providing time-series data for:
    - Outdoor air temperature
    - Global solar irradiation
    - Outdoor CO2 concentration

    The model reads weather data from 3 separate CSV files (one for each parameter) and can
    optionally apply a linear correction to the temperature data. The model is designed to be
    used as a boundary condition for building energy simulations.

    Args:
        df: Input DataFrame containing weather data.
            Must have columns 'outdoorTemperature', 'globalIrradiation', and 'outdoorCo2Concentration'.
        useSpreadsheet: Whether to use spreadsheet files for data loading.
        useDatabase: Whether to use database for data loading.
        usedf: Whether to use the provided DataFrame for data loading.
        filename_outdoorTemperature: Path to CSV file containing outdoor temperature data.
        datecolumn_outdoorTemperature: Name of the date column in temperature file.
        valuecolumn_outdoorTemperature: Name of the temperature value column.
        filename_globalIrradiation: Path to CSV file containing global irradiation data.
        datecolumn_globalIrradiation: Name of the date column in irradiation file.
        valuecolumn_globalIrradiation: Name of the irradiation value column.
        filename_outdoorCo2Concentration: Path to CSV file containing CO2 concentration data.
        datecolumn_outdoorCo2Concentration: Name of the date column in CO2 file.
        valuecolumn_outdoorCo2Concentration: Name of the CO2 value column.
        a: Correction factor for linear correction of temperature data.
        b: Correction offset for linear correction of temperature data.
        apply_correction: Whether to apply linear correction to temperature data.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        use_spreadsheet: Optional[bool] = False,
        use_database: Optional[bool] = False,
        use_df: Optional[bool] = False,
        filename_outdoorTemperature: Optional[str] = None,
        datecolumn_outdoorTemperature: Optional[int] = 0,
        valuecolumn_outdoorTemperature: Optional[int] = 1,
        filename_globalIrradiation: Optional[str] = None,
        datecolumn_globalIrradiation: Optional[int] = 0,
        valuecolumn_globalIrradiation: Optional[int] = 1,
        filename_outdoorCo2Concentration: Optional[str] = None,
        datecolumn_outdoorCo2Concentration: Optional[int] = 0,
        valuecolumn_outdoorCo2Concentration: Optional[int] = 1,
        uuid_outdoorTemperature: Optional[str] = None,
        dbconfig_outdoorTemperature: Optional[dict] = None,
        uuid_globalIrradiation: Optional[str] = None,
        dbconfig_globalIrradiation: Optional[dict] = None,
        uuid_outdoorCo2Concentration: Optional[str] = None,
        dbconfig_outdoorCo2Concentration: Optional[dict] = None,
        a: Optional[float] = 1,
        b: Optional[float] = 0,
        apply_correction: Optional[bool] = False,
        **kwargs,
    ):
        # Handle deprecated camelCase arguments
        deprecated_args = ["useSpreadsheet", "useDatabase", "usedf"]
        new_args = ["use_spreadsheet", "use_database", "use_df"]
        positions = [None, None, None]
        value_map = deprecate_args(deprecated_args, new_args, positions, kwargs)
        use_spreadsheet = value_map.get("use_spreadsheet", use_spreadsheet)
        use_database = value_map.get("use_database", use_database)
        use_df = value_map.get("use_df", use_df)

        # Count how many data sources are provided
        has_df = df is not None
        has_filename = (
            filename_outdoorTemperature is not None
            or filename_globalIrradiation is not None
            or filename_outdoorCo2Concentration is not None
        )
        has_database = (
            dbconfig_outdoorTemperature is not None
            or dbconfig_globalIrradiation is not None
            or dbconfig_outdoorCo2Concentration is not None
            or uuid_outdoorTemperature is not None
            or uuid_globalIrradiation is not None
            or uuid_outdoorCo2Concentration is not None
        )
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
        nn.Module.__init__(self)

        if (
            df is None
            and (
                filename_outdoorTemperature is None
                or filename_globalIrradiation is None
                or filename_outdoorCo2Concentration is None
            )
            and (
                uuid_outdoorTemperature is None
                or uuid_globalIrradiation is None
                or uuid_outdoorCo2Concentration is None
            )
        ):
            warnings.warn(
                'Neither "df", "filename", nor "uuid" was provided as argument. The component will not be able to provide any output.'
            )

        # Define inputs and outputs as private variables
        self._input = {}
        self._output = {
            "outdoorTemperature": tps.Scalar(is_leaf=True),
            "globalIrradiation": tps.Scalar(is_leaf=True),
            "outdoorCo2Concentration": tps.Scalar(is_leaf=True),
        }

        # Store as private variables for property access
        self._use_spreadsheet = use_spreadsheet
        self._use_database = use_database
        self._use_df = use_df

        self._filename_outdoorTemperature = filename_outdoorTemperature
        self.datecolumn_outdoorTemperature = datecolumn_outdoorTemperature
        self.valuecolumn_outdoorTemperature = valuecolumn_outdoorTemperature

        self._filename_globalIrradiation = filename_globalIrradiation
        self.datecolumn_globalIrradiation = datecolumn_globalIrradiation
        self.valuecolumn_globalIrradiation = valuecolumn_globalIrradiation

        self._filename_outdoorCo2Concentration = filename_outdoorCo2Concentration
        self.datecolumn_outdoorCo2Concentration = datecolumn_outdoorCo2Concentration
        self.valuecolumn_outdoorCo2Concentration = valuecolumn_outdoorCo2Concentration

        self._uuid_outdoorTemperature = uuid_outdoorTemperature
        self._dbconfig_outdoorTemperature = dbconfig_outdoorTemperature

        self._uuid_globalIrradiation = uuid_globalIrradiation
        self._dbconfig_globalIrradiation = dbconfig_globalIrradiation

        self._uuid_outdoorCo2Concentration = uuid_outdoorCo2Concentration
        self._dbconfig_outdoorCo2Concentration = dbconfig_outdoorCo2Concentration

        self._df = df
        self.a = tps.Parameter(
            torch.tensor(a, dtype=torch.float64), requires_grad=False
        )
        self.b = tps.Parameter(
            torch.tensor(b, dtype=torch.float64), requires_grad=False
        )
        self.apply_correction = apply_correction
        self.cached_initialize_arguments = []
        self.cache_root = get_main_dir()

        self._config = {
            "parameters": [
                "a",
                "b",
                "apply_correction",
                "use_spreadsheet",
                "use_database",
                "use_df",
            ],
            "spreadsheet": [
                "filename_outdoorTemperature",
                "datecolumn_outdoorTemperature",
                "valuecolumn_outdoorTemperature",
                "filename_globalIrradiation",
                "datecolumn_globalIrradiation",
                "valuecolumn_globalIrradiation",
                "filename_outdoorCo2Concentration",
                "datecolumn_outdoorCo2Concentration",
                "valuecolumn_outdoorCo2Concentration",
            ],
            "database": [
                "uuid_outdoorTemperature",
                "dbconfig_outdoorTemperature",
                "uuid_globalIrradiation",
                "dbconfig_globalIrradiation",
                "uuid_outdoorCo2Concentration",
                "dbconfig_outdoorCo2Concentration",
            ],
        }

    @property
    def config(self):
        """Get the configuration parameters.

        Returns:
            dict: Dictionary containing configuration parameters and file reading settings.
        """
        return self._config

    # ==================== Data Source Flags ====================

    @property
    def use_spreadsheet(self):
        return self._use_spreadsheet

    @use_spreadsheet.setter
    def use_spreadsheet(self, value):
        self._use_spreadsheet = value

    @property
    def use_database(self):
        return self._use_database

    @use_database.setter
    def use_database(self, value):
        self._use_database = value

    @property
    def use_df(self):
        return self._use_df

    @use_df.setter
    def use_df(self, value):
        self._use_df = value

    # ==================== Deprecated Properties (camelCase) ====================

    @property
    def useSpreadsheet(self):
        """Deprecated: Use use_spreadsheet instead."""
        warnings.warn(
            "Property 'useSpreadsheet' is deprecated. Use 'use_spreadsheet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_spreadsheet

    @useSpreadsheet.setter
    def useSpreadsheet(self, value):
        warnings.warn(
            "Property 'useSpreadsheet' is deprecated. Use 'use_spreadsheet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_spreadsheet = value

    @property
    def useDatabase(self):
        """Deprecated: Use use_database instead."""
        warnings.warn(
            "Property 'useDatabase' is deprecated. Use 'use_database' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_database

    @useDatabase.setter
    def useDatabase(self, value):
        warnings.warn(
            "Property 'useDatabase' is deprecated. Use 'use_database' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_database = value

    @property
    def usedf(self):
        """Deprecated: Use use_df instead."""
        warnings.warn(
            "Property 'usedf' is deprecated. Use 'use_df' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_df

    @usedf.setter
    def usedf(self, value):
        warnings.warn(
            "Property 'usedf' is deprecated. Use 'use_df' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_df = value

    # ==================== DataFrame Property ====================

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value
        if value is not None:
            self._use_df = True
            self._use_spreadsheet = False
            self._use_database = False

    # ==================== Filename Properties ====================

    @property
    def filename_outdoorTemperature(self):
        return self._filename_outdoorTemperature

    @filename_outdoorTemperature.setter
    def filename_outdoorTemperature(self, value):
        self._filename_outdoorTemperature = value
        if value is not None:
            self._use_spreadsheet = True
            self._use_database = False
            self._use_df = False

    @property
    def filename_globalIrradiation(self):
        return self._filename_globalIrradiation

    @filename_globalIrradiation.setter
    def filename_globalIrradiation(self, value):
        self._filename_globalIrradiation = value
        if value is not None:
            self._use_spreadsheet = True
            self._use_database = False
            self._use_df = False

    @property
    def filename_outdoorCo2Concentration(self):
        return self._filename_outdoorCo2Concentration

    @filename_outdoorCo2Concentration.setter
    def filename_outdoorCo2Concentration(self, value):
        self._filename_outdoorCo2Concentration = value
        if value is not None:
            self._use_spreadsheet = True
            self._use_database = False
            self._use_df = False

    # ==================== UUID Properties ====================

    @property
    def uuid_outdoorTemperature(self):
        return self._uuid_outdoorTemperature

    @uuid_outdoorTemperature.setter
    def uuid_outdoorTemperature(self, value):
        self._uuid_outdoorTemperature = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    @property
    def uuid_globalIrradiation(self):
        return self._uuid_globalIrradiation

    @uuid_globalIrradiation.setter
    def uuid_globalIrradiation(self, value):
        self._uuid_globalIrradiation = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    @property
    def uuid_outdoorCo2Concentration(self):
        return self._uuid_outdoorCo2Concentration

    @uuid_outdoorCo2Concentration.setter
    def uuid_outdoorCo2Concentration(self, value):
        self._uuid_outdoorCo2Concentration = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    # ==================== DBConfig Properties ====================

    @property
    def dbconfig_outdoorTemperature(self):
        return self._dbconfig_outdoorTemperature

    @dbconfig_outdoorTemperature.setter
    def dbconfig_outdoorTemperature(self, value):
        self._dbconfig_outdoorTemperature = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    @property
    def dbconfig_globalIrradiation(self):
        return self._dbconfig_globalIrradiation

    @dbconfig_globalIrradiation.setter
    def dbconfig_globalIrradiation(self, value):
        self._dbconfig_globalIrradiation = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    @property
    def dbconfig_outdoorCo2Concentration(self):
        return self._dbconfig_outdoorCo2Concentration

    @dbconfig_outdoorCo2Concentration.setter
    def dbconfig_outdoorCo2Concentration(self, value):
        self._dbconfig_outdoorCo2Concentration = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    @property
    def input(self) -> dict:
        """
        Get the input ports of the outdoor environment system.

        Returns:
            dict: Dictionary containing input ports (empty for leaf systems)
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the outdoor environment system.

        Returns:
            dict: Dictionary containing output ports:
                - "outdoorTemperature": Outdoor air temperature [°C]
                - "globalIrradiation": Global solar irradiation [W/m²]
                - "outdoorCo2Concentration": Outdoor CO2 concentration [ppmv]
        """
        return self._output

    def validate(self, p):
        """Validate the system configuration.

        This method checks if the required data source (either DataFrame or filename parameters)
        is provided. If not, it issues a warning and marks the system as invalid for
        simulation, estimation, evaluation, and monitoring.

        Args:
            p (object): Printer object for outputting validation messages.

        Returns:
            tuple: Three boolean values indicating validation status for:
                - Simulator
                - Estimator
                - Optimizer
        """
        validated_for_simulator = True
        validated_for_estimator = True
        validated_for_optimizer = True

        if self.use_df and self.df is None:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: df must be provided when use_df=True to enable use of Simulator, Estimator, and Optimizer."
            p(message, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False
        elif self.df is None:
            if self.use_spreadsheet and (
                self.filename_outdoorTemperature is None
                or self.filename_globalIrradiation is None
                or self.filename_outdoorCo2Concentration is None
            ):
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: All three filename parameters must be provided if use_spreadsheet is True to enable use of Simulator, Estimator, and Optimizer."
                p(message, status="WARNING")
                validated_for_simulator = False
                validated_for_estimator = False
                validated_for_optimizer = False
            elif self.use_database and (
                self.uuid_outdoorTemperature is None
                or self.uuid_globalIrradiation is None
                or self.uuid_outdoorCo2Concentration is None
            ):
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: uuid parameters must be provided for all three data types if use_database is True to enable use of Simulator, Estimator, and Optimizer."
                p(message, status="WARNING")
                validated_for_simulator = False
                validated_for_estimator = False
                validated_for_optimizer = False
            elif not self.use_spreadsheet and not self.use_database and not self.use_df:
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Either df with use_df=True, use_spreadsheet=True, or use_database=True must be provided to enable use of Simulator, Estimator, and Optimizer."
                p(message, status="WARNING")
                validated_for_simulator = False
                validated_for_estimator = False
                validated_for_optimizer = False

        return (
            validated_for_simulator,
            validated_for_estimator,
            validated_for_optimizer,
        )

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
    ) -> None:
        """Initialize the outdoor environment system.

        This method performs the following initialization steps:
        1. Validates and resolves the weather data file paths
        2. Loads weather data from 3 separate files/database or DataFrame
        3. Verifies required data columns are present

        Args:
            start_time (List[datetime.datetime]): List of start times for the simulation periods.
            end_time (List[datetime.datetime]): List of end times for the simulation periods.
            step_size (List[int]): List of time step sizes in seconds.

        Raises:
            ValueError: If the weather data files cannot be found or required columns are missing.
        """
        if self.use_spreadsheet or self.use_database or self.use_df:
            # Validate df if use_df is True
            if self.use_df:
                if self.df is None:
                    raise ValueError("df must be provided when use_df=True.")
                required_keys = [
                    "outdoorTemperature",
                    "globalIrradiation",
                    "outdoorCo2Concentration",
                ]
                is_included = np.array(
                    [key in self.df.columns for key in required_keys]
                )
                assert np.all(
                    is_included
                ), f"The following required columns \"{', '.join(list(np.array(required_keys)[is_included==False]))}\" are not included in the provided data."
                # Create separate DataFrames for each weather parameter
                df_temp = self.df[["outdoorTemperature"]]
                df_irrad = self.df[["globalIrradiation"]]
                df_co2 = self.df[["outdoorCo2Concentration"]]
            else:
                df_temp = df_irrad = df_co2 = None

            # Use TimeSeriesInputSystem for each weather parameter (handles batching correctly)
            time_series_temp = TimeSeriesInputSystem(
                id=f"time series input - outdoorTemperature - {self.id}",
                df=df_temp,
                filename=self.filename_outdoorTemperature,
                datecolumn=self.datecolumn_outdoorTemperature,
                valuecolumn=(
                    "outdoorTemperature"
                    if self.use_df
                    else self.valuecolumn_outdoorTemperature
                ),
                use_spreadsheet=self.use_spreadsheet,
                use_database=self.use_database,
                uuid=self.uuid_outdoorTemperature,
                dbconfig=self.dbconfig_outdoorTemperature,
                # cache=False,  # Cache is disabled for outdoor temperature data to avoid loading the same column multiple times form the same file
            )

            time_series_temp.initialize(start_time, end_time, step_size)

            time_series_irrad = TimeSeriesInputSystem(
                id=f"time series input - globalIrradiation - {self.id}",
                df=df_irrad,
                filename=self.filename_globalIrradiation,
                datecolumn=self.datecolumn_globalIrradiation,
                valuecolumn=(
                    "globalIrradiation"
                    if self.use_df
                    else self.valuecolumn_globalIrradiation
                ),
                use_spreadsheet=self.use_spreadsheet,
                use_database=self.use_database,
                uuid=self.uuid_globalIrradiation,
                dbconfig=self.dbconfig_globalIrradiation,
                # cache=False,
            )
            time_series_irrad.initialize(start_time, end_time, step_size)

            time_series_co2 = TimeSeriesInputSystem(
                id=f"time series input - outdoorCo2Concentration - {self.id}",
                df=df_co2,
                filename=self.filename_outdoorCo2Concentration,
                datecolumn=self.datecolumn_outdoorCo2Concentration,
                valuecolumn=(
                    "outdoorCo2Concentration"
                    if self.use_df
                    else self.valuecolumn_outdoorCo2Concentration
                ),
                use_spreadsheet=self.use_spreadsheet,
                use_database=self.use_database,
                uuid=self.uuid_outdoorCo2Concentration,
                dbconfig=self.dbconfig_outdoorCo2Concentration,
                # cache=False,
            )
            time_series_co2.initialize(start_time, end_time, step_size)

            # Initialize outputs using the values from TimeSeriesInputSystem
            self.output["outdoorTemperature"].initialize(
                n_t=time_series_temp.n_timesteps,
                n_s=time_series_temp.batch_size,
                n_c=1,
                values=time_series_temp.values,
            )
            self.output["globalIrradiation"].initialize(
                n_t=time_series_irrad.n_timesteps,
                n_s=time_series_irrad.batch_size,
                n_c=1,
                values=time_series_irrad.values,
            )
            self.output["outdoorCo2Concentration"].initialize(
                n_t=time_series_co2.n_timesteps,
                n_s=time_series_co2.batch_size,
                n_c=1,
                values=time_series_co2.values,
            )

            # Expand parameters to n_c dimension for vectorization
            self.a = self.a.expand_to_n_c(self.n_c)
            self.b = self.b.expand_to_n_c(self.n_c)
        else:
            raise ValueError(
                "No data source provided. Set use_spreadsheet=True, use_database=True, or use_df=True."
            )

    def _apply(self, x):
        return x * self.a.get() + self.b.get()

    def do_step(
        self,
        second_time: Optional[float] = None,
        date_time: Optional[datetime.datetime] = None,
        step_size: Optional[float] = None,
        step_index: Optional[int] = None,
    ) -> None:
        """Perform one simulation step.

        This method reads the current weather data and applies optional linear corrections
        to the temperature values. The irradiation and CO2 concentration values are passed through
        without modification.

        Args:
            second_time (float, optional): Current simulation time in seconds.
            date_time (date_time, optional): Current simulation date and time.
            step_size (float, optional): Time step size in seconds.
            step_index (int, optional): Current simulation step index.
        """
        # Set the values for each output
        if self.apply_correction:
            self._output["outdoorTemperature"]._set(
                i_t=step_index, transformation=self._apply
            )
        else:
            self._output["outdoorTemperature"]._set(i_t=step_index)

        self._output["globalIrradiation"]._set(i_t=step_index)
        self._output["outdoorCo2Concentration"]._set(i_t=step_index)


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the outdoor environment component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the outdoor environment component.
    """
    node0 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    sp = SignaturePattern(id="outdoor_environment_signature_pattern")
    sp.add_modeled_node(node0)
    return sp


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the outdoor environment component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the outdoor environment component.
    """
    weather_station = Node(cls=core.namespace.BRICK.Weather_Station)
    temp = Node(cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor)
    irrad = Node(cls=core.namespace.BRICK.Global_Solar_Irradiation_Sensor)
    externalref_temp = Node(cls=core.namespace.BRICKREF.ExternalReference)
    externalref_irrad = Node(cls=core.namespace.BRICKREF.ExternalReference)
    timeseriesid_temp = Node(cls=core.namespace.XSD.string)
    timeseriesid_irrad = Node(cls=core.namespace.XSD.string)
    senaps_temp = Node(cls=core.namespace.XSD.string)
    senaps_irrad = Node(cls=core.namespace.XSD.string)

    # node3 = Node(cls=core.namespace.BRICK.Outdoor_CO2_Concentration_Sensor)
    sp = SignaturePattern(id="outdoor_environment_signature_pattern_brick")
    sp.add_triple(
        Optional_(
            subject=weather_station,
            object=temp,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )
    sp.add_triple(
        Optional_(
            subject=weather_station,
            object=irrad,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )
    sp.add_triple(
        Optional_(
            subject=temp,
            object=externalref_temp,
            predicate=core.namespace.BRICKREF.hasExternalReference,
        )  # Used in mortar
    )
    sp.add_triple(
        Optional_(
            subject=irrad,
            object=externalref_irrad,
            predicate=core.namespace.BRICKREF.hasExternalReference,
        )  # Used in mortar
    )
    sp.add_triple(
        Optional_(
            subject=externalref_temp,
            object=timeseriesid_temp,
            predicate=core.namespace.BRICKREF.hasTimeseriesId,
        )  # Used in mortar
    )
    sp.add_triple(
        Optional_(
            subject=externalref_irrad,
            object=timeseriesid_irrad,
            predicate=core.namespace.BRICKREF.hasTimeseriesId,
        )  # Used in mortar
    )
    sp.add_triple(
        Optional_(
            subject=temp, object=senaps_temp, predicate=core.namespace.SENAPS.senaps_id
        )  # Used in bts
    )
    sp.add_triple(
        Optional_(
            subject=irrad,
            object=senaps_irrad,
            predicate=core.namespace.SENAPS.senaps_id,
        )  # Used in bts
    )
    sp.add_modeled_node(temp)
    # sp.add_modeled_node(irrad)
    sp.add_parameter("_uuid_outdoorTemperature", senaps_temp)
    sp.add_parameter("_uuid_globalIrradiation", senaps_irrad)
    sp.add_parameter("_uuid_outdoorTemperature", timeseriesid_temp)
    sp.add_parameter("_uuid_globalIrradiation", timeseriesid_irrad)

    return sp


OutdoorEnvironmentSystem.add_signature_pattern(brick_signature_pattern())
OutdoorEnvironmentSystem.add_signature_pattern(saref_signature_pattern())
