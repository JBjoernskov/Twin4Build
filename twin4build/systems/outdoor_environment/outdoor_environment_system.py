# Standard library imports
import datetime
import os
import warnings
from typing import Optional

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    Exact,
    Node,
    Optional_,
    SignaturePattern,
    SinglePath,
)
from twin4build.utils.data_loaders.load import load_from_database, load_from_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir


def get_signature_pattern():
    node0 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="OutdoorEnvironmentSystem"
    )
    sp.add_modeled_node(node0)
    return sp


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

    sp = [get_signature_pattern()]

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        useSpreadsheet: Optional[bool] = False,
        useDatabase: Optional[bool] = False,
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
        name_outdoorTemperature: Optional[str] = None,
        dbconfig_outdoorTemperature: Optional[dict] = None,
        uuid_globalIrradiation: Optional[str] = None,
        name_globalIrradiation: Optional[str] = None,
        dbconfig_globalIrradiation: Optional[dict] = None,
        uuid_outdoorCo2Concentration: Optional[str] = None,
        name_outdoorCo2Concentration: Optional[str] = None,
        dbconfig_outdoorCo2Concentration: Optional[dict] = None,
        a: Optional[float] = 1,
        b: Optional[float] = 0,
        apply_correction: Optional[bool] = False,
        **kwargs,
    ):
        assert (
            useSpreadsheet == False or useDatabase == False
        ), "useSpreadsheet and useDatabase cannot both be True."
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
        self.useSpreadsheet = useSpreadsheet
        self.useDatabase = useDatabase

        self.filename_outdoorTemperature = filename_outdoorTemperature
        self.datecolumn_outdoorTemperature = datecolumn_outdoorTemperature
        self.valuecolumn_outdoorTemperature = valuecolumn_outdoorTemperature

        self.filename_globalIrradiation = filename_globalIrradiation
        self.datecolumn_globalIrradiation = datecolumn_globalIrradiation
        self.valuecolumn_globalIrradiation = valuecolumn_globalIrradiation

        self.filename_outdoorCo2Concentration = filename_outdoorCo2Concentration
        self.datecolumn_outdoorCo2Concentration = datecolumn_outdoorCo2Concentration
        self.valuecolumn_outdoorCo2Concentration = valuecolumn_outdoorCo2Concentration

        self.uuid_outdoorTemperature = uuid_outdoorTemperature
        self.name_outdoorTemperature = name_outdoorTemperature
        self.dbconfig_outdoorTemperature = dbconfig_outdoorTemperature

        self.uuid_globalIrradiation = uuid_globalIrradiation
        self.name_globalIrradiation = name_globalIrradiation
        self.dbconfig_globalIrradiation = dbconfig_globalIrradiation

        self.uuid_outdoorCo2Concentration = uuid_outdoorCo2Concentration
        self.name_outdoorCo2Concentration = name_outdoorCo2Concentration
        self.dbconfig_outdoorCo2Concentration = dbconfig_outdoorCo2Concentration

        self.df = df
        self.a = tps.Parameter(
            torch.tensor(a, dtype=torch.float64), requires_grad=False
        )
        self.b = tps.Parameter(
            torch.tensor(b, dtype=torch.float64), requires_grad=False
        )
        self.apply_correction = apply_correction
        self.cached_initialize_arguments = None
        self.cache_root = get_main_dir()

        self._config = {
            "parameters": [
                "a",
                "b",
                "apply_correction",
                "useSpreadsheet",
                "useDatabase",
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
                "name_outdoorTemperature",
                "dbconfig_outdoorTemperature",
                "uuid_globalIrradiation",
                "name_globalIrradiation",
                "dbconfig_globalIrradiation",
                "uuid_outdoorCo2Concentration",
                "name_outdoorCo2Concentration",
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
                - "outdoorCo2Concentration": Outdoor CO2 concentration [ppm]
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

        if self.df is None:
            if self.useSpreadsheet and (
                self.filename_outdoorTemperature is None
                or self.filename_globalIrradiation is None
                or self.filename_outdoorCo2Concentration is None
            ):
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: All three filename parameters must be provided if useSpreadsheet is True to enable use of Simulator, Estimator, and Optimizer."
                p(message, plain=True, status="WARNING")
                validated_for_simulator = False
                validated_for_estimator = False
                validated_for_optimizer = False
            elif (
                self.useDatabase
                and (
                    self.uuid_outdoorTemperature is None
                    and self.name_outdoorTemperature is None
                )
                and (
                    self.uuid_globalIrradiation is None
                    and self.name_globalIrradiation is None
                )
                and (
                    self.uuid_outdoorCo2Concentration is None
                    and self.name_outdoorCo2Concentration is None
                )
            ):
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: uuid or name parameters must be provided for all three data types if useDatabase is True to enable use of Simulator, Estimator, and Optimizer."
                p(message, plain=True, status="WARNING")
                validated_for_simulator = False
                validated_for_estimator = False
                validated_for_optimizer = False
            elif not self.useSpreadsheet and not self.useDatabase:
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Either df or useSpreadsheet=True or useDatabase=True must be provided to enable use of Simulator, Estimator, and Optimizer."
                p(message, plain=True, status="WARNING")
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
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
        simulator: core.Simulator,
    ) -> None:
        """Initialize the outdoor environment system.

        This method performs the following initialization steps:
        1. Validates and resolves the weather data file paths
        2. Loads weather data from 3 separate files or DataFrame
        3. Verifies required data columns are present

        Args:
            start_time (datetime.datetime): Start time of the simulation period.
            end_time (datetime.datetime): End time of the simulation period.
            step_size (int): Time step size in seconds.
            simulator (core.Simulator): Simulation model object.

        Raises:
            ValueError: If the weather data files cannot be found or required columns are missing.
        """
        if self.df is None or (
            self.cached_initialize_arguments != (start_time, end_time, step_size)
            and self.cached_initialize_arguments is not None
        ):
            if self.useSpreadsheet:
                # Load from 3 separate files
                self.df = self._load_from_separate_files(
                    start_time, end_time, step_size
                )
            elif self.useDatabase:
                # Load from database
                self.df = self._load_from_database(start_time, end_time, step_size)
            else:
                # Use provided DataFrame
                if self.df is None:
                    raise ValueError(
                        "No data source provided. Set useSpreadsheet=True or useDatabase=True or provide df."
                    )

        self.cached_initialize_arguments = (start_time, end_time, step_size)
        required_keys = [
            "outdoorTemperature",
            "globalIrradiation",
            "outdoorCo2Concentration",
        ]
        is_included = np.array([key in self.df.columns for key in required_keys])
        assert np.all(
            is_included
        ), f"The following required columns \"{', '.join(list(np.array(required_keys)[is_included==False]))}\" are not included in the provided data."

        for key, output in self._output.items():
            output.initialize(
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                simulator=simulator,
                values=self.df[key].values,
            )

    def _load_from_separate_files(self, start_time, end_time, step_size):
        """Load data from 3 separate CSV files and combine them."""
        # Validate that all required filename parameters are provided
        if (
            self.filename_outdoorTemperature is None
            or self.filename_globalIrradiation is None
            or self.filename_outdoorCo2Concentration is None
        ):
            raise ValueError(
                "All three filename parameters (filename_outdoorTemperature, filename_globalIrradiation, filename_outdoorCo2Concentration) must be provided when useSpreadsheet=True"
            )

        # Load each file
        df_temp = load_from_spreadsheet(
            filename=self.filename_outdoorTemperature,
            datecolumn=self.datecolumn_outdoorTemperature,
            valuecolumn=self.valuecolumn_outdoorTemperature,
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            cache_root=self.cache_root,
        )
        df_irrad = load_from_spreadsheet(
            filename=self.filename_globalIrradiation,
            datecolumn=self.datecolumn_globalIrradiation,
            valuecolumn=self.valuecolumn_globalIrradiation,
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            cache_root=self.cache_root,
        )
        df_co2 = load_from_spreadsheet(
            filename=self.filename_outdoorCo2Concentration,
            datecolumn=self.datecolumn_outdoorCo2Concentration,
            valuecolumn=self.valuecolumn_outdoorCo2Concentration,
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            cache_root=self.cache_root,
        )

        # Create combined DataFrame
        # When valuecolumn is specified, load_from_spreadsheet returns a pandas Series with DatetimeIndex
        df = pd.DataFrame(
            {
                "time": df_temp.index,
                "outdoorTemperature": df_temp.values,
                "globalIrradiation": df_irrad.values,
                "outdoorCo2Concentration": df_co2.values,
            }
        )

        return df

    def _load_from_database(self, start_time, end_time, step_size):
        """Load data from database and combine them."""

        # Validate that all required database parameters are provided
        if (
            (
                self.uuid_outdoorTemperature is None
                and self.name_outdoorTemperature is None
            )
            or (
                self.uuid_globalIrradiation is None
                and self.name_globalIrradiation is None
            )
            or (
                self.uuid_outdoorCo2Concentration is None
                and self.name_outdoorCo2Concentration is None
            )
        ):
            raise ValueError(
                "uuid or name parameters must be provided for all three data types (outdoorTemperature, globalIrradiation, outdoorCo2Concentration) when useDatabase=True"
            )

        # Load each parameter from database
        df_temp = load_from_database(
            uuid=self.uuid_outdoorTemperature,
            name=self.name_outdoorTemperature,
            dbconfig=self.database_config_outdoorTemperature,
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            dt_limit=1200,
        )

        df_irrad = load_from_database(
            uuid=self.uuid_globalIrradiation,
            name=self.name_globalIrradiation,
            dbconfig=self.database_config_globalIrradiation,
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            dt_limit=1200,
        )

        df_co2 = load_from_database(
            uuid=self.uuid_outdoorCo2Concentration,
            name=self.name_outdoorCo2Concentration,
            dbconfig=self.dbconfig_outdoorCo2Concentration,
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            dt_limit=1200,
        )

        # Create combined DataFrame
        df = pd.DataFrame(
            {
                "time": df_temp.index,
                "outdoorTemperature": df_temp.values,
                "globalIrradiation": df_irrad.values,
                "outdoorCo2Concentration": df_co2.values,
            }
        )

        return df

    def _apply(self, x):
        return x * self.a.get() + self.b.get()

    def do_step(
        self,
        secondTime: Optional[float] = None,
        dateTime: Optional[datetime.datetime] = None,
        step_size: Optional[float] = None,
        stepIndex: Optional[int] = None,
    ) -> None:
        """Perform one simulation step.

        This method reads the current weather data and applies optional linear corrections
        to the temperature values. The irradiation and CO2 concentration values are passed through
        without modification.

        Args:
            secondTime (float, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation date and time.
            step_size (float, optional): Time step size in seconds.
            stepIndex (int, optional): Current simulation step index.
        """
        # Set the values for each output
        if self.apply_correction:
            self._output["outdoorTemperature"].set(
                stepIndex=stepIndex, apply=self._apply
            )
        else:
            self._output["outdoorTemperature"].set(stepIndex=stepIndex)

        self._output["globalIrradiation"].set(stepIndex=stepIndex)
        self._output["outdoorCo2Concentration"].set(stepIndex=stepIndex)
