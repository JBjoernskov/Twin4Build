# Standard library imports
import datetime
from typing import Any, Dict, List, Optional, Union

# Third party imports
import pandas as pd

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.translator.translator import Exact, Node, SignaturePattern, SinglePath


def get_signature_pattern_input():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="SensorSystem", priority=-10
    )
    sp.add_modeled_node(node0)
    return sp


def get_flow_signature_pattern_after_coil_air_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.Coil))  # waterside
    node3 = Node(cls=(core.namespace.S4BLDG.Coil))  # airside
    node4 = Node(cls=(core.namespace.S4BLDG.Coil))  # supersystem
    node5 = Node(cls=core.namespace.S4SYST.System)  # before waterside
    node6 = Node(cls=core.namespace.S4SYST.System)  # after waterside
    node7 = Node(cls=core.namespace.S4SYST.System)  # before airside
    node8 = Node(cls=core.namespace.S4SYST.System)  # after airside
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node5, object=node2, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(subject=node2, object=node6, predicate=core.namespace.FSO.returnsFluidTo)
    )
    sp.add_triple(
        Exact(subject=node7, object=node3, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(subject=node3, object=node8, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(subject=node2, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_triple(
        Exact(subject=node3, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_triple(
        SinglePath(
            subject=node3, object=node0, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_input("measuredValue", node4, ("outletAirTemperature"))
    sp.add_modeled_node(node0)
    return sp


def get_flow_signature_pattern_after_coil_air_side_simple():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node3 = Node(cls=(core.namespace.S4BLDG.Coil))  # airside
    node4 = Node(cls=(core.namespace.S4BLDG.Coil))  # supersystem
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="SensorSystem", priority=-1
    )
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node3, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_triple(
        SinglePath(
            subject=node3, object=node0, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_input("measuredValue", node4, ("outletAirTemperature"))
    sp.add_modeled_node(node0)
    return sp


def get_flow_signature_pattern_after_coil_water_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.Coil))  # waterside
    node3 = Node(cls=(core.namespace.S4BLDG.Coil))  # airside
    node4 = Node(cls=(core.namespace.S4BLDG.Coil))  # supersystem
    node5 = Node(cls=core.namespace.S4SYST.System)  # before waterside
    node6 = Node(cls=core.namespace.S4SYST.System)  # after waterside
    node7 = Node(cls=core.namespace.S4SYST.System)  # before airside
    node8 = Node(cls=core.namespace.S4SYST.System)  # after airside
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node5, object=node2, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(subject=node2, object=node6, predicate=core.namespace.FSO.returnsFluidTo)
    )
    sp.add_triple(
        Exact(subject=node7, object=node3, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(subject=node3, object=node8, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(subject=node2, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_triple(
        Exact(subject=node3, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_triple(
        SinglePath(
            subject=node2, object=node0, predicate=core.namespace.FSO.returnsFluidTo
        )
    )
    sp.add_input("measuredValue", node4, ("outletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp


def get_flow_signature_pattern_before_coil_water_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.Coil))  # waterside
    node3 = Node(cls=(core.namespace.S4BLDG.Coil))  # airside
    node4 = Node(cls=(core.namespace.S4BLDG.Coil))  # supersystem
    node6 = Node(cls=core.namespace.S4SYST.System)  # after waterside
    node7 = Node(cls=core.namespace.S4SYST.System)  # before airside
    node8 = Node(cls=core.namespace.S4SYST.System)  # after airside
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    # sp.add_triple(Exact(subject=node5, object=node2, predicate="suppliesFluidTo"))
    sp.add_triple(
        Exact(subject=node2, object=node6, predicate=core.namespace.FSO.returnsFluidTo)
    )
    sp.add_triple(
        Exact(subject=node7, object=node3, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(subject=node3, object=node8, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(subject=node2, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_triple(
        Exact(subject=node3, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_triple(
        SinglePath(
            subject=node2, object=node0, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_input("measuredValue", node4, ("inletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp


# Properties of spaces
def get_space_temperature_signature_pattern():
    node0 = Node(cls=(core.namespace.SAREF.Sensor))
    node1 = Node(cls=(core.namespace.SAREF.Temperature))
    node2 = Node(cls=(core.namespace.S4BLDG.BuildingSpace))
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node1, object=node2, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_input("measuredValue", node2, ("indoorTemperature"))
    sp.add_modeled_node(node0)
    return sp


# Properties of spaces
def get_space_co2_signature_pattern():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Co2,))
    node2 = Node(cls=(core.namespace.S4BLDG.BuildingSpace,))
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node1, object=node2, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_input("measuredValue", node2, ("indoorCO2"))
    sp.add_modeled_node(node0)
    return sp


def get_position_signature_pattern():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.OpeningPosition,))
    node2 = Node(
        cls=(
            core.namespace.S4BLDG.Valve,
            core.namespace.S4BLDG.Damper,
        )
    )
    node3 = Node(cls=(core.namespace.S4BLDG.Controller))
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node1, object=node2, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_triple(
        Exact(subject=node3, object=node1, predicate=core.namespace.SAREF.controls)
    )
    sp.add_input("measuredValue", node3, ("inputSignal", "inputSignal"))
    sp.add_modeled_node(node0)
    return sp


def get_temperature_before_air_to_air_supply_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery,))  # AirToAirPrimary
    node9 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirSuper
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")

    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        SinglePath(
            subject=node2, object=node0, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_triple(
        Exact(subject=node2, object=node9, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_input("measuredValue", node2, ("primaryTemperatureIn"))
    sp.add_modeled_node(node0)

    return sp


def get_temperature_before_air_to_air_exhaust_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirPrimary

    node9 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirSuper

    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        SinglePath(
            subject=node0, object=node2, predicate=core.namespace.FSO.returnsFluidTo
        )
    )
    sp.add_triple(
        Exact(subject=node2, object=node9, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_input("measuredValue", node2, ("secondaryTemperatureIn"))
    sp.add_modeled_node(node0)

    return sp


def get_temperature_after_air_to_air_supply_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirPrimary
    node9 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirSuper

    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(
            subject=node0, object=node2, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_triple(
        Exact(subject=node2, object=node9, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_input("measuredValue", node2, ("primaryTemperatureOut"))
    sp.add_modeled_node(node0)

    return sp


def get_temperature_after_air_to_air_exhaust_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirPrimary

    node9 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirSuper

    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="SensorSystem")
    sp.add_triple(
        Exact(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node2, object=node0, predicate=core.namespace.FSO.returnsFluidTo)
    )
    sp.add_triple(
        Exact(subject=node2, object=node9, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_input("measuredValue", node2, ("secondaryTemperatureOut"))
    sp.add_modeled_node(node0)

    return sp


class SensorSystem(core.System):
    """A system representing a physical or virtual sensor in the building.

    This class implements sensor functionality, supporting both physical sensors
    (reading from time series data) and virtual sensors (computing values from
    other inputs). It integrates with TimeSeriesInputSystem for data handling.

    Attributes:
        filename (Optional[str]): Path to sensor readings file.
        df (Optional[pd.DataFrame]): Direct DataFrame input of sensor readings.
        datecolumn (int): Column index for datetime values. Defaults to 0.
        valuecolumn (int): Column index for sensor readings. Defaults to 1.
        is_leaf (bool): True if sensor reads from file/DataFrame, False if virtual.
        physicalSystem (Optional[TimeSeriesInputSystem]): Data handling system for physical sensors.
        _config (Dict[str, Any]): Configuration parameters and reading specifications.

    Note:
        A sensor must either have connections to other systems (virtual sensor) or
        have data input through filename/df (physical sensor).
    """

    sp = [
        get_temperature_before_air_to_air_supply_side(),
        get_temperature_before_air_to_air_exhaust_side(),
        get_temperature_after_air_to_air_supply_side(),
        get_temperature_after_air_to_air_exhaust_side(),
        get_signature_pattern_input(),
        get_flow_signature_pattern_after_coil_air_side(),
        get_flow_signature_pattern_after_coil_water_side(),
        get_flow_signature_pattern_before_coil_water_side(),
        get_space_temperature_signature_pattern(),
        get_space_co2_signature_pattern(),
        get_position_signature_pattern(),
    ]

    def __init__(
        self,
        filename: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        uuid: Optional[str] = None,
        name: Optional[str] = None,
        dbconfig: Optional[Dict[str, Any]] = None,
        useSpreadsheet: bool = False,
        useDatabase: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the sensor system.

        Args:
            filename (Optional[str], optional): Path to sensor readings file.
                Defaults to None.
            df (Optional[pd.DataFrame], optional): DataFrame containing readings.
                Defaults to None.
            useSpreadsheet (bool, optional): Whether to use a spreadsheet for input.
                Defaults to False.
            useDatabase (bool, optional): Whether to use a database for input.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to parent class.

        Note:
            Either filename/df must be provided for physical sensors, or
            the sensor must have connections defined for virtual sensors.
        """
        assert (
            useSpreadsheet == False or useDatabase == False
        ), "useSpreadsheet and useDatabase cannot both be True."
        super().__init__(**kwargs)
        self.input = {"measuredValue": tps.Scalar()}
        self.output = {
            "measuredValue": tps.Scalar(0)
        }  # TODO: Not necessary to be a leaf scalar, if the sensor has inputs. Need to implement check in initialize()
        self.useSpreadsheet = useSpreadsheet
        self.useDatabase = useDatabase
        self.filename = filename
        self.df = df
        self.datecolumn = 0
        self.valuecolumn = 1
        self.uuid = uuid
        self.name = name
        self.dbconfig = dbconfig
        self._config = {
            "parameters": ["useSpreadsheet", "useDatabase"],
            "spreadsheet": ["filename", "datecolumn", "valuecolumn"],
            "database": ["uuid", "name", "dbconfig"],
        }

    @property
    def config(self):
        return self._config

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

        if len(self.connectsAt) == 0 and self.filename is None:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename or df must be provided to enable use of Simulator, Estimator, and Optimizer."
            p(message, plain=True, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif len(self.connectsAt) > 0 and self.filename is None:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename or df must be provided to enable use of Estimator."
            p(message, plain=True, status="WARNING")
            validated_for_estimator = False

        self.is_leaf = len(self.connectsAt) == 0
        self.output["measuredValue"].is_leaf = self.is_leaf

        return (
            validated_for_simulator,
            validated_for_estimator,
            validated_for_optimizer,
        )

    def initialize(
        self,
        startTime: Optional[datetime.datetime] = None,
        endTime: Optional[datetime.datetime] = None,
        stepSize: Optional[float] = None,
        simulator: Optional[Any] = None,
    ) -> None:
        """Initialize the sensor system.

        Sets up the physical or virtual sensor system and initializes the step instance.

        Args:
            startTime (Optional[datetime.datetime]): Start time for the simulation.
            endTime (Optional[datetime.datetime]): End time for the simulation.
            stepSize (Optional[float]): Time step size in seconds.
            model (Optional[Any]): Model object (not used in this class).
        """

        if (
            self.filename is not None
            or self.df is not None
            or self.dbconfig is not None
        ):
            if self.df is None:
                assert (
                    self.useSpreadsheet == True or self.useDatabase == True
                ), "useSpreadsheet or useDatabase must be True if df is not provided."
            self.physicalSystem = TimeSeriesInputSystem(
                id=f"time series input - {self.id}",
                df=self.df,
                filename=self.filename,
                datecolumn=self.datecolumn,
                valuecolumn=self.valuecolumn,
                useSpreadsheet=self.useSpreadsheet,
                useDatabase=self.useDatabase,
                uuid=self.uuid,
                name=self.name,
                dbconfig=self.dbconfig,
            )
            self.physicalSystem.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )

        else:
            self.physicalSystem = None

        assert (
            len(self.connectsAt) == 0 and self.physicalSystem is None
        ) == False, f'Sensor object "{self.id}" has no inputs and and holds no data.'

        if self.is_leaf:
            self.output["measuredValue"].initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
                values=self.physicalSystem.df.values,
            )
        else:
            self.input["measuredValue"].initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )
            self.output["measuredValue"].initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )

    def do_step(
        self,
        secondTime: Optional[float] = None,
        dateTime: Optional[datetime.datetime] = None,
        stepSize: Optional[float] = None,
        stepIndex: Optional[int] = None,
    ) -> None:
        """Execute one time step of the sensor system.

        Updates sensor outputs based on either physical readings or virtual calculations.

        Args:
            secondTime (Optional[float]): Current simulation time in seconds.
            dateTime (Optional[datetime.datetime]): Current simulation datetime.
            stepSize (Optional[float]): Time step size in seconds.
        """
        if self.is_leaf:
            self.output["measuredValue"].set(stepIndex=stepIndex)
        else:
            self.output["measuredValue"].set(
                self.input["measuredValue"].get(), stepIndex
            )

    def get_physical_readings(
        self,
        startTime: Optional[datetime.datetime] = None,
        endTime: Optional[datetime.datetime] = None,
        stepSize: Optional[float] = None,
    ) -> pd.DataFrame:
        """Retrieve physical sensor readings for a specified time period.

        Args:
            startTime (Optional[datetime.datetime]): Start time for readings.
            endTime (Optional[datetime.datetime]): End time for readings.
            stepSize (Optional[float]): Time step size in seconds.

        Returns:
            pd.DataFrame: DataFrame containing sensor readings.

        Raises:
            AssertionError: If called on a virtual sensor (no physical readings available).
        """
        assert (
            self.physicalSystem is not None
        ), f'Cannot return physical readings for Sensor with id "{self.id}" as the argument "filename" was not provided when the object was initialized.'
        self.physicalSystem.initialize(startTime, endTime, stepSize)
        return self.physicalSystem.df
