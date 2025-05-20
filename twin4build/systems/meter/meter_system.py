from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath
import twin4build.core as core
import twin4build.systems as systems
import numpy as np
import twin4build.utils.input_output_types as tps
import pandas as pd
import datetime
from typing import Optional, Any

def get_signature_pattern():
    node0 = Node(cls=(core.SAREF.Meter))
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="MeterSystem", priority=-1)
    sp.add_modeled_node(node0)
    return sp

def get_fan_power_signature_pattern():
    node0 = Node(cls=(core.SAREF.Meter,))
    node1 = Node(cls=(core.SAREF.Power))
    node2 = Node(cls=(core.S4BLDG.Fan))
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="MeterSystem")
    sp.add_triple(Exact(subject=node0, object=node1, predicate="observes"))
    sp.add_triple(Exact(subject=node1, object=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, "Power")
    sp.add_modeled_node(node0)
    return sp

def get_flow_supply_fan_signature_pattern():
    node0 = Node(cls=(core.SAREF.Meter))
    node1 = Node(cls=(core.SAREF.Flow))
    node2 = Node(cls=(core.S4BLDG.Fan))
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="MeterSystem")
    sp.add_triple(Exact(subject=node0, object=node1, predicate="observes"))
    sp.add_triple(Exact(subject=node1, object=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, "airFlowRate")
    sp.add_modeled_node(node0)
    return sp

def get_space_heater_energy_signature_pattern():
    node0 = Node(cls=(core.SAREF.Meter,))
    node1 = Node(cls=(core.SAREF.Energy))
    node2 = Node(cls=(core.S4BLDG.SpaceHeater))
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="MeterSystem")
    sp.add_triple(Exact(subject=node0, object=node1, predicate="observes"))
    sp.add_triple(Exact(subject=node1, object=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, "spaceHeaterEnergy")
    sp.add_modeled_node(node0)
    return sp


class MeterSystem(core.System):
    r"""
    A system representing a physical or virtual meter in the building.

    This class implements meter functionality, supporting both physical meters
    (reading from time series data) and virtual meters (computing values from
    other inputs). It integrates with TimeSeriesInputSystem for data handling.

    Mathematical Formulation
    -----------------------

    For a physical meter:
       The measured value is read directly from a time series file or DataFrame:

        .. math::

            y_{meter}(t) = y_{file}(t)

       where:
          - :math:`y_{meter}(t)` is the meter output at time :math:`t`
          - :math:`y_{file}(t)` is the value read from file/DataFrame at time :math:`t`

    For a virtual meter:
       The measured value is computed or passed through from connected system outputs:

        .. math::

            y_{meter}(t) = y_{input}(t)

       where:
          - :math:`y_{meter}(t)` is the meter output at time :math:`t`
          - :math:`y_{input}(t)` is the input value from connected system at time :math:`t`

    Parameters
    ----------
    filename : Optional[str]
        Path to meter readings file.
    df_input : Optional[pd.DataFrame]
        Direct DataFrame input of meter readings.
    **kwargs
        Additional keyword arguments passed to the parent System class.

    Attributes
    ----------
    filename : Optional[str]
        Path to meter readings file.
    df_input : Optional[pd.DataFrame]
        Direct DataFrame input of meter readings.
    datecolumn : int
        Column index for datetime values. Defaults to 0.
    valuecolumn : int
        Column index for meter readings. Defaults to 1.
    isPhysicalSystem : bool
        True if meter reads from file/DataFrame, False if virtual.
    physicalSystem : Optional[TimeSeriesInputSystem]
        Data handling system for physical meters.
    _config : Dict[str, Any]
        Configuration parameters and reading specifications.

    Notes
    -----
    - For physical meters, readings must be provided via either filename or df_input
    - For virtual meters, readings are computed from connected system outputs
    - The system supports both time series data and direct DataFrame inputs
    - Validation checks ensure proper configuration for different operational modes
    """

    sp = [get_signature_pattern(), get_fan_power_signature_pattern(), get_space_heater_energy_signature_pattern(), get_flow_supply_fan_signature_pattern()]

    def __init__(self,
                 filename: Optional[str] = None,
                 df_input: Optional[pd.DataFrame] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input = {"measuredValue": tps.Scalar()}
        self.output = {"measuredValue": tps.Scalar(is_leaf=True)}
        self.filename = filename
        self.df_input = df_input
        self.datecolumn = 0
        self.valuecolumn = 1
        self._config = {
            "parameters": [],
            "readings": {
                "filename": self.filename,
                "datecolumn": self.datecolumn,
                "valuecolumn": self.valuecolumn
            }
        }

    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def validate(self, p) -> tuple[bool, bool, bool, bool]:
        """Validate the meter system configuration.

        Checks if the meter has proper inputs for different operational modes.

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
        validated_for_evaluator = True
        validated_for_monitor = True

        if len(self.connectsAt)==0 and self.filename is None and self.df_input is None:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename or df_input must be provided to enable use of Simulator, Estimator, Evaluator, and Monitor."
            p(message, plain=True, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_evaluator = False
            validated_for_monitor = False

        elif len(self.connectsAt)>0 and self.filename is None and self.df_input is None:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename or df_input must be provided to enable use of Estimator, and Monitor."
            p(message, plain=True, status="WARNING")
            validated_for_estimator = False
            validated_for_monitor = False
            
        return (validated_for_simulator, validated_for_estimator, 
                validated_for_evaluator, validated_for_monitor)

    def initialize(self,
                  startTime: Optional[datetime.datetime] = None,
                  endTime: Optional[datetime.datetime] = None,
                  stepSize: Optional[float] = None,
                  simulator: Optional[Any] = None) -> None:
        """Initialize the meter system.

        Sets up the physical or virtual meter system and initializes the step instance.

        Args:
            startTime (Optional[datetime.datetime]): Start time for the simulation.
            endTime (Optional[datetime.datetime]): End time for the simulation.
            stepSize (Optional[float]): Time step size in seconds.
            model (Optional[Any]): Model object (not used in this class).
        """
        assert (len(self.connectsAt)==0 and self.filename is None and self.df_input is None)==False, \
            f'Meter object "{self.id}" has no inputs and the argument "filename" or "df_input" in the constructor was not provided.'
        
        self.isPhysicalSystem = len(self.connectsAt) == 0

        if self.filename is not None:
            self.physicalSystem = TimeSeriesInputSystem(id=f"time series input - {self.id}", filename=self.filename, datecolumn=self.datecolumn, valuecolumn=self.valuecolumn)
            self.output["measuredValue"].initialize(startTime=startTime,
                                                endTime=endTime,
                                                stepSize=stepSize,
                                                simulator=simulator,
                                                values=self.physicalSystem.physicalSystemReadings.values)
        elif self.df_input is not None:
            self.physicalSystem = TimeSeriesInputSystem(id=f"time series input - {self.id}", df_input=self.df_input, datecolumn=self.datecolumn, valuecolumn=self.valuecolumn)
            self.output["measuredValue"].initialize(startTime=startTime,
                                                    endTime=endTime,
                                                    stepSize=stepSize,
                                                    simulator=simulator,
                                                    values=self.physicalSystem.physicalSystemReadings.values)
        else:
            self.physicalSystem = None
            self.output["measuredValue"].is_leaf = False
            self.input["measuredValue"].initialize(startTime=startTime,
                                                    endTime=endTime,
                                                    stepSize=stepSize,
                                                    simulator=simulator)
            self.output["measuredValue"].initialize(startTime=startTime,
                                                    endTime=endTime,
                                                    stepSize=stepSize,
                                                    simulator=simulator)

    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        """Execute one time step of the meter system.

        Updates meter outputs based on either physical readings or virtual calculations.

        Args:
            secondTime (Optional[float]): Current simulation time in seconds.
            dateTime (Optional[datetime.datetime]): Current simulation datetime.
            stepSize (Optional[float]): Time step size in seconds.
        """
        if self.isPhysicalSystem:
            self.output["measuredValue"].set(stepIndex=stepIndex)
        else:
            self.output["measuredValue"].set(self.input["measuredValue"].get(), stepIndex)

    def get_physical_readings(self, startTime: Optional[datetime.datetime] = None,
                            endTime: Optional[datetime.datetime] = None,
                            stepSize: Optional[float] = None) -> pd.DataFrame:
        """Retrieve physical meter readings for a specified time period.

        Args:
            startTime (Optional[datetime.datetime]): Start time for readings.
            endTime (Optional[datetime.datetime]): End time for readings.
            stepSize (Optional[float]): Time step size in seconds.

        Returns:
            pd.DataFrame: DataFrame containing meter readings.

        Raises:
            AssertionError: If called on a virtual meter (no physical readings available).
        """
        assert self.physicalSystem is not None, f"Cannot return physical readings for Meter with id \"{self.id}\" as the argument \"filename\" was not provided when the object was initialized."
        self.physicalSystem.initialize(startTime,
                                        endTime,
                                        stepSize)
        return self.physicalSystem.physicalSystemReadings