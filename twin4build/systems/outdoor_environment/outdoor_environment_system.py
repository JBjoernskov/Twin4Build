import numpy as np
import twin4build.core as core
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.get_main_dir import get_main_dir
import os
import warnings
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
import warnings
import twin4build.utils.input_output_types as tps
import datetime
from typing import Optional

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.OutdoorEnvironment)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="OutdoorEnvironmentSystem")
    sp.add_modeled_node(node0)
    return sp

class OutdoorEnvironmentSystem(core.System):
    """An outdoor environment system model that provides weather data for building simulations.
    
    This model represents the outdoor environment by providing time-series data for:
    - Outdoor air temperature
    - Global solar irradiation
    - Outdoor CO2 concentration (constant value)
    
    The model reads weather data from CSV files and can optionally apply a linear
    correction to the data. The model is designed to be used as a boundary condition
    for building energy simulations.
    
    Args:
        df_input (pandas.DataFrame, optional): Input DataFrame containing weather data.
            Must have columns 'outdoorTemperature' and 'globalIrradiation'.
        filename (str, optional): Path to CSV file containing weather data.
            Either df_input or filename must be provided.
        a (float, optional): Correction factor for linear correction of weather data.
        b (float, optional): Correction offset for linear correction of weather data.
        apply_correction (bool, optional): Whether to apply linear correction to weather data.
    """
    sp = [get_signature_pattern()]
    def __init__(self,
                 df_input=None,
                 filename=None,
                 a = None,
                 b = None,
                 apply_correction = None,
                **kwargs):
        super().__init__(**kwargs)
    
        if df_input is None and filename is None:
            warnings.warn("Neither \"df_input\" nor \"filename\" was provided as argument. The component will not be able to provide any output.")
            
        self.input = {}
        self.output = {"outdoorTemperature": tps.Scalar(),
                       "globalIrradiation": tps.Scalar(),
                       "outdoorCo2Concentration": tps.Scalar()}
        
        self.filename = filename
        self.df = df_input
        self.a = a
        self.b = b
        self.apply_correction = apply_correction
        self.cached_initialize_arguments = None
        self.cache_root = get_main_dir()

        self._config = {"parameters": ["a", "b", "apply_correction"],
                        "readings": {"filename": filename,
                                     "datecolumn": None,
                                     "valuecolumn": None}
                        }
    @property
    def config(self):
        """Get the configuration parameters.

        Returns:
            dict: Dictionary containing configuration parameters and file reading settings.
        """
        return self._config
    
    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        """Cache system data for the specified time period.
        
        This method is currently not implemented as the system does not require caching.
        
        Args:
            startTime (datetime, optional): Start time of the simulation period.
            endTime (datetime, optional): End time of the simulation period.
            stepSize (float, optional): Time step size in seconds.
        """
        pass

    def validate(self, p):
        """Validate the system configuration.
        
        This method checks if the required data source (either DataFrame or filename)
        is provided. If not, it issues a warning and marks the system as invalid for
        simulation, estimation, evaluation, and monitoring.
        
        Args:
            p (object): Printer object for outputting validation messages.
            
        Returns:
            tuple: Four boolean values indicating validation status for:
                - Simulator
                - Estimator
                - Evaluator
                - Monitor
        """
        validated_for_simulator = True
        validated_for_estimator = True
        validated_for_evaluator = True
        validated_for_monitor = True

        if self.df is None and self.filename is None:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Either df_input or filename must be provided to enable use of Simulator, Estimator, Evaluator, and Monitor."
            p(message, plain=True, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_evaluator = False
            validated_for_monitor = False

        return (validated_for_simulator, validated_for_estimator, validated_for_evaluator, validated_for_monitor)

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    simulator=None):
        """Initialize the outdoor environment system.
        
        This method performs the following initialization steps:
        1. Validates and resolves the weather data file path
        2. Loads weather data from file or DataFrame
        3. Verifies required data columns are present
        
        Args:
            startTime (datetime, optional): Start time of the simulation period.
            endTime (datetime, optional): End time of the simulation period.
            stepSize (float, optional): Time step size in seconds.
            simulator (object, optional): Simulation model object.
            
        Raises:
            ValueError: If the weather data file cannot be found or required columns are missing.
        """
        if self.filename is not None:
            if os.path.isfile(self.filename)==False: #Absolute or relative was provided
                #Check if relative path to root was provided
                filename_ = os.path.join(self.cache_root, self.filename)
                if os.path.isfile(filename_)==False:
                    raise(ValueError(f"Neither one of the following filenames exist: \n\"{self.filename}\"\n{filename_}"))
                self.filename = filename_

        if self.df is None or (self.cached_initialize_arguments!=(startTime, endTime, stepSize) and self.cached_initialize_arguments is not None):
            self.df = load_spreadsheet(filename=self.filename, stepSize=stepSize, start_time=startTime, end_time=endTime, dt_limit=1200, cache_root=self.cache_root)
        self.cached_initialize_arguments = (startTime, endTime, stepSize)
        required_keys = ["outdoorTemperature", "globalIrradiation"]
        is_included = np.array([key in np.array([self.df.columns]) for key in required_keys])
        assert np.all(is_included), f"The following required columns \"{', '.join(list(np.array(required_keys)[is_included==False]))}\" are not included in the provided weather file {self.filename}." 

    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        """Perform one simulation step.
        
        This method reads the current weather data and applies optional linear corrections
        to the temperature and irradiation values. The outdoor CO2 concentration is set
        to a constant value of 400 ppm.
        
        Args:
            secondTime (float, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation date and time.
            stepSize (float, optional): Time step size in seconds.
            stepIndex (int, optional): Current simulation step index.
        """
        temp = self.df["outdoorTemperature"].iloc[stepIndex]
        irradiation = self.df["globalIrradiation"].iloc[stepIndex]

        if self.apply_correction and self.a is not None and self.b is not None:
            temp = temp * self.a + self.b
            irradiation = irradiation * self.a + self.b

        self.output["outdoorTemperature"].set(temp, stepIndex)
        self.output["globalIrradiation"].set(irradiation, stepIndex)
        self.output["outdoorCo2Concentration"].set(400, stepIndex)

