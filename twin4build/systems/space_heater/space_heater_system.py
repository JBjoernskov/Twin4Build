from typing import Optional, Dict, List, Any, Union
import datetime
import twin4build.utils.input_output_types as tps
import twin4build.core as core

class SpaceHeaterSystem(core.System):
    """A system modeling a space heater (radiator) with thermal dynamics.
    
    This class implements a dynamic model of a space heater that calculates outlet
    water temperatures, power output, and energy consumption. The model uses a
    discretized approach with multiple segments to improve accuracy.

    Attributes:
        specificHeatCapacityWater (float): Specific heat capacity of water [J/(kg·K)].
        heatTransferCoefficient (float): Heat transfer coefficient [W/K].
        thermalMassHeatCapacity (float): Thermal mass heat capacity [J/K].
        output (Dict[str, Union[List[float], float]]): Contains:
            - outletWaterTemperature: List of temperatures for each segment
            - Energy: Cumulative energy consumption [kWh]
            - Power: Current power output [W]
        _config (Dict[str, List[str]]): Configuration parameters.

    Note:
        The heater is modeled using n=10 segments for improved accuracy of the
        temperature distribution along the radiator length.
    """

    def __init__(self, 
                 heatTransferCoefficient=None,
                 thermalMassHeatCapacity=None,
                 **kwargs) -> None:
        """Initialize the space heater system.

        Args:
            **kwargs: Keyword arguments passed to parent class.
                Should include thermal properties like heat transfer coefficient
                and thermal mass capacity.
        """
        super().__init__(**kwargs)
        self.specificHeatCapacityWater = Constants.specificHeatCapacity["water"]
        self.heatTransferCoefficient = None
        self.thermalMassHeatCapacity = None
        self._config = {"parameters": []}
        self.input = {
            "supplyWaterTemperature": tps.Scalar(),
            "waterFlowRate": tps.Scalar(),
            "indoorTemperature": tps.Scalar()
        }
        self.output = {
            "outletWaterTemperature": tps.Scalar(),
            "Power": tps.Scalar(),
            "Energy": tps.Scalar()
        }

    @property
    def config(self) -> Dict[str, List[str]]:
        """Get the configuration parameters.

        Returns:
            Dict[str, List[str]]: Dictionary containing configuration parameter names.
        """
        return self._config

    def cache(self, startTime: Optional[datetime.datetime] = None,
             endTime: Optional[datetime.datetime] = None,
             stepSize: Optional[float] = None) -> None:
        """Cache method (placeholder for interface compatibility).

        Args:
            startTime (Optional[datetime.datetime], optional): Start time. Defaults to None.
            endTime (Optional[datetime.datetime], optional): End time. Defaults to None.
            stepSize (Optional[float], optional): Time step size. Defaults to None.
        """
        pass

    def initialize(self, startTime: Optional[datetime.datetime] = None,
                  endTime: Optional[datetime.datetime] = None,
                  stepSize: Optional[float] = None,
                  model: Optional[Any] = None) -> None:
        """Initialize the space heater model.

        Sets up initial conditions for outlet water temperatures and energy counter.

        Args:
            startTime (Optional[datetime.datetime], optional): Start time. Defaults to None.
            endTime (Optional[datetime.datetime], optional): End time. Defaults to None.
            stepSize (Optional[float], optional): Time step size. Defaults to None.
            model (Optional[Any], optional): Model object. Defaults to None.
        """
        self.output["outletWaterTemperature"] = [self.output["outletWaterTemperature"] for i in range(10)]
        self.output["Energy"] = 0

    def do_step(self, secondTime: Optional[float] = None,
                dateTime: Optional[datetime.datetime] = None,
                stepSize: Optional[float] = None) -> None:
        """Advance the space heater model by one time step.

        Calculates outlet water temperatures for each segment and updates power
        and energy outputs.

        Args:
            secondTime (Optional[float], optional): Time in seconds. Defaults to None.
            dateTime (Optional[datetime.datetime], optional): Date and time. Defaults to None.
            stepSize (Optional[float], optional): Time step size. Defaults to None.
        """
        n = 10
        self.input["supplyWaterTemperature"] = [self.input["supplyWaterTemperature"] for i in range(n)]
        for i in range(n):
            # K1 = (self.input["supplyWaterTemperature"]*self.input["waterFlowRate"]*self.specificHeatCapacityWater + self.heatTransferCoefficient*self.input["indoorTemperature"])/self.thermalMassHeatCapacity + self.output["outletWaterTemperature"]/stepSize
            # K2 = 1/stepSize + (self.heatTransferCoefficient + self.input["waterFlowRate"]*self.specificHeatCapacityWater)/self.thermalMassHeatCapacity
            K1 = (self.input["supplyWaterTemperature"][i]*self.input["waterFlowRate"]*self.specificHeatCapacityWater + self.heatTransferCoefficient/n*self.input["indoorTemperature"])/(self.thermalMassHeatCapacity/n) + self.output["outletWaterTemperature"][i]/stepSize
            K2 = 1/stepSize + (self.heatTransferCoefficient/n + self.input["waterFlowRate"]*self.specificHeatCapacityWater)/(self.thermalMassHeatCapacity/n)
            self.output["outletWaterTemperature"][i] = K1/K2
            if i!=n-1:
                self.input["supplyWaterTemperature"][i+1] = self.output["outletWaterTemperature"][i]
            # print(self.output["outletWaterTemperature"])

        #Two different ways of calculating heat consumption:
        # 1. Heat delivered to room
        # Q_r = sum([self.heatTransferCoefficient/n*(self.output["outletWaterTemperature"][i]-self.input["indoorTemperature"]) for i in range(n)])

        # 2. Heat delivered to radiator from heating system
        Q_r = self.input["waterFlowRate"]*self.specificHeatCapacityWater*(self.input["supplyWaterTemperature"][0]-self.output["outletWaterTemperature"][-1])

        self.output["Power"].set(Q_r)
        self.output["Energy"].set(self.output["Energy"] + Q_r*stepSize/3600/1000)
