from typing import Optional, Dict, List, Any, Union
import datetime
from twin4build.utils.constants import Constants 
import twin4build.utils.input_output_types as tps
import twin4build.core as core
import torch.nn as nn
from scipy.optimize import fsolve
import torch

class SpaceHeaterSystem(core.System, nn.Module):
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
                 Q_flow_nominal_sh=None,
                 T_a_nominal_sh=None,
                 T_b_nominal_sh=None,
                 TAir_nominal_sh=None,
                 thermalMassHeatCapacity=None,
                 nelements=None,
                 **kwargs) -> None:
        """Initialize the space heater system.

        Args:
            **kwargs: Keyword arguments passed to parent class.
                Should include thermal properties like heat transfer coefficient
                and thermal mass capacity.
        """
        super().__init__(**kwargs)
        self.thermalMassHeatCapacity = thermalMassHeatCapacity
        self.Q_flow_nominal_sh = Q_flow_nominal_sh
        self.T_a_nominal_sh = T_a_nominal_sh
        self.T_b_nominal_sh = T_b_nominal_sh
        self.TAir_nominal_sh = TAir_nominal_sh
        self.nelements = nelements
        self.n = 1.24
        self._config = {"parameters": []}
        self.input = {
            "supplyWaterTemperature": tps.Scalar(),
            "waterFlowRate": tps.Scalar(),
            "indoorTemperature": tps.Scalar()
        }
        self.output = {
            "outletWaterTemperature": tps.Vector(tensor=torch.zeros(nelements), size=self.nelements),
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
                  simulator: Optional[Any] = None) -> None:
        """Initialize the space heater model.

        Sets up initial conditions for outlet water temperatures and energy counter.

        Args:
            startTime (Optional[datetime.datetime], optional): Start time. Defaults to None.
            endTime (Optional[datetime.datetime], optional): End time. Defaults to None.
            stepSize (Optional[float], optional): Time step size. Defaults to None.
            model (Optional[Any], optional): Model object. Defaults to None.
        """
        self.output["Energy"].set(0, stepIndex=0)

        # for i in range(self.nelements):
        
        self.output["outletWaterTemperature"].increment(self.nelements).initialize()
        self.output["outletWaterTemperature"][:] = self.TAir_nominal_sh

        self.m_flow_nominal = self.Q_flow_nominal_sh/Constants.specificHeatCapacity["water"]/(self.T_a_nominal_sh-self.T_b_nominal_sh)
        UA0 = 10 # starting guess for heat transfer coefficient
        root = fsolve(self.f_root, UA0, args=(startTime, endTime, stepSize, 0, simulator), full_output=True)
        print(root)
        self.heatTransferCoefficient = float(root[0])


    def f_root(self, UA, *args):
        self.heatTransferCoefficient = float(UA)
        self._do_step_nominal(*args)
        return self.Q_flow_nominal_sh-self.output["Power"]

    # def f_root_inner(self, T_r, T_w_in):
    #     lhs = self.m_flow_nominal*Constants.specificHeatCapacity["water"]*(T_w_in-T_r)
    #     rhs = self.heatTransferCoefficient/self.nelements*(T_r-self.TAir_nominal_sh)**self.n
    #     return lhs-rhs

    def _do_step_nominal(self, 
                         secondTime: Optional[float] = None,
                         dateTime: Optional[datetime.datetime] = None,
                         stepSize: Optional[float] = None,
                         stepIndex: Optional[int] = None,
                         simulator: Optional[core.Model] = None) -> None:
        T_w_in = self.T_a_nominal_sh
        Q_r = 0
        for i in range(self.nelements):
            T_r = (self.m_flow_nominal*Constants.specificHeatCapacity["water"]*T_w_in + self.heatTransferCoefficient/self.nelements*self.TAir_nominal_sh)/(self.m_flow_nominal*Constants.specificHeatCapacity["water"] + self.heatTransferCoefficient/self.nelements)
            Q_r += self.heatTransferCoefficient/self.nelements*(T_r-self.TAir_nominal_sh)
            T_w_in = T_r
        self.output["Power"].set(Q_r, stepIndex)
    
    def do_step(self, secondTime: Optional[float] = None,
                dateTime: Optional[datetime.datetime] = None,
                stepSize: Optional[float] = None,
                stepIndex: Optional[int] = None) -> None:
        """Advance the space heater model by one time step.

        Calculates outlet water temperatures for each segment and updates power
        and energy outputs.

        Args:
            secondTime (Optional[float], optional): Time in seconds. Defaults to None.
            dateTime (Optional[datetime.datetime], optional): Date and time. Defaults to None.
            stepSize (Optional[float], optional): Time step size. Defaults to None.
        """
        T_w_in = self.input["supplyWaterTemperature"]
        Q_r = 0
        for i in range(self.nelements):
            K1 = (T_w_in*self.input["waterFlowRate"]*Constants.specificHeatCapacity["water"] + self.heatTransferCoefficient/self.nelements*self.input["indoorTemperature"])/(self.thermalMassHeatCapacity/self.nelements) + self.output["outletWaterTemperature"][i]/stepSize
            K2 = 1/stepSize + (self.heatTransferCoefficient/self.nelements + self.input["waterFlowRate"]*Constants.specificHeatCapacity["water"])/(self.thermalMassHeatCapacity/self.nelements)
            self.output["outletWaterTemperature"][i] = K1/K2
            T_w_in = self.output["outletWaterTemperature"][i]
            Q_r += self.heatTransferCoefficient/self.nelements*(self.output["outletWaterTemperature"][i]-self.input["indoorTemperature"])

        self.output["Power"].set(Q_r, stepIndex)
        # self.output["Energy"].set(self.output["Energy"] + Q_r*stepSize)
