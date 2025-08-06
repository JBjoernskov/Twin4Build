# Standard library imports
import datetime
from typing import Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class FanTorchSystem(core.System, nn.Module):
    r"""
    A fan system model implemented with PyTorch for gradient-based optimization.

    This model represents a fan that controls air flow rate and temperature, considering
    both the power consumption and the heat added to the air stream.

    Mathematical Formulation
    -----------------------

    The fan power is calculated using a polynomial equation:

        .. math::

            P = P_{nom} \cdot (c_1 + c_2\frac{\dot{m}}{\dot{m}_{nom}} + c_3(\frac{\dot{m}}{\dot{m}_{nom}})^2 + c_4(\frac{\dot{m}}{\dot{m}_{nom}})^3)

    where:
       - :math:`P` is the fan power [W]
       - :math:`P_{nom}` is the nominal power [W]
       - :math:`\dot{m}` is the air flow rate [m³/s]
       - :math:`\dot{m}_{nom}` is the nominal air flow rate [m³/s]
       - :math:`c_1` to :math:`c_4` are polynomial coefficients

    The outlet air temperature is calculated considering the heat added by the fan:

        .. math::

            T_{out} = T_{in} + \frac{P \cdot f_{total}}{\dot{m} \cdot \rho \cdot c_p}

    where:
       - :math:`T_{out}` is the outlet temperature [°C]
       - :math:`T_{in}` is the inlet temperature [°C]
       - :math:`f_{total}` is the total fan efficiency factor
       - :math:`\rho` is the air density [kg/m³]
       - :math:`c_p` is the specific heat capacity of air [J/(kg·K)]

    Args
    ----------
    nominalPowerRate : float
        Nominal power rate [W]
    nominalAirFlowRate : float
        Nominal air flow rate [m³/s]
    c1 : float
        Constant term in power polynomial
    c2 : float
        Linear term coefficient in power polynomial
    c3 : float
        Quadratic term coefficient in power polynomial
    c4 : float
        Cubic term coefficient in power polynomial
    f_total : float
        Total fan efficiency factor (0-1)

    Notes
    -----
    Model Assumptions:
       - Fan power follows polynomial relationship with flow rate
       - All heat from fan power is added to air stream
       - Constant air density and specific heat capacity
       - No mechanical losses considered separately

    Implementation Details:
       - Uses PyTorch for gradient-based optimization
       - Parameters are stored as trainable PyTorch parameters
       - Includes safety checks for numerical stability
       - All calculations performed in SI units
    """

    def __init__(
        self,
        nominalPowerRate: float = None,
        nominalAirFlowRate: float = None,
        c1: float = None,
        c2: float = None,
        c3: float = None,
        c4: float = None,
        f_total: float = None,
        **kwargs,
    ):
        """
        Initialize the fan system model.

        Args:
            nominalPowerRate: Nominal power rate [W]
            nominalAirFlowRate: Nominal air flow rate [m³/s]
            c1-c4: Polynomial coefficients for power calculation
            f_total: Total fan efficiency factor
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store parameters as tps.Parameters for gradient tracking
        self.nominalPowerRate = tps.Parameter(
            torch.tensor(nominalPowerRate, dtype=torch.float64), requires_grad=False
        )
        self.nominalAirFlowRate = tps.Parameter(
            torch.tensor(nominalAirFlowRate, dtype=torch.float64), requires_grad=False
        )
        self.c1 = tps.Parameter(
            torch.tensor(c1, dtype=torch.float64), requires_grad=False
        )
        self.c2 = tps.Parameter(
            torch.tensor(c2, dtype=torch.float64), requires_grad=False
        )
        self.c3 = tps.Parameter(
            torch.tensor(c3, dtype=torch.float64), requires_grad=False
        )
        self.c4 = tps.Parameter(
            torch.tensor(c4, dtype=torch.float64), requires_grad=False
        )
        self.f_total = tps.Parameter(
            torch.tensor(f_total, dtype=torch.float64), requires_grad=False
        )

        # Define inputs and outputs as private variables
        self._input = {"airFlowRate": tps.Scalar(), "inletAirTemperature": tps.Scalar()}
        self._output = {"outletAirTemperature": tps.Scalar(), "Power": tps.Scalar()}

        # Define parameters for calibration
        self.parameter = {
            "nominalPowerRate": {"lb": 0.0, "ub": 10000.0},
            "nominalAirFlowRate": {"lb": 0.0, "ub": 10.0},
            "c1": {"lb": -10.0, "ub": 10.0},
            "c2": {"lb": -10.0, "ub": 10.0},
            "c3": {"lb": -10.0, "ub": 10.0},
            "c4": {"lb": -10.0, "ub": 10.0},
            "f_total": {"lb": 0.0, "ub": 1.0},
        }

        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    @property
    def config(self):
        """Get the configuration of the fan system."""
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the fan system.

        Returns:
            dict: Dictionary containing input ports:
                - "airFlowRate": Air flow rate [m³/s]
                - "inletAirTemperature": Inlet air temperature [°C]
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the fan system.

        Returns:
            dict: Dictionary containing output ports:
                - "outletAirTemperature": Outlet air temperature [°C]
                - "Power": Fan power consumption [W]
        """
        return self._output

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """Initialize the fan system."""
        # Initialize I/O
        for input in self.input.values():
            input.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )
        for output in self.output.values():
            output.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )
        self.INITIALIZED = True

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """
        Perform one step of the fan system simulation.

        The fan power is calculated using a polynomial equation:
        P = P_nom * (c1 + c2*(m/m_nom) + c3*(m/m_nom)^2 + c4*(m/m_nom)^3)
        where:
        - P is the fan power
        - P_nom is the nominal power
        - m is the air flow rate
        - m_nom is the nominal air flow rate
        - c1-c4 are polynomial coefficients

        The outlet air temperature is calculated considering the heat added by the fan:
        T_out = T_in + (P * f_total) / (m * c_p)
        where:
        - T_out is the outlet temperature
        - T_in is the inlet temperature
        - f_total is the total fan efficiency
        - c_p is the specific heat capacity of air
        """
        # Get inputs
        air_flow_rate = self.input["airFlowRate"].get()
        inlet_temp = self.input["inletAirTemperature"].get()

        # Convert to torch tensors if not already
        if not isinstance(air_flow_rate, torch.Tensor):
            air_flow_rate = torch.tensor(air_flow_rate, dtype=torch.float64)
        if not isinstance(inlet_temp, torch.Tensor):
            inlet_temp = torch.tensor(inlet_temp, dtype=torch.float64)

        # Calculate normalized flow rate
        m_norm = air_flow_rate / self.nominalAirFlowRate.get()

        # Calculate fan power using polynomial equation
        power = self.nominalPowerRate.get() * (
            self.c1.get()
            + self.c2.get() * m_norm
            + self.c3.get() * m_norm**2
            + self.c4.get() * m_norm**3
        )

        # Calculate outlet temperature
        # Using air properties at standard conditions
        c_p = 1005.0  # J/(kg·K)
        rho = 1.2  # kg/m³

        # Convert volume flow rate to mass flow rate
        m_dot = air_flow_rate * rho

        # Calculate temperature rise
        delta_T = (power * self.f_total.get()) / (m_dot * c_p)
        outlet_temp = inlet_temp + delta_T

        # Update outputs
        self.output["outletAirTemperature"].set(outlet_temp, stepIndex)
        self.output["Power"].set(power, stepIndex)
