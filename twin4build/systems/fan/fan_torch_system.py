# Standard library imports
import datetime
from typing import Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
import twin4build.utils.types as tps


class FanTorchSystem(core.System, nn.Module):
    r"""
    A fan system model implemented with PyTorch for gradient-based optimization.

    This model represents a fan that controls air flow rate and temperature, considering
    both the power consumption and the heat added to the air stream.

    Args:
        nominalPowerRate : Nominal power rate [W]
        nominalAirFlowRate : Nominal air flow rate [kg/s]
        c1 : Constant term in power polynomial
        c2 : Linear term coefficient in power polynomial
        c3 : Quadratic term coefficient in power polynomial
        c4 : Cubic term coefficient in power polynomial
        f_total : Total fan efficiency factor (0-1)

    Mathematical Formulation
    ------------------------

    The fan power is calculated using a polynomial equation:

        .. math::

            P = P_{nom} \cdot \left(c_1 + c_2\frac{\dot{m}}{\dot{m}_{nom}} + c_3\left(\frac{\dot{m}}{\dot{m}_{nom}}\right)^2 + c_4\left(\frac{\dot{m}}{\dot{m}_{nom}}\right)^3\right)

    where:
       - :math:`P` is the fan power [W]
       - :math:`P_{nom}` is the nominal power [W]
       - :math:`\dot{m}` is the air mass flow rate [kg/s]
       - :math:`\dot{m}_{nom}` is the nominal air mass flow rate [kg/s]
       - :math:`c_1` to :math:`c_4` are polynomial coefficients that can be calibrated

    The outlet air temperature is calculated considering the heat added by the fan:

        .. math::

            T_{out} = T_{in} + \frac{P \cdot f_{total}}{\dot{m} \cdot c_p}

    where:
       - :math:`T_{out}` is the outlet temperature [°C]
       - :math:`T_{in}` is the inlet temperature [°C]
       - :math:`f_{total}` is the fraction of power that is converted to heat and added to the air stream
       - :math:`c_p` is the specific heat capacity of air [J/(kg·K)]

    Notes
    -----
    Model Assumptions:
       - Fan power follows polynomial relationship with flow rate
       - A fraction :math:`f_{total}` of the fan power is added as heat to the air stream
       - Constant air density and specific heat capacity
       - No mechanical losses considered separately

    Implementation Details:
       - Uses PyTorch for gradient-based optimization
       - Parameters are stored as ``tps.Parameter`` objects (``requires_grad=False``
         by default); they are not trained directly but can be calibrated via the
         Estimator
       - All calculations performed in SI units
    """

    def __init__(
        self,
        nominalPowerRate: float = 1000,
        nominalAirFlowRate: float = 1.0,
        c1: float = 0,
        c2: float = 0.8,
        c3: float = 0.2,
        c4: float = 0.0,
        f_total: float = 0.9,
        **kwargs,
    ):
        """
        Initialize the fan system model.

        Args:
            nominalPowerRate: Nominal power rate [W]
            nominalAirFlowRate: Nominal air mass flow rate [kg/s]
            c1-c4: Polynomial coefficients for power calculation
            f_total: Total fan efficiency factor
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store parameters as tps.Parameters for gradient tracking
        self.nominalPowerRate = tps.Parameter(
            torch.tensor(nominalPowerRate, dtype=tps.float_dtype()), requires_grad=False
        )
        self.nominalAirFlowRate = tps.Parameter(
            torch.tensor(nominalAirFlowRate, dtype=tps.float_dtype()), requires_grad=False
        )
        self.c1 = tps.Parameter(
            torch.tensor(c1, dtype=tps.float_dtype()), requires_grad=False
        )
        self.c2 = tps.Parameter(
            torch.tensor(c2, dtype=tps.float_dtype()), requires_grad=False
        )
        self.c3 = tps.Parameter(
            torch.tensor(c3, dtype=tps.float_dtype()), requires_grad=False
        )
        self.c4 = tps.Parameter(
            torch.tensor(c4, dtype=tps.float_dtype()), requires_grad=False
        )
        self.f_total = tps.Parameter(
            torch.tensor(f_total, dtype=tps.float_dtype()), requires_grad=False
        )

        # Define inputs and outputs as private variables
        self._input = {"airFlowRate": tps.Scalar(), "inletAirTemperature": tps.Scalar()}
        self._output = {"outletAirTemperature": tps.Scalar(), "Power": tps.Scalar()}

        # Define parameters for calibration.  Tightened to a real AHU
        # supply / exhaust fan operating envelope so the auto-estimator
        # can't pin a fan to ``nominalPowerRate = 0`` (no fan, but
        # outlet temperature still updated from cp * m * dT) or
        # ``f_total = 0`` (zero efficiency means the model attributes
        # *all* the shaft power to fluid heat gain, which then forces
        # SAT past the coil's reach).
        self.parameter = {
            # AHU fan rated power.  Spans small fan-coil supply (~ 200
            # W) up to a 10 kW primary AHU.  Outside this range the
            # other parameters compensate in unphysical ways.
            "nominalPowerRate": {"lb": 200.0, "ub": 10000.0},
            # AHU air flow [kg/s].  ~ 0.5 kg/s is a small fan-coil,
            # ~ 10 kg/s a large central handler.
            "nominalAirFlowRate": {"lb": 0.5, "ub": 10.0},
            # Polynomial coefficients of the ``P(m)`` curve.  These
            # can legitimately be negative (curve concavity) but
            # values past ~ 5 in magnitude give pathological power
            # curves that the solver loves because they let one
            # operating point dominate the fit.
            "c1": {"lb": -1.0, "ub": 1.0},
            "c2": {"lb": -1.0, "ub": 1.0},
            "c3": {"lb": -1.0, "ub": 1.0},
            "c4": {"lb": -1.0, "ub": 1.0},
            # Total fan efficiency.  Real centrifugal AHU fans land
            # between 0.4 and 0.85; lower bound 0.3 keeps very small
            # / poorly maintained units in scope without admitting
            # the unphysical 0.
            "f_total": {"lb": 0.3, "ub": 0.9},
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
                - "airFlowRate": Air mass flow rate [kg/s]
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
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the fan system."""
        # Initialize I/O
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        if hasattr(self, "_n_c_compiled") and self._n_c_compiled > 1:
            self.n_c = self._n_c_compiled
        else:
            self.n_c = 1

        for input in self.input.values():
            input.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )
        for output in self.output.values():
            output.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )

        # Expand parameters to n_c dimension for vectorization
        self.nominalPowerRate = self.nominalPowerRate.expand_to_n_c(self.n_c)
        self.nominalAirFlowRate = self.nominalAirFlowRate.expand_to_n_c(self.n_c)
        self.c1 = self.c1.expand_to_n_c(self.n_c)
        self.c2 = self.c2.expand_to_n_c(self.n_c)
        self.c3 = self.c3.expand_to_n_c(self.n_c)
        self.c4 = self.c4.expand_to_n_c(self.n_c)
        self.f_total = self.f_total.expand_to_n_c(self.n_c)

        self.INITIALIZED = True

    PARAM_NAMES = (
        "nominalPowerRate",
        "nominalAirFlowRate",
        "c1",
        "c2",
        "c3",
        "c4",
        "f_total",
    )

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step fan model (functorch-safe, stateless).

        Fan power from the polynomial power curve:
        ``P = P_nom * (c1 + c2*(m/m_nom) + c3*(m/m_nom)^2 + c4*(m/m_nom)^3)``,
        outlet temperature from the heat added to the air stream:
        ``T_out = T_in + (P * f_total) / (m * c_p)``.
        """
        m_dot = inputs["airFlowRate"]
        inlet_temp = inputs["inletAirTemperature"]

        # Calculate normalized flow rate
        m_norm = m_dot / params["nominalAirFlowRate"]

        # Calculate fan power using polynomial equation
        power = params["nominalPowerRate"] * (
            params["c1"]
            + params["c2"] * m_norm
            + params["c3"] * m_norm**2
            + params["c4"] * m_norm**3
        )

        # Calculate temperature rise
        delta_T = (power * params["f_total"]) / (m_dot * constants.CP_AIR)
        outlet_temp = inlet_temp + delta_T
        return x, {"outletAirTemperature": outlet_temp, "Power": power}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one step of the fan system simulation.

        Thin port-I/O wrapper delegating the math to :meth:`forward`.
        """
        inputs = {
            "airFlowRate": self.input["airFlowRate"].get(),
            "inletAirTemperature": self.input["inletAirTemperature"].get(),
        }
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["outletAirTemperature"]._set(
            outs["outletAirTemperature"], i_t=step_index
        )
        self.output["Power"]._set(outs["Power"], i_t=step_index)
