# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
from twin4build.systems.saref4syst.declarations import Input, Output


class CoilSystem(core.System):
    r"""
    A coil system model implemented with PyTorch for gradient-based optimization.

    This model represents a heating/cooling coil that transfers heat between air and water,
    calculating the required heating or cooling power based on air flow rate and temperature
    differences.

    Mathematical Formulation
    ------------------------

    The heating/cooling power is calculated using the following equations:

    For heating mode (when :math:`T_{in} < T_{out,set}`):

        .. math::

            P_{heat} = \dot{m}_{air} \cdot c_{p,air} \cdot (T_{out,set} - T_{in})

        .. math::

            P_{cool} = 0

    For cooling mode (when :math:`T_{in} \geq T_{out,set}`):

        .. math::

            P_{heat} = 0

        .. math::

            P_{cool} = \dot{m}_{air} \cdot c_{p,air} \cdot (T_{in} - T_{out,set})

    where:
       - :math:`P_{heat}` is the heating power [W]
       - :math:`P_{cool}` is the cooling power [W]
       - :math:`\dot{m}_{air}` is the air flow rate [kg/s]
       - :math:`c_{p,air}` is the specific heat capacity of air [J/(kg·K)]
       - :math:`T_{in}` is the inlet air temperature [°C]
       - :math:`T_{out,set}` is the outlet air temperature setpoint [°C]

    Notes
    -----
    Model Assumptions:
       - Perfect heat transfer (outlet temperature equals setpoint)
       - Constant specific heat capacity of air
       - No heat losses to the environment
       - No water-side calculations (focus on air-side performance)

    Implementation Details:
       - If air flow rate is below threshold (1e-5 kg/s), both heating and cooling
         powers are set to zero
       - The model uses PyTorch tensors for gradient-based optimization
       - All calculations are performed in SI units
       - The specific heat capacity of air is the constant ``CP_AIR`` from
         ``twin4build.utils.constants`` (not a stored parameter)
    """

    inletAirTemperature = Input(doc="Inlet air temperature [degC]")
    outletAirTemperatureSetpoint = Input(doc="Outlet air temperature setpoint [degC]")
    airFlowRate = Input(doc="Air flow rate [kg/s]")
    heatingPower = Output(doc="Heating power [W]")
    coolingPower = Output(doc="Cooling power [W]")
    outletAirTemperature = Output(doc="Outlet air temperature [degC]")

    def forward(self, x, inputs, p, step_size):
        """Pure one-step ideal-coil model (functorch-safe, stateless).

        Heating/cooling power from the air-side energy balance with a
        perfect-transfer assumption (outlet temperature equals the setpoint);
        zero power below the flow tolerance.
        """
        inlet_air_temp = inputs["inletAirTemperature"]
        outlet_air_temp_setpoint = inputs["outletAirTemperatureSetpoint"]
        air_flow_rate = inputs["airFlowRate"]

        # Calculate heating/cooling power based on temperature difference
        tol = 1e-5
        zero = torch.zeros_like(air_flow_rate)

        # Condition: flow rate above tolerance
        has_flow = air_flow_rate > tol

        # Condition: heating mode (inlet < setpoint)
        is_heating_mode = inlet_air_temp < outlet_air_temp_setpoint

        # Calculate power magnitude (same formula, different sign interpretation)
        power = (
            air_flow_rate
            * constants.CP_AIR
            * torch.abs(outlet_air_temp_setpoint - inlet_air_temp)
        )

        # Select heating/cooling power based on mode and flow
        heating_power = torch.where(has_flow & is_heating_mode, power, zero)
        cooling_power = torch.where(has_flow & (~is_heating_mode), power, zero)
        return x, {
            "heatingPower": heating_power,
            "coolingPower": cooling_power,
            "outletAirTemperature": outlet_air_temp_setpoint,
        }


# Deprecated aliases (removed in twin4build 2.1)
CoilTorchSystem = CoilSystem
