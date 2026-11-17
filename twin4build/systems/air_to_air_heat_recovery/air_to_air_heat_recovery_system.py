# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter


class AirToAirHeatRecoverySystem(core.System):
    r"""
    An air-to-air heat recovery system model.

    This model represents a heat exchanger that recovers heat between supply and
    exhaust air streams. The effectiveness varies based on flow rates and operation
    mode (heating or cooling). The model includes temperature setpoint control and
    handles cases where heat recovery is not feasible.

    The model is implemented with the following features:
       - Flow-dependent effectiveness interpolation
       - Separate effectiveness values for heating and cooling modes
       - Temperature setpoint control
       - Energy conservation between air streams

    Args:
       eps_75_h: Effectiveness at 75% flow in heating mode
       eps_100_h: Effectiveness at 100% flow in heating mode
       eps_75_c: Effectiveness at 75% flow in cooling mode
       eps_100_c: Effectiveness at 100% flow in cooling mode
       primaryAirFlowRateMax: Maximum primary (supply) air flow rate [kg/s]
       secondaryAirFlowRateMax: Maximum secondary (exhaust) air flow rate [kg/s]

    Mathematical Formulation:

       The effectiveness :math:`\varepsilon` is interpolated based on flow rate fraction :math:`f`:

       .. math::

          f = \frac{1}{2} \frac{\dot{m}_{a,sup} + \dot{m}_{a,exh}}{\operatorname{max}(\dot{m}_{a,sup}, \dot{m}_{a,exh})}

       .. math::

          \varepsilon = \varepsilon_{75} + (\varepsilon_{100} - \varepsilon_{75}) \cdot \frac{f - 0.75}{1 - 0.75}

       where:
          - :math:`\varepsilon_{75}`: Effectiveness at 75% flow
          - :math:`\varepsilon_{100}`: Effectiveness at 100% flow
          - :math:`f`: Normalized flow rate

       The outlet temperature of the supply air stream is:

       .. math::

          T_\text{out,sup} = T_\text{in,sup} + \varepsilon(f) \cdot (T_\text{in,exh} - T_\text{in,sup}) \cdot \frac{C_\min}{C_\text{sup}}

       where:
          - :math:`T_\text{in,sup}`: Inlet temperature of the supply air
          - :math:`T_\text{in,exh}`: Inlet temperature of the exhaust air
          - :math:`C_\text{sup}=\dot{m}_{a,primary} \cdot c_p`: Heat capacity rate of the supply (primary) air
          - :math:`C_\text{exh}=\dot{m}_{a,secondary} \cdot c_p`: Heat capacity rate of the exhaust (secondary) air
          - :math:`C_\min = \min(C_\text{sup}, C_\text{exh})`: Minimum heat capacity rate


       The outlet temperature of the exhaust air stream is:

       .. math::

          T_\text{out,exh} = T_\text{in,exh} - \Delta T \cdot \frac{C_\text{sup}}{C_\text{exh}}

       where:
          - :math:`\Delta T = T_\text{out,sup} - T_\text{in,sup}`: Temperature change in supply air stream

    """

    primaryAirFlowRate = Input(doc="Primary (supply) air flow rate [kg/s]")
    secondaryAirFlowRate = Input(doc="Secondary (exhaust) air flow rate [kg/s]")
    primaryTemperatureIn = Input(doc="Primary air inlet temperature [degC]")
    secondaryTemperatureIn = Input(doc="Secondary air inlet temperature [degC]")
    primaryTemperatureOutSetpoint = Input(doc="Primary air outlet temperature setpoint [degC]")
    primaryTemperatureOut = Output(doc="Primary air outlet temperature [degC]")
    secondaryTemperatureOut = Output(doc="Secondary air outlet temperature [degC]")
    eps_75_h = Parameter(0.86, lb=0.4, ub=0.95, doc="Effectiveness at 75% flow, heating [-]")
    eps_100_h = Parameter(0.8, lb=0.4, ub=0.95, doc="Effectiveness at 100% flow, heating [-]")
    eps_75_c = Parameter(0.86, lb=0.4, ub=0.95, doc="Effectiveness at 75% flow, cooling [-]")
    eps_100_c = Parameter(0.8, lb=0.4, ub=0.95, doc="Effectiveness at 100% flow, cooling [-]")
    primaryAirFlowRateMax = Parameter(
        1.0, lb=0.5, ub=10.0, relative=(0.25, 4.0), doc="Maximum primary air flow rate [kg/s]"
    )
    secondaryAirFlowRateMax = Parameter(
        1.0, lb=0.5, ub=10.0, relative=(0.25, 4.0), doc="Maximum secondary air flow rate [kg/s]"
    )

    def forward(self, x, inputs, params, step_size):
        """Stateless: the heat recovered between the two streams.

        Effectiveness interpolated between the 75%/100% reference points,
        operation mode selected from inlet temperatures vs the setpoint,
        pass-through when there is no flow or recovery is infeasible.
        """
        tol = 1e-5

        primary_flow = inputs["primaryAirFlowRate"]
        secondary_flow = inputs["secondaryAirFlowRate"]
        primary_temp_in = inputs["primaryTemperatureIn"]
        secondary_temp_in = inputs["secondaryTemperatureIn"]
        primary_temp_setpoint = inputs["primaryTemperatureOutSetpoint"]

        # Condition: both flow rates above tolerance
        has_flow = (primary_flow > tol) & (secondary_flow > tol)

        # Determine feasible mode (heating if primary < secondary, else cooling)
        is_heating_feasible = primary_temp_in < secondary_temp_in

        # Select effectiveness values based on feasible mode
        eps_75 = torch.where(
            is_heating_feasible, params["eps_75_h"], params["eps_75_c"]
        )
        eps_100 = torch.where(
            is_heating_feasible, params["eps_100_h"], params["eps_100_c"]
        )

        # Determine operation mode (heating if primary < setpoint, else cooling)
        is_heating_mode = primary_temp_in < primary_temp_setpoint

        # Feasible matches operation mode when both are heating or both are cooling
        feasible_matches_mode = is_heating_feasible == is_heating_mode

        # Condition for heat recovery: has flow AND feasible matches mode
        do_heat_recovery = has_flow & feasible_matches_mode

        # Calculate heat recovery values (computed for all, selected later)
        m_a_max = torch.max(
            params["primaryAirFlowRateMax"], params["secondaryAirFlowRateMax"]
        )

        f_flow = 0.5 * (primary_flow + secondary_flow) / m_a_max
        eps_op = eps_75 + (eps_100 - eps_75) * (f_flow - 0.75) / (1 - 0.75)

        C_sup = primary_flow * constants.CP_AIR
        C_exh = secondary_flow * constants.CP_AIR
        C_min = torch.min(C_sup, C_exh)

        # Zero flow on either side (dampers closed, unit off) is a normal
        # state handled by ``has_flow`` above; the capacity ratios are
        # clamped so the (discarded) heat-recovery branch stays finite.
        C_sup_safe = torch.clamp(C_sup, min=1e-9 * constants.CP_AIR)
        C_exh_safe = torch.clamp(C_exh, min=1e-9 * constants.CP_AIR)

        # Calculate primary temperature out with heat recovery
        primary_temp_out_hr = primary_temp_in + eps_op * (
            secondary_temp_in - primary_temp_in
        ) * (C_min / C_sup_safe)

        # Clamp to setpoint based on operation mode
        # In heating mode: clamp if output > setpoint
        # In cooling mode: clamp if output < setpoint
        heating_exceeds = is_heating_mode & (
            primary_temp_out_hr > primary_temp_setpoint
        )
        cooling_exceeds = (~is_heating_mode) & (
            primary_temp_out_hr < primary_temp_setpoint
        )
        needs_clamping = heating_exceeds | cooling_exceeds

        primary_temp_out_hr = torch.where(
            needs_clamping, primary_temp_setpoint, primary_temp_out_hr
        )

        # Calculate secondary temperature out using energy conservation
        primary_delta_T = primary_temp_out_hr - primary_temp_in
        secondary_delta_T = primary_delta_T * (C_sup / C_exh_safe)
        secondary_temp_out_hr = secondary_temp_in - secondary_delta_T

        # Select final outputs: heat recovery values if conditions met, else pass-through
        primary_temp_out = torch.where(
            do_heat_recovery, primary_temp_out_hr, primary_temp_in
        )
        secondary_temp_out = torch.where(
            do_heat_recovery, secondary_temp_out_hr, secondary_temp_in
        )
        return x, {
            "primaryTemperatureOut": primary_temp_out,
            "secondaryTemperatureOut": secondary_temp_out,
        }
