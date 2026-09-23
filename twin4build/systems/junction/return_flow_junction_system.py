# Standard library imports
import datetime
from typing import List, Optional, Union

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    OptionalRule,
    SignaturePattern,
    PathRule,
)


class ReturnFlowJunctionSystem(core.System):
    r"""
    A return flow junction system model for combining air flow rates and temperatures.

    This model represents a junction that combines multiple return air flows and their temperatures
    into a single output flow and temperature. The total output flow is the sum of all input flows
    (plus an optional bias), and the output temperature is the flow-weighted average of the input temperatures.

    Args:
        airFlowRateBias: Bias to be added to the total flow rate [kg/s].

    Mathematical Formulation
    ========================

    The total output flow rate is:

        .. math::

            \dot{m}_{out} = \sum_{i=1}^{n} \dot{m}_i + b

    where:
       - :math:`\dot{m}_{out}` is the total output flow rate [kg/s]
       - :math:`\dot{m}_i` are the individual input flow rates [kg/s]
       - :math:`n` is the number of input flows
       - :math:`b` is the optional flow rate bias [kg/s]

    The output temperature is the flow-weighted average:

        .. math::

            T_{out} = \frac{\sum_{i=1}^{n} T_i \dot{m}_i}{\dot{m}_{out}}

    where:
       - :math:`T_{out}` is the output temperature [°C]
       - :math:`T_i` are the input temperatures [°C]
       - :math:`\dot{m}_i` are the input flow rates [kg/s]
       - :math:`\dot{m}_{out}` is the total output flow rate [kg/s]
    """

    def __init__(self, airFlowRateBias=None, branch_temperature_slots=None, **kwargs):
        """
        Args:
            airFlowRateBias: Bias added to the total flow rate [kg/s].
            branch_temperature_slots: For flow slot ``b`` (a branch), the
                slot of ``airTemperatureIn`` that carries its temperature.
                A room with several terminals publishes one temperature,
                on one slot, for all of its branches (a scalar output
                cannot be wired to several slots of one port), so the
                temperature port is one slot per room and this map joins
                the two.  ``None``: the slots are aligned one to one.
        """
        super().__init__(**kwargs)
        if airFlowRateBias is not None:
            self.airFlowRateBias = airFlowRateBias
        else:
            self.airFlowRateBias = 0
        self.n_input_ports = 2
        self._manual_setup_n_input_ports = False
        self.branch_temperature_slots = (
            None if branch_temperature_slots is None else [int(i) for i in branch_temperature_slots]
        )


        self.input = {
            "airFlowRateIn": tps.Vector(),
            "airTemperatureIn": tps.Vector(),
        }
        self.output = {
            "airFlowRateOut": tps.Scalar(),
            "airTemperatureOut": tps.Scalar(),
        }
        self._config = {"parameters": ["airFlowRateBias", "branch_temperature_slots"]}


    @property
    def n_input_ports(self):
        return self._n_input_ports

    @n_input_ports.setter
    def n_input_ports(self, n_input_ports: int):
        self._manual_setup_n_input_ports = True
        self._n_input_ports = n_input_ports

    @property
    def config(self):
        return self._config

    def setup_variable_inputs(self):

        # Check that the number of airFlowRateIn and airTemperatureIn are the same
        connection_point_airFlowRateIn = [
            cp for cp in self.connects_at if cp.input_port == "airFlowRateIn"
        ]
        connection_point_airTemperatureIn = [
            cp for cp in self.connects_at if cp.input_port == "airTemperatureIn"
        ]
        if len(connection_point_airFlowRateIn) != len(
            connection_point_airTemperatureIn
        ):
            raise ValueError(
                "The number of airFlowRateIn and airTemperatureIn must be the same"
            )

        if self._manual_setup_n_input_ports == False:
            # Assert that the number of input ports is at least 1
            connection_point = [
                cp for cp in self.connects_at if cp.input_port == "airFlowRateIn"
            ]
            if len(connection_point) == 0:
                raise ValueError("No input port found for airFlowRateIn")
            # The width is the highest wired slot plus one (one connection
            # from a batched meta carries many slots); without slot indices,
            # one slot per connection as before.
            n_input_ports = self.get_n_v_from_connections("airFlowRateIn") or len(
                connection_point[0].connects_system_through
            )
            self.n_input_ports = n_input_ports

    def initialize(
        self,
        start_time: Union[List[datetime.datetime], datetime.datetime],
        end_time: Union[List[datetime.datetime], datetime.datetime],
        step_size: Union[List[int], int],
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        self.setup_variable_inputs()
        # The temperature port may be narrower than the flow port (one slot
        # per room against one per branch, see ``branch_temperature_slots``).
        n_temperature = self.n_input_ports
        if self.branch_temperature_slots is not None:
            n_temperature = self.get_n_v_from_connections("airTemperatureIn") or (
                max(self.branch_temperature_slots) + 1
            )
        self.input["airFlowRateIn"].initialize(n_t=max_timesteps, n_s=batch_size, n_v=self.n_input_ports)
        self.input["airTemperatureIn"].initialize(n_t=max_timesteps, n_s=batch_size, n_v=n_temperature)
        self._temperature_index = (
            None
            if self.branch_temperature_slots is None
            else torch.tensor(self.branch_temperature_slots, dtype=torch.long)
        )

        for output in self.output.values():
            output.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
            )

    PARAM_NAMES = ()  # airFlowRateBias is a plain number (structural constant)

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step flow mixing (functorch-safe, stateless).

        Total flow is the sum of the input flows plus the bias; the output
        temperature is the flow-weighted average (20 °C fallback when there
        is no flow).  ``airFlowRateBias`` is a plain (non-estimable) number,
        read from ``self`` as a structural constant.
        """
        # Sum over last dimension (input flows dimension) to preserve batch dimension
        temperature = inputs["airTemperatureIn"]
        index = getattr(self, "_temperature_index", None)
        if index is not None:
            temperature = temperature[..., index.to(temperature.device)]
        m_dot_in = inputs["airFlowRateIn"].sum(dim=-1)
        Q_dot_in = (temperature * inputs["airFlowRateIn"]).sum(dim=-1)


        tol = 1e-5
        has_flow = m_dot_in > tol

        # Calculate outputs for flow case
        flow_rate_out = m_dot_in + self.airFlowRateBias
        # Avoid division by zero by using flow_rate_out (which includes bias)
        temp_out_flow = Q_dot_in / torch.clamp(flow_rate_out, min=tol)

        # Select between flow and no-flow cases
        air_flow_rate_out = torch.where(
            has_flow, flow_rate_out, torch.zeros_like(m_dot_in)
        )
        air_temp_out = torch.where(
            has_flow, temp_out_flow, torch.full_like(m_dot_in, 20.0)
        )
        return x, {
            "airFlowRateOut": air_flow_rate_out,
            "airTemperatureOut": air_temp_out,
        }

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        inputs = {
            "airFlowRateIn": self.input["airFlowRateIn"].get(),
            "airTemperatureIn": self.input["airTemperatureIn"].get(),
        }
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["airFlowRateOut"]._set(outs["airFlowRateOut"], i_t=step_index)
        self.output["airTemperatureOut"]._set(outs["airTemperatureOut"], i_t=step_index)


