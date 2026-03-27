"""
Air handling unit composed of damper, heat recovery, and coil submodels.

This module provides a vectorized AHU implementation that uses vectorized
DamperTorchSystem objects (one for supply, one for exhaust) rather than
separate damper components for each branch.
"""

# Standard library imports
import datetime

# Third party imports
import torch.nn as nn  # noqa: F401 - torch needed for tensor ops

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)
from twin4build.systems.coil.coil_torch_system import CoilTorchSystem
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.fan.fan_torch_system import FanTorchSystem
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)
from twin4build.translator.translator import (
    MultiPathRule,
    Node,
    Predicate,
    SignaturePattern,
)


class AirHandlingUnitTorchSystem(core.System, nn.Module):
    r"""
    Air handling unit (AHU) with vectorized damper components.

    The AHU orchestrates subcomponents using vectorized operations:
      - Dampers: Two DamperTorchSystem objects (supply and exhaust), each vectorized
        across n_branches with parameters (a, nominalAirFlowRate) per branch
      - Air-to-air heat recovery: preheats/precools outdoor air using return air
      - Coil: trims the supply air temperature to the setpoint and reports power
      - Fans: add temperature rise and electrical power on supply/return streams

    External interface
    ------------------
    Inputs:
      - supplyDamperPosition: Supply damper openings (vector 0-1) [n_branches]
      - exhaustDamperPosition: Exhaust damper openings (vector 0-1) [n_branches]
      - exhaustTemperature: Exhaust air temperatures per branch (vector) [°C] [n_branches]
      - supplyAirTemperatureSetpoint: Desired supply air temperature [°C]
      - outdoorAirTemperature: Outdoor air temperature [°C]

    Outputs:
      - supplyAirFlowRate: Supply air mass flow rate per branch [kg/s] [n_branches]
      - supplyAirTemperature: Supply air temperature leaving the supply fan [°C]
      - exhaustAirFlowRate: Exhaust air mass flow rate per branch [kg/s] [n_branches]
      - exhaustAirTemperatureOut: Exhaust temperature leaving heat recovery [°C]
      - heatingPower: Coil heating power [W]
      - coolingPower: Coil cooling power [W]
      - supplyFanPower: Supply fan electrical power [W]
      - exhaustFanPower: Exhaust/return fan electrical power [W]

    Notes
    -----
    - Uses vectorized DamperTorchSystem objects: each damper has n_branches parallel
      elements with individual parameters
    - The return flow defaults to the supply flow when zero/absent so that the
      heat recovery can still operate in simple configurations.
    """

    def __init__(
        self,
        supply_damper_kwargs: dict | None = None,
        exhaust_damper_kwargs: dict | None = None,
        coil_kwargs: dict | None = None,
        heat_recovery_kwargs: dict | None = None,
        junction_kwargs: dict | None = None,
        supply_fan_kwargs: dict | None = None,
        exhaust_fan_kwargs: dict | None = None,
        n_branches: int | None = None,
        **kwargs,
    ):
        """
        Initialize the vectorized AHU.

        Args:
            supply_damper_kwargs: Keyword arguments for supply DamperTorchSystem.
                Can include 'a' and 'nominalAirFlowRate' as scalars (broadcast to
                all branches) or lists/tensors per branch.
            exhaust_damper_kwargs: Keyword arguments for exhaust DamperTorchSystem.
            coil_kwargs: Keyword arguments for CoilTorchSystem.
            heat_recovery_kwargs: Keyword arguments for AirToAirHeatRecoverySystem.
            junction_kwargs: Keyword arguments for ReturnFlowJunctionSystem.
            supply_fan_kwargs: Keyword arguments for FanTorchSystem (supply).
            exhaust_fan_kwargs: Keyword arguments for FanTorchSystem (exhaust).
            n_branches: Number of branches/zones served by the AHU.
            **kwargs: Additional arguments passed to System base class.
        """
        if supply_damper_kwargs is None:
            supply_damper_kwargs = {}
        if exhaust_damper_kwargs is None:
            exhaust_damper_kwargs = {}
        if coil_kwargs is None:
            coil_kwargs = {}
        if heat_recovery_kwargs is None:
            heat_recovery_kwargs = {}
        if junction_kwargs is None:
            junction_kwargs = {}
        if supply_fan_kwargs is None:
            supply_fan_kwargs = {}
        if exhaust_fan_kwargs is None:
            exhaust_fan_kwargs = {}

        assert "id" in kwargs, "id is required for AirHandlingUnitTorchSystem"
        ahu_id = kwargs["id"]

        # Make sure each subcomponent has a unique id
        if "id" not in supply_damper_kwargs:
            supply_damper_kwargs["id"] = f"{ahu_id}_supply_damper"
        if "id" not in exhaust_damper_kwargs:
            exhaust_damper_kwargs["id"] = f"{ahu_id}_exhaust_damper"
        if "id" not in coil_kwargs:
            coil_kwargs["id"] = f"{ahu_id}_coil"
        if "id" not in heat_recovery_kwargs:
            heat_recovery_kwargs["id"] = f"{ahu_id}_heat_recovery"
        if "id" not in junction_kwargs:
            junction_kwargs["id"] = f"{ahu_id}_return_junction"

        # Create separate kwargs for supply junction with unique id
        supply_junction_kwargs = junction_kwargs.copy()
        supply_junction_kwargs["id"] = f"{ahu_id}_supply_junction"
        if "id" not in supply_fan_kwargs:
            supply_fan_kwargs["id"] = f"{ahu_id}_supply_fan"
        if "id" not in exhaust_fan_kwargs:
            exhaust_fan_kwargs["id"] = f"{ahu_id}_exhaust_fan"

        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Number of branches
        self.n_branches = n_branches if n_branches is not None else 1

        # Vectorized damper components (one supply, one exhaust)
        # n_branches is passed to initialize() at runtime
        self.supply_damper = DamperTorchSystem(**supply_damper_kwargs)
        self.exhaust_damper = DamperTorchSystem(**exhaust_damper_kwargs)

        # Junction components for combining flows
        self.supply_junction = SupplyFlowJunctionSystem(**supply_junction_kwargs)
        self.return_junction = ReturnFlowJunctionSystem(**junction_kwargs)
        # Set number of junction inputs to match branches
        self.supply_junction.n_input_ports = self.n_branches
        self.return_junction.n_input_ports = self.n_branches

        # Other subcomponents
        self.coil = CoilTorchSystem(**coil_kwargs)
        self.heat_recovery = AirToAirHeatRecoverySystem(**heat_recovery_kwargs)
        self.supply_fan = FanTorchSystem(**supply_fan_kwargs)
        self.exhaust_fan = FanTorchSystem(**exhaust_fan_kwargs)

        self._input = {
            "supplyDamperPosition": tps.Vector(),
            "exhaustDamperPosition": tps.Vector(),
            "exhaustTemperature": tps.Vector(),
            "supplyAirTemperatureSetpoint": tps.Scalar(),
            "outdoorAirTemperature": tps.Scalar(),
        }
        self._output = {
            "supplyAirFlowRate": tps.Vector(),  # Vector: one per branch
            "supplyAirTemperature": tps.Scalar(),
            "exhaustAirFlowRate": tps.Vector(),  # Vector: one per branch
            "exhaustAirTemperatureOut": tps.Scalar(),
            "heatingPower": tps.Scalar(),
            "coolingPower": tps.Scalar(),
            "supplyFanPower": tps.Scalar(),
            "exhaustFanPower": tps.Scalar(),
        }

        # Parameter configuration for calibration
        damper_params = [
            f"supply_damper.{p}" for p in self.supply_damper._config["parameters"]
        ] + [f"exhaust_damper.{p}" for p in self.exhaust_damper._config["parameters"]]
        coil_params = [f"coil.{p}" for p in self.coil._config["parameters"]]
        hr_params = [
            f"heat_recovery.{p}" for p in self.heat_recovery._config["parameters"]
        ]
        junction_params = [
            f"supply_junction.{p}" for p in self.supply_junction._config["parameters"]
        ] + [f"return_junction.{p}" for p in self.return_junction._config["parameters"]]
        fan_params = [
            f"supply_fan.{p}" for p in self.supply_fan._config["parameters"]
        ] + [f"exhaust_fan.{p}" for p in self.exhaust_fan._config["parameters"]]
        self._config = {
            "parameters": damper_params
            + coil_params
            + hr_params
            + junction_params
            + fan_params
        }

        self.INITIALIZED = False

    @property
    def input(self) -> dict:
        """Get AHU input ports."""
        return self._input

    @property
    def output(self) -> dict:
        """Get AHU output ports."""
        return self._output

    @property
    def config(self):
        """Get AHU configuration parameters."""
        return self._config

    def initialize(
        self,
        start_time: list[datetime.datetime],
        end_time: list[datetime.datetime],
        step_size: int,
    ) -> None:
        """Initialize AHU and subcomponents."""
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        # Initialize input ports - derive n_v from connection points for Vector inputs
        for name, input_port in self.input.items():
            if isinstance(input_port, tps.Vector):
                # Derive n_v from connection point indices, fall back to n_branches
                n_v = self.get_n_v_from_connections(name) or self.n_branches
                input_port.initialize(
                    n_t=max_timesteps,
                    n_s=batch_size,
                    n_v=n_v,
                )
            else:
                input_port.initialize(n_t=max_timesteps, n_s=batch_size)

        # Initialize output ports - use same n_v as corresponding inputs
        # Supply outputs use supply input n_v, exhaust outputs use exhaust input n_v
        n_v_supply = self.input["supplyDamperPosition"].n_v
        n_v_exhaust = self.input["exhaustDamperPosition"].n_v
        for name, output_port in self.output.items():
            if isinstance(output_port, tps.Vector):
                # Determine n_v based on which side this output belongs to
                if "supply" in name.lower():
                    n_v = n_v_supply
                else:
                    n_v = n_v_exhaust
                output_port.initialize(
                    n_t=max_timesteps,
                    n_s=batch_size,
                    n_v=n_v,
                )
            else:
                output_port.initialize(n_t=max_timesteps, n_s=batch_size)

        # Set n_c for damper subcomponents: n_c_ahu * n_v (flattened from Vector shape)
        # Supply and exhaust can have different n_v values
        self.supply_damper.n_c = self.n_c * n_v_supply
        self.exhaust_damper.n_c = self.n_c * n_v_exhaust
        self.supply_damper.initialize(start_time, end_time, step_size)
        self.exhaust_damper.initialize(start_time, end_time, step_size)

        # Initialize junction subcomponents - set n_input_ports based on n_v
        self.supply_junction.n_input_ports = n_v_supply
        self.return_junction.n_input_ports = n_v_exhaust
        self.supply_junction.initialize(start_time, end_time, step_size)
        self.return_junction.initialize(start_time, end_time, step_size)

        # Initialize other subcomponents
        self.coil.initialize(start_time, end_time, step_size)
        self.heat_recovery.initialize(start_time, end_time, step_size)
        self.supply_fan.initialize(start_time, end_time, step_size)
        self.exhaust_fan.initialize(start_time, end_time, step_size)
        self.INITIALIZED = True

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one simulation step for the AHU using vectorized damper objects.

        All damper calculations are performed in parallel across branches via
        the vectorized DamperTorchSystem objects.
        """
        # 1) Supply damper: vectorized position -> flow calculation
        # Vector input shape: (n_s, n_c, n_v) -> reshape to (n_s, n_c*n_v) for damper n_c
        supply_pos_vec = self.input["supplyDamperPosition"].get()
        supply_pos_flat = supply_pos_vec.reshape(
            supply_pos_vec.shape[0], -1
        )  # (n_s, n_c*n_v)
        self.supply_damper.input["damperPosition"].set(supply_pos_flat, step_index)
        self.supply_damper.do_step(second_time, date_time, step_size, step_index)
        supply_flow_flat = self.supply_damper.output[
            "airFlowRate"
        ].get()  # (n_s, n_c*n_v)
        # Reshape back to (n_s, n_c, n_v) for Vector outputs
        supply_flow_vec = supply_flow_flat.reshape(supply_pos_vec.shape)

        # 2) Supply junction: sum branch flows
        self.supply_junction.input["airFlowRateOut"].set(supply_flow_vec, step_index)
        self.supply_junction.do_step(second_time, date_time, step_size, step_index)
        supply_flow_total = self.supply_junction.output["airFlowRateIn"].get()

        # 3) Exhaust damper: vectorized position -> flow calculation
        exhaust_pos_vec = self.input["exhaustDamperPosition"].get()
        exhaust_pos_flat = exhaust_pos_vec.reshape(
            exhaust_pos_vec.shape[0], -1
        )  # (n_s, n_c*n_v)
        self.exhaust_damper.input["damperPosition"].set(exhaust_pos_flat, step_index)
        self.exhaust_damper.do_step(second_time, date_time, step_size, step_index)
        exhaust_flow_flat = self.exhaust_damper.output[
            "airFlowRate"
        ].get()  # (n_s, n_c*n_v)
        # Reshape back to (n_s, n_c, n_v) for Vector outputs
        exhaust_flow_vec = exhaust_flow_flat.reshape(exhaust_pos_vec.shape)

        # 4) Return junction: combine exhaust flows and temperatures
        exhaust_temp_vec = self.input["exhaustTemperature"].get()
        self.return_junction.input["airFlowRateIn"].set(exhaust_flow_vec, step_index)
        self.return_junction.input["airTemperatureIn"].set(exhaust_temp_vec, step_index)
        self.return_junction.do_step(second_time, date_time, step_size, step_index)
        secondary_flow = self.return_junction.output["airFlowRateOut"].get()
        return_temp = self.return_junction.output["airTemperatureOut"].get()

        # 5) Exhaust fan (on return stream before heat recovery)
        self.exhaust_fan.input["airFlowRate"].set(secondary_flow, step_index)
        self.exhaust_fan.input["inletAirTemperature"].set(return_temp, step_index)
        self.exhaust_fan.do_step(second_time, date_time, step_size, step_index)
        return_temp_fan = self.exhaust_fan.output["outletAirTemperature"].get()
        exhaust_fan_power = self.exhaust_fan.output["Power"].get()

        # 6) Heat recovery
        self.heat_recovery.input["primaryAirFlowRate"].set(
            supply_flow_total, step_index
        )
        self.heat_recovery.input["secondaryAirFlowRate"].set(secondary_flow, step_index)
        self.heat_recovery.input["primaryTemperatureIn"].set(
            self.input["outdoorAirTemperature"].get(), step_index
        )
        self.heat_recovery.input["secondaryTemperatureIn"].set(
            return_temp_fan, step_index
        )
        self.heat_recovery.input["primaryTemperatureOutSetpoint"].set(
            self.input["supplyAirTemperatureSetpoint"].get(), step_index
        )
        self.heat_recovery.do_step(second_time, date_time, step_size, step_index)
        precoil_temp = self.heat_recovery.output["primaryTemperatureOut"].get()
        exhaust_temp_out = self.heat_recovery.output["secondaryTemperatureOut"].get()

        # 7) Coil: trim to setpoint & report power
        self.coil.input["inletAirTemperature"].set(precoil_temp, step_index)
        self.coil.input["outletAirTemperatureSetpoint"].set(
            self.input["supplyAirTemperatureSetpoint"].get(), step_index
        )
        self.coil.input["airFlowRate"].set(supply_flow_total, step_index)
        self.coil.do_step(second_time, date_time, step_size, step_index)

        # 8) Supply fan after coil to add temperature rise and power
        self.supply_fan.input["airFlowRate"].set(supply_flow_total, step_index)
        self.supply_fan.input["inletAirTemperature"].set(
            self.coil.output["outletAirTemperature"].get(), step_index
        )
        self.supply_fan.do_step(second_time, date_time, step_size, step_index)
        supply_temp_out = self.supply_fan.output["outletAirTemperature"].get()
        supply_fan_power = self.supply_fan.output["Power"].get()

        # 9) Publish AHU outputs
        # Vector outputs (per branch)
        self.output["supplyAirFlowRate"]._set(supply_flow_vec, i_t=step_index)
        self.output["exhaustAirFlowRate"]._set(exhaust_flow_vec, i_t=step_index)
        # Scalar outputs
        self.output["supplyAirTemperature"]._set(supply_temp_out, i_t=step_index)
        self.output["exhaustAirTemperatureOut"]._set(exhaust_temp_out, i_t=step_index)
        self.output["heatingPower"]._set(
            self.coil.output["heatingPower"].get(), i_t=step_index
        )
        self.output["coolingPower"]._set(
            self.coil.output["coolingPower"].get(), i_t=step_index
        )
        self.output["supplyFanPower"]._set(supply_fan_power, i_t=step_index)
        self.output["exhaustFanPower"]._set(exhaust_fan_power, i_t=step_index)


def brick_signature_pattern():
    """
    Signature pattern for an AHU composed of damper, coil, and heat recovery.
    """
    sp = SignaturePattern(id="air_handling_unit_signature_pattern_brick")

    ahu = Node(cls=core.namespace.BRICK.AHU)
    spaces = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.BRICK.HVAC_Zone,
        )
    )

    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    sp.add_triple(MultiPathRule(subject=ahu, object=spaces, predicate=feeds))

    sp.add_connection(
        spaces, "indoorTemperature", "exhaustTemperature", input_port_index=spaces
    )

    sp.add_modeled_node(ahu)

    return sp


AirHandlingUnitTorchSystem.add_signature_pattern(brick_signature_pattern())
