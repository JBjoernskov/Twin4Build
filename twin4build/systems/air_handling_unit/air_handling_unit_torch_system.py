"""
Air handling unit composed of damper, heat recovery, and coil submodels.
"""

# Standard library imports
import datetime

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)
from twin4build.systems.coil.coil_torch_system import CoilTorchSystem
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)
from twin4build.systems.fan.fan_torch_system import FanTorchSystem
from twin4build.translator.translator import Exact, Node, Optional_, Predicate, MultiPathRule, SignaturePattern


class AirHandlingUnitTorchSystem(core.System, nn.Module):
    r"""
    Air handling unit (AHU) that composes a damper, heat recovery, and coil.

    The AHU orchestrates three subcomponents:
      - Damper: converts position to supply air flow rate
      - Air-to-air heat recovery: preheats/precools outdoor air using return air
      - Coil: trims the supply air temperature to the setpoint and reports power
      - Fans: add temperature rise and electrical power on supply/return streams

    External interface
    ------------------
    Inputs:
      - supplyDamperPosition: Supply damper openings (vector 0-1)
      - exhaustDamperPosition: Exhaust damper openings (vector 0-1)
      - exhaustTemperature: Exhaust air temperatures matching exhaust branches (vector) [°C]
      - supplyAirTemperatureSetpoint: Desired supply air temperature [°C]
      - outdoorAirTemperature: Outdoor air temperature [°C]

    Outputs:
      - supplyAirFlowRate: Total supply air mass flow rate [kg/s]
      - supplyAirTemperature: Supply air temperature leaving the supply fan [°C]
      - exhaustAirTemperatureOut: Exhaust temperature leaving heat recovery [°C]
      - heatingPower: Coil heating power [W]
      - coolingPower: Coil cooling power [W]
      - supplyFanPower: Supply fan electrical power [W]
      - exhaustFanPower: Exhaust/return fan electrical power [W]

    Notes
    -----
    - The return flow defaults to the supply flow when zero/absent so that the
      heat recovery can still operate in simple configurations.
    """

    def __init__(
        self,
        damper_kwargs: dict | None = None,
        coil_kwargs: dict | None = None,
        heat_recovery_kwargs: dict | None = None,
        junction_kwargs: dict | None = None,
        supply_fan_kwargs: dict | None = None,
        exhaust_fan_kwargs: dict | None = None,
        n_branches: int | None = None,
        **kwargs,
    ):
        if damper_kwargs is None:
            damper_kwargs = {}
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
        if "id" not in coil_kwargs:
            coil_kwargs["id"] = f"{ahu_id}_coil"
        if "id" not in heat_recovery_kwargs:
            heat_recovery_kwargs["id"] = f"{ahu_id}_heat_recovery"
        if "id" not in junction_kwargs:
            junction_kwargs["id"] = f"{ahu_id}_return_junction"
        if "id" not in supply_fan_kwargs:
            supply_fan_kwargs["id"] = f"{ahu_id}_supply_fan"
        if "id" not in exhaust_fan_kwargs:
            exhaust_fan_kwargs["id"] = f"{ahu_id}_exhaust_fan"

        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Per-branch dampers (supply and exhaust)
        self.n_branches = n_branches if n_branches is not None else 1
        self.supply_dampers = nn.ModuleList(
            [
                DamperTorchSystem(**{**damper_kwargs, "id": f"{ahu_id}_supply_damper_{i}"})
                for i in range(self.n_branches)
            ]
        )
        self.exhaust_dampers = nn.ModuleList(
            [
                DamperTorchSystem(
                    **{**damper_kwargs, "id": f"{ahu_id}_exhaust_damper_{i}"}
                )
                for i in range(self.n_branches)
            ]
        )
        for i, supply_damper in enumerate(self.supply_dampers):
            setattr(self, f"supply_damper_{i}", supply_damper)
        for i, exhaust_damper in enumerate(self.exhaust_dampers):
            setattr(self, f"exhaust_damper_{i}", exhaust_damper)

        self.coil = CoilTorchSystem(**coil_kwargs)
        self.heat_recovery = AirToAirHeatRecoverySystem(**heat_recovery_kwargs)
        self.return_junction = ReturnFlowJunctionSystem(**junction_kwargs)
        # Manually set the number of junction inputs to match branches
        self.return_junction.n_input_ports = self.n_branches
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
            "supplyAirFlowRate": tps.Scalar(),
            "supplyAirTemperature": tps.Scalar(),
            "exhaustAirTemperatureOut": tps.Scalar(),
            "heatingPower": tps.Scalar(),
            "coolingPower": tps.Scalar(),
            "supplyFanPower": tps.Scalar(),
            "exhaustFanPower": tps.Scalar(),
        }

        damper_params = (
            [f"supply_damper_{i}.{p}" for i in range(self.n_branches) for p in self.supply_dampers[i]._config["parameters"]]
            + [f"exhaust_damper_{i}.{p}" for i in range(self.n_branches) for p in self.exhaust_dampers[i]._config["parameters"]]
        )
        coil_params = [f"coil.{p}" for p in self.coil._config["parameters"]]
        hr_params = [
            f"heat_recovery.{p}" for p in self.heat_recovery._config["parameters"]
        ]
        junction_params = [
            f"return_junction.{p}" for p in self.return_junction._config["parameters"]
        ]
        fan_params = [f"supply_fan.{p}" for p in self.supply_fan._config["parameters"]] + [
            f"exhaust_fan.{p}" for p in self.exhaust_fan._config["parameters"]
        ]
        self._config = {
            "parameters": damper_params + coil_params + hr_params + junction_params + fan_params
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
        for name, input_port in self.input.items():
            if isinstance(input_port, tps.Vector):
                input_port.initialize(
                    n_timesteps=max_timesteps,
                    batch_size=batch_size,
                    size=self.n_branches,
                )
            else:
                input_port.initialize(n_timesteps=max_timesteps, batch_size=batch_size)
        for output_port in self.output.values():
            output_port.initialize(n_timesteps=max_timesteps, batch_size=batch_size)

        for damper in list(self.supply_dampers) + list(self.exhaust_dampers):
            damper.initialize(start_time, end_time, step_size)
        self.coil.initialize(start_time, end_time, step_size)
        self.heat_recovery.initialize(start_time, end_time, step_size)
        self.return_junction.initialize(start_time, end_time, step_size)
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
        """Perform one simulation step for the AHU."""
        tol = 1e-5

        # 1) Supply dampers: per-branch position -> flow, then total
        supply_pos_vec = self.input["supplyDamperPosition"].get()
        supply_flows = []
        for i, damper in enumerate(self.supply_dampers):
            damper.input["damperPosition"].set(supply_pos_vec[..., i], step_index)
            damper.do_step(second_time, date_time, step_size, step_index)
            supply_flows.append(damper.output["airFlowRate"].get())
        supply_flow_vec = torch.stack(supply_flows, dim=-1)
        supply_flow = supply_flow_vec.sum(dim=-1)

        # 2) Exhaust dampers: per-branch position -> flow
        exhaust_pos_vec = self.input["exhaustDamperPosition"].get()
        exhaust_flows = []
        for i, damper in enumerate(self.exhaust_dampers):
            damper.input["damperPosition"].set(exhaust_pos_vec[..., i], step_index)
            damper.do_step(second_time, date_time, step_size, step_index)
            exhaust_flows.append(damper.output["airFlowRate"].get())
        exhaust_flow_vec = torch.stack(exhaust_flows, dim=-1)

        # 3) Combine exhaust flows and temperatures (junction-like)
        exhaust_temp_vec = self.input["exhaustTemperature"].get()
        self.return_junction.input["airFlowRateIn"].set(exhaust_flow_vec, step_index)
        self.return_junction.input["airTemperatureIn"].set(exhaust_temp_vec, step_index)
        self.return_junction.do_step(second_time, date_time, step_size, step_index)
        secondary_flow = self.return_junction.output["airFlowRateOut"].get()
        return_temp = self.return_junction.output["airTemperatureOut"].get()

        # 3b) Exhaust fan (on return stream before heat recovery) using mass flow
        self.exhaust_fan.input["airFlowRate"].set(secondary_flow, step_index)
        self.exhaust_fan.input["inletAirTemperature"].set(return_temp, step_index)
        self.exhaust_fan.do_step(second_time, date_time, step_size, step_index)
        return_temp_fan = self.exhaust_fan.output["outletAirTemperature"].get()
        exhaust_fan_power = self.exhaust_fan.output["Power"].get()

        # 4) Heat recovery
        self.heat_recovery.input["primaryAirFlowRate"].set(supply_flow, step_index)
        self.heat_recovery.input["secondaryAirFlowRate"].set(
            secondary_flow, step_index
        )
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

        # 5) Coil: trim to setpoint & report power
        self.coil.input["inletAirTemperature"].set(precoil_temp, step_index)
        self.coil.input["outletAirTemperatureSetpoint"].set(
            self.input["supplyAirTemperatureSetpoint"].get(), step_index
        )
        self.coil.input["airFlowRate"].set(supply_flow, step_index)
        self.coil.do_step(second_time, date_time, step_size, step_index)

        # 6) Supply fan after coil to add temperature rise and power (mass flow)
        self.supply_fan.input["airFlowRate"].set(supply_flow, step_index)
        self.supply_fan.input["inletAirTemperature"].set(
            self.coil.output["outletAirTemperature"].get(), step_index
        )
        self.supply_fan.do_step(second_time, date_time, step_size, step_index)
        supply_temp_out = self.supply_fan.output["outletAirTemperature"].get()
        supply_fan_power = self.supply_fan.output["Power"].get()

        # 7) Publish AHU outputs (totals)
        self.output["supplyAirFlowRate"].set(supply_flow, step_index)
        self.output["supplyAirTemperature"].set(supply_temp_out, step_index)
        self.output["exhaustAirTemperatureOut"].set(exhaust_temp_out, step_index)
        self.output["heatingPower"].set(
            self.coil.output["heatingPower"].get(), step_index
        )
        self.output["coolingPower"].set(
            self.coil.output["coolingPower"].get(), step_index
        )
        self.output["supplyFanPower"].set(supply_fan_power, step_index)
        self.output["exhaustFanPower"].set(exhaust_fan_power, step_index)


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
        )
    )
    weather_station = Node(cls=core.namespace.BRICK.Weather_Station)  # outdoor temperature sensor
    solar_radiance_sensor = Node(cls=core.namespace.BRICK.Solar_Radiance_Sensor)
    outside_air_temperature_sensor = Node(cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor)

    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))


    sp.add_triple(
        MultiPathRule(subject=ahu, object=spaces, predicate=feeds)
    )

    # sp.add_triple(
    #     Exact(subject=weather_station, object=solar_radiance_sensor, predicate=core.namespace.BRICK.hasPoint)
    # )
    # sp.add_triple(
    #     Exact(subject=weather_station, object=outside_air_temperature_sensor, predicate=core.namespace.BRICK.hasPoint)
    # )

    # sp.add_connection(weather_station, "measuredValue", "outdoorAirTemperature")
    sp.add_connection(spaces, "indoorTemperature", "exhaustTemperature", input_port_index=spaces)

    sp.add_modeled_node(ahu)

    return sp


AirHandlingUnitTorchSystem.add_signature_pattern(brick_signature_pattern())
