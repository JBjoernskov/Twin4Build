"""
Air handling unit composed of damper, heat recovery, and coil submodels.

This module provides a vectorized AHU implementation that uses vectorized
DamperSystem objects (one for supply, one for exhaust) rather than
separate damper components for each branch.
"""

# Postpone evaluation of annotations (PEP 563 / PEP 649).
from __future__ import annotations

# Standard library imports
import datetime

# Third party imports
import torch
import torch.nn as nn  # noqa: F401 - torch needed for tensor ops

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.slots import slot_pairs
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)
from twin4build.systems.coil.coil_system import CoilSystem
from twin4build.systems.damper.damper_system import DamperSystem
from twin4build.systems.fan.fan_system import FanSystem
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)
from twin4build.translator.translator import (
    AnyPathRule,
    ModeledNode,
    Node,
    OptionalRule,
    Predicate,
    SetAnyPathRule,
    SetStepRule,
    SignaturePattern,
    StepRule,
)


class AirHandlingUnitSystem(core.System, nn.Module):
    r"""
    Air handling unit (AHU) with vectorized damper components.

    The AHU orchestrates subcomponents using vectorized operations:
      - Dampers: Two DamperSystem objects (supply and exhaust), each vectorized
        across n_branches with parameters (a, nominalAirFlowRate) per branch
      - Air-to-air heat recovery: preheats/precools outdoor air using return air
      - Coil: trims the supply air temperature to the setpoint and reports power
      - Fans: add temperature rise and electrical power on supply/return streams

    Args:
        supply_damper_kwargs: Keyword arguments for the supply DamperSystem.
            Can include 'a' and 'nominalAirFlowRate' as scalars (broadcast to
            all branches) or lists/tensors per branch.
        exhaust_damper_kwargs: Keyword arguments for the exhaust DamperSystem.
        coil_kwargs: Keyword arguments for CoilSystem.
        heat_recovery_kwargs: Keyword arguments for AirToAirHeatRecoverySystem.
        junction_kwargs: Keyword arguments for ReturnFlowJunctionSystem.
        supply_fan_kwargs: Keyword arguments for FanSystem (supply).
        exhaust_fan_kwargs: Keyword arguments for FanSystem (exhaust).
        n_branches: Number of branches/zones served by the AHU. Defaults to 1.
        **kwargs: Additional arguments passed to the System base class
            (must include 'id').

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
    - Uses vectorized DamperSystem objects: each damper has n_branches parallel
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
        exhaust_follows_supply: bool = False,
        exhaustFlowRatio: float = 1.0,
        exhaust_ratio_per_branch: bool = False,
        **kwargs,
    ):
        """
        Initialize the vectorized AHU.

        Args:
            supply_damper_kwargs: Keyword arguments for supply DamperSystem.
                Can include 'a' and 'nominalAirFlowRate' as scalars (broadcast to
                all branches) or lists/tensors per branch.
            exhaust_damper_kwargs: Keyword arguments for exhaust DamperSystem.
            coil_kwargs: Keyword arguments for CoilSystem.
            heat_recovery_kwargs: Keyword arguments for AirToAirHeatRecoverySystem.
            junction_kwargs: Keyword arguments for ReturnFlowJunctionSystem.
            supply_fan_kwargs: Keyword arguments for FanSystem (supply).
            exhaust_fan_kwargs: Keyword arguments for FanSystem (exhaust).
            n_branches: Number of branches/zones served by the AHU.
            exhaust_follows_supply: When True every branch's exhaust flow is
                ``exhaustFlowRatio`` times its supply flow and the exhaust
                damper model is bypassed.  Use it when the exhaust side has
                no per-branch measurement: with one exhaust meter per AHU
                the per-branch exhaust dampers are unidentifiable (only
                the total is), while the ratio is pinned by that meter.
            exhaustFlowRatio: Exhaust-to-supply flow ratio [-] used when
                ``exhaust_follows_supply`` is set (estimable).
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

        assert "id" in kwargs, "id is required for AirHandlingUnitSystem"
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
        self.supply_damper = DamperSystem(**supply_damper_kwargs)
        self.exhaust_damper = DamperSystem(**exhaust_damper_kwargs)

        # Junction components for combining flows
        self.supply_junction = SupplyFlowJunctionSystem(**supply_junction_kwargs)
        self.return_junction = ReturnFlowJunctionSystem(**junction_kwargs)
        # Set number of junction inputs to match branches
        self.supply_junction.n_input_ports = self.n_branches
        self.return_junction.n_input_ports = self.n_branches

        # Other subcomponents
        self.coil = CoilSystem(**coil_kwargs)
        self.heat_recovery = AirToAirHeatRecoverySystem(**heat_recovery_kwargs)
        self.supply_fan = FanSystem(**supply_fan_kwargs)
        self.exhaust_fan = FanSystem(**exhaust_fan_kwargs)
        self.exhaust_follows_supply = bool(exhaust_follows_supply)
        # One ratio per branch (zone-level imbalance is real: a BMS snapshot
        # showed 500 m3/h in / 250 out in one room with the floor balanced).
        self.exhaust_ratio_per_branch = bool(exhaust_ratio_per_branch)
        self.exhaustFlowRatio = tps.Parameter(
            torch.tensor(exhaustFlowRatio, dtype=tps.float_dtype()),
            requires_grad=False,
        )
        self.parameter = {"exhaustFlowRatio": {"lb": 0.3, "ub": 1.5}}

        self._input = {
            "supplyDamperPosition": tps.Vector(),
            "exhaustDamperPosition": tps.Vector(),
            "exhaustTemperature": tps.Vector(),
            "supplyAirTemperatureSetpoint": tps.Scalar(),
            "outdoorAirTemperature": tps.Scalar(),
            # Fan state, 0-1 (a speed command or status), optional: unwired
            # means the fans run.  A branch moves air only while its fan
            # runs -- a VAV damper left open after the fan stops delivers
            # nothing -- so the branch flows are gated by
            # ``_fan_gate`` (fully on above FAN_ON_SPEED).
            "supplyFanSpeed": tps.Scalar(1.0, optional=True),
            "exhaustFanSpeed": tps.Scalar(1.0, optional=True),
        }
        self._output = {
            "supplyAirFlowRate": tps.Vector(),  # Vector: one per branch
            "supplyAirTemperature": tps.Scalar(),
            # Heat-recovery outlet on the supply side, before the coil
            # (Brick ``Preheat_Supply_Air_Temperature_Sensor``).
            "preheatSupplyAirTemperature": tps.Scalar(),
            "exhaustAirFlowRate": tps.Vector(),  # Vector: one per branch
            # Totals over the branches: what the AHU's own supply / return
            # flow sensors measure.
            "totalSupplyAirFlowRate": tps.Scalar(),
            "totalExhaustAirFlowRate": tps.Scalar(),
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
            + ["exhaust_follows_supply", "exhaustFlowRatio", "exhaust_ratio_per_branch"]
        }
        self.PARAM_NAMES = tuple(
            f"{sub_name}.{param_name}"
            for sub_name in self._SUB_NAMES
            for param_name in getattr(getattr(self, sub_name), "PARAM_NAMES", ())
        ) + ("exhaustFlowRatio",)

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

        # Exhaust branches read their room's exhaust temperature through the
        # branch -> room map (None when the two are aligned one-to-one).
        self._branch_room_index = self._exhaust_branch_map(n_v_exhaust)

        # Set n_c for damper subcomponents: n_c_ahu * n_v (flattened from Vector shape)
        # Supply and exhaust can have different n_v values
        self.exhaustFlowRatio = self.exhaustFlowRatio.expand_to_n_c(
            self.n_branches if self.exhaust_ratio_per_branch else self.n_c
        )
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

    #: Fan speed (0-1) above which a fan is fully "on" for the branch flows.
    FAN_ON_SPEED = 0.1

    @classmethod
    def _fan_gate(cls, speed):
        """0 with the fan stopped, 1 once it runs (linear in between): the
        VAV branches are pressure-controlled, so the damper sets the flow
        while the fan runs, and nothing moves when it does not."""
        return torch.clamp(speed / cls.FAN_ON_SPEED, 0.0, 1.0)

    def _exhaust_branch_map(self, n_v_exhaust: int):
        """``LongTensor`` mapping each exhaust branch to the slot of
        ``exhaustTemperature`` that carries its room's temperature, or
        ``None`` when the two are aligned one-to-one (as many temperature
        slots as branches -- the hand-built and single-VAV cases).

        The map is read off the wiring: branch ``b`` belongs to the zone
        that consumes ``supplyAirFlowRate[b]``, and that zone's slot on
        ``exhaustTemperature`` is where it publishes ``indoorTemperature``.
        """
        n_v_temp = self.input["exhaustTemperature"].n_v
        if n_v_temp == n_v_exhaust or n_v_temp is None:
            return None
        # (zone component, zone instance) -> its slot on exhaustTemperature.
        # A batched zone meta publishes one temperature per instance, each
        # on its own slot, so the instance is part of the key.
        temp_slot_of = {}
        for cp in self.connects_at:
            if cp.input_port != "exhaustTemperature":
                continue
            for conn in cp.connects_system_through:
                for s_ic, _, _, in_v in slot_pairs(cp, conn):
                    instance = s_ic if isinstance(s_ic, int) else 0
                    temp_slot_of[(id(conn.connects_system), instance)] = int(
                        0 if in_v is None or isinstance(in_v, slice) else in_v
                    )
        index = [None] * n_v_exhaust
        for conn in self.connected_through:
            if conn.output_port != "supplyAirFlowRate":
                continue
            for cp in conn.connects_system_at:
                for _, r_ic, out_v, _ in slot_pairs(cp, conn):
                    instance = r_ic if isinstance(r_ic, int) else 0
                    slot = temp_slot_of.get((id(cp.connection_point_of), instance))
                    if slot is None or out_v is None or isinstance(out_v, slice):
                        continue
                    b = int(out_v)
                    if b < n_v_exhaust:
                        index[b] = slot
        if any(i is None for i in index):
            missing = [b for b, i in enumerate(index) if i is None]
            raise ValueError(
                f"|{self.__class__.__name__}|{self.id}|: exhaustTemperature has "
                f"{n_v_temp} slots for {n_v_exhaust} exhaust branches, and branches "
                f"{missing} cannot be mapped to a room (no zone consumes their "
                "supplyAirFlowRate and publishes its exhaust temperature)."
            )
        return torch.tensor(index, dtype=torch.long)

    def _per_branch_exhaust_temperature(self, exhaust_temperature):
        index = getattr(self, "_branch_room_index", None)
        if index is None:
            return exhaust_temperature
        return exhaust_temperature[..., index.to(exhaust_temperature.device)]

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
        the vectorized DamperSystem objects.
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
        supply_flow_vec = supply_flow_vec * self._fan_gate(
            self.input["supplyFanSpeed"].get()
        ).unsqueeze(-1)

        # 2) Supply junction: sum branch flows
        self.supply_junction.input["airFlowRateOut"].set(supply_flow_vec, step_index)
        self.supply_junction.do_step(second_time, date_time, step_size, step_index)
        supply_flow_total = self.supply_junction.output["airFlowRateIn"].get()

        # 3) Exhaust flows: the exhaust damper model (gated by the
        #    exhaust fan), or the supply flows scaled by the exhaust-to-
        #    supply ratio (exhaust_follows_supply; the supply flows already
        #    carry the supply fan gate)
        if self.exhaust_follows_supply:
            exhaust_flow_vec = self._scale_by_ratio(
                supply_flow_vec, self.exhaustFlowRatio.get(), per_branch=self.exhaust_ratio_per_branch
            )
        else:
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
            exhaust_flow_vec = exhaust_flow_vec * self._fan_gate(
                self.input["exhaustFanSpeed"].get()
            ).unsqueeze(-1)

        # 4) Return junction: combine exhaust flows and temperatures
        exhaust_temp_vec = self._per_branch_exhaust_temperature(
            self.input["exhaustTemperature"].get()
        )
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
        self.output["totalSupplyAirFlowRate"]._set(supply_flow_total, i_t=step_index)
        self.output["totalExhaustAirFlowRate"]._set(
            self.return_junction.output["airFlowRateOut"].get(), i_t=step_index
        )
        self.output["supplyAirTemperature"]._set(supply_temp_out, i_t=step_index)
        self.output["preheatSupplyAirTemperature"]._set(precoil_temp, i_t=step_index)
        self.output["exhaustAirTemperatureOut"]._set(exhaust_temp_out, i_t=step_index)
        self.output["heatingPower"]._set(
            self.coil.output["heatingPower"].get(), i_t=step_index
        )
        self.output["coolingPower"]._set(
            self.coil.output["coolingPower"].get(), i_t=step_index
        )
        self.output["supplyFanPower"]._set(supply_fan_power, i_t=step_index)
        self.output["exhaustFanPower"]._set(exhaust_fan_power, i_t=step_index)

    # -- composed-map support (mirrors BuildingSpaceSystem) -------------

    SUPPORTS_TRANSFORM_MODE = True
    PARAM_NAMES = ()  # all parameters live on the owned submodels (prefixed)

    _SUB_NAMES = (
        "supply_damper",
        "exhaust_damper",
        "supply_junction",
        "return_junction",
        "coil",
        "heat_recovery",
        "supply_fan",
        "exhaust_fan",
    )

    def get_estimable_parameters(self):
        """The submodels' estimable parameters: a damper's offset ``c`` only
        once untied (see ``DamperSystem.set_c``); the exhaust damper's not at
        all when the exhaust follows the supply, and the ratio only then."""
        out = []
        for entry in super().get_estimable_parameters():
            attr = str(entry[1])
            prefix, _, leaf = attr.rpartition(".")
            if leaf == "c" and getattr(getattr(self, prefix, None), "c_tied", False):
                continue
            if self.exhaust_follows_supply and attr.startswith("exhaust_damper."):
                continue
            if not self.exhaust_follows_supply and attr == "exhaustFlowRatio":
                continue
            out.append(entry)
        return out

    @staticmethod
    def _scale_by_ratio(flow_vec: torch.Tensor, ratio: torch.Tensor, per_branch: bool = False) -> torch.Tensor:
        """``ratio`` times the per-branch flows (``(n_s, n_c, n_v)`` in
        ``do_step``, ``(n_c, n_v)`` in ``forward``).  ``ratio`` is scalar,
        ``(n_c,)`` (one per parallel component) or, with ``per_branch``,
        ``(n_v,)`` (one per branch, on the last axis)."""
        ratio = torch.as_tensor(ratio, dtype=flow_vec.dtype, device=flow_vec.device)
        if ratio.dim() == 0 or ratio.numel() == 1:
            return flow_vec * ratio.reshape(())
        shape = [1] * flow_vec.dim()
        shape[-1 if per_branch else -2] = ratio.numel()
        return flow_vec * ratio.reshape(shape)

    def _inactive_parameters(self):
        """Parameters the exhaust mode leaves without effect: the exhaust
        damper's when the exhaust follows the supply, the ratio otherwise."""
        return () if self.exhaust_follows_supply else ("exhaustFlowRatio",)

    @staticmethod
    def _resolve_sub_params(sub, prefix, params):
        """Full physical-parameter dict for a submodel: estimated values from
        ``params`` (keyed ``"<prefix>.<name>"``), the rest from the submodel's
        own ``tps.Parameter`` defaults."""
        out = {}
        for name in sub.PARAM_NAMES:
            key = f"{prefix}.{name}"
            out[name] = params[key] if key in params else getattr(sub, name).get()
        return out

    def forward(self, x, inputs, params, sample_time, transform_mode=None):
        """Pure one-step of the composite AHU (functorch-safe, stateless).

        Chains the submodels' pure ``forward``s in exactly the order
        :meth:`do_step` steps them (dampers -> junctions -> exhaust fan ->
        heat recovery -> coil -> supply fan).  ``params`` is keyed by the
        composite attr path (``"supply_damper.a"``, ``"coil..."``, ...);
        non-estimated entries fall back to the submodels' defaults.
        """
        # Identity-keyed cache: a sequential rollout re-calls forward with the
        # SAME params dict every step (see OneStepComposer._params_for).
        if transform_mode:
            P = {
                n: self._resolve_sub_params(getattr(self, n), n, params)
                for n in self._SUB_NAMES
            }
        else:
            cache = getattr(self, "_fwd_param_cache", None)
            if cache is None or cache[0] is not params:
                cache = (
                    params,
                    {
                        n: self._resolve_sub_params(getattr(self, n), n, params)
                        for n in self._SUB_NAMES
                    },
                )
                self._fwd_param_cache = cache
            P = cache[1]

        # 1) Supply damper: vectorized position -> flow calculation
        supply_pos_vec = inputs["supplyDamperPosition"]
        supply_pos_flat = supply_pos_vec.reshape(supply_pos_vec.shape[0], -1)
        _, d_sup = self.supply_damper.forward(
            None, {"damperPosition": supply_pos_flat}, P["supply_damper"], sample_time
        )
        supply_flow_vec = d_sup["airFlowRate"].reshape(supply_pos_vec.shape)
        if inputs.get("supplyFanSpeed") is not None:
            supply_flow_vec = supply_flow_vec * self._fan_gate(
                inputs["supplyFanSpeed"]
            ).unsqueeze(-1)

        # 2) Supply junction: sum branch flows
        _, j_sup = self.supply_junction.forward(
            None, {"airFlowRateOut": supply_flow_vec}, P["supply_junction"],
            sample_time,
        )
        supply_flow_total = j_sup["airFlowRateIn"]

        # 3) Exhaust flows: damper model (fan-gated), or supply flows times the ratio
        if self.exhaust_follows_supply:
            ratio = params.get("exhaustFlowRatio", None)
            if ratio is None:
                ratio = self.exhaustFlowRatio.get()
            exhaust_flow_vec = self._scale_by_ratio(supply_flow_vec, ratio)
        else:
            exhaust_pos_vec = inputs["exhaustDamperPosition"]
            exhaust_pos_flat = exhaust_pos_vec.reshape(exhaust_pos_vec.shape[0], -1)
            _, d_exh = self.exhaust_damper.forward(
                None, {"damperPosition": exhaust_pos_flat}, P["exhaust_damper"],
                sample_time,
            )
            exhaust_flow_vec = d_exh["airFlowRate"].reshape(exhaust_pos_vec.shape)
            if inputs.get("exhaustFanSpeed") is not None:
                exhaust_flow_vec = exhaust_flow_vec * self._fan_gate(
                    inputs["exhaustFanSpeed"]
                ).unsqueeze(-1)

        # 4) Return junction: combine exhaust flows and temperatures
        _, j_ret = self.return_junction.forward(
            None,
            {
                "airFlowRateIn": exhaust_flow_vec,
                "airTemperatureIn": self._per_branch_exhaust_temperature(
                    inputs["exhaustTemperature"]
                ),
            },
            P["return_junction"],
            sample_time,
        )
        secondary_flow = j_ret["airFlowRateOut"]
        return_temp = j_ret["airTemperatureOut"]

        # 5) Exhaust fan (on return stream before heat recovery)
        _, f_exh = self.exhaust_fan.forward(
            None,
            {"airFlowRate": secondary_flow, "inletAirTemperature": return_temp},
            P["exhaust_fan"],
            sample_time,
        )

        # 6) Heat recovery
        _, hr = self.heat_recovery.forward(
            None,
            {
                "primaryAirFlowRate": supply_flow_total,
                "secondaryAirFlowRate": secondary_flow,
                "primaryTemperatureIn": inputs["outdoorAirTemperature"],
                "secondaryTemperatureIn": f_exh["outletAirTemperature"],
                "primaryTemperatureOutSetpoint": inputs[
                    "supplyAirTemperatureSetpoint"
                ],
            },
            P["heat_recovery"],
            sample_time,
        )

        # 7) Coil: trim to setpoint & report power
        _, coil = self.coil.forward(
            None,
            {
                "inletAirTemperature": hr["primaryTemperatureOut"],
                "outletAirTemperatureSetpoint": inputs[
                    "supplyAirTemperatureSetpoint"
                ],
                "airFlowRate": supply_flow_total,
            },
            P["coil"],
            sample_time,
        )

        # 8) Supply fan after coil to add temperature rise and power
        _, f_sup = self.supply_fan.forward(
            None,
            {
                "airFlowRate": supply_flow_total,
                "inletAirTemperature": coil["outletAirTemperature"],
            },
            P["supply_fan"],
            sample_time,
        )

        return x, {
            "supplyAirFlowRate": supply_flow_vec,
            "exhaustAirFlowRate": exhaust_flow_vec,
            "totalSupplyAirFlowRate": supply_flow_total,
            "totalExhaustAirFlowRate": secondary_flow,
            "supplyAirTemperature": f_sup["outletAirTemperature"],
            "preheatSupplyAirTemperature": hr["primaryTemperatureOut"],
            "exhaustAirTemperatureOut": hr["secondaryTemperatureOut"],
            "heatingPower": coil["heatingPower"],
            "coolingPower": coil["coolingPower"],
            "supplyFanPower": f_sup["Power"],
            "exhaustFanPower": f_exh["Power"],
        }


# NOTE: ``brick_signature_pattern_vav_dampers`` below absorbs the per-VAV
# ``Damper`` and ``Damper_Position_Setpoint`` nodes into the AHU's
# ``ModeledNode`` group so the Stage-1 -> Stage-2 controller-extraction
# merge can locate damper actuators.  Status today:
#
#   1.  RESOLVED.  Previously a ``SetStepRule`` AHU->VAVs hop combined
#       with downstream ``StepRule`` rules auto-broadcasted per element,
#       producing one AHU component per VAV (the translator's MILP then
#       kept the original ``brick_signature_pattern`` match *and* the
#       per-VAV matches, yielding duplicate AHU components).  The new
#       :class:`SetAnyPathRule` (translator.py) does the multi-hop
#       traversal AND emits a single tuple-bound branch from the AHU
#       side, so the pattern now produces exactly one match per AHU
#       with all four set-bound descendants (``vavs``, ``spaces``,
#       ``dampers``, ``damper_cmds``) aligned in parallel tuples.
#
#   3.  RESOLVED.  ``__prune_recursive`` previously initialised
#       ``valid_maps = []`` once *outside* the per-rule loop and only
#       ever extended it.  Each sibling rule re-read ``candidate_maps =
#       valid_maps`` *after* appending its own outputs, so the prior
#       rule's pre-extension snapshots stayed in the bag and propagated
#       to ``__broadcast_recurse``, which only takes ``child_maps[0]``
#       per element and would happily pick up a stale partial that
#       lacks the later siblings' bindings.  ``valid_maps`` is now
#       reset per-rule, and ``candidate_maps`` is only replaced when
#       the rule actually matched (preserving ``OptionalRule``
#       no-match semantics).  With this, the AHU-rooted Phase-1 DFS
#       produces a single fully-bound map per AHU and the broadcast
#       aggregator no longer drops ``StepRule`` siblings.
#
#   2.  OPEN (Stage-2 sensor keying, independent of the matcher).
#       Even with the AHU's ``_sim2sem_map`` carrying the damper
#       command URIs, the Stage-2 historised damper command
#       ``SensorSystem`` (matched by
#       ``brick_damper_command_sensor_pattern``, modelled at the
#       ``externalref`` BlankNode rather than at the
#       ``Damper_Position_Setpoint`` URI) is *not* keyed by the URI the
#       merge looks up.  ``_pick_best_component`` therefore returns the
#       AHU itself, which has no ``measuredValue`` output, and the
#       consumer-rewire snapshot is empty.  Closing this loop requires
#       either (a) modelling ``brick_damper_command_sensor_pattern`` on
#       a multi-member ``ModeledNode([damper_cmd, externalref])`` so the
#       sensor is also keyed by the command URI, or (b) extending the
#       merge to detect "actuator-direct" wiring (no intermediate sensor)
#       and rewire from controller -> AHU.supplyDamperPosition[i] using
#       the index that the AHU pattern recorded.  This is independent
#       of the pattern-matching fixes for (1) and (3) above.
#
#   4.  RESOLVED.  ``OptionalRule`` allows a node to remain unbound
#       when the SM lacks the predicate; the legacy heuristic Phase-4
#       merger would then absorb any incomplete partial that *did*
#       bind that node -- even one rooted from an unrelated SM
#       neighbourhood -- producing the canonical AHU01-SAT-leaks-into
#       -AHU02 cross-contamination.  After PR2.1-PR2.6 the
#       bidirectional walker (``__prune_recursive``) walks both forward
#       and backward edges from a single seed, so a connected
#       SP graph (this pattern is one weakly-connected component)
#       fills every required + optional binding from one Phase-1 seed.
#       ``_merge_incomplete_groups`` short-circuits to a no-op for
#       single-WCC patterns (PR4), so the cross-contamination path
#       no longer fires here.  When AHU02 lacks a SAT setpoint, the
#       walker terminates the seed without binding ``sat_setpoint``,
#       Phase 5 (isolated-optional fill) only transfers an optional
#       binding when ``_optional_binding_compatible`` confirms it
#       agrees with the complete map's structural context, and
#       AHU01's SAT setpoint stays attached to AHU01.


# ``brick_signature_pattern`` is intentionally NOT registered.
# ``brick_signature_pattern_vav_dampers`` subsumes it for the topologies
# we care about (VAV-based AHUs in BMS-grade BRICK graphs), and the two
# patterns cannot coexist on the same SM AHU node today: the multi-member
# ``ModeledNode`` group on the dampers pattern is non-exclusive for *every*
# member (including the AHU itself), so the simple pattern's
# singleton-modeled AHU match is allowed to bind the same AHU URI
# alongside the dampers pattern.  That yields TWO
# AirHandlingUnitSystem components per real AHU, and the
# BuildingSpace pattern's ``ahu.supplyAirFlowRate[output_port_index=vav]``
# connection ends up resolved against the simple AHU (which never declared
# Vector-output indexing), tripping the "input port Scalar / output port
# Vector" assertion in ``simulation_model.add_connection``.
#
# If a non-VAV topology (direct AHU -> Room without a VAV equipment
# layer) is ever needed, define a sibling pattern that uses singleton
# ``add_modeled_node(ahu)`` and is mutually exclusive with the VAV
# variant via the matcher's existing exclusion machinery (rather than
# relying on the broken mixed-mutex semantics today).


# Deprecated aliases (removed in twin4build 2.1)
AirHandlingUnitTorchSystem = AirHandlingUnitSystem
