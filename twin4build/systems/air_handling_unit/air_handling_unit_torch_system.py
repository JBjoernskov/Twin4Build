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

    # ``SetAnyPathRule`` collapses every Room/HVAC_Zone reachable from
    # the AHU via any feeds-path (e.g. AHU -> VAV -> Space) into a single
    # tuple binding on ``spaces``.  The earlier ``AnyPathRule`` produced
    # one scalar branch per reachable endpoint, yielding ``N_zones``
    # separate AHU components per AHU; the set-bound binding instead
    # gives one AHU component whose ``exhaustTemperature`` Vector input
    # is indexed by the served zones.
    sp.add_rule(SetAnyPathRule(subject=ahu, object=spaces, predicate=feeds))

    sp.add_connection(
        spaces, "indoorTemperature", "exhaustTemperature", input_port_index=spaces
    )

    sp.add_modeled_node(ahu)

    return sp


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

def brick_signature_pattern_vav_dampers():
    """AHU pattern for VAV systems with per-zone dampers + AHU-level points.

    Mirrors the building's actual topology:

    - AHU ``feeds`` N VAVs (each VAV serves one space).
    - Each VAV has a ``Damper`` (``isPartOf``) with a
      ``Damper_Position_Setpoint`` command point.  That command drives
      the AHU's per-branch ``supplyDamperPosition`` / ``exhaustDamperPosition``
      ``Vector`` inputs (the AHU torch model owns supply/exhaust damper
      sub-systems internally and reshapes the per-branch positions
      across the vectorised :class:`DamperTorchSystem`).
    - The AHU's ``hasPoint`` ``Outside_Air_Temperature_Sensor`` matches
      :class:`OutdoorEnvironmentSystem` (modelled at the same URI), which
      provides the ``outdoorTemperature`` source.
    - The AHU's ``hasPoint`` ``Supply_Air_Temperature_Setpoint`` (when
      present -- not every AHU has one in real BMS data; e.g. Mortar
      bldg1 AHU02 only has ``Supply_Air_Temp``) provides the supply-air
      temperature setpoint.

    The ``Damper`` and ``Damper_Position_Setpoint`` nodes are absorbed
    into the AHU's ``ModeledNode`` group rather than being matched by a
    separate t4b component: the supply/exhaust damper torch sub-systems
    are owned by the AHU and BRICK's per-VAV ``Damper`` equipment does
    not deserve its own component.  Including ``damper_cmds`` in the
    group also means Stage-2 ``_sim2sem_map[ahu]`` carries every per-VAV
    damper-command URI, so the Stage-1 -> Stage-2 controller-extraction
    merge (see
    :func:`twin4build.systems.controller.controller_identification.extractor.wire_extracted_controllers`)
    can locate the historised damper-command ``SensorSystem`` (matched
    by ``brick_sensor_leaf_pattern`` at the same URI) and rewire its
    consumers to the extracted PI controller.
    """
    sp = SignaturePattern(id="air_handling_unit_signature_pattern_brick_vav_dampers")

    ahu = Node(cls=core.namespace.BRICK.AHU)
    spaces = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.BRICK.HVAC_Zone,
        )
    )
    vavs = Node(cls=core.namespace.BRICK.VAV)
    dampers = Node(cls=core.namespace.BRICK.Damper)
    damper_cmds = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    # ``sat_setpoint`` deliberately matches ONLY ``Supply_Air_Temperature_Setpoint``,
    # never ``Supply_Air_Temperature_Sensor``.  Allowing the sensor class here
    # used to close a positive-feedback loop in buildings that only carry a
    # ``Supply_Air_Temp`` measurement (e.g. Mortar bldg1 AHU02):
    #
    #   * ``brick_signature_pattern_ahu_supply_air_temp`` (sensor_system.py)
    #     wires ``ahu.supplyAirTemperature -> sensor.measuredValue``, i.e. the
    #     sensor is a virtual pass-through of the simulated AHU output.
    #   * If the same sensor URI also bound ``sat_setpoint`` here, the AHU
    #     pattern would wire ``sensor.measuredValue -> ahu.supplyAirTemperature
    #     Setpoint``, and the coil + supply-fan chain would feed
    #     ``setpoint[k+1] = AHU.supplyAirTemperature[k]`` back into itself.
    #     Each step the supply fan adds ``delta_T = P_fan * f_total /
    #     (m_dot * c_p)``, so the loop becomes a pure accumulator
    #     ``AHU.supplyAirTemperature[k+1] = AHU.supplyAirTemperature[k] +
    #     delta_T``.  Over a 10-day run at 600 s step that drifts to ~1000 K.
    #
    # A real BMS setpoint is an input (commanded value), not a measurement of
    # the AHU's own output, so this match is structurally correct.  For AHUs
    # without a setpoint URI the ``OptionalRule`` below simply leaves
    # ``supplyAirTemperatureSetpoint`` unwired; ``fill_missing_inputs`` (or
    # the user) is expected to supply a default.
    sat_setpoint = Node(cls=core.namespace.BRICK.Supply_Air_Temperature_Setpoint)
    oat_sensor = Node(cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor)

    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    # AHU -> VAVs (multi-hop, set-bound).  ``SetAnyPathRule`` BFS-walks
    # the ``feeds`` predicate from the AHU and bundles every reachable
    # VAV into a single tuple binding on ``vavs``; downstream
    # ``StepRule``\ s on the set-bound ``vavs`` SP node auto-broadcast
    # per element (see SetStepRule docs), so we get parallel set-bound
    # bindings for ``spaces``, ``dampers`` and ``damper_cmds`` -- one
    # per VAV -- without the unsupported nested-set chaining.
    #
    # Using ``SetAnyPathRule`` instead of ``SetStepRule`` matters when
    # the BMS graph models AHU -> air-handler-segment -> VAV (i.e. the
    # AHU does not feed VAVs in a single hop): the multi-hop traversal
    # still discovers them, while remaining set-bound at the AHU level
    # so each AHU yields one match instead of one per VAV.
    #
    # We walk:
    #   vav  --feeds--> space          (each VAV serves one space)
    #   vav  --hasPart--> damper       (inverse of ``damper isPartOf vav``;
    #                                    BRICK declares ``hasPart owl:inverseOf
    #                                    isPartOf`` and the SemanticModel
    #                                    reasoner materialises both directions)
    #   damper --hasPoint--> damper_cmd
    #
    # All four set-bound nodes (``vavs``, ``spaces``, ``dampers``,
    # ``damper_cmds``) end up indexed in lockstep, so the translator can
    # align AHU's ``supplyAirFlowRate`` Vector output (indexed by
    # ``spaces`` from the BuildingSpace pattern that consumes it) with
    # ``supplyDamperPosition`` Vector input (indexed by ``spaces`` here).
    sp.add_rule(SetAnyPathRule(subject=ahu, object=vavs, predicate=feeds))
    sp.add_rule(StepRule(subject=vavs, object=spaces, predicate=feeds))
    sp.add_rule(
        StepRule(subject=vavs, object=dampers, predicate=core.namespace.BRICK.hasPart)
    )
    sp.add_rule(
        StepRule(subject=dampers, object=damper_cmds, predicate=core.namespace.BRICK.hasPoint)
    )
    # Outside-air temperature sensor: free-floating optional, NOT tied
    # to ``ahu hasPoint oat_sensor``.  Every AHU in the dataset has its
    # own per-unit ``Outside_Air_Temp`` ``hasPoint`` (e.g.
    # ``bldg1.AHU.AHU01.Outside_Air_Temp`` and ``bldg1.AHU.AHU02.Outside_Air_Temp``),
    # but only one ``OutdoorEnvironmentSystem`` instance typically
    # materialises per building -- its ``brick_signature_pattern_standalone``
    # pairs an ``Outside_Air_Temperature_Sensor`` with a
    # ``Global_Solar_Irradiation_Sensor`` (both required) and the MILP
    # gets to claim each modeled URI at most once.  If we anchored
    # ``oat_sensor`` to the AHU via ``hasPoint``, AHU02 would bind to
    # its own ``Outside_Air_Temp`` URI -- at which no ``OutdoorEnvironment
    # System`` is modelled -- and ``outdoorAirTemperature`` would stay
    # dangling.  By declaring ``oat_sensor`` as a structurally-free
    # optional node we let the matcher bind it to whichever
    # ``Outside_Air_Temperature_Sensor`` URI an ``OutdoorEnvironmentSystem``
    # actually claims, so both AHUs end up reading from the same shared
    # outdoor-environment instance (and so do all BuildingSpaces -- the
    # ``BuildingSpace`` pattern uses the same free-floating optional
    # convention for outdoor temperature and solar irradiation).
    sp.add_node(oat_sensor, optional=True)
    # Supply-air temperature setpoint: optional -- some AHUs only carry
    # the Supply_Air_Temp measurement without a separate setpoint point
    # (e.g. Mortar bldg1 AHU02).  Wrap in OptionalRule so the pattern
    # still matches; the AHU's supplyAirTemperatureSetpoint port simply
    # stays unwired in that case (validation will warn but not fail).
    sp.add_rule(
        OptionalRule(
            subject=ahu,
            object=sat_setpoint,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )

    # All Vector inputs are indexed by ``spaces`` so the BuildingSpace
    # pattern (which uses ``output_port_index=space`` when reading from
    # the AHU's ``supplyAirFlowRate`` Vector output) shares a common
    # SP-side node identity with this pattern.  Mixing index keys
    # (``vavs`` here, ``spaces`` over there) leaves the translator
    # unable to map BuildingSpace.space to an AHU output slot, which
    # surfaces as a ``Vector -> Scalar with no index`` assertion in
    # ``add_connection`` for unrelated edges like
    # ``ahu.supplyAirFlowRate -> RM107A.supplyAirFlowRate``.
    sp.add_connection(
        spaces, "indoorTemperature", "exhaustTemperature", input_port_index=spaces
    )
    # Per-VAV damper command -> AHU supply / exhaust damper vectors.
    # The source-side port name is ``inputSignal``, which is the output
    # port a CITS / extracted PI controller produces at the
    # ``Damper_Position_Setpoint`` URI.  When ``ControllerIdentificationPI
    # TorchSystem`` is in Stage-2's ``systems_``, the controller component
    # is matched at the same command URI as the historised
    # ``SensorSystem`` and provides this ``inputSignal`` output, closing
    # the control loop natively during translation -- no separate
    # extract/wire post-process is needed.
    #
    # Both vectors are driven by the same per-VAV command because the
    # bldg1 BRICK graph models a single damper per VAV; for VAV systems
    # without dedicated exhaust modulation the return-air branch tracks
    # supply 1:1.  If a building genuinely models separate exhaust
    # dampers, register a sibling pattern that binds them separately.
    # ``output_port_index=damper_cmds`` picks the CITS actuator slot
    # corresponding to this damper command -- CITS.inputSignal is a
    # Vector (one slot per actuator) so we need a key on the sender side
    # too, matching the convention in ``brick_damper_command_sensor_pattern``.
    sp.add_connection(
        damper_cmds,
        "inputSignal",
        "supplyDamperPosition",
        output_port_index=damper_cmds,
        input_port_index=spaces,
    )
    sp.add_connection(
        damper_cmds,
        "inputSignal",
        "exhaustDamperPosition",
        output_port_index=damper_cmds,
        input_port_index=spaces,
    )
    sp.add_connection(sat_setpoint, "measuredValue", "supplyAirTemperatureSetpoint")
    sp.add_connection(oat_sensor, "outdoorTemperature", "outdoorAirTemperature")

    ModeledNode([ahu, vavs, dampers, damper_cmds])
    return sp


# ``brick_signature_pattern`` is intentionally NOT registered.
# ``brick_signature_pattern_vav_dampers`` subsumes it for the topologies
# we care about (VAV-based AHUs in BMS-grade BRICK graphs), and the two
# patterns cannot coexist on the same SM AHU node today: the multi-member
# ``ModeledNode`` group on the dampers pattern is non-exclusive for *every*
# member (including the AHU itself), so the simple pattern's
# singleton-modeled AHU match is allowed to bind the same AHU URI
# alongside the dampers pattern.  That yields TWO
# AirHandlingUnitTorchSystem components per real AHU, and the
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
AirHandlingUnitTorchSystem.add_signature_pattern(brick_signature_pattern_vav_dampers())
