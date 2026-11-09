"""
Air handling unit with its terminals: a damper per branch, the two
junctions, the heat recovery, the coil and the two fans, in one System.

:mod:`twin4build.systems.air_handling_unit.air_handling_unit_core_system`
holds the device on its totals, which this unit shares, and the factored form
(:func:`twin4build.model.factor_units.factor_air_handling_unit`) splits this
unit into its terminals, its junctions and the core.
"""

# Postpone evaluation of annotations (PEP 563 / PEP 649).
from __future__ import annotations

# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.air_handling_unit.air_handling_unit_core_system import (
    CORE_SUB_NAMES,
    resolve_sub_params,
    unit_chain,
)
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
from twin4build.systems.saref4syst.declarations import Child, Input, Output, Parameter, Setting
from twin4build.utils.slots import wired_width


class AirHandlingUnitSystem(core.System):
    r"""
    Air handling unit (AHU) with its terminals.

    The branches are the slots of ``supplyDamperPosition``: one per VAV box.
    Each branch has a supply and an exhaust damper (the children
    ``supply_damper`` and ``exhaust_damper``, stepped once per branch, so each
    of their parameters holds a value per branch).  The branch flows are
    summed by the supply junction and mixed with their exhaust temperatures
    by the return junction; the device on the totals -- exhaust fan, heat
    recovery, coil, supply fan -- is
    :func:`~twin4build.systems.air_handling_unit.air_handling_unit_core_system.unit_chain`,
    shared with
    :class:`~twin4build.systems.air_handling_unit.air_handling_unit_core_system.AirHandlingUnitCoreSystem`.

    A child's values are given as a dict:
    ``AirHandlingUnitSystem(id="ahu", supply_damper=dict(nominalAirFlowRate=0.5))``;
    a damper value may be one per branch (a list).

    With ``exhaust_follows_supply`` every branch's exhaust flow is
    ``exhaustFlowRatio`` times its supply flow and the exhaust dampers are not
    read.  Use it when the exhaust side has no per-branch measurement: with one
    exhaust meter per AHU the per-branch exhaust dampers are unidentifiable
    (only the total is), while the ratio is pinned by that meter.
    """

    #: The old names of the children's constructor values (removed in 2.1).
    DEPRECATED_KEYWORDS = {
        "supply_damper_kwargs": "supply_damper",
        "exhaust_damper_kwargs": "exhaust_damper",
        "coil_kwargs": "coil",
        "heat_recovery_kwargs": "heat_recovery",
        "supply_fan_kwargs": "supply_fan",
        "exhaust_fan_kwargs": "exhaust_fan",
    }

    exhaust_follows_supply = Setting(False, doc="Each branch's exhaust flow is exhaustFlowRatio times its supply flow")
    # Zone-level imbalance is real: a BMS snapshot showed 500 m3/h in and 250
    # out in one room with the floor balanced.
    exhaust_ratio_per_branch = Setting(False, doc="One exhaustFlowRatio per branch instead of one per unit")
    supplyDamperPosition = Input(size="wired", doc="Supply damper opening of each branch (0-1)")
    exhaustDamperPosition = Input(
        size=supplyDamperPosition, default=0.0,
        doc="Exhaust damper opening of each branch (0-1); not read when the exhaust follows the supply",
    )
    exhaustTemperature = Input(
        size=supplyDamperPosition,
        doc="Exhaust air temperature of each branch [degC]: a room on several branches fills the slot of each",
    )
    supplyAirTemperatureSetpoint = Input(doc="Supply air temperature setpoint [degC]")
    outdoorAirTemperature = Input(doc="Outdoor air temperature [degC]")
    # A branch moves air only while its fan runs: a VAV damper left open after
    # the fan stops delivers nothing (``DamperSystem.fan_gate``).
    supplyFanSpeed = Input(default=1.0, doc="Supply fan speed or status (0-1); unwired: running")
    exhaustFanSpeed = Input(default=1.0, doc="Exhaust fan speed or status (0-1); unwired: running")
    supplyAirFlowRate = Output(size=supplyDamperPosition, doc="Supply flow of each branch [kg/s]")
    supplyAirTemperature = Output(doc="Supply air temperature, after the supply fan [degC]")
    # Brick ``Preheat_Supply_Air_Temperature_Sensor``.
    preheatSupplyAirTemperature = Output(doc="Supply air temperature after the heat recovery, before the coil [degC]")
    exhaustAirFlowRate = Output(size=supplyDamperPosition, doc="Exhaust flow of each branch [kg/s]")
    totalSupplyAirFlowRate = Output(doc="The branches' supply flows summed: what the unit's flow meter reads [kg/s]")
    totalExhaustAirFlowRate = Output(doc="The branches' exhaust flows summed [kg/s]")
    exhaustAirTemperatureOut = Output(doc="Exhaust air temperature after the heat recovery [degC]")
    heatingPower = Output(doc="Coil heating power [W]")
    coolingPower = Output(doc="Coil cooling power [W]")
    supplyFanPower = Output(doc="Supply fan power [W]")
    exhaustFanPower = Output(doc="Exhaust fan power [W]")
    exhaustFlowRatio = Parameter(
        1.0, lb=0.3, ub=1.5, per_slot=supplyDamperPosition, per_slot_if=exhaust_ratio_per_branch,
        doc="Exhaust over supply flow [-], with exhaust_follows_supply",
    )
    supply_damper = Child(DamperSystem, per_slot=supplyDamperPosition)
    exhaust_damper = Child(DamperSystem, per_slot=exhaustDamperPosition)
    supply_junction = Child(SupplyFlowJunctionSystem)
    return_junction = Child(ReturnFlowJunctionSystem)
    coil = Child(CoilSystem)
    heat_recovery = Child(AirToAirHeatRecoverySystem)
    supply_fan = Child(FanSystem)
    exhaust_fan = Child(FanSystem)

    #: The children, in the order ``forward`` steps them.
    _CHILDREN = ("supply_damper", "exhaust_damper", "supply_junction", "return_junction") + CORE_SUB_NAMES

    @property
    def n_branches(self) -> int:
        """The number of branches: the wired supply damper slots, or the width
        set by hand on a unit stepped on its own."""
        return wired_width(self, "supplyDamperPosition") or self.input["supplyDamperPosition"]._n_v or 1

    @n_branches.setter
    def n_branches(self, n: int) -> None:
        self.input["supplyDamperPosition"]._n_v = int(n)

    def size_declared_ports(self) -> None:
        """Size the ports, and check that every branch has its exhaust
        temperature (until the checks at load of ADR 0001, decision 10)."""
        super().size_declared_ports()
        wired = wired_width(self, "exhaustTemperature")
        n = self.input["supplyDamperPosition"]._n_v
        if wired not in (0, n):
            raise ValueError(
                f"|{type(self).__name__}|{self.id}|: exhaustTemperature has {wired} slots wired for "
                f"{n} branches; wire each room's temperature to the slot of each of its branches "
                "(add_connection(room, ahu, 'indoorTemperature', 'exhaustTemperature', "
                "input_port_index=[the branches' slots]))"
            )

    def get_estimable_parameters(self):
        """The children's estimable parameters: a damper's offset ``c`` only
        once given; the dampers' own exhaust ratio never (the unit's
        ``exhaustFlowRatio`` is the one); the exhaust damper's not at all when
        the exhaust follows the supply, and the ratio only then."""
        out = []
        for entry in super().get_estimable_parameters():
            attr = str(entry[1])
            prefix, _, leaf = attr.rpartition(".")
            if leaf == "exhaustFlowRatio" and prefix.endswith("damper"):
                continue
            if self.exhaust_follows_supply and attr.startswith("exhaust_damper."):
                continue
            if not self.exhaust_follows_supply and attr == "exhaustFlowRatio":
                continue
            out.append(entry)
        return out

    def _inactive_parameters(self):
        """Parameters the exhaust mode leaves without effect: the exhaust
        damper's when the exhaust follows the supply, the ratio otherwise."""
        return () if self.exhaust_follows_supply else ("exhaustFlowRatio",)

    @staticmethod
    def _on_branches(value, like: torch.Tensor, per_branch: bool) -> torch.Tensor:
        """``value`` laid out to multiply the branch values ``like`` ``(rows,
        n_v)``: a shared value as it is; one per copy (``per_branch`` false) on
        the rows; one per branch of every copy (a per-slot parameter, ``n_c *
        n_v`` values, held once for every simulation in object mode) on the rows
        and the branches."""
        value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        if value.numel() == 1:
            return value.reshape(())
        if not per_branch:
            return value.reshape(-1, 1)
        value = value.reshape(-1, like.shape[-1])
        if value.shape[0] != like.shape[0]:
            value = value.repeat(like.shape[0] // value.shape[0], 1)
        return value

    def _damper_flows(self, damper, P, positions, fan_speed, step_size) -> torch.Tensor:
        """The branch flows of ``damper`` at ``positions`` ``(rows, n_v)``,
        gated by its fan."""
        params = {k: self._on_branches(v, positions, per_branch=True) for k, v in P.items()}
        _, out = damper.forward(None, {"damperPosition": positions}, params, step_size)
        flows = out["airFlowRate"]
        if fan_speed is not None:
            flows = flows * DamperSystem.fan_gate(fan_speed).unsqueeze(-1)
        return flows

    def derive(self, p, step_size):
        """The children's parameters (estimated values from ``p``, the rest the
        children's own), resolved once per run or rollout."""
        return {
            "sub_params": {
                n: getattr(self, n).full_params(resolve_sub_params(getattr(self, n), n, p), step_size)
                for n in self._CHILDREN
            }
        }

    def forward(self, x, inputs, p, step_size):
        """Stateless: the dampers, the junctions and the device on the totals
        (:func:`unit_chain`)."""
        P = p["sub_params"]
        supply_flow = self._damper_flows(
            self.supply_damper, P["supply_damper"], inputs["supplyDamperPosition"],
            inputs.get("supplyFanSpeed"), step_size,
        )
        _, j_sup = self.supply_junction.forward(
            None, {"airFlowRateOut": supply_flow}, P["supply_junction"], step_size
        )
        if self.exhaust_follows_supply:
            # the supply flows already carry the supply fan's gate
            exhaust_flow = supply_flow * self._on_branches(
                p["exhaustFlowRatio"], supply_flow, per_branch=self.exhaust_ratio_per_branch
            )
        else:
            exhaust_flow = self._damper_flows(
                self.exhaust_damper, P["exhaust_damper"], inputs["exhaustDamperPosition"],
                inputs.get("exhaustFanSpeed"), step_size,
            )
        _, j_ret = self.return_junction.forward(
            None,
            {"airFlowRateIn": exhaust_flow, "airTemperatureIn": inputs["exhaustTemperature"]},
            P["return_junction"],
            step_size,
        )
        outs = unit_chain(
            {n: getattr(self, n) for n in CORE_SUB_NAMES},
            P,
            j_sup["airFlowRateIn"],
            j_ret["airFlowRateOut"],
            j_ret["airTemperatureOut"],
            inputs["outdoorAirTemperature"],
            inputs["supplyAirTemperatureSetpoint"],
            step_size,
        )
        outs["supplyAirFlowRate"] = supply_flow
        outs["exhaustAirFlowRate"] = exhaust_flow
        return x, outs


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
