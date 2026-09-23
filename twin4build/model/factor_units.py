"""Factor composite air handling units into terminals, junctions and the device.

A translated :class:`AirHandlingUnitSystem` owns its terminals: one damper
per branch inside the unit, vector ports per branch, the junctions' sums
inside its step.  :func:`factor_air_handling_units` rewrites such a unit,
in place on the model, into the factored form::

    command_b --> damper_b --airFlowRate--> (whatever read unit.supplyAirFlowRate[b])
                                        --> supply junction[b]
                           --exhaustAirFlowRate--> (whatever read unit.exhaustAirFlowRate[b])
                                                --> return junction[b]
    room of branch b --indoorTemperature--> return junction[b]
    supply junction --> core.totalSupplyAirFlowRate
    return junction --> core.totalExhaustAirFlowRate, core.returnAirTemperature
    core (the unit's id, coil, heat recovery and fans) --> whatever read the unit's scalar outputs

The damper parameters, the exhaust ratio and the device's submodels are the
unit's own, so the factored model simulates exactly as the composite did
(tested step for step).  What changes is the structure the estimator sees:
each terminal is a component of its own, so a room's block is bounded by
its commands, the weather and the unit's supply temperature instead of
being joined to every other room through the unit's parameters.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from twin4build.systems.air_handling_unit.air_handling_unit_core_system import (
    AirHandlingUnitCoreSystem,
)
from twin4build.systems.air_handling_unit.air_handling_unit_system import (
    AirHandlingUnitSystem,
)
from twin4build.systems.damper.damper_system import DamperSystem
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)
from twin4build.utils.slots import slot_pairs, wired_width

LOGGER = logging.getLogger(__name__)

#: The unit's scalar outputs the core carries under the same names.
CORE_OUTPUTS = (
    "supplyAirTemperature",
    "preheatSupplyAirTemperature",
    "exhaustAirTemperatureOut",
    "totalSupplyAirFlowRate",
    "totalExhaustAirFlowRate",
    "heatingPower",
    "coolingPower",
    "supplyFanPower",
    "exhaustFanPower",
)
#: The unit's scalar inputs the core takes under the same names.
CORE_INPUTS = ("supplyAirTemperatureSetpoint", "outdoorAirTemperature")


def _slot(index) -> Optional[int]:
    if index is None or isinstance(index, slice):
        return None
    return int(index)


def _sources(comp, port) -> List[Tuple[object, str, Optional[int], Optional[int]]]:
    """``(sender, output port, output slot, input slot)`` per pair wired
    into ``comp.input[port]``."""
    out = []
    for cp in comp.connects_at:
        if cp.input_port != port:
            continue
        for conn in cp.connects_system_through:
            for _, _, out_v, in_v in slot_pairs(cp, conn):
                out.append((conn.connects_system, conn.output_port, _slot(out_v), _slot(in_v)))
    return out


def _receivers(comp, port) -> List[Tuple[object, str, Optional[int], Optional[int]]]:
    """``(receiver, input port, output slot, input slot)`` per pair wired
    from ``comp.output[port]``."""
    out = []
    for conn in comp.connected_through:
        if conn.output_port != port:
            continue
        for cp in conn.connects_system_at:
            for _, _, out_v, in_v in slot_pairs(cp, conn):
                out.append((cp.connection_point_of, cp.input_port, _slot(out_v), _slot(in_v)))
    return out


def _per_branch(param, b: int, n: int) -> float:
    """Branch ``b``'s value of a parameter stored per branch, or the shared
    value when the parameter is a scalar."""
    t = torch.as_tensor(param.get()).detach().reshape(-1)
    if t.numel() == n:
        return float(t[b])
    if t.numel() == 1:
        return float(t[0])
    raise ValueError(f"a parameter with {t.numel()} values for {n} branches")


def _connect(model, sender, out_port, receiver, in_port, out_v=None, in_v=None):
    kwargs = {}
    if out_v is not None:
        kwargs["output_port_index"] = out_v
    if in_v is not None:
        kwargs["input_port_index"] = in_v
    model.add_connection(sender, receiver, out_port, in_port, **kwargs)


def factor_air_handling_unit(model, unit: AirHandlingUnitSystem) -> Dict[str, object]:
    """Replace ``unit`` on ``model`` by its factored form.  Returns
    ``{"core", "dampers", "exhaust_dampers", "supply_junction",
    "return_junction"}``; the core keeps the unit's id."""
    n = wired_width(unit, "supplyDamperPosition")
    if n == 0:
        raise ValueError(f"|{unit.id}|: no damper command is wired, there is no branch to factor")
    follows = bool(unit.exhaust_follows_supply)
    per_branch_ratio = bool(unit.exhaust_ratio_per_branch)

    # --- read the wiring before anything is removed ---------------------------------
    commands = {in_v: (s, p, out_v) for s, p, out_v, in_v in _sources(unit, "supplyDamperPosition")}
    exhaust_commands = {
        in_v: (s, p, out_v) for s, p, out_v, in_v in _sources(unit, "exhaustDamperPosition")
    }
    supply_fan = [(s, p, out_v) for s, p, out_v, _ in _sources(unit, "supplyFanSpeed")]
    exhaust_fan = [(s, p, out_v) for s, p, out_v, _ in _sources(unit, "exhaustFanSpeed")]
    temperatures = {in_v: (s, p, out_v) for s, p, out_v, in_v in _sources(unit, "exhaustTemperature")}
    scalar_in = {port: [(s, p, out_v) for s, p, out_v, _ in _sources(unit, port)] for port in CORE_INPUTS}
    scalar_out = {port: [(r, ip, in_v) for r, ip, _, in_v in _receivers(unit, port)] for port in CORE_OUTPUTS}
    supply_readers: Dict[int, list] = {}
    for r, ip, out_v, in_v in _receivers(unit, "supplyAirFlowRate"):
        supply_readers.setdefault(out_v if out_v is not None else 0, []).append((r, ip, in_v))
    exhaust_readers: Dict[int, list] = {}
    for r, ip, out_v, in_v in _receivers(unit, "exhaustAirFlowRate"):
        exhaust_readers.setdefault(out_v if out_v is not None else 0, []).append((r, ip, in_v))
    missing = [b for b in range(n) if b not in commands]
    if missing:
        # The composite fed such a slot its initial value (a closed damper):
        # the terminal keeps that, with its position unwired.
        LOGGER.warning("%s: branches %s carry no damper command; their terminals stay closed", unit.id, missing)

    # The room of branch b: what reads its supply flow and also publishes its
    # temperature into the unit; else the aligned temperature slot.
    temperature_of_sender = {id(s): (s, p, out_v) for s, p, out_v in temperatures.values()}
    room_temperature: Dict[int, Tuple[object, str, Optional[int]]] = {}
    for b in range(n):
        for r, _, _ in supply_readers.get(b, []):
            if id(r) in temperature_of_sender:
                room_temperature[b] = temperature_of_sender[id(r)]
                break
        else:
            if b in temperatures:
                room_temperature[b] = temperatures[b]

    # --- the parts ----------------------------------------------------------------
    unit_id = unit.id
    dampers: List[DamperSystem] = []
    exhaust_dampers: List[DamperSystem] = []
    for b in range(n):
        ratio = (
            _per_branch(unit.exhaustFlowRatio, b, n) if (follows and per_branch_ratio)
            else float(torch.as_tensor(unit.exhaustFlowRatio.get()).reshape(-1)[0])
        )
        sd = unit.supply_damper
        dampers.append(
            DamperSystem(
                id=f"{unit_id}_terminal{b}",
                a=_per_branch(sd.a, b, n),
                nominalAirFlowRate=_per_branch(sd.nominalAirFlowRate, b, n),
                c=None if sd.c_tied else _per_branch(sd.c, b, n),
                exhaustFlowRatio=ratio if follows else 1.0,
            )
        )
        if not follows:
            ed = unit.exhaust_damper
            exhaust_dampers.append(
                DamperSystem(
                    id=f"{unit_id}_exhaust_terminal{b}",
                    a=_per_branch(ed.a, b, n),
                    nominalAirFlowRate=_per_branch(ed.nominalAirFlowRate, b, n),
                    c=None if ed.c_tied else _per_branch(ed.c, b, n),
                )
            )
    supply_junction = SupplyFlowJunctionSystem(id=f"{unit_id}_supply_junction")
    return_junction = ReturnFlowJunctionSystem(id=f"{unit_id}_return_junction")
    core = AirHandlingUnitCoreSystem(id=unit_id)
    # the device's own submodels, fitted parameters included
    core.coil = unit.coil
    core.heat_recovery = unit.heat_recovery
    core.supply_fan = unit.supply_fan
    core.exhaust_fan = unit.exhaust_fan

    # --- rewire -------------------------------------------------------------------------
    model.remove_component(unit)
    temperature_slots: Dict[Tuple[int, str], Tuple[object, str, list, list]] = {}
    for c in dampers + exhaust_dampers + [supply_junction, return_junction, core]:
        model.add_component(c)
    for b, d in enumerate(dampers):
        if b in commands:
            s, p, out_v = commands[b]
            _connect(model, s, p, d, "damperPosition", out_v=out_v)
        for s, p, out_v in supply_fan:
            _connect(model, s, p, d, "fanSpeed", out_v=out_v)
        _connect(model, d, "airFlowRate", supply_junction, "airFlowRateOut", in_v=b)
        for r, ip, in_v in supply_readers.get(b, []):
            _connect(model, d, "airFlowRate", r, ip, in_v=in_v)
        exhaust_source = (d, "exhaustAirFlowRate")
        if not follows:
            e = exhaust_dampers[b]
            if b in exhaust_commands:
                s, p, out_v = exhaust_commands[b]
                _connect(model, s, p, e, "damperPosition", out_v=out_v)
            for s, p, out_v in exhaust_fan:
                _connect(model, s, p, e, "fanSpeed", out_v=out_v)
            exhaust_source = (e, "airFlowRate")
        for r, ip, in_v in exhaust_readers.get(b, []):
            _connect(model, exhaust_source[0], exhaust_source[1], r, ip, in_v=in_v)
        if b in room_temperature:
            _connect(model, exhaust_source[0], exhaust_source[1], return_junction, "airFlowRateIn", in_v=b)
            s, p, out_v = room_temperature[b]
            temperature_slots.setdefault((id(s), p), (s, p, [], []))[2].append(b)
            temperature_slots[(id(s), p)][3].append(out_v)
        else:
            LOGGER.warning(
                "%s: branch %d has no room temperature; its exhaust is left out of the return junction",
                unit_id, b,
            )
    # a room with several terminals publishes its temperature on several
    # slots: one connection per room, carrying its slots
    for s, p, slots, out_vs in temperature_slots.values():
        in_v = slots[0] if len(slots) == 1 else torch.tensor(slots, dtype=torch.long)
        out_v = None
        if all(v is not None for v in out_vs):
            out_v = out_vs[0] if len(out_vs) == 1 else torch.tensor(out_vs, dtype=torch.long)
        _connect(model, s, p, return_junction, "airTemperatureIn", out_v=out_v, in_v=in_v)
    _connect(model, supply_junction, "airFlowRateIn", core, "totalSupplyAirFlowRate")
    _connect(model, return_junction, "airFlowRateOut", core, "totalExhaustAirFlowRate")
    _connect(model, return_junction, "airTemperatureOut", core, "returnAirTemperature")
    for port, sources in scalar_in.items():
        for s, p, out_v in sources:
            _connect(model, s, p, core, port, out_v=out_v)
    for port, readers in scalar_out.items():
        for r, ip, in_v in readers:
            _connect(model, core, port, r, ip, in_v=in_v)
    LOGGER.info(
        "%s: factored into %d terminal(s)%s, two junctions and the core unit",
        unit_id, n, "" if follows else f" with {n} exhaust damper(s)",
    )
    return {
        "core": core,
        "dampers": dampers,
        "exhaust_dampers": exhaust_dampers,
        "supply_junction": supply_junction,
        "return_junction": return_junction,
    }


def factor_air_handling_units(model) -> List[Dict[str, object]]:
    """Factor every composite :class:`AirHandlingUnitSystem` on ``model``
    (see the module docstring).  Returns one parts dict per unit."""
    units = list(model.get_components_by_class(AirHandlingUnitSystem))
    return [factor_air_handling_unit(model, unit) for unit in units]
