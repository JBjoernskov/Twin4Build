"""Partition a model into groups bounded by measured signals, and cut the
edges between them.

An estimation problem is separable into blocks when no block's parameters
reach another block's residuals.  On a model that is a question of
wiring: a connection whose signal is *measured* (a data-bearing sensor reads
that very port and slot) can be replaced by a leaf that replays the
measurement, after which the receiver no longer depends on the sender's
parameters.  A connection whose signal is not measured ties its two ends
together.

:func:`measured_partition` classifies every connection and takes the
connected components of the graph over the unmeasured ones: those are the
minimal groups the measurements allow.  A data-bearing sensor that reads a
port is a sink and stays with its source (it carries that group's residual
columns); what the sensor itself sends on (a controller reading it) is
measured too, by its own series.  A leaf with no inputs (weather,
schedules, replayed commands) is exogenous and joins nothing.
:func:`cut_measured_edges` then replaces every measured connection by a
replay leaf, one per sensor.  The connections that cross between groups
make the groups separable; the measured connections inside a group leave
the groups as they are and hand the receiver the measurement instead of
the simulated signal.  Classing what a sensor sends on as measured is
what keeps a controller chain from gluing a room to its terminals: a
controller in playback (its recorded command replayed) does not read the
sensor at all, so the cut is exact, and the terminal's measured flow into
the room then crosses between groups instead of staying inside one.
Afterwards the estimator's own structure walk
(``FunctionalModel.index_coupling``) finds one block per group that
carries parameters.

Signals that are a known function of a measurement (a terminal's exhaust
flow as a fixed ratio times its measured supply flow) count as measured
when the caller says so through ``derived``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.utils.slots import slot_pairs

LOGGER = logging.getLogger(__name__)

Signal = Tuple[str, str, Optional[int]]  # (component id, output port, output slot)


@dataclass
class Edge:
    """One (sender port, slot) -> (receiver port, slot) pair of a connection."""

    sender: object
    output_port: str
    output_slot: Optional[int]
    receiver: object
    input_port: str
    input_slot: Optional[int]
    #: The data-bearing sensor whose series stands for the signal, or None.
    sensor: Optional[SensorSystem] = None
    #: A function of the sensor's series that gives the signal (derived).
    transform: Optional[Callable] = None

    @property
    def signal(self) -> Signal:
        return (self.sender.id, self.output_port, self.output_slot)

    @property
    def pairing(self) -> Tuple[str, str, str, str]:
        return (self.sender.id, self.output_port, self.receiver.id, self.input_port)


@dataclass
class Partition:
    groups: List[List[str]]
    group_of: Dict[str, int]
    #: Measured edges between two groups: what separates the groups.
    crossing: List[Edge]
    #: Measured edges inside a group: cut too, groups unchanged.
    internal: List[Edge]
    #: Unmeasured edges, all inside groups by construction.
    binding: List[Edge]
    #: Component ids that carry free parameters, per group.
    free: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> str:
        sizes = sorted((len(g) for g in self.groups), reverse=True)
        with_theta = sum(1 for g in self.groups if any(c in self.free for c in g))
        return (
            f"{len(self.groups)} groups ({with_theta} with parameters), sizes {sizes[:8]}"
            f"{'...' if len(sizes) > 8 else ''}; {len(self.crossing)} measured edges cross, "
            f"{len(self.internal)} measured edges stay inside, {len(self.binding)} unmeasured edges bind"
        )


def _slot(index) -> Optional[int]:
    if index is None or isinstance(index, slice):
        return None
    return int(index)


def _edges(model) -> List[Edge]:
    out = []
    for comp in model.components.values():
        for conn in comp.connected_through:
            for cp in conn.connects_system_at:
                for _, _, out_v, in_v in slot_pairs(cp, conn):
                    out.append(
                        Edge(comp, conn.output_port, _slot(out_v), cp.connection_point_of, cp.input_port, _slot(in_v))
                    )
    return out


def _is_data_sensor(comp, allowed: Optional[Set[str]]) -> bool:
    if not isinstance(comp, SensorSystem) or not comp.has_data:
        return False
    return allowed is None or comp.id in allowed


def _is_fusable(e: "Edge") -> bool:
    """Whether the edge is a fusable arc: its output port in the sender's
    ``FUSABLE_OUTPUT_PORTS`` and its input port in the receiver's
    ``FUSABLE_INPUT_PORTS`` (see ``FusedStateSpaceSystem``)."""
    outs = getattr(type(e.sender), "FUSABLE_OUTPUT_PORTS", frozenset())
    ins = getattr(type(e.receiver), "FUSABLE_INPUT_PORTS", frozenset())
    return e.output_port in outs and e.input_port in ins


def _has_inputs(comp) -> bool:
    return any(cp.connects_system_through for cp in comp.connects_at)


def _free_ids(model, free) -> Dict[str, List[str]]:
    """``{component id: [free attrs]}`` from ``free``: a mapping of ids to
    attrs, an iterable of ids, a predicate, or None (every component's
    ``get_estimable_parameters``)."""
    if free is None:
        out = {}
        for comp in model.components.values():
            getter = getattr(comp, "get_estimable_parameters", None)
            attrs = [str(e[1]) for e in getter()] if callable(getter) else []
            if attrs:
                out[comp.id] = attrs
        return out
    if isinstance(free, Mapping):
        return {k: list(v) for k, v in free.items() if v}
    if callable(free):
        return {c.id: ["*"] for c in model.components.values() if free(c)}
    return {cid: ["*"] for cid in free}


def measured_partition(
    model,
    free=None,
    measured: Optional[Iterable] = None,
    derived: Optional[Mapping[Tuple[str, str], Tuple[SensorSystem, Optional[Callable]]]] = None,
) -> Partition:
    """The minimal groups the measurements allow (module docstring).

    ``free`` names the components with free parameters (see :func:`_free_ids`);
    it only labels the groups, the partition itself is a property of the
    wiring.  ``measured`` restricts the sensors that count (by component or
    id) to those the fit will score, so a dead duplicate does not cut an
    edge.  ``derived`` maps ``(component id, output port)`` to ``(sensor,
    transform)``: that port equals ``transform`` of the sensor's series
    (``None`` for identity) and counts as measured.
    """
    allowed = None
    if measured is not None:
        allowed = {c if isinstance(c, str) else c.id for c in measured}
    derived = dict(derived or {})
    # data sensors by the signal they read
    readers: Dict[Signal, List[SensorSystem]] = {}
    for e in _edges(model):
        if e.input_port == "measuredValue" and _is_data_sensor(e.receiver, allowed):
            readers.setdefault(e.signal, []).append(e.receiver)

    parent: Dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cid in model.components:
        find(cid)
    free_ids = _free_ids(model, free)
    crossing_candidates: List[Edge] = []
    binding: List[Edge] = []
    for e in _edges(model):
        r = e.receiver
        if isinstance(r, SensorSystem) and e.input_port == "measuredValue":
            union(e.sender.id, r.id)  # a sink (or a pass-through) stays with its source
            continue
        if not _has_inputs(e.sender) and e.sender.id not in free_ids:
            continue  # a leaf without parameters (data, a schedule, the weather): exogenous
        if _is_fusable(e):
            # A fusable arc (a zone and its wall, opening or radiator) is a
            # stiff algebraic coupling that the fused block eliminates
            # exactly; replayed, the receiver would step explicitly against
            # a lagged signal and diverge.  It binds whether or not a sensor
            # reads the signal.
            binding.append(e)
            union(e.sender.id, r.id)
            continue
        if _is_data_sensor(e.sender, allowed) and e.output_port == "measuredValue":
            # the sender is a data sensor passing its reading on: its own
            # series stands for the signal (the sensor stays with its source)
            e.sensor = e.sender
            crossing_candidates.append(e)
            continue
        sensors = readers.get(e.signal) or readers.get((e.sender.id, e.output_port, None))
        if sensors:
            e.sensor = sensors[0]
            crossing_candidates.append(e)
            continue
        key = (e.sender.id, e.output_port)
        if key in derived:
            e.sensor, e.transform = derived[key]
            crossing_candidates.append(e)
            continue
        binding.append(e)
        union(e.sender.id, r.id)

    roots: Dict[str, int] = {}
    group_of: Dict[str, int] = {}
    groups: List[List[str]] = []
    for cid in model.components:
        root = find(cid)
        if root not in roots:
            roots[root] = len(groups)
            groups.append([])
        group_of[cid] = roots[root]
        groups[roots[root]].append(cid)
    for g in groups:
        g.sort()
    crossing = [e for e in crossing_candidates if group_of[e.sender.id] != group_of[e.receiver.id]]
    internal = [e for e in crossing_candidates if group_of[e.sender.id] == group_of[e.receiver.id]]
    return Partition(groups, group_of, crossing, internal, binding, free_ids)


def replay_leaf(sensor: SensorSystem, leaf_id: str, transform: Optional[Callable] = None) -> SensorSystem:
    """A data leaf with ``sensor``'s data source (database, DataFrame or
    file), optionally with ``transform`` composed onto its unit conversion."""
    base = sensor.transformation
    if transform is None:
        transformation = base
    elif base is None:
        transformation = transform
    else:
        def transformation(x, _base=base, _t=transform):
            return _t(_base(x))
    return SensorSystem(
        id=leaf_id,
        uuid=sensor.uuid,
        dbconfig=sensor.dbconfig,
        use_database=bool(sensor.use_database),
        df=sensor.df,
        use_df=bool(sensor.use_df),
        filename=sensor.filename,
        use_spreadsheet=bool(sensor.use_spreadsheet),
        datecolumn=getattr(sensor, "_datecolumn", 0),
        valuecolumn=getattr(sensor, "_valuecolumn", 1),
        transformation=transformation,
    )


def cut_measured_edges(
    model, partition: Partition, suffix: str = "__replay", internal: bool = True
) -> List[SensorSystem]:
    """Replace every measured edge of ``partition`` by a replay leaf of its
    sensor (one leaf per sensor and transform), in place: the crossing edges
    and, with ``internal`` (the default), the measured edges inside a group
    as well (module docstring).  A pairing (sender port, receiver port)
    whose slots are not all measured is left whole and reported.  Returns
    the leaves added; call ``model.load`` afterwards."""
    by_pairing: Dict[Tuple[str, str, str, str], List[Edge]] = {}
    for e in partition.crossing + (partition.internal if internal else []):
        by_pairing.setdefault(e.pairing, []).append(e)
    all_slots: Dict[Tuple[str, str, str, str], int] = {}
    for e in _edges(model):
        all_slots[e.pairing] = all_slots.get(e.pairing, 0) + 1
    leaves: Dict[Tuple[str, int], SensorSystem] = {}
    added: List[SensorSystem] = []
    for pairing, edges in by_pairing.items():
        if all_slots.get(pairing, 0) != len(edges):
            LOGGER.warning(
                "partition: %s.%s -> %s.%s has unmeasured slots, left whole", *pairing
            )
            continue
        sender, receiver = edges[0].sender, edges[0].receiver
        model.remove_connection(sender, receiver, pairing[1], pairing[3])
        # one connection per leaf into the receiver's port, carrying every
        # slot that leaf replays (a scalar into several slots is one
        # connection with a slot tensor)
        # one leaf per (sensor, transform); a scalar output reaches one slot
        # of a port per connection, so a pairing with several slots gets one
        # leaf per slot
        per_leaf: Dict[Tuple[str, int], List[Edge]] = {}
        for e in edges:
            per_leaf.setdefault((e.sensor.id, id(e.transform)), []).append(e)
        for key, group in per_leaf.items():
            for k, e in enumerate(group):
                leaf_key = key if k == 0 else (key[0], key[1], k)
                leaf = leaves.get(leaf_key)
                if leaf is None:
                    n = sum(1 for lk in leaves if lk[0] == e.sensor.id)
                    leaf = replay_leaf(e.sensor, f"{e.sensor.id}{suffix}{n if n else ''}", e.transform)
                    leaves[leaf_key] = leaf
                    model.add_component(leaf)
                    added.append(leaf)
                kwargs = {}
                if e.input_slot is not None:
                    kwargs["input_port_index"] = e.input_slot
                model.add_connection(leaf, receiver, "measuredValue", pairing[3], **kwargs)

    LOGGER.info("partition: %d pairings cut, %d replay leaves added", len(by_pairing), len(added))
    return added


def _repair_return_junctions(model) -> List[str]:
    """Make every return junction's ``branch_temperature_slots`` follow its
    surviving connections; returns the ids of the junctions changed.

    The map names, per flow slot (a branch), the temperature slot that
    carries its temperature, and the junction sizes its ports from the
    connections it has.  With components removed the map is cut to the
    surviving branches, and a branch that lost its flow or its temperature
    points at a surviving temperature slot (a branch without a connection
    carries no flow, so the temperature it points at does not count).

    Removing a connection leaves its slot indices on the connection point,
    and a component sizes a vector port by the indices it finds there; the
    junction's are dropped with their connections, so that its ports and
    its map are sized by the same connections."""
    from twin4build.systems.junction.return_flow_junction_system import (
        ReturnFlowJunctionSystem,
    )

    connected: Dict[Tuple[str, str], Set[int]] = {}
    for e in _edges(model):
        if isinstance(e.receiver, ReturnFlowJunctionSystem) and e.input_slot is not None:
            connected.setdefault((e.receiver.id, e.input_port), set()).add(e.input_slot)
    repaired = []
    for junction in model.components.values():
        if not isinstance(junction, ReturnFlowJunctionSystem):
            continue
        slots = list(junction.branch_temperature_slots)
        branches = connected.get((junction.id, "airFlowRateIn"), set())
        temperatures = connected.get((junction.id, "airTemperatureIn"), set())
        if not slots or not branches or not temperatures:
            continue
        for cp in junction.connects_at:
            for indices in (
                cp.input_port_index, cp.output_port_index,
                cp.input_component_index, cp.output_component_index,
            ):
                for connection in [c for c in indices if c not in cp.connects_system_through]:
                    del indices[connection]
        fallback = min(temperatures)
        new = [
            slots[b] if b in branches and b < len(slots) and slots[b] in temperatures else fallback
            for b in range(max(branches) + 1)
        ]
        if new != slots:
            junction.branch_temperature_slots = new
            repaired.append(junction.id)
    return repaired


def keep_groups(
    model,
    partition: Partition,
    groups: Iterable[int],
    with_senders: bool = True,
    stop_at: Iterable[int] = (),
) -> Dict:
    """Keep the given groups of ``partition`` and remove the rest of the
    model, in place: the smaller model that has the same structure per
    group (a few rooms of a building, to fit them alone or to reproduce a
    fault on a model of a size that can be looked at).

    Kept are the components of ``groups``; with ``with_senders`` also the
    groups that send into a kept group over a measured edge
    (``partition.crossing``: a room's terminals, their controllers),
    transitively; and every leaf that feeds a kept component (data,
    schedules, the weather, the replay leaves of :func:`cut_measured_edges`):
    a leaf carries no parameters and joins no group, and without it the
    kept component has no input.  Everything else is removed, and what the
    removal breaks is repaired: a ``ReturnFlowJunctionSystem``'s
    ``branch_temperature_slots`` follow the surviving connections.

    Args:
        model: The model the partition was made of, cut
            (:func:`cut_measured_edges`) or not.
        partition: The measured partition (:func:`measured_partition`).
        groups: The indices of the groups to keep (``partition.group_of``
            gives a component's).
        with_senders: Keep the groups that send into a kept group as well.
        stop_at: Groups ``with_senders`` does not take in (unless they are
            in ``groups``): the air handling unit's group, which every room
            receives from and which receives from every room, so that
            through it a few rooms would keep the whole building.

    Returns:
        ``{"kept": [...], "removed": [...], "groups": [...],
        "repaired": [...]}``: the ids of the components kept and removed,
        the indices of the groups kept and the ids of the junctions
        repaired.  Call ``model.load`` afterwards.

    Raises:
        ValueError: If ``groups`` names a group the partition does not have.
    """
    kept_groups = {int(g) for g in groups}
    unknown = sorted(g for g in kept_groups if not 0 <= g < len(partition.groups))
    if unknown:
        raise ValueError(
            f"The partition has {len(partition.groups)} groups; there is no group {unknown}"
        )
    if with_senders:
        barred = {int(g) for g in stop_at} - kept_groups
        senders: Dict[int, Set[int]] = {}
        for e in partition.crossing:
            senders.setdefault(partition.group_of[e.receiver.id], set()).add(
                partition.group_of[e.sender.id]
            )
        frontier = set(kept_groups)
        while frontier:
            for g in senders.get(frontier.pop(), ()):
                if g not in kept_groups and g not in barred:
                    kept_groups.add(g)
                    frontier.add(g)
    keep = {cid for cid, g in partition.group_of.items() if g in kept_groups and cid in model.components}
    for cid, comp in model.components.items():
        if cid in keep or _has_inputs(comp):
            continue
        if any(
            cp.connection_point_of.id in keep
            for conn in comp.connected_through
            for cp in conn.connects_system_at
        ):
            keep.add(cid)
    removed = [cid for cid in model.components if cid not in keep]
    for cid in removed:
        model.remove_component(model.components[cid])
    repaired = _repair_return_junctions(model)
    LOGGER.info(
        "partition: %d groups kept, %d components kept, %d removed, %d junctions repaired",
        len(kept_groups), len(keep), len(removed), len(repaired),
    )
    return {
        "kept": sorted(keep),
        "removed": removed,
        "groups": sorted(kept_groups),
        "repaired": repaired,
    }
