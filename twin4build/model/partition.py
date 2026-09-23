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
columns); a leaf with no inputs (weather, schedules, replayed commands) is
exogenous and joins nothing.  :func:`cut_measured_edges` then replaces the
measured connections that cross between groups by replay leaves, one per
sensor, and leaves the connections inside a group as they are, so the
physics inside a group is untouched.  Afterwards the estimator's own
structure walk (``FunctionalModel.index_coupling``) finds one block per
group that carries parameters.

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
    #: Measured edges between two groups: what :func:`cut_measured_edges` replaces.
    crossing: List[Edge]
    #: Measured edges inside a group: left alone.
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


def cut_measured_edges(model, partition: Partition, suffix: str = "__replay") -> List[SensorSystem]:
    """Replace every crossing measured edge of ``partition`` by a replay leaf
    of its sensor (one leaf per sensor and transform), in place.  A pairing
    (sender port, receiver port) whose slots are not all measured is left
    whole and reported.  Returns the leaves added; call ``model.load``
    afterwards."""
    by_pairing: Dict[Tuple[str, str, str, str], List[Edge]] = {}
    for e in partition.crossing:
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
