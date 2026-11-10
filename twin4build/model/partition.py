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
schedules, replayed commands) is exogenous and joins nothing; so is any
component without free parameters that reads only data sensors' series
and other exogenous components.
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

Grouping by influence (``scored``): an unmeasured connection ties its two
ends only where the receiver's signals reach a *scored* sensor.  Two
components then share a group exactly when their parameters reach a
common scored residual through unmeasured connections; a component whose
outputs reach no scored sensor (a storey's interior read only by
measured stand-ins) no longer glues together the components that feed
it.  Without ``scored`` every unmeasured connection ties its ends (the
connected components).  A data sensor's own inputs other than
``measuredValue`` (the signals a stand-in is derived from) tie nothing:
the sensor replays its series.
"""

from __future__ import annotations

import copy as _copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


from twin4build.model import signal_dependencies as sd
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.utils.data_loaders.load import DEFAULT_DT_LIMIT
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
    #: The connection is read without its gradient (``ConnectionPoint.detached_of``; split_worlds' cuts from a
    #: provided component): the receiver reads the signal as data.
    detached: bool = False

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
    #: Unmeasured edges: inside groups by construction, except, when grouped
    #: by influence, the edges into a component that reaches no scored sensor.
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
                cut = cp.detached_of(conn)
                detached = bool(cut.any()) if hasattr(cut, "any") else bool(cut)
                for _, _, out_v, in_v in slot_pairs(cp, conn):
                    out.append(
                        Edge(comp, conn.output_port, _slot(out_v), cp.connection_point_of, cp.input_port, _slot(in_v),
                             detached=detached)
                    )
    return out


def _is_data_sensor(comp, allowed: Optional[Set[str]]) -> bool:
    if not isinstance(comp, SensorSystem) or not comp.has_data:
        return False
    return allowed is None or comp.id in allowed


def _is_fusable(e: "Edge") -> bool:
    """Whether the edge is a fusable arc: its output port in the sender's
    ``FUSABLE_OUTPUT_PORTS`` and its input port in the receiver's
    ``FUSABLE_INPUT_PORTS`` (see ``FusedStateSpaceSystem``), neither of them
    set ``fusable=False``."""
    if not getattr(e.sender, "fusable", True) or not getattr(e.receiver, "fusable", True):
        return False
    outs = getattr(type(e.sender), "FUSABLE_OUTPUT_PORTS", frozenset())
    ins = getattr(type(e.receiver), "FUSABLE_INPUT_PORTS", frozenset())
    return e.output_port in outs and e.input_port in ins


def _has_inputs(comp) -> bool:
    return any(cp.connects_system_through for cp in comp.connects_at)


def _exogenous(model, edges, free_ids, allowed=None) -> Set[str]:
    """The components that are functions of data: without free parameters, and every input of theirs a data
    sensor's series or the output of another such component.  A leaf without parameters (the weather, a schedule, a
    replayed command) is the simplest case; a component that reads only replay leaves is the next (2 Nov 2026: the
    envelope leakage at a fixed prior, read by every room, which :func:`split_worlds` evaluates once on replays).
    A signal that a sensor merely measures is still the simulated signal here: the rule never hands a component a
    measurement in place of what it reads.  Their outputs tie nothing: every group reads them as it reads data."""
    ins: Dict[str, List[Edge]] = {}
    for e in edges:
        ins.setdefault(e.receiver.id, []).append(e)

    def data_input(e: Edge) -> bool:
        # a connection read without its gradient (split_worlds' cut from a provided component) is read as data
        return e.detached or (_is_data_sensor(e.sender, allowed) and e.output_port == "measuredValue")

    out: Set[str] = set()
    pending = []
    for cid, comp in model.components.items():
        if cid in free_ids:
            continue
        if not _has_inputs(comp):
            out.add(cid)
        elif not isinstance(comp, SensorSystem):
            pending.append(cid)
    changed = True
    while changed:
        changed = False
        for cid in pending:
            if cid not in out and all(e.sender.id in out or data_input(e) for e in ins.get(cid, ())):
                out.add(cid)
                changed = True
    return out


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


@dataclass
class _Classified:
    """Every edge of a model classified for the partition (:func:`_classify`)."""

    #: (sender id, sensor id): a data sensor stays with the signal it reads.
    sinks: List[Tuple[str, str]]
    #: Unmeasured edges (fusable arcs among them).
    binding: List[Edge]
    fusable: List[Edge]
    #: Measured edges (by a sensor or ``derived``): what a cut replays.
    measured: List[Edge]
    readers: Dict[Signal, List[SensorSystem]]
    free_ids: Dict[str, List[str]]
    scored_ids: Optional[Set[str]]
    #: Components whose signals reach a scored sensor (grouping by influence; ``None`` without ``scored``).
    reaches: Optional[Set[str]]


def _classify(model, free=None, measured=None, derived=None, scored=None) -> _Classified:
    """The edges of ``model`` by what they do to the partition (arguments as :func:`measured_partition`)."""
    allowed = None
    if measured is not None:
        allowed = {c if isinstance(c, str) else c.id for c in measured}
    scored_ids = None if scored is None else {c if isinstance(c, str) else c.id for c in scored}
    derived = dict(derived or {})
    edges = _edges(model)
    # data sensors by the signal they read
    readers: Dict[Signal, List[SensorSystem]] = {}
    for e in edges:
        if e.input_port == "measuredValue" and _is_data_sensor(e.receiver, allowed):
            readers.setdefault(e.signal, []).append(e.receiver)
    free_ids = _free_ids(model, free)
    exogenous = _exogenous(model, edges, free_ids, allowed)
    sinks: List[Tuple[str, str]] = []
    measured_edges: List[Edge] = []
    binding: List[Edge] = []
    fusable: List[Edge] = []
    for e in edges:
        r = e.receiver
        if isinstance(r, SensorSystem) and e.input_port == "measuredValue":
            sinks.append((e.sender.id, r.id))  # a sink (or a pass-through) stays with its source
            continue
        if e.detached:
            continue  # read without its gradient (split_worlds' cut from a provided component): as data, ties nothing
        if isinstance(r, SensorSystem) and r.has_data:
            continue  # a data sensor replays its series: its other inputs tie nothing
        if e.sender.id in exogenous:
            continue  # a function of data without parameters (a schedule, the weather, ...): exogenous
        if _is_fusable(e):
            # A fusable arc (a zone and its wall, opening or radiator) is a
            # stiff algebraic coupling that the fused block eliminates
            # exactly; replayed, the receiver would step explicitly against
            # a lagged signal and diverge.  It binds whether or not a sensor
            # reads the signal.
            binding.append(e)
            fusable.append(e)
            continue
        if _is_data_sensor(e.sender, allowed) and e.output_port == "measuredValue":
            # the sender is a data sensor passing its reading on: its own
            # series stands for the signal (the sensor stays with its source)
            e.sensor = e.sender
            measured_edges.append(e)
            continue
        sensors = readers.get(e.signal) or readers.get((e.sender.id, e.output_port, None))
        if sensors:
            e.sensor = sensors[0]
            measured_edges.append(e)
            continue
        key = (e.sender.id, e.output_port)
        slot_key = (e.sender.id, e.output_port, e.output_slot)
        if slot_key in derived or key in derived:
            e.sensor, e.transform = derived[slot_key] if slot_key in derived else derived[key]
            measured_edges.append(e)
            continue
        binding.append(e)
    reaches = None
    if scored_ids is not None:
        # Grouping by influence: an unmeasured edge ties its ends where the
        # receiver reaches a scored sensor (through unmeasured edges, both
        # ways along fusable arcs, which are one block).
        upstream: Dict[str, List[str]] = {}
        for e in binding:
            upstream.setdefault(e.receiver.id, []).append(e.sender.id)
        for e in fusable:
            upstream.setdefault(e.sender.id, []).append(e.receiver.id)
        reaches = set()
        frontier = []
        for e in edges:
            if e.input_port == "measuredValue" and e.receiver.id in scored_ids and e.sender.id not in reaches:
                reaches.add(e.sender.id)
                frontier.append(e.sender.id)
        while frontier:
            cid = frontier.pop()
            for up in upstream.get(cid, ()):
                if up not in reaches:
                    reaches.add(up)
                    frontier.append(up)
    return _Classified(sinks, binding, fusable, measured_edges, readers, free_ids, scored_ids, reaches)


def _ties(c: _Classified, e: Edge) -> bool:
    """Whether the unmeasured edge ``e`` ties its two ends into one group."""
    if _is_fusable(e) or c.reaches is None:
        return True
    return e.receiver.id in c.reaches


def _union_find(ids: Iterable[str]):
    """``(find, union)`` over ``ids``."""
    parent: Dict[str, str] = {cid: cid for cid in ids}

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

    return find, union


def measured_partition(
    model,
    free=None,
    measured: Optional[Iterable] = None,
    derived: Optional[Mapping[tuple, Tuple[SensorSystem, Optional[Callable]]]] = None,
    scored: Optional[Iterable] = None,
) -> Partition:
    """The minimal groups the measurements allow (module docstring).

    ``free`` names the components with free parameters (see :func:`_free_ids`);
    it only labels the groups, the partition itself is a property of the
    wiring.  ``measured`` restricts the sensors that count (by component or
    id) to those the fit will score, so a dead duplicate does not cut an
    edge.  ``derived`` maps ``(component id, output port)`` -- or ``(component
    id, output port, output slot)`` for one slot of a vector output -- to
    ``(sensor, transform)``: that port equals ``transform`` of the sensor's
    series (``None`` for identity) and counts as measured.  ``scored`` (by
    component or id) groups by influence (module docstring): the sensors
    whose residuals the fit scores; ``measured`` still decides what is cut.
    """
    c = _classify(model, free, measured, derived, scored)
    find, union = _union_find(model.components)
    for a, b in c.sinks:
        union(a, b)
    for e in c.binding:
        if _ties(c, e):
            union(e.sender.id, e.receiver.id)

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
    crossing = [e for e in c.measured if group_of[e.sender.id] != group_of[e.receiver.id]]
    internal = [e for e in c.measured if group_of[e.sender.id] == group_of[e.receiver.id]]
    return Partition(groups, group_of, crossing, internal, c.binding, c.free_ids)


@dataclass
class Worlds:
    """What :func:`split_worlds` did: the components it duplicated and the parameters their copies share."""

    #: ``{component id: [ids of its copies]}`` (the component keeps one world, each copy another).
    copies: Dict[str, List[str]] = field(default_factory=dict)
    #: ``{(component id, attr): [(component or copy id, attr), ...]}``: the copies of each free parameter of a
    #: duplicated component, the component's own first.  A fit keeps them in agreement (consensus).
    shared: Dict[Tuple[str, str], List[Tuple[str, str]]] = field(default_factory=dict)
    #: Components that more than one world reads but that an unmeasured input from another world keeps whole, and
    #: components whose class joins the worlds of what they connect (``JOINS_WORLDS``):
    #: ``{component id: the worlds' representative components merged for it}``.
    merged: Dict[str, List[str]] = field(default_factory=dict)
    #: Components that several worlds read but that have no free parameters and read nothing from any world:
    #: evaluated once, on the replays of what they read (exogenous afterwards), not copied.
    once: List[str] = field(default_factory=list)
    #: ``{component id: the ids of its projections}``: a component with a state whose state-free outputs another world
    #: needs, computed there by the class's projection (``OUTPUT_PROJECTIONS``) from that world's members.
    projections: Dict[str, List[str]] = field(default_factory=dict)
    #: ``{component id: [(output port, reader id, input port), ...]}``: a provided component (:func:`split_worlds`),
    #: evaluated in its home world only; every other world that needs it reads its lagged outputs through these cut
    #: connections, without their gradient (``Model.add_connection(..., detached=True)``).
    provided: Dict[str, List[str]] = field(default_factory=dict)


def _clone(model, comp, new_id: str):
    """A copy of ``comp`` (its ports, parameters and sub-models) under ``new_id``, wired to nothing: everything
    else in ``model`` (the other components, the model itself) is referred to, never copied."""

    memo = {id(other): other for other in model.components.values() if other is not comp}
    memo[id(model)] = model
    for name in ("simulation_model", "semantic_model"):
        if getattr(model, name, None) is not None:
            memo[id(getattr(model, name))] = getattr(model, name)
    through, at = comp.connected_through, comp.connects_at
    comp.connected_through, comp.connects_at = [], []
    try:
        clone = _copy.deepcopy(comp, memo)
    finally:
        comp.connected_through, comp.connects_at = through, at
    clone.connected_through, clone.connects_at = [], []
    clone.id = new_id
    return clone


def split_worlds(
    model,
    free=None,
    measured: Optional[Iterable] = None,
    derived: Optional[Mapping[tuple, Tuple[SensorSystem, Optional[Callable]]]] = None,
    scored: Optional[Iterable] = None,
    replay_suffix: str = "__world_replay",
) -> Worlds:
    """Give each group of the partition its own world (in place, before :func:`measured_partition`).

    A world is fitted with its own states simulated and everything outside it measured.  The worlds follow signals,
    not components (:mod:`twin4build.model.signal_dependencies`): from each scored sensor the walk goes upstream,
    output by output, through what each output depends on (its inputs, slot by slot where the instance says so, and,
    when it reads the component's state, what the state update reads), and stops at a measured or exogenous signal.

    * A component with scored sensors of its own is its world's.  A world that needs a signal of it that depends on
      its state (unmeasured) cannot be told apart from it: the two merge (:attr:`Worlds.merged` names the
      component and the worlds' members).  A signal that depends on neither its state nor a parameter is computed
      in the other world by the class's projection (``OUTPUT_PROJECTIONS``), from that world's copies of what it
      reads; without a projection the worlds merge.
    * Any other component is evaluated once per world that needs it: the original in one, a copy (:func:`_clone`)
      in each other, state and all (a wall between two measured rooms), or the projection where only state-free
      outputs are needed.  The free parameters the copied signals depend on are one parameter evaluated in several
      worlds: :attr:`Worlds.shared` lists their copies for the fit to keep in agreement (consensus).
    * A component without free parameters that reads nothing from any world (only measured or exogenous signals)
      is not copied: every world would evaluate the same function of measurements, so it is evaluated once on their
      replays (:attr:`Worlds.once`) and is exogenous afterwards.
    * A component with free parameters whose outputs the other worlds read are all lagged (its class's
      ``LAGGED_OUTPUT_PORTS``: the start-of-step state a hub publishes, a building's hidden interior node) and that
      one world needs without reading anything of that world's own -- there every path upstream of its state ends at
      a measured or exogenous signal -- is provided (7 Nov 2026): it lives in that world (its home), on those
      measurements, and every other world reads its lagged outputs through a cut connection, without their gradient
      (``detached=True``, :attr:`Worlds.provided`), instead of copying it.  It is simulated once and fitted to its home's measurements; its readers' residuals
      move only their own parameters.  A wall or a radiator (outputs not lagged) keeps its copies.
    * A component whose class sets :attr:`~twin4build.systems.saref4syst.system.System.JOINS_WORLDS` (an opening
      between the zones of one room) joins the worlds of the scored spaces it reads or feeds: they are one world
      (:attr:`Worlds.merged` names it), so it is neither copied nor fed a measurement of either side.  Its coupling
      is free and can be strong; a measured side would act as a reservoir that the fit can push and pull without
      cost.

    Every member reads from its own world, from measurements (cut by :func:`cut_measured_edges` as usual) or from
    exogenous signals; the slots of a copy's vector inputs that its world does not need read zero, so a copy keeps
    its original's widths (and batches with it).  Which components these are follows from the wiring, the
    measurements and the classes' declarations; no class is named.  Arguments as :func:`measured_partition`; ``free``
    also decides which parameters are shared.  Call :func:`measured_partition` on the model afterwards.

    The model records the result (``model.worlds``): a multiple-shooting fit of it starts every component and its
    copies from one box and one start, as they are one state.
    """

    c = _classify(model, free, measured, derived, scored)
    edges = _edges(model)
    binding_keys = {_edge_key(e) for e in c.binding}
    measured_keys = {_edge_key(e) for e in c.measured}
    binding_in: Dict[str, Dict[str, List[Edge]]] = {}
    for e in c.binding:
        binding_in.setdefault(e.receiver.id, {}).setdefault(e.input_port, []).append(e)
    in_edges: Dict[str, List[Edge]] = {}
    for e in edges:
        in_edges.setdefault(e.receiver.id, []).append(e)

    # the seeds: each scored sensor's source, with the signals it scores
    seeds: Dict[str, Set[Tuple[str, Optional[int]]]] = {}
    for e in edges:
        if e.input_port != "measuredValue" or not isinstance(e.receiver, SensorSystem):
            continue
        if not _is_data_sensor(e.receiver, None if measured is None else
                               {m if isinstance(m, str) else m.id for m in measured}):
            continue
        if c.scored_ids is not None and e.receiver.id not in c.scored_ids:
            continue
        if isinstance(e.sender, SensorSystem):
            continue
        seeds.setdefault(e.sender.id, set()).add((e.output_port, e.output_slot))

    table_cache: Dict[str, Dict[str, "sd.Dependency"]] = {}

    def deps(cid: str) -> Dict[str, "sd.Dependency"]:
        if cid not in table_cache:
            table_cache[cid] = sd.declared(model.components[cid])
        return table_cache[cid]

    def dependency(cid: str, port: str) -> "sd.Dependency":
        table = deps(cid)
        return table[port] if port in table else sd.everything(model.components[cid])

    def projectable(cid: str, port: str) -> bool:
        comp = model.components[cid]
        return not sd.has_state(comp) or sd.projection(comp, port) is not None

    def upstream(cid: str, port: str, slot: Optional[int], dep: "sd.Dependency"):
        """The binding edges an output (slot) of ``cid`` reads."""
        comp = model.components[cid]
        for inp in dep.inputs:
            allowed = None if port == sd.STATE or slot is None else sd.slots(comp, port, inp)
            for e in binding_in.get(cid, {}).get(inp, ()):
                if allowed is not None and e.input_slot is not None and e.input_slot not in allowed.get(slot, ()):
                    continue
                yield e

    def measured_signal(cid: str, port: str, slot: Optional[int]) -> bool:
        return bool(c.readers.get((cid, port, slot)) or c.readers.get((cid, port, None)))

    find, union = _union_find(seeds)
    merged_for: Dict[str, Set[str]] = {}

    # A component whose class joins the worlds of the spaces it connects (``JOINS_WORLDS``) is never split: those
    # spaces are one world, so it reads none of them as a measurement.  8 Nov 2026: an opening between the zones of one
    # room exchanges air freely; with each zone in its own world reading the other's measured CO2, its exchange summed
    # over the zones was -m_x/2 (bias_a + bias_b), a sink no building has, and a fit pushed m_x to its bound and the
    # zones' CO2 generation up with it (a large building's room: +150 ppm once both zones were simulated).
    out_edges: Dict[str, List[Edge]] = {}
    for e in edges:
        out_edges.setdefault(e.sender.id, []).append(e)
    for comp in model.components.values():
        if not type(comp).JOINS_WORLDS:
            continue
        sides = sorted({e.sender.id for e in in_edges.get(comp.id, ()) if e.sender.id in seeds}
                       | {e.receiver.id for e in out_edges.get(comp.id, ()) if e.receiver.id in seeds})
        if len(sides) < 2:
            continue
        merged_for.setdefault(comp.id, set()).update(sides)
        for other in sides[1:]:
            union(sides[0], other)

    def walk(provided: Mapping[str, str]) -> Dict[str, Dict[str, Dict]]:
        """``needs[world][cid] = {"ports": {(port, slot)}, "state": bool}``: what each world simulates, merging the
        worlds that cannot be told apart; a provided component (``{cid: home}``) is needed by its home only."""
        while True:
            needs: Dict[str, Dict[str, Dict]] = {}
            merges: List[Tuple[str, str, str, str]] = []
            for seed in sorted(seeds):
                w = find(seed)
                world = needs.setdefault(w, {})
                stack = [(seed, port, slot, seed) for port, slot in sorted(seeds[seed], key=lambda t: (t[0], -1 if t[1] is None else t[1]))]
                seen: Set[Tuple[str, str, Optional[int]]] = set()
                while stack:
                    cid, port, slot, reader = stack.pop()
                    if (cid, port, slot) in seen:
                        continue
                    seen.add((cid, port, slot))
                    if cid in provided and provided[cid] != w:
                        continue  # read through a cut connection from its home
                    dep = dependency(cid, port)
                    foreign = cid in seeds and find(cid) != w
                    if foreign and port != sd.STATE and measured_signal(cid, port, slot):
                        continue  # another world's measured signal (a fusable arc's far room): replayed in this world
                    if foreign and (dep.state or (port != sd.STATE and not projectable(cid, port))):
                        merges.append((w, find(cid), reader, port))
                        continue
                    state_needed = dep.state or (port != sd.STATE and not projectable(cid, port))
                    entry = world.setdefault(cid, {"ports": set(), "state": False})
                    entry["ports"].add((port, slot))
                    if state_needed and not foreign:
                        entry["state"] = True
                        if port != sd.STATE:
                            stack.append((cid, sd.STATE, None, cid))
                    for e in upstream(cid, port, slot, dep):
                        stack.append((e.sender.id, e.output_port, e.output_slot, cid))
            if not merges:
                return needs
            for a, b, reader, _port in merges:
                if find(a) != find(b):
                    merged_for.setdefault(reader, set()).update({a, b})
                    union(a, b)

    def by_component(needs) -> Dict[str, Dict[str, Dict]]:
        out: Dict[str, Dict[str, Dict]] = {}
        for w, comps in needs.items():
            for cid, entry in comps.items():
                out.setdefault(cid, {})[w] = entry
        return out

    def anchor(w: str) -> str:
        """A world's name in its copies' ids: its smallest seed."""
        return min(s for s in seeds if find(s) == w)

    def lagged_ports(cid: str) -> Set[str]:
        comp = model.components[cid]
        decide = getattr(comp, "lagged_output_ports", None)
        return set(decide() if callable(decide) else getattr(type(comp), "LAGGED_OUTPUT_PORTS", frozenset()))

    def reads_own(cid: str, w: str) -> bool:
        """Whether ``cid``'s state, in world ``w``, reads anything simulated there: a seed of ``w`` upstream of it
        through unmeasured signals (or another world's unmeasured state, which would merge).  Another world's
        measured signal is replayed; its state-free output (a box's draw) is computed in ``w`` from its inputs, as
        :func:`walk` does, and the walk goes on through them."""
        stack = [(cid, sd.STATE, None)]
        seen: Set[Tuple[str, str, Optional[int]]] = set()
        while stack:
            x, port, slot = stack.pop()
            if (x, port, slot) in seen:
                continue
            seen.add((x, port, slot))
            dep = dependency(x, port)
            if x in seeds:
                if find(x) == w:
                    return True
                if port != sd.STATE and measured_signal(x, port, slot):
                    continue  # another world's measured signal: replayed
                if port == sd.STATE or dep.state or not projectable(x, port):
                    return True  # another world's unmeasured state: the worlds would merge
                for e in upstream(x, port, slot, dep):  # its state-free output, computed here from its inputs
                    stack.append((e.sender.id, e.output_port, e.output_slot))
                continue
            if port != sd.STATE and (dep.state or not projectable(x, port)):
                stack.append((x, sd.STATE, None))
            for e in upstream(x, port, slot, dep):
                stack.append((e.sender.id, e.output_port, e.output_slot))
        return False

    def providers(worlds_of) -> Dict[str, str]:
        """``{cid: home world}`` of the provided components (docstring)."""
        out: Dict[str, str] = {}
        for cid in sorted(worlds_of):
            by_world = worlds_of[cid]
            lagged = lagged_ports(cid)
            if cid in seeds or cid not in c.free_ids or not lagged or len(by_world) < 2:
                continue
            homes = [w for w in sorted(by_world, key=anchor) if not reads_own(cid, w)]
            if not homes:
                continue
            read_ports = {p for w, entry in by_world.items() if w not in homes for p, _ in entry["ports"] if p != sd.STATE}
            if not read_ports <= lagged:
                continue
            # every world that reads it only through measurements is its home: they are one world with it (the
            # units' return air and the atrium's heating circuit both measure a building's hidden interior)
            for other in homes[1:]:
                if find(other) != find(homes[0]):
                    merged_for.setdefault(cid, set()).update({homes[0], other})
                    union(homes[0], other)
            out[cid] = find(homes[0])
        return out

    needs = walk({})
    provided = providers(by_component(needs))
    if provided:
        needs = walk(provided)
    result = Worlds()
    worlds_of = by_component(needs)

    leaves: Dict[str, SensorSystem] = {}
    # the leaves this split adds (replays, the zero) by id -> the id of their family's first; a scalar leaf reaches
    # one slot of a port per connection, so a second slot of the same port takes a sibling (as cut_measured_edges)
    family: Dict[str, str] = {}
    makers: Dict[str, Callable[[str], object]] = {}
    wired: Set[Tuple[str, str, str, str]] = set()

    def new_leaf(leaf_id: str, make: Callable[[str], object]):
        if leaf_id not in model.components:
            model.add_component(make(leaf_id))
            family[leaf_id] = leaf_id
            makers[leaf_id] = make
        return model.components[leaf_id]

    def zero_leaf():

        return new_leaf("__world_zero", lambda i: ScheduleSystem(weekday_ruleset={"ruleset_default_value": 0.0}, id=i))

    def replay_of(sensor):
        leaf_id = f"{sensor.id}{replay_suffix}"
        leaf = new_leaf(leaf_id, lambda i, _s=sensor: replay_leaf(_s, i))
        leaves[leaf_id] = leaf
        return leaf

    def wire(sender, receiver, e: Edge, output_port: Optional[str] = None, detached: bool = False):
        kwargs = {"detached": True} if detached else {}
        out_port = output_port or e.output_port
        if (sender.id, receiver.id, out_port, e.input_port) in wired and sender.id in family:
            base, k = family[sender.id], 1
            while (f"{base}{k}", receiver.id, out_port, e.input_port) in wired:
                k += 1
            sibling_id = f"{base}{k}"
            if sibling_id not in model.components:
                model.add_component(makers[base](sibling_id))
                family[sibling_id] = base
            sender = model.components[sibling_id]
        wired.add((sender.id, receiver.id, out_port, e.input_port))
        out_spec = getattr(sender, "output", {}).get(out_port)
        if e.output_slot is not None and getattr(out_spec, "__class__", None).__name__ == "Vector":
            kwargs["output_port_index"] = e.output_slot
        if e.input_slot is not None:
            kwargs["input_port_index"] = e.input_slot
        model.add_connection(sender, receiver, out_port, e.input_port, **kwargs)

    # once: no free parameters, nothing read from any world
    for cid in sorted(worlds_of):
        comp = model.components[cid]
        if cid in seeds or cid in c.free_ids or binding_in.get(cid) or len(worlds_of[cid]) < 2:
            continue
        replayed = [e for e in in_edges.get(cid, ()) if _edge_key(e) in measured_keys]
        sensors = [c.readers.get(e.signal) or c.readers.get((e.sender.id, e.output_port, None)) for e in replayed]
        if any(not s for s in sensors):
            continue  # a derived input keeps the copies
        done = set()
        for e, s in zip(replayed, sensors):
            if e.pairing not in done:
                model.remove_connection(e.sender, comp, e.output_port, e.input_port)
                done.add(e.pairing)
            wire(replay_of(s[0]), comp, e, output_port="measuredValue")
        result.once.append(cid)
        del worlds_of[cid]

    once = set(result.once)

    # members: who holds the original, who gets a copy or a projection
    members: Dict[str, Dict[str, object]] = {}
    kind: Dict[Tuple[str, str], str] = {}  # (cid, world) -> "original" | "copy" | "projection"
    for cid in sorted(worlds_of):
        comp = model.components[cid]
        by_world = worlds_of[cid]
        order = sorted(by_world, key=anchor)
        if cid in seeds:
            home = find(cid)
        else:
            stateful_worlds = [w for w in order if by_world[w]["state"]]
            home = (stateful_worlds or order)[0]
        members[cid] = {home: comp}
        kind[(cid, home)] = "original"
        for w in order:
            if w == home:
                continue
            entry = by_world[w]
            ports = {p for p, _ in entry["ports"] if p != sd.STATE}
            if sd.has_state(comp) and not entry["state"] and ports:
                classes = {sd.projection(comp, p) for p in ports}
                if len(classes) == 1 and None not in classes:
                    member = next(iter(classes))(id=f"{cid}__projection_in_{anchor(w)}")
                    model.add_component(member)
                    members[cid][w] = member
                    kind[(cid, w)] = "projection"
                    result.projections.setdefault(cid, []).append(member.id)
                    continue
            member = _clone(model, comp, f"{cid}__in_{anchor(w)}")
            model.add_component(member)
            members[cid][w] = member
            kind[(cid, w)] = "copy"
            result.copies.setdefault(cid, []).append(member.id)

    def foreign_measurement(e: Edge, w: str):
        """The replay leaf for a measured input whose sender is another world's member, else None."""
        if _edge_key(e) not in measured_keys and not (_edge_key(e) in binding_keys and measured_signal(
                e.sender.id, e.output_port, e.output_slot)):
            return None
        held = members.get(e.sender.id)
        if held is None:
            return None
        if w in held and e.output_port in getattr(held[w], "output", {}):
            return None  # this world's own member gives it
        # (a projection in this world has the state-free ports only: the measured one is replayed)
        sensors = c.readers.get(e.signal) or c.readers.get((e.sender.id, e.output_port, None))
        if not sensors:
            return None  # derived: the cut replays it
        return replay_of(sensors[0])

    for cid in provided:
        result.provided[cid] = []

    def cut_from(e: Edge, receiver):
        """A provided component read outside its home: the same connection, cut (without its gradient)."""
        wire(e.sender, receiver, e, detached=True)
        entry = (e.output_port, receiver.id, e.input_port)
        if entry not in result.provided[e.sender.id]:
            result.provided[e.sender.id].append(entry)

    # wiring: every member reads its own world's members, measurements and exogenous signals; a provided component
    # outside its home, through a cut connection
    for cid in sorted(members):
        comp = model.components[cid]
        for w, member in members[cid].items():
            k = kind[(cid, w)]
            ins = in_edges.get(cid, ())
            if k == "original":
                removed, keep = set(), []
                for e in ins:
                    leaf = foreign_measurement(e, w)
                    if leaf is not None:
                        if e.pairing not in removed:
                            keep.extend(x for x in ins if x.pairing == e.pairing and x is not e)
                            model.remove_connection(e.sender, comp, e.output_port, e.input_port)
                            removed.add(e.pairing)
                        wire(leaf, comp, e, output_port="measuredValue")
                        continue
                    if _edge_key(e) not in binding_keys or e.sender.id in once:
                        continue  # measured (the cut replays it), exogenous, or evaluated once: as it is
                    source = members.get(e.sender.id, {}).get(w)
                    if source is e.sender:
                        continue  # same world, the original: as it is
                    if e.pairing not in removed:
                        # rewire the whole connection: its other slots are put back below
                        keep.extend(x for x in ins if x.pairing == e.pairing and x is not e)
                        model.remove_connection(e.sender, comp, e.output_port, e.input_port)
                        removed.add(e.pairing)
                    if source is not None:
                        wire(source, comp, e)
                    elif e.sender.id in provided:
                        cut_from(e, comp)
                    elif e.input_slot is not None:
                        wire(zero_leaf(), comp, e, output_port="scheduleValue")
                for x in keep:
                    if (_edge_key(x) in binding_keys and x.sender.id not in once) or foreign_measurement(x, w) is not None:
                        continue  # handled above (each such slot on its own)
                    wire(x.sender, comp, x)
                continue
            for e in ins:
                if k == "projection" and e.input_port not in member.input:
                    continue
                leaf = foreign_measurement(e, w)
                if leaf is not None:  # another world's measured signal: its replay
                    wire(leaf, member, e, output_port="measuredValue")
                elif _edge_key(e) in binding_keys and e.sender.id not in once:
                    source = members.get(e.sender.id, {}).get(w)
                    if source is not None:
                        wire(source, member, e)
                    elif e.sender.id in provided:
                        cut_from(e, member)
                    elif e.input_slot is not None:
                        wire(zero_leaf(), member, e, output_port="scheduleValue")
                else:  # measured (the cut replays it) or exogenous: as it is
                    wire(e.sender, member, e)
        copies = [m.id for w, m in members[cid].items() if kind[(cid, w)] == "copy"]
        if not copies:
            continue
        needed_params: Set[str] = set()
        for w, entry in worlds_of[cid].items():
            for port, _ in entry["ports"]:
                if port != sd.STATE:
                    needed_params |= set(dependency(cid, port).parameters)
            if entry["state"]:
                needed_params |= set(dependency(cid, sd.STATE).parameters)
        for attr in c.free_ids.get(cid, ()):
            if attr == "*":
                getter = getattr(comp, "get_estimable_parameters", None)
                attrs = [str(p[1]) for p in getter()] if callable(getter) else []
            else:
                attrs = [attr]
            for a in attrs:
                if a in needed_params or a not in sd.parameter_names(comp):
                    result.shared[(cid, a)] = [(cid, a)] + [(copy_id, a) for copy_id in copies]
    result.merged = {cid: sorted(ids) for cid, ids in merged_for.items()}
    LOGGER.info("partition: %d components split into worlds (%d copies, %d projections), %d shared parameters, %d merged, "
                "%d provided (%d cut connections)",
                len(result.copies), sum(len(v) for v in result.copies.values()),
                sum(len(v) for v in result.projections.values()), len(result.shared), len(result.merged),
                len(result.provided), sum(len(v) for v in result.provided.values()))
    model.worlds = result
    return result


def _edge_key(e: Edge) -> tuple:
    return (e.sender.id, e.output_port, e.output_slot, e.receiver.id, e.input_port, e.input_slot)


def _edges_between(sender, receiver, conn, cp) -> List[Edge]:
    """The slot pairs of one connection as edges."""
    return [Edge(sender, conn.output_port, _slot(out_v), receiver, cp.input_port, _slot(in_v))
            for _, _, out_v, in_v in slot_pairs(cp, conn)]


def replay_leaf(sensor: SensorSystem, leaf_id: str, transform: Optional[Callable] = None) -> SensorSystem:
    """A data leaf with ``sensor``'s reader (database, DataFrame or file),
    optionally with ``transform`` composed onto its unit conversion."""
    base = sensor.transformation
    if transform is None:
        transformation = base
    elif base is None:
        transformation = transform
    else:
        def transformation(x, _base=base, _t=transform):
            return _t(_base(x))
    return SensorSystem(id=leaf_id, uuid=sensor.uuid, source=sensor.reader, transformation=transformation)


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
    """Give every flow slot of a return junction that lost its temperature
    (the room was removed, its terminal kept) the temperature of the lowest
    surviving temperature slot, by a connection; returns the ids of the
    junctions changed.

    Removing a connection leaves its slot indices on the connection point,
    and the junction sizes its ports by the indices it finds there; those of
    removed connections are dropped first."""
    from twin4build.systems.junction.return_flow_junction_system import (
        ReturnFlowJunctionSystem,
    )

    repaired = []
    for junction in model.components.values():
        if not isinstance(junction, ReturnFlowJunctionSystem):
            continue
        for cp in junction.connects_at:
            for indices in (
                cp.input_port_index, cp.output_port_index,
                cp.input_component_index, cp.output_component_index,
            ):
                for connection in [c for c in indices if c not in cp.connects_system_through]:
                    del indices[connection]
        edges = [e for e in _edges(model) if e.receiver is junction and e.input_slot is not None]
        branches = {e.input_slot for e in edges if e.input_port == "airFlowRateIn"}
        temperatures = {e.input_slot: e for e in edges if e.input_port == "airTemperatureIn"}
        if not branches or not temperatures:
            continue
        source = temperatures[min(temperatures)]
        missing = sorted(b for b in branches if b not in temperatures)
        for b in missing:
            kwargs = {"input_port_index": b}
            if source.output_slot is not None:
                kwargs["output_port_index"] = source.output_slot
            model.add_connection(source.sender, junction, source.output_port, "airTemperatureIn", **kwargs)
        if missing:
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
    removal breaks is repaired: a ``ReturnFlowJunctionSystem``'s branch whose
    room was removed takes a surviving room's temperature, by a connection.

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
    # the functions of data that feed what is kept (a leaf, or a component that reads only replays), and theirs
    exogenous = _exogenous(model, _edges(model), partition.free)
    exogenous |= {cid for cid, comp in model.components.items() if not _has_inputs(comp)}  # every leaf, as before
    changed = True
    while changed:
        changed = False
        for cid in exogenous:
            if cid in keep or cid not in model.components:
                continue
            if any(
                cp.connection_point_of.id in keep
                for conn in model.components[cid].connected_through
                for cp in conn.connects_system_at
            ):
                keep.add(cid)
                changed = True
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
