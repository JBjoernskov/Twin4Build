"""Functional-map simulation: component ``forward``s fused into one pure step.

The object-graph engine (:meth:`Simulator.simulate`) steps every component's
``do_step`` through the ports -- flexible, but each step pays Python dispatch
(Gauss-Seidel traversal, ``tps`` port bookkeeping, history logging) and each
run pays ``model.initialize``.  Since every composable component's ``do_step``
is a thin port-I/O wrapper that DELEGATES its math to a pure ``forward``
method (single source of truth -- see the developer reference's
``do_step``/``forward`` contract), the same model can also be simulated as a
plain sequential torch rollout of one functional function:

    y_{t+1}, meas_t = F_aug(y_t, theta, EXOGENOUS[t])

This module provides that machinery, consumed through the
:class:`~twin4build.simulator.simulator.Simulator` facade
(``build_functional_model`` / ``record_exogenous_inputs`` /
``rollout_functional``) by

* the Estimator's single-shooting objective
  (:mod:`twin4build.estimator._single_shooting`),
* the Estimator's collocation transcription
  (:mod:`twin4build.estimator._collocation`, which consumes ``F`` under
  ``vmap(jacrev(...))`` for the sparse NLP Jacobian), and
* the Optimizer's control objective
  (:mod:`twin4build.optimizer._single_shooting`).

**How the functional assembly works.** :class:`FunctionalModel` analyzes the model
graph once and threads the ``forward`` methods in execution order.  For each
input port of a functional component it uses one of

* a **fresh** value -- the output of an upstream component functional earlier in
  the same step (pass-through sensors followed to their source), so parameter
  and state couplings are exact;
* a **feedback** value -- a cut cycle edge (the producer executes *later* in
  the Gauss-Seidel order, e.g. ``office.heatGain <- space_heater.Power``).
  These are one-step *lag variables*: :meth:`FunctionalModel.F_aug` appends
  them to the state, reproducing ``do_step``'s one-step-delayed feedback
  semantics exactly;
* an **exogenous** constant -- truly exogenous drivers (weather, schedules,
  data-driven occupancy), frozen from one reference ``do_step`` rollout
  (:func:`record_exogenous_inputs`).  Exogenous signals are independent of
  the unknowns by definition, so capturing once is valid for every parameter
  or control iterate (callers that *do* optimize an exogenous trajectory --
  the Optimizer -- override the corresponding exogenous slots per step).

The functional map is pure and functorch-traceable: sequential rollouts
differentiate with plain autograd, and collocation maps ``vmap(jacrev(F))``
over segments.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.utils.rgetattr import rgetattr


def _content_signature(value):
    """Return an exact, device-independent signature for tensor-like data."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous()
        raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
        return (
            tuple(tensor.shape),
            str(tensor.dtype),
            hashlib.sha256(raw).digest(),
        )
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            tuple(array.shape),
            str(array.dtype),
            hashlib.sha256(array.tobytes()).digest(),
        )
    return None


def recording_data_signature(model):
    """Fingerprint initialized values that can affect an exogenous recording.

    This intentionally covers populated/leaf port histories, component
    parameters and direct tensor-like data publishers (including ``.values``
    wrappers used by time-series systems). It lets repeated solves reuse a
    recording while invalidating on changed periods, source data, parameters,
    or initialized state.
    """
    sim_model = getattr(model, "_simulation_model", None) or model
    components = dict(model.components)
    components.update(getattr(sim_model, "_fused_components", None) or {})
    signature = []
    content_by_identity = {}

    def append_value(component_id, name, value):
        identity = id(value)
        if identity in content_by_identity:
            content = content_by_identity[identity]
        else:
            content = _content_signature(value)
            content_by_identity[identity] = content
        if content is None:
            return
        signature.append((component_id, name, content))

    for component_id, component in sorted(components.items()):
        for direction, ports in (("input", component.input), ("output", component.output)):
            for port_name, port in sorted(ports.items()):
                if port._history is not None and (
                    port._history_is_populated or port.is_leaf
                ):
                    append_value(
                        component_id,
                        f"{direction}:{port_name}:history",
                        port._history,
                    )
        for name, value in sorted(vars(component).items()):
            wrapped_values = getattr(value, "values", None)
            if not name.startswith("_"):
                append_value(component_id, f"attribute:{name}", value)
            if name.endswith("_ts") or not name.startswith("_"):
                append_value(
                    component_id,
                    f"attribute:{name}.values",
                    wrapped_values,
                )
    return tuple(signature)


def recording_parameter_signature(model):
    """Fingerprint parameters that can affect initial state or warm values."""
    order = getattr(model, "_flat_execution_order", None)
    if order is None:
        order = model.flat_execution_order
    signature = []
    for component in order:
        for name in getattr(component, "PARAM_NAMES", ()):
            signature.append(
                (component.id, name, _content_signature(rgetattr(component, name).get()))
            )
    return tuple(signature)


def _clone_recording(recording):
    """Clone a cached recording so callers cannot mutate the cache entry."""
    def clone_list(values):
        if values is None:
            return None
        return [value.detach().clone() for value in values]

    return SimpleNamespace(
        state0=clone_list(recording.state0),
        Y0=clone_list(recording.Y0),
        exogenous_tape=clone_list(recording.exogenous_tape),
        feedback_tape=clone_list(recording.feedback_tape),
        measurement_tape=clone_list(recording.measurement_tape),
        n_timesteps=list(recording.n_timesteps),
    )


def _has_real_forward(comp) -> bool:
    """True iff ``comp``'s class overrides ``forward`` (not the ``nn.Module`` base
    stub, which every ``nn.Module`` inherits).  Components without their own
    ``forward`` (sensors, schedules, weather, occupancy, ...) are treated as
    exogenous/pass-through by the functional_model."""
    f = getattr(type(comp), "forward", None)
    return f is not None and f is not nn.Module.forward


def _is_passthrough_sensor(comp) -> bool:
    """A SensorSystem whose ``measuredValue`` is driven by another component
    (not by its own data source) just forwards that value."""
    if not isinstance(comp, systems.SensorSystem):
        return False
    for cp in comp.connects_at:
        if cp.input_port == "measuredValue" and cp.connects_system_through:
            return True
    return False


def _single_source(comp, port) -> Optional[Tuple[object, str]]:
    """The ``(producer, output_port)`` feeding ``comp.input[port]`` (first
    connection), or ``None`` if the port is unconnected (exogenous / data)."""
    for cp in comp.connects_at:
        if cp.input_port == port:
            conns = cp.connects_system_through
            if conns:
                conn = conns[0]
                return conn.connects_system, conn.output_port
    return None


def collect_stateful(model) -> List:
    """Stateful components in execution order (composites only, not their subs).

    A composite like ``BuildingSpaceSystem`` owns state (thermal|mass) and
    its submodels are not separate nodes in the execution order, so iterating
    ``_flat_execution_order`` and taking ``System.is_stateful()`` (which walks the
    owned ``tps.State``) yields each state exactly once.
    """
    order = getattr(model, "_flat_execution_order", None)
    if order is None:
        # Model wrapper: delegate to the simulation model's execution order.
        # This must NOT fall back to ``components`` -- fused state-space
        # clusters execute (and own their joint state) through a
        # FusedStateSpaceSystem that only appears in the execution order.
        order = getattr(model, "flat_execution_order", None)
    if order is None:
        order = list(model.components.values())
    return [c for c in order if c.is_stateful()]


class StateLayout:
    """Flat <-> per-component packing of the stateful-component states.

    Enumerates the model's stateful components (those owning a ``tps.State``, in
    execution order) and lays their states out contiguously into a single vector
    of width ``D = sum_c (n_c * state_size_c)``.
    """

    def __init__(self, components: List):
        self.components = components
        self.slices: List[Tuple[int, int]] = []  # (start, stop) into flat vector
        self.shapes: List[Tuple[int, int]] = []  # (n_c, state_size) per component
        offset = 0
        for comp in components:
            state = comp.get_state()  # (n_s, n_c, state_size)
            n_c, ss = state.shape[1], state.shape[2]
            width = n_c * ss
            self.slices.append((offset, offset + width))
            self.shapes.append((n_c, ss))
            offset += width
        self.width = offset  # D

    def gather(self, n_s_index: int = 0) -> torch.Tensor:
        """Flatten current component states at sim-batch index into ``(D,)``."""
        parts = []
        for comp, (n_c, ss) in zip(self.components, self.shapes):
            s = comp.get_state()[n_s_index]  # (n_c, state_size)
            parts.append(s.reshape(-1))
        return torch.cat(parts) if parts else torch.zeros(0, dtype=tps.float_dtype())

    def scatter(self, seg_states: torch.Tensor) -> None:
        """Write per-segment states into every component via ``set_state``.

        ``seg_states`` has shape ``(K, D)`` (one flat state per segment); it is
        unpacked and each component's ``set_state`` receives ``(K, n_c,
        state_size)`` -- i.e. K segments live on the simulator's n_s axis.
        """
        K = seg_states.shape[0]
        for comp, (start, stop), (n_c, ss) in zip(
            self.components, self.slices, self.shapes
        ):
            block = seg_states[:, start:stop].reshape(K, n_c, ss)
            comp.set_state(block)

    def end_states(self, K: int) -> torch.Tensor:
        """Collect each segment's *final* state into ``(K, D)`` after a sim."""
        parts = []
        for comp, (n_c, ss) in zip(self.components, self.shapes):
            s = comp.get_state()  # (K, n_c, state_size)
            parts.append(s.reshape(K, -1))
        return (
            torch.cat(parts, dim=1)
            if parts
            else torch.zeros((K, 0), dtype=tps.float_dtype())
        )


class FunctionalModel:
    """Builds and evaluates the pure one-step map ``F`` for a model.

    Parameters
    ----------
    model : SimulationModel
        The (initialized) simulation model.
    stateful : list
        Stateful components in execution order (each owns ``tps.State``).
    theta_spec : list of (component, attr) or (component, attr, theta_index)
        Estimated parameters.  ``attr`` is the component-relative path (e.g.
        ``"thermal.C_air"``, ``"kp"``).  Two-element entries take their theta
        index from their list position (one-to-one theta).  Three-element
        entries carry it explicitly, so several entries may point at the SAME
        theta slot -- that is how *shared* parameters (one decision variable
        driving the same attribute on several components) are functional.
    sample_time : float
        Segment step size in seconds.
    measurements : list, optional
        Measuring devices whose modelled ``measuredValue`` the map must return
        (the Estimator's data-fit signals).
    outputs : list of (component, out_port), optional
        Arbitrary component outputs the map must additionally return (the
        Optimizer's objective/constraint signals).  Their producers seed the
        influence cone, so a purely *downstream* component (e.g. a cost
        sensor multiplying heater power by an electricity price) is functional
        even though no state depends on it.  Outputs whose producer cannot be
        functional are returned as ``("external", comp_id, port)`` specs and
        evaluate to zero inside ``F`` -- the caller decides whether it can
        supply them (e.g. a decision-variable trajectory) or must reject.
    """

    def __init__(
        self,
        model,
        stateful,
        theta_spec,
        sample_time,
        measurements=None,
        outputs=None,
    ):
        self.model = model
        self.sample_time = float(sample_time)
        # Accept either the SimulationModel (``_flat_execution_order``) or the
        # Model wrapper (``flat_execution_order`` property).
        order = getattr(model, "_flat_execution_order", None)
        if order is None:
            order = model.flat_execution_order
        order = list(order)
        self.pos = {c.id: i for i, c in enumerate(order)}
        self.order = order
        self.forward_ids = {c.id for c in order if _has_real_forward(c)}
        # Fused-cluster members are not executing nodes; their produced
        # signals resolve to the fused block's namespaced outputs (_follow)
        # and their theta associations to the fused block's id.
        sim_model = getattr(model, "_simulation_model", None) or model
        self._fusion_alias = dict(
            getattr(sim_model, "_fusion_member_to_fused", None) or {}
        )

        # Stateful components and their flat state layout (widths, offsets).
        self.stateful = list(stateful)
        self.state_shapes = [
            (int(c.get_state().shape[1]), int(c.get_state().shape[2]))
            for c in self.stateful
        ]
        self.state_widths = [n_c * state_size for n_c, state_size in self.state_shapes]
        self.state_offsets = np.cumsum([0] + self.state_widths).tolist()
        self.D = int(self.state_offsets[-1])
        self.state_index = {c.id: i for i, c in enumerate(self.stateful)}

        # theta routing: per component id -> {attr: theta_index}.  Entries are
        # (comp, attr) -- index = position -- or (comp, attr, theta_index);
        # shared parameters route several (comp, attr) pairs to one index.
        self.theta_spec = theta_spec
        self.theta_by_comp: Dict[str, Dict[str, object]] = {}
        for i, spec in enumerate(theta_spec):
            comp, attr = spec[0], spec[1]
            idx = spec[2] if len(spec) > 2 else i
            self.theta_by_comp.setdefault(comp.id, {})[attr] = idx

        # Which components must be evaluated by F: the stateful ones (plus any
        # requested-output producers) plus every forward-component
        # reverse-reachable from them over fresh edges.
        seed_extra = {comp.id for comp, _ in (outputs or []) if _has_real_forward(comp)}
        self.cone = self._influence_cone(seed_extra)
        self._default_params = {
            comp.id: {
                name: rgetattr(comp, name).get()
                for name in getattr(comp, "PARAM_NAMES", ())
            }
            for comp in self.cone
        }

        # Static input wiring for every cone component: port -> source spec.
        # A port's source is one of:
        #   ("fresh",    producer_id, out_port) -- produced earlier in F this step;
        #   ("feedback", fb_index)              -- a *cut feedback edge*: the source
        #        is a cone forward-component that executes LATER (the cycle-broken
        #        edge).  Its value is a decision variable, NOT frozen, because it
        #        is a function of the states/params (e.g. office.heatGain <-
        #        space_heater.Power).  A defect ties it to the producer's output.
        #   ("exogenous", cap_index)             -- truly exogenous (weather,
        #        schedules): frozen from a reference sim (correct -- independent of
        #        the unknowns).
        self.cone_ids = {c.id for c in self.cone}
        self._exogenous_keys: List[Tuple[str, str]] = []  # exogenous (comp_id, port)
        self._exogenous_index: Dict[Tuple[str, str], slice] = {}
        self._exogenous_widths: List[int] = []
        self._n_exogenous = 0
        self._feedback_keys: List[Tuple[str, str]] = (
            []
        )  # cut-feedback (consumer_id, port)
        self._fb_index: Dict[Tuple[str, str], slice] = {}
        self._feedback_widths: List[int] = []
        self._feedback_masks: List[torch.Tensor] = []
        self._n_feedback = 0
        self._fb_producer: List[List[tuple]] = []
        self.wiring: Dict[str, List[Tuple[str, tuple]]] = {}
        for c in self.cone:
            self.wiring[c.id] = self._resolve_inputs(c)

        # Measurement sources: for each measurement sensor, where does its value
        # come from?  ("fresh", producer_id, out_port) if produced by a cone
        # component (F computes it), else ("exogenous", cap_index) sampled from a
        # reference sim.  Lets F return the modelled measured outputs for the
        # data-fit objective.
        self._theta_param_cache = None  # (theta_ref, {comp_id: params dict})

        self.meas_sources: List[tuple] = []
        self.meas_slices: List[slice] = []
        meas_offset = 0
        for md in measurements or []:
            src = self._trace_source(md, "measuredValue")
            if src is not None and src[0].id in {x.id for x in self.cone}:
                width = self._route_width(src[2], src[0], src[1])
                self.meas_sources.append(("fresh", src[0].id, src[1], src[2]))
            else:
                key = (md.id, "measuredValue")
                if key not in self._exogenous_index:
                    self._register_exogenous(key, self._component_n_c(md))
                self.meas_sources.append(("exogenous", self._exogenous_index[key]))
                width = (
                    self._exogenous_index[key].stop - self._exogenous_index[key].start
                )
            self.meas_slices.append(slice(meas_offset, meas_offset + width))
            meas_offset += width

        # Requested outputs are appended to the same measurement vector: fresh
        # when the producer is functional, ("external", ...) otherwise (F returns
        # zero there; the caller supplies or rejects).  ``_follow`` translates
        # fused-cluster members to the executing FusedStateSpaceSystem, which
        # publishes their outputs under namespaced port names -- without it,
        # every output on a fused zone would be wrongly classified external.
        for comp, out_port in outputs or []:
            resolved = self._follow(comp, out_port)
            if resolved is not None and resolved[0].id in self.cone_ids:
                width = self._route_width(resolved[2], resolved[0], resolved[1])
                self.meas_sources.append(
                    ("fresh", resolved[0].id, resolved[1], resolved[2])
                )
            else:
                width = self._output_width(comp, out_port)
                self.meas_sources.append(("external", comp.id, out_port, width))
            self.meas_slices.append(slice(meas_offset, meas_offset + width))
            meas_offset += width
        self.n_meas = meas_offset

        # Gradients must not be silently lost: every theta path through the
        # real system has to be threaded through F, never frozen into a
        # exogenous constant.
        self._validate_theta_influence()

    # -- static graph analysis ----------------------------------------------
    def _validate_theta_influence(self):
        """Refuse to compose when theta gradients would silently be lost.

        The functional map freezes every input without a functional producer into
        a exogenous constant.  That is exact -- in value AND gradient -- only
        when the frozen signal is truly exogenous (weather, schedules).  Two
        structural violations make the frozen value theta-dependent, so the
        functional objective would MATCH the object graph in value at the
        reference theta while its gradient silently loses the paths through
        the frozen signal (observed historically: an ``OccupancySystem`` --
        before it grew its pure ``forward`` -- with estimated
        ``V``/``G_occ``/``m_inf`` feeding ``numberOfPeople``):

        1. a theta component outside the influence cone (not functional at
           all -- e.g. no functorch-safe ``forward``, or only reachable
           through a non-composable component);
        2. a exogenous input whose upstream object-graph ancestry contains a
           theta component (theta leaks into the "constant" through a
           non-composable intermediary).

        Raises ``RuntimeError``; callers treat that as "fall back to the
        object-graph engine".
        """
        missing = sorted(cid for cid in self.theta_by_comp if cid not in self.cone_ids)
        if missing:
            raise RuntimeError(
                "theta components not composable (their gradient paths would "
                f"be frozen into exogenous constants): {missing}"
            )

        by_id = {c.id: c for c in self.order}
        for key in self._exogenous_keys:
            comp = by_id.get(key[0])
            if comp is None:
                continue
            if len(key) == 3:
                starts = [
                    (src[0], src[1])
                    for slot, sources in self._vector_slot_sources(comp, key[1])
                    if slot == key[2]
                    for src in sources
                ]
            else:
                src = _single_source(comp, key[1])
                starts = [src] if src is not None else []
            hit = self._upstream_theta_component([p for p, _ in starts])
            if hit is not None:
                raise RuntimeError(
                    f"exogenous input {key} depends on theta component "
                    f"'{hit}' (freezing it would drop its gradient)"
                )

    def _upstream_theta_component(self, start_comps):
        """Walk the object graph upstream (over ALL edges, composable or not)
        from ``start_comps``; return the id of the first theta component
        reached, or ``None``."""
        stack = list(start_comps)
        visited = set()
        while stack:
            c = stack.pop()
            if c is None or c.id in visited:
                continue
            visited.add(c.id)
            if c.id in self.theta_by_comp:
                return c.id
            # A fused-cluster member carries its theta under the fused id.
            fused = self._fusion_alias.get(c.id)
            if fused is not None and fused.id in self.theta_by_comp:
                return c.id
            for port in list(c.input.keys()):
                if isinstance(c.input[port], tps.Vector):
                    for _, sources in self._vector_slot_sources(c, port):
                        stack.extend(src[0] for src in sources)
                else:
                    src = _single_source(c, port)
                    if src is not None:
                        stack.append(src[0])
        return None

    def _influence_cone(self, seed_extra=()) -> List:
        """Forward-components reverse-reachable (over fresh edges, following
        pass-through sensors) from the stateful components -- plus any extra
        seed ids (requested-output producers) -- in execution order."""
        keep = set(c.id for c in self.stateful) | set(seed_extra)
        changed = True
        while changed:
            changed = False
            for c in self.order:
                if c.id not in keep or c.id not in self.forward_ids:
                    continue
                for port in list(c.input.keys()):
                    if isinstance(c.input[port], tps.Vector):
                        srcs = [
                            (src[0], src[1])
                            for _, sources in self._vector_slot_sources(c, port)
                            for src in sources
                        ]
                    else:
                        srcs = [
                            (src[0], src[1]) for src in self._trace_sources(c, port)
                        ]
                    for prod, _ in srcs:
                        if (
                            prod.id in self.forward_ids
                            and self.pos[prod.id] < self.pos[c.id]
                            and prod.id not in keep
                        ):
                            keep.add(prod.id)
                            changed = True
        return [c for c in self.order if c.id in keep and c.id in self.forward_ids]

    @staticmethod
    def _component_n_c(comp) -> int:
        return int(getattr(comp, "n_c", None) or getattr(comp, "_n_c_batched", 1) or 1)

    def _output_width(self, comp, port) -> int:
        value = comp.output[port].get()
        if value is not None:
            return int(value[0].numel())
        n_c = self._component_n_c(comp)
        n_v = int(getattr(comp.output[port], "n_v", 1) or 1)
        return n_c * n_v

    def _connection_sources(self, comp, port):
        """Immediate sources with exact compiled ``i_c`` routing metadata."""
        sources = []
        target_n_c = self._component_n_c(comp)
        for cp in comp.connects_at:
            if cp.input_port != port:
                continue
            for conn in cp.connects_system_through:
                producer = conn.connects_system
                route = (
                    cp.output_port_index.get(conn, slice(None)),
                    cp.output_component_index.get(conn, slice(None)),
                    cp.input_component_index.get(conn, slice(None)),
                    target_n_c,
                    isinstance(producer.output[conn.output_port], tps.Vector),
                )
                sources.append((producer, conn.output_port, (route,)))
        return sources

    def _trace_sources(self, comp, port):
        result = []
        for producer, out_port, immediate_routes in self._connection_sources(
            comp, port
        ):
            followed = self._follow(producer, out_port)
            if followed is not None:
                result.append(
                    (followed[0], followed[1], followed[2] + immediate_routes)
                )
        return result

    def _trace_source(self, comp, port):
        """First resolved source, retained for scalar/pass-through callers."""
        sources = self._trace_sources(comp, port)
        return sources[0] if sources else None

    def _follow(self, producer, out_port):
        """Follow pass-through sensors from a ``(producer, out_port)`` pair,
        and translate fused-cluster members to their executing
        ``FusedStateSpaceSystem`` (which publishes the member's outputs under
        namespaced port names)."""
        if _is_passthrough_sensor(producer):
            return self._trace_source(producer, "measuredValue")
        fused = self._fusion_alias.get(producer.id)
        if fused is not None:
            return fused, f"{producer.id}.{out_port}", ()
        return producer, out_port, ()

    def _vector_slot_sources(self, comp, port):
        """Per vector slot, all sources with compiled branch routing."""
        slots = {}
        for cp in comp.connects_at:
            if cp.input_port != port:
                continue
            for conn in cp.connects_system_through:
                idx = cp.input_port_index.get(conn, 0)
                idx = int(idx.item()) if hasattr(idx, "item") else int(idx)
                immediate = self._connection_sources(comp, port)
                for src in immediate:
                    if src[0] is conn.connects_system and src[1] == conn.output_port:
                        followed = self._follow(src[0], src[1])
                        if followed is not None:
                            slots.setdefault(idx, []).append(
                                (
                                    followed[0],
                                    followed[1],
                                    followed[2] + src[2],
                                )
                            )
                        break
        return sorted(slots.items())

    def _register_exogenous(self, key, width):
        start = self._n_exogenous
        self._exogenous_index[key] = slice(start, start + width)
        self._exogenous_keys.append(key)
        self._exogenous_widths.append(width)
        self._n_exogenous += width

    def _register_feedback(self, key, width, sources, mask=None):
        start = self._n_feedback
        self._fb_index[key] = slice(start, start + width)
        self._feedback_keys.append(key)
        self._feedback_widths.append(width)
        if mask is None:
            mask = torch.ones(width, dtype=tps.float_dtype())
        self._feedback_masks.append(mask.reshape(-1))
        self._fb_producer.append(sources)
        self._n_feedback += width

    def _classify_sources(self, comp, key, sources):
        """Classify all connections populating one consumer scalar/vector slot."""
        width = self._component_n_c(comp)
        categories = []
        for src in sources:
            in_cone = src[0].id in self.forward_ids and src[0].id in self.cone_ids
            if in_cone and self.pos[src[0].id] < self.pos[comp.id]:
                categories.append("fresh")
            elif in_cone:
                categories.append("feedback")
            else:
                categories.append("exogenous")
        category_set = set(categories)
        if not sources or category_set == {"exogenous"}:
            if key not in self._exogenous_index:
                self._register_exogenous(key, width)
            return ("exogenous", self._exogenous_index[key])
        if category_set == {"fresh"}:
            return ("fresh", sources)
        if category_set == {"feedback"}:
            # Source is a cone forward-component that executes LATER: this is
            # the cycle-broken (feedback) edge.  Its value is a function of the
            # states/params, so it becomes a decision variable tied to the
            # producer's output by a defect -- NOT a frozen constant.
            if key not in self._fb_index:
                self._register_feedback(key, width, sources)
            return ("feedback", self._fb_index[key])
        if "exogenous" in category_set:
            raise RuntimeError(
                f"mixed fresh/exogenous routing for {key} is not composable"
            )
        parts = []
        for i, (category, src) in enumerate(zip(categories, sources)):
            if category == "fresh":
                parts.append(("fresh", [src]))
            else:
                part_key = key + (f"branch-part-{i}",)
                source_n_c = self._component_n_c(src[0])
                marker = torch.ones(source_n_c, dtype=tps.float_dtype())
                mask = self._apply_routes(marker, src[2])
                self._register_feedback(part_key, width, [src], mask=mask)
                parts.append(("feedback", self._fb_index[part_key]))
        return ("mixed", parts)

    def _resolve_inputs(self, comp):
        """Per-input-port source spec for one cone component.

        Scalar ports resolve to a single fresh/feedback/exogenous spec.  Vector
        ports (e.g. ``MaxSystem.inputs``) resolve **per slot** to a ``("vector",
        [slot_spec, ...])`` spec, so a producer inside the cone (e.g. the CO2
        controller feeding the damper max) is threaded fresh instead of frozen.
        Unconnected vector ports (n_v=0, e.g. ``wallHeatGain``) are
        skipped.
        """
        specs = []
        for port in comp.input.keys():
            if isinstance(comp.input[port], tps.Vector):
                populated = dict(self._vector_slot_sources(comp, port))
                n_v = int(getattr(comp.input[port], "n_v", 0) or 0)
                if n_v == 0:
                    continue
                slot_specs = [
                    self._classify_sources(
                        comp, (comp.id, port, slot), populated.get(slot, [])
                    )
                    for slot in range(n_v)
                ]
                specs.append((port, ("vector", slot_specs)))
                continue
            sources = self._trace_sources(comp, port)
            specs.append(
                (
                    port,
                    self._classify_sources(comp, (comp.id, port), sources),
                )
            )
        return specs

    # -- the pure one-step map ----------------------------------------------
    # (Exogenous-input sampling lives in :func:`record_exogenous_inputs` --
    # the single continuous-rollout source of truth; a per-segment capture
    # would evaluate stateful/data-indexed signals like the OccupancySystem's
    # ``previousIndoorCo2Measured`` at the wrong step.)
    @staticmethod
    def _as_index(index, device):
        if isinstance(index, slice):
            return index
        return torch.as_tensor(index, dtype=torch.long, device=device).reshape(-1)

    def _apply_routes(self, value, routes):
        """Apply object-graph output/input branch mappings without mutation."""
        result = value
        for out_v, source_i_c, target_i_c, target_n_c, output_is_vector in routes:
            if output_is_vector:
                result = result[..., out_v]
            if result.ndim == 0:
                result = result.reshape(1)
            source_i_c = self._as_index(source_i_c, result.device)
            selected = result[source_i_c]
            if isinstance(target_i_c, slice):
                if selected.shape[0] == target_n_c:
                    result = selected
                elif selected.shape[0] == 1:
                    result = selected.expand((target_n_c,) + selected.shape[1:])
                else:
                    result = selected
            else:
                target_i_c = self._as_index(target_i_c, result.device)
                shape = (target_n_c,) + selected.shape[1:]
                base = torch.zeros(shape, dtype=selected.dtype, device=selected.device)
                result = torch.index_copy(base, 0, target_i_c, selected)
        return result

    def _route_width(self, routes, producer, out_port):
        value = producer.output[out_port].get()
        if value is None:
            return self._output_width(producer, out_port)
        return int(self._apply_routes(value[0], routes).numel())

    def _merge_sources(self, sources, produced):
        values = [
            self._apply_routes(produced[src[0].id][src[1]], src[2]) for src in sources
        ]
        if not values:
            raise RuntimeError("cannot merge an empty source list")
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result

    def _params_for(self, comp, theta, use_cache=True):
        """Physical-parameter dict for ``comp``: estimated entries from ``theta``
        (a 1-D tensor in theta_spec order), the rest from the component's
        defaults (``getattr(comp, name).get()``).

        Cached per ``theta`` object: a sequential rollout calls ``F`` hundreds
        of times with the SAME theta tensor, and rebuilding the dict each step
        re-slices theta and re-denormalizes every default parameter -- pure
        overhead in both the forward pass and the autograd graph.  Identity
        keying (``is``) makes a stale hit impossible: holding the theta
        reference in the cache also pins its id.  Downstream, stable parameter
        -tensor identities let the components' ``_build_matrices`` cache the
        (theta-only, step-independent) state-space matrices the same way.
        """
        cache = None
        if use_cache:
            cache = self._theta_param_cache
            if cache is None or cache[0] is not theta:
                cache = (theta, {})
                self._theta_param_cache = cache
            hit = cache[1].get(comp.id)
            if hit is not None:
                return hit
        p = {}
        est = self.theta_by_comp.get(comp.id, {})
        # For composites the attrs are prefixed (thermal.C_air); pass them
        # through and let the composite's forward route/resolve.  For leaf
        # components fill PARAM_NAMES.
        if hasattr(comp, "PARAM_NAMES"):
            for name in comp.PARAM_NAMES:
                if name in est:
                    p[name] = theta[est[name]].reshape(-1)
                else:
                    p[name] = self._default_params[comp.id][name]
        # Prefixed estimated params (composite: "thermal.C_air") -> pass through.
        for attr, idx in est.items():
            if "." in attr:
                p[attr] = theta[idx].reshape(-1)
        if cache is not None:
            cache[1][comp.id] = p
        return p

    def F(self, states_flat, theta, exogenous, feedback=None, transform_mode=None):
        """One pure step for a single segment.

        Args:
            states_flat: ``(D,)`` concatenated stateful-component states.
            theta: ``(n_theta,)`` physical estimated parameters (theta_spec order).
            exogenous: ``(n_exogenous,)`` exogenous input values for this segment.
            feedback: ``(n_feedback,)`` cut-feedback input values (decision
                variables) for this segment; ``None`` -> zeros (n_feedback==0).

        Returns:
            ``(x_next_flat (D,), meas (n_meas,), fb_out (n_feedback,))`` -- the
            next state, the modelled measured outputs, and the producer outputs
            that the feedback variables must match (for the feedback defects).
        """
        if feedback is None:
            feedback = torch.zeros(
                self._n_feedback,
                dtype=states_flat.dtype,
                device=states_flat.device,
            )
        # Unpack per-component states.
        states = {}
        for i, c in enumerate(self.stateful):
            a, b = self.state_offsets[i], self.state_offsets[i + 1]
            n_c, state_size = self.state_shapes[i]
            states[c.id] = states_flat[a:b].reshape(n_c, state_size)

        produced: Dict[str, Dict[str, torch.Tensor]] = {}
        x_next_parts = [None] * len(self.stateful)
        for c in self.cone:

            def _input_value(spec):
                if spec[0] == "fresh":
                    return self._merge_sources(spec[1], produced)
                if spec[0] == "feedback":
                    return feedback[spec[1]]
                if spec[0] == "exogenous":
                    return exogenous[spec[1]]
                if spec[0] == "mixed":
                    values = [_input_value(part) for part in spec[1]]
                    result = values[0]
                    for value in values[1:]:
                        result = result + value
                    return result
                raise RuntimeError(f"unknown functional input spec {spec[0]!r}")

            inputs = {}
            for port, spec in self.wiring[c.id]:
                if spec[0] == "vector":
                    vals = []
                    for s in spec[1]:
                        vals.append(_input_value(s))
                    inputs[port] = torch.stack(vals, dim=-1)
                else:
                    inputs[port] = _input_value(spec)
            params = self._params_for(c, theta, use_cache=not transform_mode)
            st = states.get(c.id, None)
            if getattr(c, "SUPPORTS_TRANSFORM_MODE", False):
                x_next_c, outs = c.forward(
                    st,
                    inputs,
                    params,
                    self.sample_time,
                    transform_mode=transform_mode,
                )
            else:
                x_next_c, outs = c.forward(st, inputs, params, self.sample_time)
            produced[c.id] = outs
            if c.id in self.state_index:
                x_next_parts[self.state_index[c.id]] = x_next_c.reshape(-1)
        x_next = torch.cat(x_next_parts)
        # Producer outputs the feedback decision variables must equal.
        if self._fb_producer:
            fb_out = torch.cat(
                [
                    self._merge_sources(sources, produced).reshape(-1)
                    for sources in self._fb_producer
                ]
            )
        else:
            fb_out = torch.zeros(0, dtype=x_next.dtype, device=x_next.device)
        meas = []
        for spec in self.meas_sources:
            if spec[0] == "fresh":
                meas.append(
                    self._apply_routes(produced[spec[1]][spec[2]], spec[3]).reshape(-1)
                )
            elif spec[0] == "external":
                # Not producible by the functional map; the caller supplies the
                # signal (e.g. a decision-variable trajectory) or rejects.
                meas.append(
                    torch.zeros(spec[3], dtype=x_next.dtype, device=x_next.device)
                )
            else:
                meas.append(exogenous[spec[1]].reshape(-1))
        meas = (
            torch.cat(meas)
            if meas
            else torch.zeros(0, dtype=x_next.dtype, device=x_next.device)
        )
        return x_next, meas, fb_out

    @property
    def n_feedback(self) -> int:
        return self._n_feedback

    # -- coupling structure ------------------------------------------------------
    def index_coupling(self):
        """Independent blocks of the estimation problem, from the wiring alone.

        Nodes are ``(component id, batch index)``; two nodes are joined when a
        connection routes one into the other (the exact ``_apply_routes``
        semantics, applied to index vectors), when a feedback lag connects
        them, or when they share a theta entry.  Theta entries and residual
        columns attach to the nodes they belong to.  Connected components are
        the blocks: a batched layout of independent zones yields one block per
        zone; an air-handling unit feeding every zone joins them all; a fully
        coupled model yields a single block, which is the unstructured case.

        Returns ``(theta_block, column_block, n_blocks)``: ``theta_block[t]``
        is the block id of theta entry ``t`` (every entry has one);
        ``column_block[c]`` is the block id of residual column ``c`` or ``-1``
        when no estimated parameter reaches it (exogenous / external columns).
        """
        parent: Dict[object, object] = {}

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

        def owners(producer, out_port, routes):
            """(producer index or -1) per routed slot, via the real route code."""
            n = self._component_n_c(producer)
            width = max(1, int(self._output_width(producer, out_port)))
            tags = torch.arange(1, n + 1, dtype=torch.float64)[:, None].expand(n, width)
            mapped = self._apply_routes(tags, routes)
            if mapped.ndim == 1:
                mapped = mapped[:, None]
            rows = mapped.reshape(mapped.shape[0], -1)
            out = []
            for row in rows:
                out.append(sorted({int(v) - 1 for v in row.tolist() if v > 0}))
            return out  # list over slots -> producer indices feeding that slot

        # feedback vector layout: position -> (producer id, index) set
        fb_owner: List[List[tuple]] = []
        for sources in self._fb_producer:
            merged: List[set] = []
            for src in sources:
                for slot, idxs in enumerate(owners(src[0], src[1], src[2])):
                    while len(merged) <= slot:
                        merged.append(set())
                    merged[slot].update((src[0].id, i) for i in idxs)
            fb_owner.extend([sorted(m) for m in merged])

        def bind(consumer_id, spec):
            if spec[0] == "fresh":
                for src in spec[1]:
                    for slot, idxs in enumerate(owners(src[0], src[1], src[2])):
                        for i in idxs:
                            union((consumer_id, slot), (src[0].id, i))
            elif spec[0] == "feedback":
                sl = spec[1]
                for slot, pos in enumerate(range(sl.start, sl.stop)):
                    if pos < len(fb_owner):
                        for node in fb_owner[pos]:
                            union((consumer_id, slot), node)
            elif spec[0] == "mixed":
                for part in spec[1]:
                    bind(consumer_id, part)
            elif spec[0] == "vector":
                for part in spec[1]:
                    bind(consumer_id, part)
            # "exogenous": data, no coupling

        for cid, ports in self.wiring.items():
            for _port, spec in ports:
                bind(cid, spec)

        # theta entries: per-index slices attach entry start+i to index i; a
        # single shared entry couples every index of the component.
        n_theta = 0
        for cid, attrs in self.theta_by_comp.items():
            comp = next((c for c in self.cone if c.id == cid), None)
            n_c = self._component_n_c(comp) if comp is not None else 1
            for _attr, idx in attrs.items():
                if isinstance(idx, slice):
                    entries = list(range(idx.start, idx.stop))
                    n_theta = max(n_theta, idx.stop)
                    if len(entries) == n_c:
                        for i, t in enumerate(entries):
                            union(("theta", t), (cid, i))
                    else:
                        for t in entries:
                            for i in range(n_c):
                                union(("theta", t), (cid, i))
                else:
                    t = int(idx)
                    n_theta = max(n_theta, t + 1)
                    for i in range(n_c):
                        union(("theta", t), (cid, i))

        # residual columns
        col_root: List[object] = [None] * self.n_meas
        for src, sl in zip(self.meas_sources, self.meas_slices):
            if src[0] == "fresh":
                comp = next(c for c in self.cone if c.id == src[1])
                for slot, idxs in enumerate(owners(comp, src[2], src[3])):
                    c = sl.start + slot
                    if c < self.n_meas:
                        for i in idxs:
                            union(("col", c), (comp.id, i))
        roots = {}
        theta_block = np.full(n_theta, -1, dtype=np.int64)
        for t in range(n_theta):
            r = find(("theta", t))
            theta_block[t] = roots.setdefault(r, len(roots))
        column_block = np.full(self.n_meas, -1, dtype=np.int64)
        for c in range(self.n_meas):
            if ("col", c) in parent:
                r = find(("col", c))
                if r in roots:
                    column_block[c] = roots[r]
        return theta_block, column_block, len(roots)

    @property
    def compiled_batched_step(self):
        """``vmap`` of the transform-mode ``F_aug`` over a batch, compiled.

        ``step(Y, Theta, u) -> (Y_next, meas)`` with ``Y (B, D_aug)``,
        ``Theta (B, n_theta)`` and one shared exogenous row ``u``.  The vmap
        sits *inside* ``torch.compile`` -- Dynamo inlines it -- because
        ``vmap`` applied from eager code to a compiled function is not
        supported.  One compiled object serves every batch size (Dynamo
        re-specializes per shape).  Used by the batched single-shooting
        bundles (multi-start SQP) instead of ``vmap`` over the scalar rollout.
        """
        step = self.__dict__.get("_compiled_batched_step")
        if step is None:
            model = self

            def _step(y, theta, u):
                return model.F_aug(y, theta, u, transform_mode=True)

            batched = torch.func.vmap(_step, in_dims=(0, 0, None))
            step = torch.compile(batched, fullgraph=True, dynamic=False)
            self.__dict__["_compiled_batched_step"] = step
        return step

    @property
    def compiled_step(self):
        """``F_aug`` in transform mode, compiled once with ``torch.compile``.

        The whole step traces as one Dynamo graph (no graph breaks), so
        ``fullgraph=True`` guards against silent fallbacks; ``dynamic=False``
        because every call has the same shapes.  Inductor fuses the step's
        elementwise chains into a few hundred Triton kernels instead of the
        ~2000 ATen kernels of the eager step, which shrinks the captured CUDA
        graph and its replay time several-fold (issue #134).  Compiled on
        first use (tens of seconds, cached on disk by Inductor); the callable
        is cached on the model.  Requires a Triton-capable torch build
        (``torch.utils._triton.has_triton()``).
        """
        step = self.__dict__.get("_compiled_step")
        if step is None:
            model = self

            def _step(y, theta, u):
                return model.F_aug(y, theta, u, transform_mode=True)

            step = torch.compile(_step, fullgraph=True, dynamic=False)
            self.__dict__["_compiled_step"] = step
        return step

    @property
    def D_aug(self) -> int:
        """Augmented-state width: component states + cut-feedback lag variables."""
        return self.D + self._n_feedback

    def F_aug(self, y_flat, theta, exogenous, transform_mode=None):
        """Augmented one-step map over ``y = [state | feedback]``.

        The cut-feedback signals are one-step *lag variables* -- i.e. state in a
        discrete-time sense -- so appending them to the state turns the feedback
        loop into ordinary state continuity: ``y_{t+1} = [F(s_t, w_t),
        producer_output(s_t, w_t)]``.  The producer output computed at step ``t``
        becomes the feedback consumed at ``t+1``, exactly ``do_step``'s
        one-step-delayed (Gauss-Seidel) semantics -- with no separate defect type.

        Returns ``(y_next (D_aug,), meas (n_meas,))``.
        """
        n_fb = self._n_feedback
        s = y_flat[: self.D]
        w = y_flat[self.D :] if n_fb else None
        x_next, meas, fb_out = self.F(
            s, theta, exogenous, w, transform_mode=transform_mode
        )
        y_next = torch.cat([x_next, fb_out]) if n_fb else x_next
        return y_next, meas


def record_exogenous_inputs(
    simulator,
    functional_model,
    start_time,
    end_time,
    step_size,
    layout=None,
    meas_ids=(),
):
    """Record the reference rollout's exogenous inputs.

    One batched, CONTINUOUS ``do_step`` rollout over all periods at the
    model's *current* parameters and inputs, sampling every relevant input
    port right after each step. Every functional caller obtains its frozen
    map obtains its frozen exogenous inputs and warm values through this
    function.

    Two properties only a continuous ``do_step`` run provides:

    * **Step-indexed exogenous drivers** are evaluated correctly: e.g. the
      ``OccupancySystem``'s ``previousIndoorCo2Measured`` is the data sample
      one step back -- stepping segments in isolation (every segment at its
      own ``step_index = 0``) would freeze wrong values.
    * **Gauss-Seidel consumption semantics**: an input port read right after
      step ``t`` holds exactly the value ``do_step`` consumed at step ``t``
      (the producer's current- or previous-step output depending on execution
      order), so lag warm values and lagged step-0 sensor readings come out
      right by construction.

    The exogenous signals are independent of the unknowns by construction
    (anything the estimated parameters influence is inside the functional_model's
    cone, hence fresh or feedback; decision-variable slots are overridden by
    the Optimizer), so capturing once at the current model state is valid for
    every iterate.

    The run is batched over the periods (all periods share one
    ``model.initialize``), exactly like the corresponding object-graph
    objectives. Decision-variable ports with ``requires_grad`` set keep
    their current (initial-iterate) trajectories through the re-initialization,
    which is what the Optimizer requires.

    Parameters
    ----------
    simulator, functional_model
        The simulator (model already configured) and the ``FunctionalModel``
        whose ``_exogenous_keys`` / ``_feedback_keys`` define what to sample.
    start_time, end_time, step_size
        Period lists (one entry per period).
    layout : StateLayout, optional
        When given, the initial component states (right after ``initialize``)
        are gathered into per-period ``state0`` vectors and per-period
        augmented initial states ``Y0 = [state0 | FB[0]]`` are assembled.
    meas_ids : sequence of str, optional
        Measuring-device ids whose ``measuredValue`` input to sample per step.

    Returns
    -------
    types.SimpleNamespace
        Per-period lists, indexed by period:
        ``state0`` (each ``(D,)``; ``None`` without ``layout``),
        ``Y0`` (each ``(D_aug,)``; ``None`` without ``layout``),
        ``exogenous_tape`` (each ``(n_t_p, n_exogenous)``),
        ``feedback_tape`` (each ``(n_t_p, n_feedback)``),
        ``measurement_tape`` (each ``(n_t_p, len(meas_ids))``),
        ``n_timesteps`` (each ``int``).
    """
    model = simulator.model
    comps = dict(model.components)
    # Fused state-space clusters consume their members' exogenous inputs
    # under the fused component's id (namespaced ports).
    sim_model = getattr(model, "_simulation_model", None) or model
    comps.update(getattr(sim_model, "_fused_components", None) or {})
    exogenous_keys = functional_model._exogenous_keys
    fb_keys = functional_model._feedback_keys
    meas_keys = [(mid, "measuredValue") for mid in meas_ids]

    starts, ends, steps = list(start_time), list(end_time), list(step_size)
    simulator.get_simulation_timesteps(starts, ends, steps)
    model.initialize(starts, ends, steps)
    sec, dts, max_t, n_ts = simulator.get_simulation_timesteps(starts, ends, steps)
    n_s = len(starts)
    n_t = [int(n) for n in n_ts]

    state0 = (
        [layout.gather(p).detach().clone() for p in range(n_s)]
        if layout is not None
        else None
    )
    cache_key = (
        id(model),
        tuple(repr(value) for value in starts),
        tuple(repr(value) for value in ends),
        tuple(int(value) for value in steps),
        tuple(repr(key) for key in exogenous_keys),
        tuple(repr(key) for key in fb_keys),
        tuple(meas_ids),
        getattr(layout, "width", None),
        recording_parameter_signature(model),
        recording_data_signature(model),
    )
    cached = getattr(simulator, "_exogenous_recording_cache", None)
    if cached is not None and cached[0] == cache_key:
        simulator._exogenous_recording_cache_hit = True
        simulator._exogenous_recording_cache_hits += 1
        return _clone_recording(cached[1])
    simulator._exogenous_recording_cache_hit = False
    simulator._exogenous_recording_cache_misses += 1
    device = getattr(sim_model, "device", None) or torch.device("cpu")
    exogenous_tape = torch.zeros(
        (int(max_t), n_s, functional_model._n_exogenous),
        dtype=tps.float_dtype(),
        device=device,
    )
    feedback_tape = torch.zeros(
        (int(max_t), n_s, functional_model.n_feedback),
        dtype=tps.float_dtype(),
        device=device,
    )
    measurement_tape = torch.zeros(
        (int(max_t), n_s, len(meas_keys)), dtype=tps.float_dtype(), device=device
    )

    def _sample(dst, keys, slices, masks=None):
        if masks is None:
            masks = [None] * len(keys)
        for key, target_slice, mask in zip(keys, slices, masks):
            cid, port = key[0], key[1]
            val = comps[cid].input[port].get()
            if isinstance(comps[cid].input[port], tps.Vector):
                slot = key[2]
                val = val[..., slot]
            val = val.reshape(n_s, -1).detach()
            if mask is not None:
                val = val * mask.to(device=val.device, dtype=val.dtype)
            dst[:, target_slice] = val

    with torch.no_grad():
        for t in range(int(max_t)):
            simulator._do_system_time_step(
                model, sec[:, t], dts[:, t], steps, t, "gauss-seidel"
            )
            _sample(
                exogenous_tape[t],
                exogenous_keys,
                [functional_model._exogenous_index[k] for k in exogenous_keys],
            )
            if fb_keys:
                _sample(
                    feedback_tape[t],
                    fb_keys,
                    [functional_model._fb_index[k] for k in fb_keys],
                    functional_model._feedback_masks,
                )
            if meas_keys:
                for k, (cid, port) in enumerate(meas_keys):
                    val = comps[cid].input[port].get()
                    if val.reshape(n_s, -1).shape[1] != 1:
                        raise RuntimeError(
                            f"exogenous measurement {cid}.{port} has multiple branches"
                        )
                    measurement_tape[t, :, k] = val.reshape(n_s, -1)[:, 0].detach()

    n_fb = functional_model.n_feedback
    Y0 = None
    if layout is not None:
        Y0 = [
            torch.cat([state0[p], feedback_tape[0, p]]) if n_fb else state0[p]
            for p in range(n_s)
        ]
    recording = SimpleNamespace(
        state0=state0,
        Y0=Y0,
        exogenous_tape=[exogenous_tape[: n_t[p], p, :] for p in range(n_s)],
        feedback_tape=[feedback_tape[: n_t[p], p, :] for p in range(n_s)],
        measurement_tape=[measurement_tape[: n_t[p], p, :] for p in range(n_s)],
        n_timesteps=n_t,
    )
    simulator._exogenous_recording_cache = (cache_key, _clone_recording(recording))
    return recording


def functional_rollout(
    functional_model, y0, theta, exogenous_tape, *, transform_mode=False, step=None
):
    """Roll the functional map over one period; returns ``(n_t, n_meas)``.

    ``step`` optionally replaces ``functional_model.F_aug`` with a callable
    ``step(y, theta, u) -> (y_next, meas)`` -- the compiled transform-mode
    step from :meth:`FunctionalModel.compiled_step` -- for every time step.

    A plain Python loop, NOT ``vmap``: with a handful of periods the vmap
    dispatch overhead exceeds the batching gain, and staying in ordinary eager
    mode lets the state-space components use the fused ``torch.matrix_exp``
    (which has no vmap rule) instead of the unrolled scaling-and-squaring
    fallback.  Differentiable w.r.t. ``theta``, ``y0`` and ``exogenous_tape``.

    Args:
        functional_model: The :class:`FunctionalModel`.
        y0: ``(D_aug,)`` augmented initial state (``[state0 | FB[0]]``).
        theta: ``(n_theta,)`` physical parameters (theta_spec order).
        exogenous_tape: ``(n_t, n_exogenous)`` exogenous inputs for the period.
    """
    y = y0
    rows = []
    if step is None:
        for t in range(exogenous_tape.shape[0]):
            y, meas = functional_model.F_aug(
                y, theta, exogenous_tape[t], transform_mode=transform_mode
            )
            rows.append(meas)
    else:
        for t in range(exogenous_tape.shape[0]):
            y, meas = step(y, theta, exogenous_tape[t])
            rows.append(meas)
    if not rows:
        return torch.zeros(
            (0, functional_model.n_meas),
            dtype=exogenous_tape.dtype,
            device=exogenous_tape.device,
        )
    return torch.stack(rows)


def functional_rollout_batched(functional_model, Y0, Theta, exogenous_tape, *, step=None):
    """Roll a *batch* of parameter vectors over one period.

    ``Y0 (B, D_aug)``, ``Theta (B, n_theta)``; returns ``(B, n_t, n_meas)``.
    With ``step`` (the compiled batched step) every time step is one call
    over the whole batch; without it the scalar transform-mode rollout is
    ``vmap``-ed, which is what the batched bundles always did.
    """
    if step is None:
        return torch.func.vmap(
            lambda y0, th: functional_rollout(
                functional_model, y0, th, exogenous_tape, transform_mode=True
            )
        )(Y0, Theta)
    Y = Y0
    rows = []
    for t in range(exogenous_tape.shape[0]):
        Y, meas = step(Y, Theta, exogenous_tape[t])
        rows.append(meas)
    if not rows:
        return torch.zeros(
            (Y0.shape[0], 0, functional_model.n_meas),
            dtype=exogenous_tape.dtype,
            device=exogenous_tape.device,
        )
    return torch.stack(rows, dim=1)


def functional_rollout_tape(
    functional_model, y0, theta, exogenous_tape, *, transform_mode=False
):
    """Roll out ``F_aug`` and return tensor-only state and output tapes.

    ``states`` includes the initial augmented state at index zero and therefore
    has shape ``(n_t + 1, D_aug)``. ``outputs`` contains every output requested
    when constructing ``FunctionalModel`` and has shape ``(n_t, n_meas)``.
    This deliberately contains no dictionaries so the complete fixed-shape
    rollout can be captured by :class:`torch.cuda.CUDAGraph`.
    """
    y = y0
    states = [y]
    outputs = []
    for t in range(exogenous_tape.shape[0]):
        y, row = functional_model.F_aug(
            y, theta, exogenous_tape[t], transform_mode=transform_mode
        )
        states.append(y)
        outputs.append(row)
    if outputs:
        output_tape = torch.stack(outputs)
    else:
        output_tape = torch.zeros(
            (0, functional_model.n_meas),
            dtype=exogenous_tape.dtype,
            device=exogenous_tape.device,
        )
    return torch.stack(states), output_tape
