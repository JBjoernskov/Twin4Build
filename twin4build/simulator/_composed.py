"""Composed-map simulation: component ``forward``s fused into one pure step.

The object-graph engine (:meth:`Simulator.simulate`) steps every component's
``do_step`` through the ports -- flexible, but each step pays Python dispatch
(Gauss-Seidel traversal, ``tps`` port bookkeeping, history logging) and each
run pays ``model.initialize``.  Since every composable component's ``do_step``
is a thin port-I/O wrapper that DELEGATES its math to a pure ``forward``
method (single source of truth -- see the developer reference's
``do_step``/``forward`` contract), the same model can also be simulated as a
plain sequential torch rollout of one composed function:

    y_{t+1}, meas_t = F_aug(y_t, theta, CAP[t])

This module provides that machinery, consumed through the
:class:`~twin4build.simulator.simulator.Simulator` facade
(``compose`` / ``capture_rollout`` / ``rollout_composed``) by

* the Estimator's fast single-shooting objective
  (:mod:`twin4build.estimator._shooting`),
* the Estimator's collocation transcription
  (:mod:`twin4build.estimator._transcription`, which consumes ``F`` under
  ``vmap(jacrev(...))`` for the sparse NLP Jacobian), and
* the Optimizer's fast control objective
  (:mod:`twin4build.optimizer._fast_objective`).

**How the composition works.**  :class:`OneStepComposer` analyzes the model
graph once and threads the ``forward`` methods in execution order.  For each
input port of a composed component it uses one of

* a **fresh** value -- the output of an upstream component composed earlier in
  the same step (pass-through sensors followed to their source), so parameter
  and state couplings are exact;
* a **feedback** value -- a cut cycle edge (the producer executes *later* in
  the Gauss-Seidel order, e.g. ``office.heatGain <- space_heater.Power``).
  These are one-step *lag variables*: :meth:`OneStepComposer.F_aug` appends
  them to the state, reproducing ``do_step``'s one-step-delayed feedback
  semantics exactly;
* a **captured** constant -- truly exogenous drivers (weather, schedules,
  data-driven occupancy), frozen from one reference ``do_step`` rollout
  (:func:`capture_reference_rollout`).  Exogenous signals are independent of
  the unknowns by definition, so capturing once is valid for every parameter
  or control iterate (callers that *do* optimize an exogenous trajectory --
  the Optimizer -- override the corresponding captured slots per step).

The composed map is pure and functorch-traceable: sequential rollouts
differentiate with plain autograd, and collocation maps ``vmap(jacrev(F))``
over segments.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.utils.rgetattr import rgetattr


def _has_real_forward(comp) -> bool:
    """True iff ``comp``'s class overrides ``forward`` (not the ``nn.Module`` base
    stub, which every ``nn.Module`` inherits).  Components without their own
    ``forward`` (sensors, schedules, weather, occupancy, ...) are treated as
    exogenous/pass-through by the composer."""
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

    A composite like ``BuildingSpaceTorchSystem`` owns state (thermal|mass) and
    its submodels are not separate nodes in the execution order, so iterating
    ``_flat_execution_order`` and taking ``System.is_stateful()`` (which walks the
    owned ``tps.State``) yields each state exactly once.
    """
    order = getattr(model, "_flat_execution_order", None) or list(
        model.components.values()
    )
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
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)

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
        return torch.cat(parts, dim=1) if parts else torch.zeros((K, 0), dtype=torch.float64)


class OneStepComposer:
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
        driving the same attribute on several components) are composed.
    sample_time : float
        Segment step size in seconds.
    measurements : list, optional
        Measuring devices whose modelled ``measuredValue`` the map must return
        (the Estimator's data-fit signals).
    outputs : list of (component, out_port), optional
        Arbitrary component outputs the map must additionally return (the
        Optimizer's objective/constraint signals).  Their producers seed the
        influence cone, so a purely *downstream* component (e.g. a cost
        sensor multiplying heater power by an electricity price) is composed
        even though no state depends on it.  Outputs whose producer cannot be
        composed are returned as ``("external", comp_id, port)`` specs and
        evaluate to zero inside ``F`` -- the caller decides whether it can
        supply them (e.g. a decision-variable trajectory) or must reject.
    """

    def __init__(
        self, model, stateful, theta_spec, sample_time, measurements=None,
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

        # Stateful components and their flat state layout (widths, offsets).
        self.stateful = list(stateful)
        self.state_widths = [c.get_state().shape[-1] for c in self.stateful]
        self.state_offsets = np.cumsum([0] + self.state_widths).tolist()
        self.D = int(self.state_offsets[-1])
        self.state_index = {c.id: i for i, c in enumerate(self.stateful)}

        # theta routing: per component id -> {attr: theta_index}.  Entries are
        # (comp, attr) -- index = position -- or (comp, attr, theta_index);
        # shared parameters route several (comp, attr) pairs to one index.
        self.theta_spec = theta_spec
        self.theta_by_comp: Dict[str, Dict[str, int]] = {}
        for i, spec in enumerate(theta_spec):
            comp, attr = spec[0], spec[1]
            idx = spec[2] if len(spec) > 2 else i
            self.theta_by_comp.setdefault(comp.id, {})[attr] = int(idx)

        # Which components must be evaluated by F: the stateful ones (plus any
        # requested-output producers) plus every forward-component
        # reverse-reachable from them over fresh edges.
        seed_extra = {
            comp.id for comp, _ in (outputs or []) if _has_real_forward(comp)
        }
        self.cone = self._influence_cone(seed_extra)

        # Static input wiring for every cone component: port -> source spec.
        # A port's source is one of:
        #   ("fresh",    producer_id, out_port) -- produced earlier in F this step;
        #   ("feedback", fb_index)              -- a *cut feedback edge*: the source
        #        is a cone forward-component that executes LATER (the cycle-broken
        #        edge).  Its value is a decision variable, NOT frozen, because it
        #        is a function of the states/params (e.g. office.heatGain <-
        #        space_heater.Power).  A defect ties it to the producer's output.
        #   ("captured", cap_index)             -- truly exogenous (weather,
        #        schedules): frozen from a reference sim (correct -- independent of
        #        the unknowns).
        self.cone_ids = {c.id for c in self.cone}
        self._captured_keys: List[Tuple[str, str]] = []  # exogenous (comp_id, port)
        self._cap_index: Dict[Tuple[str, str], int] = {}
        self._feedback_keys: List[Tuple[str, str]] = []  # cut-feedback (consumer_id, port)
        self._fb_index: Dict[Tuple[str, str], int] = {}
        self._fb_producer: List[Tuple[str, str]] = []    # (producer_id, out_port) per fb
        self.wiring: Dict[str, List[Tuple[str, tuple]]] = {}
        for c in self.cone:
            self.wiring[c.id] = self._resolve_inputs(c)

        # Measurement sources: for each measurement sensor, where does its value
        # come from?  ("fresh", producer_id, out_port) if produced by a cone
        # component (F computes it), else ("captured", cap_index) sampled from a
        # reference sim.  Lets F return the modelled measured outputs for the
        # data-fit objective.
        self._theta_param_cache = None  # (theta_ref, {comp_id: params dict})

        self.meas_sources: List[tuple] = []
        for md in (measurements or []):
            src = self._trace_source(md, "measuredValue")
            if src is not None and src[0].id in {x.id for x in self.cone}:
                self.meas_sources.append(("fresh", src[0].id, src[1]))
            else:
                key = (md.id, "measuredValue")
                if key not in self._cap_index:
                    self._cap_index[key] = len(self._captured_keys)
                    self._captured_keys.append(key)
                self.meas_sources.append(("captured", self._cap_index[key]))

        # Requested outputs are appended to the same measurement vector: fresh
        # when the producer is composed, ("external", ...) otherwise (F returns
        # zero there; the caller supplies or rejects).
        for comp, out_port in (outputs or []):
            if comp.id in self.cone_ids:
                self.meas_sources.append(("fresh", comp.id, out_port))
            else:
                self.meas_sources.append(("external", comp.id, out_port))

        # Gradients must not be silently lost: every theta path through the
        # real system has to be threaded through F, never frozen into a
        # captured constant.
        self._validate_theta_influence()

    # -- static graph analysis ----------------------------------------------
    def _validate_theta_influence(self):
        """Refuse to compose when theta gradients would silently be lost.

        The composed map freezes every input without a composed producer into
        a captured constant.  That is exact -- in value AND gradient -- only
        when the frozen signal is truly exogenous (weather, schedules).  Two
        structural violations make the frozen value theta-dependent, so the
        composed objective would MATCH the object graph in value at the
        reference theta while its gradient silently loses the paths through
        the frozen signal (observed historically: an ``OccupancySystem`` --
        before it grew its pure ``forward`` -- with estimated
        ``V``/``G_occ``/``m_inf`` feeding ``numberOfPeople``):

        1. a theta component outside the influence cone (not composed at
           all -- e.g. no functorch-safe ``forward``, or only reachable
           through a non-composable component);
        2. a captured input whose upstream object-graph ancestry contains a
           theta component (theta leaks into the "constant" through a
           non-composable intermediary).

        Raises ``RuntimeError``; callers treat that as "fall back to the
        object-graph engine".
        """
        missing = sorted(
            cid for cid in self.theta_by_comp if cid not in self.cone_ids
        )
        if missing:
            raise RuntimeError(
                "theta components not composable (their gradient paths would "
                f"be frozen into captured constants): {missing}"
            )

        by_id = {c.id: c for c in self.order}
        for key in self._captured_keys:
            comp = by_id.get(key[0])
            if comp is None:
                continue
            if len(key) == 3:
                starts = [
                    (prod, oport)
                    for slot, prod, oport in self._vector_slot_sources(comp, key[1])
                    if slot == key[2]
                ]
            else:
                src = _single_source(comp, key[1])
                starts = [src] if src is not None else []
            hit = self._upstream_theta_component([p for p, _ in starts])
            if hit is not None:
                raise RuntimeError(
                    f"captured input {key} depends on theta component "
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
            for port in list(c.input.keys()):
                if isinstance(c.input[port], tps.Vector):
                    for _, prod, _ in self._vector_slot_sources(c, port):
                        stack.append(prod)
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
                            (p, o) for _, p, o in self._vector_slot_sources(c, port)
                        ]
                    else:
                        src = self._trace_source(c, port)
                        srcs = [src] if src is not None else []
                    for prod, _ in srcs:
                        if (
                            prod.id in self.forward_ids
                            and self.pos[prod.id] < self.pos[c.id]
                            and prod.id not in keep
                        ):
                            keep.add(prod.id)
                            changed = True
        return [c for c in self.order if c.id in keep and c.id in self.forward_ids]

    def _trace_source(self, comp, port):
        """Resolve ``comp.input[port]`` to a ``(producer, out_port)``, following
        pass-through sensors to their upstream source.  ``None`` if exogenous."""
        src = _single_source(comp, port)
        if src is None:
            return None
        return self._follow(*src)

    def _follow(self, producer, out_port):
        """Follow pass-through sensors from a ``(producer, out_port)`` pair."""
        if _is_passthrough_sensor(producer):
            return self._trace_source(producer, "measuredValue")
        return producer, out_port

    def _vector_slot_sources(self, comp, port):
        """Per-slot ``(slot_index, producer, out_port)`` list for a Vector input
        port (e.g. ``MaxSystem.inputs``), pass-through sensors followed, sorted
        by slot index."""
        slots = []
        for cp in comp.connects_at:
            if cp.input_port != port:
                continue
            for conn in cp.connects_system_through:
                idx = cp.input_port_index.get(conn, 0)
                idx = int(idx.item()) if hasattr(idx, "item") else int(idx)
                src = self._follow(conn.connects_system, conn.output_port)
                if src is not None:
                    slots.append((idx, src[0], src[1]))
        return sorted(slots, key=lambda t: t[0])

    def _classify_source(self, comp, key, src):
        """Spec for one resolved input source.

        ``key`` identifies the consumer slot -- ``(comp_id, port)`` for scalar
        ports, ``(comp_id, port, slot)`` for vector-port slots -- and indexes
        the feedback / captured tables.
        """
        src_in_cone = (
            src is not None
            and src[0].id in self.forward_ids
            and src[0].id in self.cone_ids
        )
        if src_in_cone and self.pos[src[0].id] < self.pos[comp.id]:
            # Produced earlier in this step -> thread it fresh.
            return ("fresh", src[0].id, src[1])
        if src_in_cone:
            # Source is a cone forward-component that executes LATER: this is
            # the cycle-broken (feedback) edge.  Its value is a function of the
            # states/params, so it becomes a decision variable tied to the
            # producer's output by a defect -- NOT a frozen constant.
            if key not in self._fb_index:
                self._fb_index[key] = len(self._feedback_keys)
                self._feedback_keys.append(key)
                self._fb_producer.append((src[0].id, src[1]))
            return ("feedback", self._fb_index[key])
        # Truly exogenous (no cone producer): frozen capture is correct.
        if key not in self._cap_index:
            self._cap_index[key] = len(self._captured_keys)
            self._captured_keys.append(key)
        return ("captured", self._cap_index[key])

    def _resolve_inputs(self, comp):
        """Per-input-port source spec for one cone component.

        Scalar ports resolve to a single fresh/feedback/captured spec.  Vector
        ports (e.g. ``MaxSystem.inputs``) resolve **per slot** to a ``("vector",
        [slot_spec, ...])`` spec, so a producer inside the cone (e.g. the CO2
        controller feeding the damper max) is threaded fresh instead of frozen.
        Unconnected vector ports (n_v=0, e.g. ``wallHeatGain``) are
        skipped.
        """
        specs = []
        for port in comp.input.keys():
            if isinstance(comp.input[port], tps.Vector):
                slot_srcs = self._vector_slot_sources(comp, port)
                if not slot_srcs:
                    continue
                slot_specs = [
                    self._classify_source(comp, (comp.id, port, slot), (prod, oport))
                    for slot, prod, oport in slot_srcs
                ]
                specs.append((port, ("vector", slot_specs)))
                continue
            src = self._trace_source(comp, port)
            specs.append((port, self._classify_source(comp, (comp.id, port), src)))
        return specs

    # -- the pure one-step map ----------------------------------------------
    # (Captured-input sampling lives in :func:`capture_reference_rollout` --
    # the single continuous-rollout source of truth; a per-segment capture
    # would evaluate stateful/data-indexed signals like the OccupancySystem's
    # ``previousIndoorCo2Measured`` at the wrong step.)
    def _params_for(self, comp, theta):
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
                    # Estimated params come from ``theta`` as 0-dim scalars; shape
                    # them to ``(n_c=1,)`` to match the default ``.get()`` values
                    # (some components read n_c from a parameter's shape).
                    p[name] = theta[est[name]].reshape(1)
                else:
                    # rgetattr: PARAM_NAMES may contain dotted paths into
                    # owned sub-objects (e.g. "supply_damper.a").
                    p[name] = rgetattr(comp, name).get()
        # Prefixed estimated params (composite: "thermal.C_air") -> pass through.
        for attr, idx in est.items():
            if "." in attr:
                p[attr] = theta[idx].reshape(1)
        cache[1][comp.id] = p
        return p

    def F(self, states_flat, theta, captured, feedback=None):
        """One pure step for a single segment.

        Args:
            states_flat: ``(D,)`` concatenated stateful-component states.
            theta: ``(n_theta,)`` physical estimated parameters (theta_spec order).
            captured: ``(n_captured,)`` exogenous input values for this segment.
            feedback: ``(n_feedback,)`` cut-feedback input values (decision
                variables) for this segment; ``None`` -> zeros (n_feedback==0).

        Returns:
            ``(x_next_flat (D,), meas (n_meas,), fb_out (n_feedback,))`` -- the
            next state, the modelled measured outputs, and the producer outputs
            that the feedback variables must match (for the feedback defects).
        """
        if feedback is None:
            feedback = torch.zeros(len(self._feedback_keys), dtype=states_flat.dtype)
        # Unpack per-component states.
        states = {}
        for i, c in enumerate(self.stateful):
            a, b = self.state_offsets[i], self.state_offsets[i + 1]
            states[c.id] = states_flat[a:b].unsqueeze(0)  # (n_c=1, width)

        produced: Dict[str, Dict[str, torch.Tensor]] = {}
        x_next_parts = [None] * len(self.stateful)
        for c in self.cone:
            inputs = {}
            for port, spec in self.wiring[c.id]:
                if spec[0] == "fresh":
                    inputs[port] = produced[spec[1]][spec[2]]
                elif spec[0] == "feedback":
                    inputs[port] = feedback[spec[1]].reshape(1)  # (n_c=1,)
                elif spec[0] == "vector":
                    vals = []
                    for s in spec[1]:
                        if s[0] == "fresh":
                            vals.append(produced[s[1]][s[2]].reshape(-1)[0])
                        elif s[0] == "feedback":
                            vals.append(feedback[s[1]].reshape(-1)[0])
                        else:
                            vals.append(captured[s[1]].reshape(-1)[0])
                    inputs[port] = torch.stack(vals).reshape(1, -1)  # (n_c=1, n_v)
                else:
                    inputs[port] = captured[spec[1]].reshape(1)  # (n_c=1,)
            params = self._params_for(c, theta)
            st = states.get(c.id, None)
            x_next_c, outs = c.forward(st, inputs, params, self.sample_time)
            produced[c.id] = outs
            if c.id in self.state_index:
                x_next_parts[self.state_index[c.id]] = x_next_c.reshape(-1)
        x_next = torch.cat(x_next_parts)
        # Producer outputs the feedback decision variables must equal.
        if self._fb_producer:
            fb_out = torch.stack([
                produced[pid][port].reshape(-1)[0] for (pid, port) in self._fb_producer
            ])
        else:
            fb_out = torch.zeros(0, dtype=x_next.dtype)
        meas = []
        for spec in self.meas_sources:
            if spec[0] == "fresh":
                meas.append(produced[spec[1]][spec[2]].reshape(-1)[0])
            elif spec[0] == "external":
                # Not producible by the composed map; the caller supplies the
                # signal (e.g. a decision-variable trajectory) or rejects.
                meas.append(torch.zeros((), dtype=x_next.dtype))
            else:
                meas.append(captured[spec[1]].reshape(-1)[0])
        meas = torch.stack(meas) if meas else torch.zeros(0, dtype=x_next.dtype)
        return x_next, meas, fb_out

    @property
    def n_feedback(self) -> int:
        return len(self._feedback_keys)

    @property
    def D_aug(self) -> int:
        """Augmented-state width: component states + cut-feedback lag variables."""
        return self.D + len(self._feedback_keys)

    def F_aug(self, y_flat, theta, captured):
        """Augmented one-step map over ``y = [state | feedback]``.

        The cut-feedback signals are one-step *lag variables* -- i.e. state in a
        discrete-time sense -- so appending them to the state turns the feedback
        loop into ordinary state continuity: ``y_{t+1} = [F(s_t, w_t),
        producer_output(s_t, w_t)]``.  The producer output computed at step ``t``
        becomes the feedback consumed at ``t+1``, exactly ``do_step``'s
        one-step-delayed (Gauss-Seidel) semantics -- with no separate defect type.

        Returns ``(y_next (D_aug,), meas (n_meas,))``.
        """
        n_fb = len(self._feedback_keys)
        s = y_flat[: self.D]
        w = y_flat[self.D:] if n_fb else None
        x_next, meas, fb_out = self.F(s, theta, captured, w)
        y_next = torch.cat([x_next, fb_out]) if n_fb else x_next
        return y_next, meas


def capture_reference_rollout(
    simulator, composer, start_time, end_time, step_size,
    layout=None, meas_ids=(),
):
    """THE reference-rollout capture (single source of truth).

    One batched, CONTINUOUS ``do_step`` rollout over all periods at the
    model's *current* parameters and inputs, sampling every relevant input
    port right after each step.  Every fast path that consumes the composed
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

    The captured signals are independent of the unknowns by construction
    (anything the estimated parameters influence is inside the composer's
    cone, hence fresh or feedback; decision-variable slots are overridden by
    the Optimizer), so capturing once at the current model state is valid for
    every iterate.

    The run is batched over the periods (all periods share one
    ``model.initialize``), exactly like the object-graph objectives the fast
    paths replace.  Decision-variable ports with ``requires_grad`` set keep
    their current (initial-iterate) trajectories through the re-initialization,
    which is what the Optimizer requires.

    Parameters
    ----------
    simulator, composer
        The simulator (model already configured) and the ``OneStepComposer``
        whose ``_captured_keys`` / ``_feedback_keys`` define what to sample.
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
        ``CAP`` (each ``(n_t_p, n_captured)``),
        ``FB`` (each ``(n_t_p, n_feedback)``),
        ``MEAS`` (each ``(n_t_p, len(meas_ids))``),
        ``n_t`` (each ``int``).
    """
    import twin4build.core as core

    model = simulator.model
    comps = model.components
    cap_keys = composer._captured_keys
    fb_keys = composer._feedback_keys
    meas_keys = [(mid, "measuredValue") for mid in meas_ids]

    starts, ends, steps = list(start_time), list(end_time), list(step_size)
    simulator.get_simulation_timesteps(starts, ends, steps)
    model.initialize(starts, ends, steps)
    sec, dts, max_t, n_ts = core.Simulator.get_simulation_timesteps(
        starts, ends, steps
    )
    n_s = len(starts)
    n_t = [int(n) for n in n_ts]

    state0 = (
        [layout.gather(p).detach().clone() for p in range(n_s)]
        if layout is not None
        else None
    )
    CAP = torch.zeros((int(max_t), n_s, len(cap_keys)), dtype=torch.float64)
    FB = torch.zeros((int(max_t), n_s, len(fb_keys)), dtype=torch.float64)
    MEAS = torch.zeros((int(max_t), n_s, len(meas_keys)), dtype=torch.float64)

    def _sample(dst, keys):
        for k, key in enumerate(keys):
            cid, port = key[0], key[1]
            slot = key[2] if len(key) > 2 else 0
            val = comps[cid].input[port].get()
            dst[:, k] = val.reshape(n_s, -1)[:, slot].detach()

    with torch.no_grad():
        for t in range(int(max_t)):
            simulator._do_system_time_step(
                model, sec[:, t], dts[:, t], steps, t, "gauss-seidel"
            )
            _sample(CAP[t], cap_keys)
            if fb_keys:
                _sample(FB[t], fb_keys)
            if meas_keys:
                _sample(MEAS[t], meas_keys)

    n_fb = len(fb_keys)
    Y0 = None
    if layout is not None:
        Y0 = [
            torch.cat([state0[p], FB[0, p]]) if n_fb else state0[p]
            for p in range(n_s)
        ]
    return SimpleNamespace(
        state0=state0,
        Y0=Y0,
        CAP=[CAP[: n_t[p], p, :] for p in range(n_s)],
        FB=[FB[: n_t[p], p, :] for p in range(n_s)],
        MEAS=[MEAS[: n_t[p], p, :] for p in range(n_s)],
        n_t=n_t,
    )


def sequential_rollout(composer, y0, theta, cap):
    """Roll the composed map over one period; returns ``(n_t, n_meas)``.

    A plain Python loop, NOT ``vmap``: with a handful of periods the vmap
    dispatch overhead exceeds the batching gain, and staying in ordinary eager
    mode lets the state-space components use the fused ``torch.matrix_exp``
    (which has no vmap rule) instead of the unrolled scaling-and-squaring
    fallback.  Differentiable w.r.t. ``theta``, ``y0`` and ``cap``.

    Args:
        composer: The :class:`OneStepComposer`.
        y0: ``(D_aug,)`` augmented initial state (``[state0 | FB[0]]``).
        theta: ``(n_theta,)`` physical parameters (theta_spec order).
        cap: ``(n_t, n_captured)`` captured inputs for the period.
    """
    y = y0
    rows = []
    for t in range(cap.shape[0]):
        y, meas = composer.F_aug(y, theta, cap[t])
        rows.append(meas)
    if not rows:
        return torch.zeros((0, len(composer.meas_sources)), dtype=torch.float64)
    return torch.stack(rows)
