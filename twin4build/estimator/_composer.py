"""Compose component ``forward``s into one pure one-step map for fast collocation.

Stage-3 collocation keeps the **defects** exact (computed by the real batched
simulator, with full coupling/feedback) and replaces only the **Jacobian** with a
single ``vmap(jacrev(F))`` call.  This module builds that ``F``.

``F(states_flat, theta, captured) -> x_next_flat`` runs the model's stateful (and
feeding algebraic) components' ``forward`` methods in execution order for one
segment, threading outputs to inputs.  For each input port it uses either

* the **fresh** output of an upstream component that (a) has a ``forward`` and
  (b) executes earlier in the order -- following pass-through sensors to their
  source -- so the estimated-parameter and strong-state couplings are exact; or
* a **captured** constant sampled from a reference simulation at that segment's
  timestep -- for exogenous drivers (weather/schedules) and the few cycle-broken
  feedback edges (e.g. ``office.heatGain <- space_heater.Power``).

The captured feedback makes the Jacobian *approximate* only in the weak one-step
feedback terms, which is immaterial for IPOPT convergence (it consumes
approximate Jacobians routinely) while the defects stay exact.

The composed map is pure and functorch-traceable, so ``vmap(jacrev(F))`` yields
the block-bidiagonal collocation Jacobian in one shot.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import twin4build.systems as systems
import twin4build.utils.types as tps


def _has_real_forward(comp) -> bool:
    """True iff ``comp``'s class overrides ``forward`` (not the ``nn.Module`` base
    stub, which every ``nn.Module`` inherits).  Components without their own
    ``forward`` (sensors, schedules, weather, occupancy, Max, ...) are treated as
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


class OneStepComposer:
    """Builds and evaluates the pure one-step map ``F`` for a model.

    Parameters
    ----------
    model : SimulationModel
        The (initialized) simulation model.
    stateful : list
        Stateful components in execution order (each owns ``tps.State``).
    theta_spec : list of (component, attr)
        Estimated parameters, in the decision-vector order.  ``attr`` is the
        component-relative path (e.g. ``"thermal.C_air"``, ``"kp"``).
    sample_time : float
        Segment step size in seconds.
    """

    def __init__(self, model, stateful, theta_spec, sample_time, measurements=None):
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

        # theta routing: per component id -> {attr: theta_index}
        self.theta_spec = theta_spec
        self.theta_by_comp: Dict[str, Dict[str, int]] = {}
        for i, (comp, attr) in enumerate(theta_spec):
            self.theta_by_comp.setdefault(comp.id, {})[attr] = i

        # Which components must be evaluated by F: the stateful ones plus every
        # forward-component reverse-reachable from them over fresh edges.
        self.cone = self._influence_cone()

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

    # -- static graph analysis ----------------------------------------------
    def _influence_cone(self) -> List:
        """Forward-components reverse-reachable (over fresh edges, following
        pass-through sensors) from the stateful components, in execution order."""
        keep = set(c.id for c in self.stateful)
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
        Unconnected vector ports (n_v=0, e.g. ``adjacentZoneTemperature``) are
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

    # -- capture from a reference simulation --------------------------------
    def capture(self, simulator, seg_starts, seg_ends, seg_steps) -> np.ndarray:
        """Run one reference simulation over the segments and sample every
        captured input-port value at each segment's first step.

        Returns an array ``(n_seg, n_captured)`` aligned with ``self._captured_keys``.
        """
        simulator.simulate(
            start_time=seg_starts, end_time=seg_ends, step_size=seg_steps,
            show_progress_bar=False,
        )
        n_seg = len(seg_starts)
        cap = torch.zeros((n_seg, len(self._captured_keys)), dtype=torch.float64)
        comps = self.model.components
        for j, key in enumerate(self._captured_keys):
            comp_id, port = key[0], key[1]
            slot = key[2] if len(key) > 2 else 0  # vector-port slot
            val = comps[comp_id].input[port].history(i_t=0)  # (n_s, n_c[, n_v]) at step 0
            cap[:, j] = torch.as_tensor(np.asarray(val)).reshape(n_seg, -1)[:, slot]
        return cap

    # -- the pure one-step map ----------------------------------------------
    def _params_for(self, comp, theta):
        """Physical-parameter dict for ``comp``: estimated entries from ``theta``
        (a 1-D tensor in theta_spec order), the rest from the component's
        defaults (``getattr(comp, name).get()``)."""
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
                    p[name] = getattr(comp, name).get()
        # Prefixed estimated params (composite: "thermal.C_air") -> pass through.
        for attr, idx in est.items():
            if "." in attr:
                p[attr] = theta[idx].reshape(1)
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
