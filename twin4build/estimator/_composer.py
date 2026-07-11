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
        # source is ("fresh", producer_id, out_port) or ("captured", cap_index).
        self._captured_keys: List[Tuple[str, str]] = []  # (comp_id, port) needing capture
        self._cap_index: Dict[Tuple[str, str], int] = {}
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
                    src = self._trace_source(c, port)
                    if src is not None:
                        prod, _ = src
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
        producer, out_port = src
        if _is_passthrough_sensor(producer):
            return self._trace_source(producer, "measuredValue")
        return producer, out_port

    def _resolve_inputs(self, comp):
        """Per-input-port source spec for one cone component.

        Vector input ports are skipped -- in the example the only one
        (``adjacentZoneTemperature``) has ``n_v=0`` and is unused by ``forward``;
        components that do use vector inputs would need per-slot capture (future).
        """
        specs = []
        for port in comp.input.keys():
            if isinstance(comp.input[port], tps.Vector):
                continue
            src = self._trace_source(comp, port)
            if (
                src is not None
                and src[0].id in self.forward_ids
                and src[0].id in {c.id for c in self.cone}
                and self.pos[src[0].id] < self.pos[comp.id]
            ):
                specs.append((port, ("fresh", src[0].id, src[1])))
            else:
                key = (comp.id, port)
                if key not in self._cap_index:
                    self._cap_index[key] = len(self._captured_keys)
                    self._captured_keys.append(key)
                specs.append((port, ("captured", self._cap_index[key])))
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
        for j, (comp_id, port) in enumerate(self._captured_keys):
            val = comps[comp_id].input[port].history(i_t=0)  # (n_s, n_c) at step 0
            cap[:, j] = torch.as_tensor(np.asarray(val)).reshape(n_seg, -1)[:, 0]
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

    def F(self, states_flat, theta, captured):
        """One pure step for a single segment.

        Args:
            states_flat: ``(D,)`` concatenated stateful-component states.
            theta: ``(n_theta,)`` physical estimated parameters (theta_spec order).
            captured: ``(n_captured,)`` captured input values for this segment.

        Returns:
            ``x_next_flat (D,)`` -- the next concatenated stateful state.
        """
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
                else:
                    inputs[port] = captured[spec[1]].reshape(1)  # (n_c=1,)
            params = self._params_for(c, theta)
            st = states.get(c.id, None)
            x_next_c, outs = c.forward(st, inputs, params, self.sample_time)
            produced[c.id] = outs
            if c.id in self.state_index:
                x_next_parts[self.state_index[c.id]] = x_next_c.reshape(-1)
        x_next = torch.cat(x_next_parts)
        if not self.meas_sources:
            return x_next
        meas = []
        for spec in self.meas_sources:
            if spec[0] == "fresh":
                meas.append(produced[spec[1]][spec[2]].reshape(-1)[0])
            else:
                meas.append(captured[spec[1]].reshape(-1)[0])
        return x_next, torch.stack(meas)
