"""Functional single-shooting objective for ``Optimizer``.

The optimizer's object-graph objective costs one full ``simulator.simulate()``
per evaluation, and its gradient runs ``torch.func.jacrev`` around that whole
simulation -- a functorch-traced object-graph rollout per iteration.  As with
the Estimator's functional single-shooting path
(:mod:`twin4build.estimator._single_shooting`),
profiling shows the cost is dominated by Python dispatch (per-step Gauss-Seidel
traversal, ``tps`` port bookkeeping, ``model.initialize`` re-reading inputs),
not tensor math.

This module builds on the Simulator's functional API
(:meth:`Simulator.build_functional_model` /
:meth:`Simulator.record_exogenous_inputs` /
:meth:`Simulator.rollout_functional`, implemented in
:mod:`twin4build.simulator._functional`): the model is composed into a pure
one-step map ``F_aug(y, theta, cap)`` and the loss becomes a plain sequential
torch rollout.  Two things differ from the Estimator's fast path:

* **The decision variables are exogenous trajectories** (actuator schedules),
  which the composer would normally freeze as captured constants.  Instead,
  the captured slots fed by a decision variable are *overridden per step* with
  the (denormalized) decision vector, so the rollout is differentiable w.r.t.
  the control trajectory.
* **The loss reads arbitrary component outputs** (power, temperatures, cost
  sensors) rather than measuring devices.  The composer's ``outputs``
  parameter seeds the influence cone with those producers, so even purely
  downstream signals (e.g. a cost sensor multiplying heater power by an
  electricity price) are composed.

The penalty loss itself (objective terms, equality/inequality soft penalties,
min-max normalization with the ports' cached ranges) replicates
``Optimizer.__obj_ad`` term by term, using the exact same cached normalization
floats -- populated from the reference rollout at the initial iterate, exactly
when the object-graph path would populate them.  The gradient is one
``torch.autograd.grad`` pass through the rollout (value and gradient in a
single evaluation) instead of ``jacrev`` around a full simulation.

Exactness: **by construction**, for the same reason as the Estimator's fast
path -- every composable component's ``do_step`` is a thin port-I/O wrapper
delegating to the same ``forward`` the composer threads, and the captured
exogenous inputs are theta-independent by definition (anything the decision
variables influence is inside the cone or explicitly overridden).
Construction performs structural checks and raises on any model feature the
composed map cannot express (no stateful components, a loss output the map
cannot produce, a decision variable with no influence on the
composed map, a lagged decision edge); callers treat that as "use the
object-graph objective".  ``tests/optimizer/test_functional_objectiveective.py``
regression-checks end-to-end value + gradient parity;
Functional execution is selected exclusively on ``Simulator``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from twin4build.utils._cuda_graph import (
    CudaGraphCallable,
    is_cuda_graph_capture_invalidated,
)
from twin4build.utils.logger import LOGGER
import twin4build.utils.types as tps


def _norm_floats(port):
    """The affine map ``normalize`` applies, as plain floats.

    Replicates ``tps.Scalar.normalize``: cached raw ``(min, max)`` of the
    port's history, remapped to ``(0, 1)`` when degenerate (min == max, by
    ``torch.allclose`` semantics).
    """
    mn, mx = float(port._min_history), float(port._max_history)
    if math.isclose(mn, mx, rel_tol=1e-5, abs_tol=1e-8):
        mn, mx = 0.0, 1.0
    return mn, mx


def _denorm_floats(port):
    """The affine map ``denormalize`` applies (raw cached floats, no
    degenerate-case remap -- mirroring ``tps.Scalar.denormalize``)."""
    return float(port._min_history), float(port._max_history)


class FunctionalControlObjective:
    """Functional-model drop-in for ``Optimizer.__obj_ad`` and its gradient.

    Built inside ``Optimizer._solve_scipy`` after the setup is complete
    (variables' ``do_normalization`` set, model initialized at the initial
    iterate, constraint values precomputed, ``x0`` extracted).  Construction
    raises ``RuntimeError`` on any incompatibility; callers treat that as
    "fall back to the object-graph objective".
    """

    def __init__(self, optimizer):
        self.opt = opt = optimizer
        model = opt.simulator.model

        # -- loss signals -----------------------------------------------------
        # Unique (component, output_port) pairs the loss reads, in first-use
        # order: objectives, then equality, then inequality constraints.
        self.loss_ports = []
        self._out_index = {}
        referenced = (
            [(c, p) for c, p, _ in opt._objectives]
            + [(c, p) for c, p, _ in opt._eq_cons]
            + [(c, p) for c, p, _, _ in opt._ineq_cons]
        )
        for comp, port in referenced:
            if (comp.id, port) not in self._out_index:
                self._out_index[(comp.id, port)] = len(self.loss_ports)
                self.loss_ports.append((comp, port))
        for comp, port in self.loss_ports:
            if not isinstance(comp.output[port], tps.Scalar):
                raise RuntimeError(f"loss output {comp.id}.{port} is not a Scalar port")

        self.vars = [(comp, port) for comp, port, *_ in opt._variables]
        var_index = {(c.id, p): v for v, (c, p) in enumerate(self.vars)}

        # -- compose ----------------------------------------------------------
        # Structural checks (stateful components exist, uniform step size,
        # state-width match) live in Simulator.compose.
        layout, composer = opt.simulator.build_functional_model(
            outputs=self.loss_ports, step_size=opt._stepSize
        )

        # Every loss output must be producible by the composed map, or BE a
        # decision variable (its trajectory is theta directly).  A frozen
        # ("captured"-style) loss signal would be theta-independent -- wrong.
        self.out_kind = []  # per loss port: ("meas", slice) | ("theta", var_idx)
        for j, (spec, meas_slice) in enumerate(
            zip(composer.meas_sources, composer.meas_slices)
        ):
            if spec[0] == "fresh":
                self.out_kind.append(("meas", meas_slice))
            else:  # ("external", comp_id, port)
                v = var_index.get((spec[1], spec[2]))
                if v is None:
                    raise RuntimeError(
                        f"loss output {spec[1]}.{spec[2]} is not producible "
                        "by the composed map"
                    )
                self.out_kind.append(("theta", v))

        # -- decision variables -> captured slots ------------------------------
        # A decision variable is an exogenous leaf trajectory; the composer
        # freezes every exogenous consumer-input as a captured slot.  Find the
        # slots whose traced source is a decision variable: those columns of
        # the captured matrix are overridden with theta at every step.
        # Fused state-space clusters consume their members' exogenous inputs
        # under the fused component's id (namespaced ports), so include them.
        comps = dict(model.components)
        sim_model = getattr(model, "_simulation_model", None) or model
        comps.update(getattr(sim_model, "_fused_components", None) or {})
        self.var_slots = [[] for _ in self.vars]
        for key in composer._exogenous_keys:
            consumer_id, port = key[0], key[1]
            slot = key[2] if len(key) > 2 else None
            consumer = comps[consumer_id]
            if slot is None:
                sources = composer._trace_sources(consumer, port)
            else:
                sources = []
                for s_idx, slot_sources in composer._vector_slot_sources(
                    consumer, port
                ):
                    if s_idx == slot:
                        sources = slot_sources
                        break
            cap_slice = composer._exogenous_index[key]
            for src in sources:
                v = var_index.get((src[0].id, src[1]))
                if v is None:
                    continue
                # Gauss-Seidel lag: a consumer executing BEFORE the variable's
                # producer would consume the PREVIOUS step's value.
                if composer.pos[consumer_id] < composer.pos[src[0].id]:
                    raise RuntimeError(
                        f"decision variable {src[0].id}.{src[1]} is consumed "
                        f"with a one-step lag by {consumer_id}.{port}"
                    )
                src_width = composer._component_n_c(src[0])
                marker = torch.ones(
                    src_width, dtype=tps.float_dtype(), device=model.device
                )
                routed = composer._apply_routes(marker, src[2]).reshape(-1)
                columns = list(range(cap_slice.start, cap_slice.stop))
                active = [
                    col for col, value in zip(columns, routed) if bool(value != 0)
                ]
                self.var_slots[v].extend(active)

        for v, ((comp, port), slots) in enumerate(zip(self.vars, self.var_slots)):
            drives_loss = any(
                kind == "theta" and idx == v for kind, idx in self.out_kind
            )
            if not slots and not drives_loss:
                raise RuntimeError(
                    f"decision variable {comp.id}.{port} has no influence on "
                    "the composed map"
                )

        self.layout = layout
        self.composer = composer
        self._theta_empty = torch.zeros(0, dtype=tps.float_dtype(), device=model.device)

        # Functional slot override: instead of in-place column writes into a
        # cloned captured matrix (which torch.func.vmap cannot batch), the
        # theta-driven columns are selected with a boolean mask and a
        # (n_vars, n_captured) 0/1 matrix S so that ``theta_phys @ S`` scatters
        # each variable's trajectory into its captured columns.  Values are
        # identical to the in-place form; the loss becomes vmap-able over an
        # extra leading theta batch dimension (used by the Pareto prepass).
        n_cap = composer._n_exogenous
        slot_mask = torch.zeros(n_cap, dtype=torch.bool, device=model.device)
        S = torch.zeros(
            (len(self.vars), n_cap), dtype=tps.float_dtype(), device=model.device
        )
        for v, slots in enumerate(self.var_slots):
            for j in slots:
                slot_mask[j] = True
                S[v, j] = 1.0
        self._slot_mask = slot_mask
        self._slot_matrix = S
        # Shape/topology invariant. Keep the branch predicate on the host so
        # direct CUDA Graph capture never performs a forbidden device-to-host
        # synchronization inside the captured region.
        self._has_slots = bool(slot_mask.any())

        # -- reference rollout (captures exogenous inputs, initial state) ------
        self._record_exogenous_inputs()

        # -- normalization floats ----------------------------------------------
        # Populate each loss port's normalization cache from the reference
        # history -- exactly the history the object-graph path would cache from
        # on its first evaluation (both are the model at the initial iterate).
        for comp, port in self.loss_ports:
            p = comp.output[port]
            if p._min_history is None:
                p.normalize()
        self._out_norm = [
            _norm_floats(comp.output[port]) for comp, port in self.loss_ports
        ]
        # Decision-variable denormalization (theta -> physical), mirroring
        # ``port.denormalize`` (raw floats).  Identity when the solver did not
        # enable normalization for that port.
        self._var_denorm = []
        for comp, port in self.vars:
            p = comp.output[port]
            if p.do_normalization:
                self._var_denorm.append(_denorm_floats(p))
            else:
                self._var_denorm.append((0.0, 1.0))

        # Constraint targets, normalized with the constrained port's floats
        # (same affine ``normalize(desired)`` applies in the object graph),
        # pre-sliced per period.
        self._eq_terms = []  # (out_j, [desired_norm_per_period])
        for comp, port, _ in opt._eq_cons:
            j = self._out_index[(comp.id, port)]
            mn, mx = self._out_norm[j]
            d = opt.equality_constraint_values[(comp, port)]
            width = (
                self.composer.meas_slices[j].stop - self.composer.meas_slices[j].start
            )
            self._eq_terms.append(
                (
                    j,
                    [
                        (
                            d[: self.n_t[p], p, :]
                            .reshape(self.n_t[p], -1)
                            .expand(-1, width)
                            .reshape(-1)
                            - mn
                        )
                        / (mx - mn)
                        for p in range(self.n_periods)
                    ],
                )
            )
        self._ineq_terms = []  # (out_j, ctype, [desired_norm_per_period])
        for comp, port, ctype, _ in opt._ineq_cons:
            j = self._out_index[(comp.id, port)]
            mn, mx = self._out_norm[j]
            d = opt.inequality_constraint_values[(comp, port, ctype)]
            width = (
                self.composer.meas_slices[j].stop - self.composer.meas_slices[j].start
            )
            self._ineq_terms.append(
                (
                    j,
                    ctype,
                    [
                        (
                            d[: self.n_t[p], p, :]
                            .reshape(self.n_t[p], -1)
                            .expand(-1, width)
                            .reshape(-1)
                            - mn
                        )
                        / (mx - mn)
                        for p in range(self.n_periods)
                    ],
                )
            )
        self._obj_terms = [
            (self._out_index[(comp.id, port)], objective_type)
            for comp, port, objective_type in opt._objectives
        ]

        LOGGER.config(
            "Functional optimizer objective: %d period(s) x %s steps | cone=%d | "
            "captured inputs=%d (%d theta-driven) | feedback lags=%d",
            self.n_periods,
            self.n_t,
            len(composer.cone),
            composer._n_exogenous,
            sum(len(s) for s in self.var_slots),
            composer.n_feedback,
        )

    # -- one-time reference rollout -------------------------------------------
    def _record_exogenous_inputs(self):
        """One batched reference ``do_step`` rollout over all periods at the
        initial iterate via :meth:`Simulator.record_exogenous_inputs` (batched matters
        here: the decision-variable ports hold the solver's initial
        trajectories, which a per-period re-initialization would disturb).

        The sampled signals are theta-independent by construction: anything a
        decision variable influences is inside the cone (fresh/feedback) or is
        one of the overridden slots.
        """
        opt = self.opt
        R = opt.simulator.record_exogenous_inputs(
            self.composer,
            opt._start_time,
            opt._end_time,
            opt._stepSize,
            layout=self.layout,
        )
        self.CAP = R.exogenous_tape
        self.Y0 = R.Y0
        self.n_t = R.n_timesteps
        self.n_periods = len(R.n_timesteps)

    # -- the differentiable loss ------------------------------------------------
    def _rollout(self, theta: torch.Tensor, *, transform_mode: bool = False):
        """Composed-map rollout at ``theta``: per-period loss-port
        trajectories ``OUT[p][j]`` of shape ``(n_t_p, n_c_j)`` in PHYSICAL
        units, matching each object-graph port history including branches."""
        opt = self.opt
        n_vars = len(self.vars)
        theta_m = theta.reshape(-1, n_vars)  # (sum n_t, n_vars)

        # Per-period physical trajectories per variable: theta rows are
        # period-0 timesteps, then period-1, ... (the solver's layout).
        theta_phys = []  # per period: (n_t_p, n_vars)
        row = 0
        for p in range(self.n_periods):
            block = theta_m[row : row + self.n_t[p], :]
            row += self.n_t[p]
            cols = []
            for v in range(n_vars):
                mn, mx = self._var_denorm[v]
                cols.append(block[:, v] * (mx - mn) + mn)
            theta_phys.append(torch.stack(cols, dim=1))

        # Rollout: per period, override the theta-driven captured slots
        # (functionally -- see __init__) and step the composed map (the shared
        # sequential rollout, :meth:`Simulator.rollout_functional`).
        sim = opt.simulator
        OUT = []
        for p in range(self.n_periods):
            cap = self.CAP[p]
            if self._has_slots:
                theta_cap = theta_phys[p] @ self._slot_matrix  # (n_t_p, n_cap)
                cap = torch.where(self._slot_mask, theta_cap, cap)
            M = sim.rollout_functional(
                self.composer,
                self.Y0[p],
                self._theta_empty,
                cap,
                transform_mode=transform_mode,
            )  # (n_t_p, n_meas)
            ports = []
            for kind, idx in self.out_kind:
                if kind == "meas":
                    ports.append(M[:, idx])
                else:  # theta-driven loss output
                    ports.append(theta_phys[p][:, idx : idx + 1])
            OUT.append(ports)
        return OUT

    def parts(
        self, theta: torch.Tensor, *, transform_mode: bool = False
    ) -> SimpleNamespace:
        """The loss decomposed into its differentiable components (one rollout).

        Returns a namespace with:

        - ``eq``: list of k-weighted equality-penalty scalars (one per
          equality constraint, in ``_eq_terms`` order),
        - ``ineq``: the k-weighted inequality-penalty scalar (``None`` when
          there are no inequality constraints),
        - ``objs``: per-objective *min-oriented* normalized means (``+mean``
          for "min", ``-mean`` for "max"), in ``opt._objectives`` order,
        - ``phys``: per-objective PHYSICAL (unnormalized, unsigned) means --
          reporting values for e.g. Pareto fronts.

        ``loss()`` reassembles exactly the object-graph accumulation order, so
        existing callers see bit-identical values.
        """
        opt = self.opt
        OUT = self._rollout(theta, transform_mode=transform_mode)

        # Means over the per-period concatenation equal the object graph's
        # masked means (the mask selects the same entries; means are
        # order-invariant).
        def _norm_col(j):
            mn, mx = self._out_norm[j]
            return torch.cat(
                [
                    ((OUT[p][j] - mn) / (mx - mn)).reshape(-1)
                    for p in range(self.n_periods)
                ]
            )

        k = opt._constraint_penalty

        eq = []
        for j, desired in self._eq_terms:
            y_norm = _norm_col(j)
            d_norm = torch.cat(desired)
            eq.append(k * torch.mean(torch.abs(y_norm - d_norm)))

        ineq = None
        if self._ineq_terms:
            upper = theta.new_zeros(())
            lower = theta.new_zeros(())
            for j, ctype, desired in self._ineq_terms:
                y_norm = _norm_col(j)
                d_norm = torch.cat(desired)
                if ctype == "upper":
                    upper = upper + torch.mean(torch.relu(y_norm - d_norm))
                else:
                    lower = lower + torch.mean(torch.relu(d_norm - y_norm))
            ineq = k * (upper + lower)

        objs = []
        phys = []
        for j, objective_type in self._obj_terms:
            y_norm = _norm_col(j)
            m = torch.mean(y_norm)
            objs.append(m if objective_type == "min" else -m)
            phys.append(
                torch.mean(
                    torch.cat([OUT[p][j].reshape(-1) for p in range(self.n_periods)])
                )
            )

        return SimpleNamespace(eq=eq, ineq=ineq, objs=objs, phys=phys)

    def penalty(self, p: SimpleNamespace) -> torch.Tensor:
        """Sum of the constraint-penalty components of ``parts()`` output, in
        the same accumulation order as ``loss()``."""
        total = p.objs[0].new_zeros(())
        for e in p.eq:
            total = total + e
        if p.ineq is not None:
            total = total + p.ineq
        return total

    def loss(
        self, theta: torch.Tensor, *, transform_mode: bool = False
    ) -> torch.Tensor:
        """Same value as ``Optimizer.__obj_ad(theta)``; differentiable w.r.t.
        ``theta`` (the solver's flattened, interleaved, normalized decision
        vector)."""
        p = self.parts(theta, transform_mode=transform_mode)
        loss = theta.new_zeros(())
        for e in p.eq:
            loss = loss + e
        if p.ineq is not None:
            loss = loss + p.ineq
        for o in p.objs:
            loss = loss + o
        return loss

    def value_and_grad(self, theta: torch.Tensor):
        """(loss, gradient) in one reverse pass -- the fast replacement for
        ``jacrev`` around a full simulation."""
        z = theta.detach().clone().requires_grad_(True)
        f = self.loss(z)
        (g,) = torch.autograd.grad(f, z)
        return f.detach(), g

    def captured_value_and_grad(self, theta: torch.Tensor) -> torch.Tensor:
        """Fixed-shape scalar value/gradient bundle for direct graph capture."""
        z = theta.detach().clone().requires_grad_(True)
        f = self.loss(z, transform_mode=True)
        (g,) = torch.autograd.grad(f, z)
        return torch.cat((f.detach().reshape(1), g.detach()))


class CapturedControlObjective:
    """Own the CUDA Graph used by SciPy's functional SLSQP callbacks."""

    def __init__(self, objective: FunctionalControlObjective):
        self.objective = objective
        self.enabled = (
            objective.opt.simulator.execution_backend == "cuda_graph"
            and objective.opt._device.type == "cuda"
        )
        self.graph = None
        self.stats = {
            "requested": objective.opt.simulator.execution_backend == "cuda_graph",
            "enabled": self.enabled,
            "calls": 0,
            "captures": 0,
            "replays": 0,
            "fallback_reason": None,
        }

    def value_and_grad(self, theta: torch.Tensor):
        self.stats["calls"] += 1
        if not self.enabled:
            return self.objective.value_and_grad(theta)
        try:
            capturing = self.graph is None
            if self.graph is None:
                self.graph = CudaGraphCallable(self.objective.captured_value_and_grad)
            # The graph output aliases static storage; SciPy's split callbacks
            # retain both values across subsequent solver evaluations.
            bundle = self.graph(theta).clone()
            self.stats["captures" if capturing else "replays"] += 1
            return bundle[0], bundle[1:]
        except Exception as exc:
            if is_cuda_graph_capture_invalidated(exc):
                raise
            self.stats["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            self.stats["enabled"] = False
            self.enabled = False
            self.close()
            LOGGER.warning(
                "CUDA Graph optimizer bundle unavailable (%s); using eager.",
                self.stats["fallback_reason"],
            )
            return self.objective.value_and_grad(theta)

    def close(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None
