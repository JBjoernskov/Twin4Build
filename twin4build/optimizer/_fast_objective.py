"""Fast composed objective for :class:`~twin4build.optimizer.optimizer.Optimizer`.

The optimizer's object-graph objective costs one full ``simulator.simulate()``
per evaluation, and its gradient runs ``torch.func.jacrev`` around that whole
simulation -- a functorch-traced object-graph rollout per iteration.  As with
the Estimator's fast single-shooting path (:mod:`twin4build.estimator._shooting`),
profiling shows the cost is dominated by Python dispatch (per-step Gauss-Seidel
traversal, ``tps`` port bookkeeping, ``model.initialize`` re-reading inputs),
not tensor math.

This module builds on the Simulator's composed-map API
(:meth:`Simulator.compose` / :meth:`Simulator.capture_rollout` /
:meth:`Simulator.rollout_composed`, implemented in
:mod:`twin4build.simulator._composed`): the model is composed into a pure
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
composed map cannot express (no stateful components, ``n_c > 1``, a loss
output the map cannot produce, a decision variable with no influence on the
composed map, a lagged decision edge); callers treat that as "use the
object-graph objective".  ``tests/optimizer/test_fast_objective.py``
regression-checks end-to-end value + gradient parity;
``options={"fast_validate": True}`` re-enables the runtime cross-check as a
debugging aid.

Enabled by default: ``optimizer.optimize(..., options={"fast": False})`` opts
out.
"""

from __future__ import annotations

import math

import torch

from twin4build.utils.print_progress import LOGGER
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


class FastControlObjective:
    """Composed-map drop-in for ``Optimizer.__obj_ad`` (+ its gradient).

    Built inside ``Optimizer._scipy_solver`` AFTER the setup is complete
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
                raise RuntimeError(
                    f"loss output {comp.id}.{port} is not a Scalar port"
                )

        self.vars = [(comp, port) for comp, port, *_ in opt._variables]
        var_index = {(c.id, p): v for v, (c, p) in enumerate(self.vars)}

        # -- compose ----------------------------------------------------------
        # Structural checks (stateful components exist, n_c == 1, uniform
        # step size, state-width match) live in Simulator.compose.
        layout, composer = opt.simulator.compose(
            outputs=self.loss_ports, step_size=opt._stepSize
        )

        # Every loss output must be producible by the composed map, or BE a
        # decision variable (its trajectory is theta directly).  A frozen
        # ("captured"-style) loss signal would be theta-independent -- wrong.
        self.out_kind = []  # per loss port: ("meas", col) | ("theta", var_idx)
        for j, spec in enumerate(composer.meas_sources):
            if spec[0] == "fresh":
                self.out_kind.append(("meas", j))
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
        for j, key in enumerate(composer._captured_keys):
            consumer_id, port = key[0], key[1]
            slot = key[2] if len(key) > 2 else None
            consumer = comps[consumer_id]
            if slot is None:
                src = composer._trace_source(consumer, port)
            else:
                src = None
                for s_idx, prod, oport in composer._vector_slot_sources(
                    consumer, port
                ):
                    if s_idx == slot:
                        src = (prod, oport)
                        break
            if src is None:
                continue
            v = var_index.get((src[0].id, src[1]))
            if v is None:
                continue
            # Gauss-Seidel lag: a consumer executing BEFORE the variable's
            # producer would consume the PREVIOUS step's value; the composed
            # map would mis-align theta by one step.  Schedules execute first
            # in practice, so reject the exotic case instead of approximating.
            if composer.pos[consumer_id] < composer.pos[src[0].id]:
                raise RuntimeError(
                    f"decision variable {src[0].id}.{src[1]} is consumed "
                    f"with a one-step lag by {consumer_id}.{port}"
                )
            self.var_slots[v].append(j)

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

        # -- reference rollout (captures exogenous inputs, initial state) ------
        self._capture()

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
            self._eq_terms.append(
                (j, [
                    (d[: self.n_t[p], p, :].reshape(-1) - mn) / (mx - mn)
                    for p in range(self.n_periods)
                ])
            )
        self._ineq_terms = []  # (out_j, ctype, [desired_norm_per_period])
        for comp, port, ctype, _ in opt._ineq_cons:
            j = self._out_index[(comp.id, port)]
            mn, mx = self._out_norm[j]
            d = opt.inequality_constraint_values[(comp, port, ctype)]
            self._ineq_terms.append(
                (j, ctype, [
                    (d[: self.n_t[p], p, :].reshape(-1) - mn) / (mx - mn)
                    for p in range(self.n_periods)
                ])
            )
        self._obj_terms = [
            (self._out_index[(comp.id, port)], objective_type)
            for comp, port, objective_type in opt._objectives
        ]

        LOGGER.config(
            "Fast optimizer objective: %d period(s) x %s steps | cone=%d | "
            "captured inputs=%d (%d theta-driven) | feedback lags=%d",
            self.n_periods, self.n_t, len(composer.cone),
            len(composer._captured_keys),
            sum(len(s) for s in self.var_slots), composer.n_feedback,
        )

    # -- one-time reference rollout -------------------------------------------
    def _capture(self):
        """One batched reference ``do_step`` rollout over all periods at the
        initial iterate via :meth:`Simulator.capture_rollout` (batched matters
        here: the decision-variable ports hold the solver's initial
        trajectories, which a per-period re-initialization would disturb).

        The sampled signals are theta-independent by construction: anything a
        decision variable influences is inside the cone (fresh/feedback) or is
        one of the overridden slots.
        """
        opt = self.opt
        R = opt.simulator.capture_rollout(
            self.composer, opt._start_time, opt._end_time, opt._stepSize,
            layout=self.layout,
        )
        self.CAP, self.Y0, self.n_t = R.CAP, R.Y0, R.n_t
        self.n_periods = len(R.n_t)

    # -- the differentiable loss ------------------------------------------------
    def loss(self, theta: torch.Tensor) -> torch.Tensor:
        """Same value as ``Optimizer.__obj_ad(theta)``; differentiable w.r.t.
        ``theta`` (the solver's flattened, interleaved, normalized decision
        vector)."""
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

        # Rollout: per period, override the theta-driven captured slots and
        # step the composed map (the shared sequential rollout,
        # :meth:`Simulator.rollout_composed`).  OUT[p] is (n_t_p, n_loss_ports)
        # in PHYSICAL units, matching the object-graph history reads.
        sim = opt.simulator
        OUT = []
        for p in range(self.n_periods):
            cap = self.CAP[p]
            if any(self.var_slots):
                cap = cap.clone()
                for v, slots in enumerate(self.var_slots):
                    for j in slots:
                        cap[:, j] = theta_phys[p][:, v]
            M = sim.rollout_composed(
                self.composer, self.Y0[p], self._theta_empty, cap
            )  # (n_t_p, n_meas)
            cols = []
            for kind, idx in self.out_kind:
                if kind == "meas":
                    cols.append(M[:, idx])
                else:  # theta-driven loss output
                    cols.append(theta_phys[p][:, idx])
            OUT.append(torch.stack(cols, dim=1) if cols else M[:, :0])

        # Loss: replicates Optimizer.__obj_ad term by term.  Means over the
        # per-period concatenation equal the object graph's masked means (the
        # mask selects the same entries; means are order-invariant).
        def _norm_col(j):
            mn, mx = self._out_norm[j]
            return torch.cat(
                [(OUT[p][:, j] - mn) / (mx - mn) for p in range(self.n_periods)]
            )

        loss = torch.tensor(0.0, dtype=tps.float_dtype(), device=opt._device)
        k = opt._constraint_penalty

        for j, desired in self._eq_terms:
            y_norm = _norm_col(j)
            d_norm = torch.cat(desired)
            loss = loss + k * torch.mean(torch.abs(y_norm - d_norm))

        if self._ineq_terms:
            upper = torch.tensor(0.0, dtype=tps.float_dtype(), device=opt._device)
            lower = torch.tensor(0.0, dtype=tps.float_dtype(), device=opt._device)
            for j, ctype, desired in self._ineq_terms:
                y_norm = _norm_col(j)
                d_norm = torch.cat(desired)
                if ctype == "upper":
                    upper = upper + torch.mean(torch.relu(y_norm - d_norm))
                else:
                    lower = lower + torch.mean(torch.relu(d_norm - y_norm))
            loss = loss + k * (upper + lower)

        for j, objective_type in self._obj_terms:
            y_norm = _norm_col(j)
            if objective_type == "min":
                loss = loss + torch.mean(y_norm)
            else:
                loss = loss - torch.mean(y_norm)

        return loss

    def value_and_grad(self, theta: torch.Tensor):
        """(loss, gradient) in one reverse pass -- the fast replacement for
        ``jacrev`` around a full simulation."""
        z = theta.detach().clone().requires_grad_(True)
        f = self.loss(z)
        (g,) = torch.autograd.grad(f, z)
        return f.detach(), g
