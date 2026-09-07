"""Sparse one-step collocation for IPOPT Pareto sweeps.

The transcription reuses the simulator's composed augmented one-step map.  Its
decision vector is ``z = [u_norm | y_norm]`` where ``u`` is the ordinary
interleaved control trajectory and ``y`` contains one augmented boundary state
per timestep.  Compiled same-class batches remain inside the flattened
``StateLayout``/``F_aug`` state, including ``n_c > 1`` components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.func import jvp, vjp, vmap

import twin4build.utils.types as tps
from twin4build.estimator._collocation import _IterateCache
from twin4build.simulator._replica_layout import ReplicaLayout
from twin4build.solvers.ipopt import solve_ipopt_constrained
from twin4build.optimizer._pareto_common import (
    ParetoResult,
    _pareto_mask,
    batched_prepass,
)
from twin4build.utils._cuda_graph import CudaGraphCallable


@dataclass
class _Solve:
    result: object
    values: dict


class ParetoCollocation:
    """Fixed-shape sparse callbacks for one composed Pareto NLP."""

    def __init__(self, opt, controls0, bounds, *, delta=0.0, options=None):
        if opt._functional_objective is None:
            raise ValueError(
                "IPOPT Pareto collocation requires composed execution; construct "
                "Simulator(model, execution_mode='functional')."
            )
        self.opt = opt
        self.fast = fast = opt._functional_objective
        self.dev = opt._device
        self.dtype = tps.float_dtype()
        self.delta = float(delta)
        self.options = dict(options or {})
        self.n_vars = len(fast.vars)
        self.n_u = int(np.asarray(controls0).size)
        self.n_seg = sum(fast.n_t)
        self.Da = fast.composer.D_aug
        self.replica_layout = ReplicaLayout.from_functional_model(fast.composer)
        self.n_replicas = self.replica_layout.n_replicas
        self.d_zone = self.replica_layout.local_width
        self.d_global = self.replica_layout.shared_width
        self._global_idx, self._replica_idx = self.replica_layout.device_indices(
            self.dev
        )
        self._local_color_basis = self.replica_layout.colored_local_basis(
            dtype=self.dtype, device=self.dev
        )
        self._global_color_basis = torch.zeros(
            (self.d_global, self.Da), dtype=self.dtype, device=self.dev
        )
        if self.d_global:
            self._global_color_basis[
                torch.arange(self.d_global, device=self.dev), self._global_idx
            ] = 1
        self.n_y = self.n_seg * self.Da
        self.n_z = self.n_u + self.n_y
        if self.n_u != self.n_seg * self.n_vars:
            raise ValueError(
                "Control trajectory shape does not match collocation segments"
            )

        self.CAP = torch.cat(fast.CAP, dim=0)
        self.period_starts = []
        self.links = []
        offset = 0
        for count in fast.n_t:
            self.period_starts.append(offset)
            self.links.extend((i, i + 1) for i in range(offset, offset + count - 1))
            offset += count
        self.cp_i = torch.tensor(
            [i for i, _ in self.links], dtype=torch.long, device=self.dev
        )
        self.cp_j = torch.tensor(
            [j for _, j in self.links], dtype=torch.long, device=self.dev
        )
        self.n_links = len(self.links)

        controls0_t = torch.as_tensor(controls0, dtype=self.dtype, device=self.dev)
        y0_phys = self._lift_physical(controls0_t)
        self.center = y0_phys.mean(0)
        scale = y0_phys.std(0) if self.n_seg > 1 else torch.zeros_like(self.center)
        floor = 0.1 * self.center.abs() + 1e-3
        self.scale = torch.maximum(scale, floor)
        y0 = self.y_to_norm(y0_phys)
        self.z0 = np.concatenate(
            [np.asarray(controls0, dtype=np.float64), y0.reshape(-1).cpu().numpy()]
        )
        if bounds is None:
            ulb = np.full(self.n_u, -np.inf)
            uub = np.full(self.n_u, np.inf)
        else:
            ulb = np.asarray(bounds.lb, dtype=np.float64)
            uub = np.asarray(bounds.ub, dtype=np.float64)
        self.lb = np.concatenate([ulb, np.full(self.n_y, -30.0)])
        self.ub = np.concatenate([uub, np.full(self.n_y, 30.0)])
        # ParetoResult.apply stores controls only, so the collocation manifold
        # must use the same model initial condition as a normal rollout.
        # Otherwise IPOPT could optimize a free initial temperature and the
        # reported point would change when applied.
        y0_np = y0.detach().cpu().numpy()
        for start in self.period_starts:
            a = self.n_u + start * self.Da
            self.lb[a : a + self.Da] = y0_np[start]
            self.ub[a : a + self.Da] = y0_np[start]

        self.obj_index = 0
        self.con_index = 1
        self.ideal2 = 0.0
        self.range2 = 1.0
        self.cache = _IterateCache()
        self._bundle_graph = None
        self._jacobian_graph = None
        self._hessian_graph = None
        self.stats = {
            "bundle_calls": 0,
            "bundle_replays": 0,
            "bundle_captured": False,
            "jacobian_calls": 0,
            "jacobian_replays": 0,
            "jacobian_captured": False,
            "hessian_calls": 0,
            "hessian_replays": 0,
            "hessian_captured": False,
        }
        legacy = {
            "exact_hessian",
            "gauss_newton",
            "capture",
            "capture_derivatives",
            "capture_hessian",
        }.intersection(self.options)
        if legacy:
            raise TypeError(
                f"Removed Pareto collocation option(s): {', '.join(sorted(legacy))}. "
                "Use hessian='exact'|'gauss_newton'|'limited_memory'; capture "
                "is selected by Simulator.execution_backend."
            )
        self.hessian_mode = str(self.options.pop("hessian", "exact")).lower()
        if self.hessian_mode not in {
            "exact",
            "gauss_newton",
            "limited_memory",
        }:
            raise ValueError(
                "hessian must be 'exact', 'gauss_newton', or 'limited_memory'"
            )
        self.exact_hessian = self.hessian_mode == "exact"
        self.uses_hessian = self.hessian_mode != "limited_memory"
        self.capture_requested = opt.simulator.execution_backend == "cuda_graph"
        self.capture_derivatives = self.capture_requested and self.dev.type == "cuda"
        self.capture_hessian = self.capture_derivatives and self.uses_hessian
        target_columns = []
        self._eq_target_slices = []
        self._ineq_target_slices = []
        target_offset = 0
        for _j, desired_periods in self.fast._eq_terms:
            desired = torch.cat(desired_periods).reshape(self.n_seg, -1)
            target_columns.append(desired)
            self._eq_target_slices.append(
                (target_offset, target_offset + desired.shape[1])
            )
            target_offset += desired.shape[1]
        for _j, _ctype, desired_periods in self.fast._ineq_terms:
            desired = torch.cat(desired_periods).reshape(self.n_seg, -1)
            target_columns.append(desired)
            self._ineq_target_slices.append(
                (target_offset, target_offset + desired.shape[1])
            )
            target_offset += desired.shape[1]
        self.targets = (
            torch.cat(target_columns, dim=1)
            if target_columns
            else torch.zeros((self.n_seg, 0), dtype=self.dtype, device=self.dev)
        )
        validate_sparsity = bool(
            self.options.pop("validate_replica_sparsity", __debug__)
        )
        if validate_sparsity and self.n_seg:
            self.replica_layout.validate_step(
                lambda state: self._step(
                    torch.as_tensor(
                        controls0, dtype=self.dtype, device=self.dev
                    ).reshape(self.n_seg, self.n_vars)[0],
                    state,
                    self.CAP[0],
                )[0],
                y0[0],
            )
        self._make_sparsity()

    def y_to_norm(self, y):
        return (y - self.center) / self.scale

    def y_from_norm(self, y):
        return y * self.scale + self.center

    def _control_physical(self, u):
        cols = []
        for v, (mn, mx) in enumerate(self.fast._var_denorm):
            cols.append(u[:, v] * (mx - mn) + mn)
        return torch.stack(cols, dim=1)

    def _caps(self, u):
        if not self.fast._has_slots:
            return self.CAP
        injected = self._control_physical(u) @ self.fast._slot_matrix
        return torch.where(self.fast._slot_mask, injected, self.CAP)

    def _lift_physical(self, controls):
        u = controls.reshape(self.n_seg, self.n_vars)
        caps = self._caps(u)
        rows = []
        cursor = 0
        empty = self.fast._theta_empty
        with torch.no_grad():
            for p, count in enumerate(self.fast.n_t):
                y = self.fast.Y0[p]
                for k in range(count):
                    rows.append(y)
                    y, _ = self.fast.composer.F_aug(y, empty, caps[cursor + k])
                cursor += count
        return torch.stack(rows)

    def lift(self, controls):
        controls = np.asarray(controls, dtype=np.float64)
        ct = torch.as_tensor(controls, dtype=self.dtype, device=self.dev)
        yn = self.y_to_norm(self._lift_physical(ct))
        return np.concatenate([controls, yn.reshape(-1).cpu().numpy()])

    def configure(self, obj_index, con_index, delta, ideal2=0.0, nadir2=1.0):
        self.close()
        self.obj_index = int(obj_index)
        self.con_index = int(con_index)
        self.delta = float(delta)
        self.ideal2 = float(ideal2)
        self.range2 = float(nadir2 - ideal2)
        self.cache = _IterateCache()
        self._bundle_graph = None
        self._jacobian_graph = None
        self._hessian_graph = None

    def close(self):
        """Release captured callback graphs; safe to call repeatedly."""
        for name in ("_bundle_graph", "_jacobian_graph", "_hessian_graph"):
            graph = getattr(self, name, None)
            if graph is not None:
                graph.close()
                setattr(self, name, None)

    def _step(self, u_i, y_i, cap_i):
        cap = cap_i
        if self.fast._has_slots:
            injected = self._control_physical(u_i.reshape(1, -1))[0] @ (
                self.fast._slot_matrix
            )
            cap = torch.where(self.fast._slot_mask, injected, cap)
        yn, meas = self.fast.composer.F_aug(
            self.y_from_norm(y_i),
            self.fast._theta_empty,
            cap,
            transform_mode=True,
        )
        return self.y_to_norm(yn), meas

    def _all(self, u, y):
        return vmap(self._step)(u, y, self.CAP)

    def _ports(self, u, meas):
        physical_u = self._control_physical(u)
        values = []
        for kind, index in self.fast.out_kind:
            values.append(
                meas[:, index] if kind == "meas" else physical_u[:, index : index + 1]
            )
        return values

    def _parts(self, u, y):
        yn, meas = self._all(u, y)
        ports = self._ports(u, meas)

        def norm(j):
            mn, mx = self.fast._out_norm[j]
            return (ports[j] - mn) / (mx - mn)

        eq = []
        for j, desired_periods in self.fast._eq_terms:
            desired = torch.cat(desired_periods).reshape_as(norm(j))
            eq.append(self.opt._constraint_penalty * (norm(j) - desired).abs().mean())
        ineq = None
        if self.fast._ineq_terms:
            ineq = u.new_zeros(())
            for j, ctype, desired_periods in self.fast._ineq_terms:
                desired = torch.cat(desired_periods).reshape_as(norm(j))
                residual = norm(j) - desired
                if ctype == "lower":
                    residual = -residual
                ineq = ineq + self.opt._constraint_penalty * torch.relu(residual).mean()
        objs, phys = [], []
        for j, orientation in self.fast._obj_terms:
            mean = norm(j).mean()
            objs.append(mean if orientation == "min" else -mean)
            phys.append(ports[j].mean())
        return yn, eq, ineq, objs, phys

    def _tensors(self, z):
        zt = torch.as_tensor(z, dtype=self.dtype, device=self.dev)
        return zt[: self.n_u].reshape(self.n_seg, self.n_vars), zt[self.n_u :].reshape(
            self.n_seg, self.Da
        )

    def _values_tensor(self, z):
        u, y = self._tensors(z)
        yn, eq, ineq, objs, phys = self._parts(u, y)
        penalty = u.new_zeros(())
        for term in eq:
            penalty = penalty + term
        if ineq is not None:
            penalty = penalty + ineq
        f2n = (objs[self.con_index] - self.ideal2) / self.range2
        objective = objs[self.obj_index] + penalty + self.delta * f2n
        defect = yn[self.cp_i] - y[self.cp_j]
        return objective, f2n, defect, objs, phys

    def _bundle_tensor(self, zt):
        zt = zt.detach().clone().requires_grad_(True)
        objective, f2n, defect, objs, phys = self._values_tensor(zt)
        (gf,) = torch.autograd.grad(objective, zt, retain_graph=True)
        (ge,) = torch.autograd.grad(f2n, zt)
        return torch.cat(
            [
                torch.stack([objective.detach(), f2n.detach(), *objs, *phys]),
                defect.detach().reshape(-1),
                gf.detach(),
                ge.detach(),
            ]
        )

    def _compute(self, z):
        def calculate(array):
            zt = torch.as_tensor(array, dtype=self.dtype, device=self.dev)
            if self.capture_derivatives:
                if self._bundle_graph is None:
                    self._bundle_graph = CudaGraphCallable(self._bundle_tensor)
                    self.stats["bundle_captured"] = True
                else:
                    self.stats["bundle_replays"] += 1
                bundle = self._bundle_graph(zt).clone()
            else:
                bundle = self._bundle_tensor(zt)
            n_defect = self.n_links * self.Da
            header = bundle[:6]
            defect = bundle[6 : 6 + n_defect]
            gf = bundle[6 + n_defect : 6 + n_defect + self.n_z]
            ge = bundle[6 + n_defect + self.n_z :]
            return {
                "f": float(header[0].detach()),
                "f2n": float(header[1].detach()),
                "gf": gf.detach().cpu().numpy().astype(np.float64),
                "ge": ge.detach().cpu().numpy().astype(np.float64),
                "defect": defect.detach().cpu().numpy().astype(np.float64),
                "objs": [float(x.detach()) for x in header[2:4]],
                "phys": [float(x.detach()) for x in header[4:6]],
            }

        self.stats["bundle_calls"] += 1
        return self.cache.forward(z, calculate)

    def fun(self, z):
        return self._compute(z)["f"]

    def grad(self, z):
        return self._compute(z)["gf"]

    def g(self, z):
        data = self._compute(z)
        return np.concatenate([data["defect"], [data["f2n"]]])

    def jac(self, z):
        u, y = self._tensors(np.asarray(z, dtype=np.float64))

        self.stats["jacobian_calls"] += 1
        if self.capture_derivatives:
            if self._jacobian_graph is None:
                self._jacobian_graph = CudaGraphCallable(self._jacobian_tensor)
                self.stats["jacobian_captured"] = True
            else:
                self.stats["jacobian_replays"] += 1
            packed = self._jacobian_graph(u, y).clone()
        else:
            packed = self._jacobian_tensor(u, y)

        n_ju = self.n_seg * self.Da * self.n_vars
        n_jg = self.n_seg * self.Da * self.d_global
        Ju = packed[:n_ju].reshape(self.n_seg, self.Da, self.n_vars)
        Jg = packed[n_ju : n_ju + n_jg].reshape(
            self.n_seg, self.Da, self.d_global
        )
        Jl = packed[n_ju + n_jg :].reshape(self.n_seg, self.Da, self.d_zone)
        vals = []
        for i, _j in self.links:
            for r in range(self.Da):
                vals.extend(Ju[i, r].detach().cpu().tolist())
                vals.extend(Jg[i, r].detach().cpu().tolist())
                replica = int(self._row_replica[r])
                if replica >= 0:
                    vals.extend(Jl[i, r].detach().cpu().tolist())
                vals.append(-1.0)
        vals.extend(self._compute(z)["ge"].tolist())
        return np.asarray(vals, dtype=np.float64)

    def _jacobian_tensor(self, u, y):
        """Fixed-shape transition derivatives, before host sparse assembly."""
        def one(ui, yi, ci):
            transition_u = lambda value: self._step(value, yi, ci)[0]
            transition_y = lambda value: self._step(ui, value, ci)[0]
            u_basis = torch.eye(self.n_vars, dtype=self.dtype, device=self.dev)
            Ju = vmap(lambda direction: jvp(transition_u, (ui,), (direction,))[1])(
                u_basis
            ).transpose(0, 1)
            if self.d_global:
                Jg = vmap(
                    lambda direction: jvp(
                        transition_y, (yi,), (direction,)
                    )[1]
                )(self._global_color_basis).transpose(0, 1)
            else:
                Jg = yi.new_zeros((self.Da, 0))
            if self.d_zone:
                Jl = vmap(
                    lambda direction: jvp(
                        transition_y, (yi,), (direction,)
                    )[1]
                )(self._local_color_basis).transpose(0, 1)
            else:
                Jl = yi.new_zeros((self.Da, 0))
            return Ju, Jg, Jl

        Ju, Jg, Jl = vmap(one)(u, y, self.CAP)
        return torch.cat((Ju.reshape(-1), Jg.reshape(-1), Jl.reshape(-1)))

    def _make_sparsity(self):
        self._row_replica = np.full(self.Da, -1, dtype=np.int64)
        for replica, indices in enumerate(self.replica_layout.replica_indices):
            self._row_replica[indices] = replica
        jr, jc = [], []
        for link, (i, j) in enumerate(self.links):
            for r in range(self.Da):
                row = link * self.Da + r
                jr.extend([row] * self.n_vars)
                jc.extend(range(i * self.n_vars, (i + 1) * self.n_vars))
                jr.extend([row] * self.d_global)
                jc.extend(
                    (self.n_u + i * self.Da + self.replica_layout.global_indices)
                    .astype(int)
                    .tolist()
                )
                replica = int(self._row_replica[r])
                if replica >= 0:
                    local = self.replica_layout.replica_indices[replica]
                    jr.extend([row] * self.d_zone)
                    jc.extend((self.n_u + i * self.Da + local).astype(int).tolist())
                jr.append(row)
                jc.append(self.n_u + j * self.Da + r)
        eps_row = self.n_links * self.Da
        jr.extend([eps_row] * self.n_z)
        jc.extend(range(self.n_z))
        self.jac_rows = np.asarray(jr, dtype=np.int64)
        self.jac_cols = np.asarray(jc, dtype=np.int64)

        hr, hc = [], []
        source_kind, source_row, target_col = [], [], []
        iu_u = np.triu_indices(self.n_vars)
        for i in range(self.n_seg):
            ub = i * self.n_vars
            yb = self.n_u + i * self.Da
            hr.extend((ub + iu_u[0]).tolist())
            hc.extend((ub + iu_u[1]).tolist())
            source_kind.extend([0] * len(iu_u[0]))
            source_row.extend(iu_u[0].tolist())
            target_col.extend(iu_u[1].tolist())
            for a in range(self.n_vars):
                hr.extend([ub + a] * self.Da)
                hc.extend(range(yb, yb + self.Da))
                source_kind.extend([0] * self.Da)
                source_row.extend([a] * self.Da)
                target_col.extend(range(self.n_vars, self.n_vars + self.Da))
            iu_g = np.triu_indices(self.d_global)
            for ga, gb in zip(*iu_g):
                a = int(self.replica_layout.global_indices[ga])
                b = int(self.replica_layout.global_indices[gb])
                hr.append(yb + min(a, b))
                hc.append(yb + max(a, b))
                source_kind.append(0)
                source_row.append(self.n_vars + ga)
                target_col.append(self.n_vars + b)
            for ga, global_column in enumerate(self.replica_layout.global_indices):
                for local in self.replica_layout.replica_indices:
                    for local_column in local:
                        a, b = int(global_column), int(local_column)
                        hr.append(yb + min(a, b))
                        hc.append(yb + max(a, b))
                        source_kind.append(0)
                        source_row.append(self.n_vars + ga)
                        target_col.append(self.n_vars + b)
            iu_l = np.triu_indices(self.d_zone)
            for local in self.replica_layout.replica_indices:
                for la, lb in zip(*iu_l):
                    a, b = int(local[la]), int(local[lb])
                    hr.append(yb + min(a, b))
                    hc.append(yb + max(a, b))
                    source_kind.append(1)
                    source_row.append(int(la))
                    target_col.append(self.n_vars + b)
        self.hess_rows = np.asarray(hr, dtype=np.int64)
        self.hess_cols = np.asarray(hc, dtype=np.int64)
        self.segment_hessian_nnz = len(source_kind) // self.n_seg if self.n_seg else 0
        source_kind = source_kind[: self.segment_hessian_nnz]
        source_row = source_row[: self.segment_hessian_nnz]
        target_col = target_col[: self.segment_hessian_nnz]
        self._h_source_kind = torch.as_tensor(
            source_kind, dtype=torch.bool, device=self.dev
        )
        self._h_source_row = torch.as_tensor(
            source_row, dtype=torch.long, device=self.dev
        )
        self._h_target_col = torch.as_tensor(
            target_col, dtype=torch.long, device=self.dev
        )

    def _curvature(self, u, y, sigma, eps_lam, dyn):
        # Build a segment-index-aware scalar so time-varying comfort targets are
        # included exactly.  This duplicates only loss algebra, not dynamics.
        def scalar(ui, yi, ci, targets_i, sig, elam, dlam):
            yn, meas = self._step(ui, yi, ci)
            ports = self._ports(ui.reshape(1, -1), meas.reshape(1, -1))

            def normalized(j):
                mn, mx = self.fast._out_norm[j]
                return (ports[j] - mn) / (mx - mn)

            objs = []
            for j, orientation in self.fast._obj_terms:
                value = normalized(j).sum() / (self.n_seg * normalized(j).numel())
                objs.append(value if orientation == "min" else -value)
            penalty = ui.new_zeros(())
            for (j, _desired_periods), (a, b) in zip(
                self.fast._eq_terms, self._eq_target_slices
            ):
                desired = targets_i[a:b]
                penalty = penalty + self.opt._constraint_penalty * (
                    normalized(j).reshape(-1) - desired
                ).abs().sum() / (self.n_seg * desired.numel())
            for (j, ctype, _desired_periods), (a, b) in zip(
                self.fast._ineq_terms, self._ineq_target_slices
            ):
                desired = targets_i[a:b]
                residual = normalized(j).reshape(-1) - desired
                if ctype == "lower":
                    residual = -residual
                penalty = penalty + self.opt._constraint_penalty * torch.relu(
                    residual
                ).sum() / (self.n_seg * desired.numel())
            f2 = (objs[self.con_index] - self.ideal2 / self.n_seg) / self.range2
            return (
                sig * (objs[self.obj_index] + penalty + self.delta * f2)
                + elam * f2
                + (dlam * yn).sum()
            )

        grad_fn = torch.func.grad(scalar, argnums=(0, 1))

        def one(ui, yi, ci, targets_i, sig, elam, dlam):
            args = (ui, yi, ci, targets_i, sig, elam, dlam)
            _, pullback = vjp(grad_fn, *args)
            u_zero = torch.zeros_like(ui)
            y_zero = torch.zeros_like(yi)
            shared_u = torch.cat(
                [
                    torch.eye(self.n_vars, dtype=self.dtype, device=self.dev),
                    torch.zeros(
                        (self.d_global, self.n_vars),
                        dtype=self.dtype,
                        device=self.dev,
                    ),
                ],
                dim=0,
            )
            shared_y = torch.zeros(
                (self.n_vars + self.d_global, self.Da),
                dtype=self.dtype,
                device=self.dev,
            )
            if self.d_global:
                shared_y[
                    self.n_vars + torch.arange(self.d_global, device=self.dev),
                    self._global_idx,
                ] = 1
            shared = vmap(lambda bu, by: torch.cat(pullback((bu, by))[:2], dim=0))(
                shared_u, shared_y
            )
            shared_values = shared[
                self._h_source_row.clamp_max(self.n_vars + self.d_global - 1),
                self._h_target_col,
            ]
            if self.d_zone:
                local = vmap(lambda by: torch.cat(pullback((u_zero, by))[:2], dim=0))(
                    self._local_color_basis
                )
                local_values = local[
                    self._h_source_row.clamp_max(self.d_zone - 1),
                    self._h_target_col,
                ]
            else:
                local_values = torch.zeros_like(shared_values)
            return torch.where(self._h_source_kind, local_values, shared_values)

        return vmap(one)(u, y, self.CAP, self.targets, sigma, eps_lam, dyn)

    def hessian(self, z, sigma, lam_g):
        self.stats["hessian_calls"] += 1
        u, y = self._tensors(np.asarray(z, dtype=np.float64))
        lam = torch.as_tensor(lam_g, dtype=self.dtype, device=self.dev)
        dyn = torch.zeros((self.n_seg, self.Da), dtype=self.dtype, device=self.dev)
        if self.n_links and self.exact_hessian:
            dyn.index_add_(0, self.cp_i, lam[:-1].reshape(self.n_links, self.Da))
        sigma_v = torch.full(
            (self.n_seg,), float(sigma), dtype=self.dtype, device=self.dev
        )
        eps_v = torch.full(
            (self.n_seg,), float(lam[-1]), dtype=self.dtype, device=self.dev
        )
        fn = self._curvature
        if self.capture_hessian:
            if self._hessian_graph is None:
                self._hessian_graph = CudaGraphCallable(fn)
                self.stats["hessian_captured"] = True
            else:
                self.stats["hessian_replays"] += 1
            blocks = self._hessian_graph(u, y, sigma_v, eps_v, dyn).clone()
        else:
            blocks = fn(u, y, sigma_v, eps_v, dyn)
        return blocks.reshape(-1).detach().cpu().numpy().astype(np.float64)

    def values(self, z):
        return self._compute(z)

    def solve(self, initial, eps=None):
        n_g = self.n_links * self.Da + 1
        lbg = np.concatenate([np.zeros(n_g - 1), [-np.inf]])
        ubg = np.concatenate(
            [np.zeros(n_g - 1), [np.inf if eps is None else float(eps)]]
        )
        result = solve_ipopt_constrained(
            initial,
            self.lb,
            self.ub,
            self.fun,
            self.grad,
            n_g,
            self.g,
            self.jac,
            self.jac_rows,
            self.jac_cols,
            options=self.options,
            hess_vals=self.hessian if self.uses_hessian else None,
            hess_rows=self.hess_rows if self.uses_hessian else None,
            hess_cols=self.hess_cols if self.uses_hessian else None,
            lbg=lbg,
            ubg=ubg,
        )
        return _Solve(result, self.values(result.x))


def pareto_front_collocation(
    opt,
    *,
    n_points,
    delta,
    method,
    use_prepass,
    prepass_options,
    options,
):
    """Run IPOPT AUGMECON with sparse one-step collocation only."""
    options = dict(options or {})
    controls0, bounds = opt._prepare_scipy_problem(method, options)
    opt._pareto_prepass_stats = {
        "requested": False,
        "enabled": False,
        "captured": False,
        "replays": 0,
        "fallback_reason": "batched prepass not run",
    }
    problem = ParetoCollocation(opt, controls0, bounds, delta=delta, options=options)

    problem.configure(0, 1, delta)
    a1 = problem.solve(problem.z0)
    z1 = a1.result.x
    problem.configure(1, 0, delta)
    a2 = problem.solve(problem.lift(z1[: problem.n_u]))
    z2 = a2.result.x
    ideal = (a1.values["objs"][0], a2.values["objs"][1])
    nadir = (a2.values["objs"][0], a1.values["objs"][1])
    range2 = nadir[1] - ideal[1]

    if abs(range2) < 1e-9:
        eps_grid = np.array([1.0, 0.0])
        solved = [a1, a2]
    else:
        eps_grid = np.linspace(1.0, 0.0, n_points)
        problem.configure(0, 1, delta, ideal[1], nadir[1])
        # Existing prepass remains a controls-only warm-start generator.  Lift
        # its first point to a dynamically consistent full z; thereafter every
        # epsilon point starts from its neighbouring full collocation solution.
        initial = problem.lift(z1[: problem.n_u])
        if use_prepass:
            try:
                warm = batched_prepass(
                    opt,
                    eps_grid,
                    z1[: problem.n_u],
                    z2[: problem.n_u],
                    ideal[1],
                    range2,
                    delta,
                    bounds,
                    **(prepass_options or {}),
                )
                initial = problem.lift(warm[0])
            except Exception:
                pass
        solved = []
        for eps in eps_grid:
            current = problem.solve(initial, float(eps))
            solved.append(current)
            initial = current.result.x

    f1m = np.asarray([s.values["objs"][0] for s in solved])
    f2m = np.asarray([s.values["objs"][1] for s in solved])
    f1 = np.asarray([s.values["phys"][0] for s in solved])
    f2 = np.asarray([s.values["phys"][1] for s in solved])
    full_z = np.stack([np.asarray(s.result.x) for s in solved])
    theta = full_z[:, : problem.n_u]
    boundary = full_z[:, problem.n_u :].reshape(len(solved), problem.n_seg, problem.Da)
    defects = np.asarray(
        [
            np.max(np.abs(s.values["defect"])) if len(s.values["defect"]) else 0.0
            for s in solved
        ]
    )
    rollout_parity = []
    for control, s in zip(theta, solved):
        parts = opt._functional_objective.parts(
            torch.as_tensor(control, dtype=tps.float_dtype(), device=opt._device)
        )
        rollout_parity.append(
            {
                "objective_abs": [
                    abs(float(parts.objs[k].detach()) - s.values["objs"][k])
                    for k in range(2)
                ],
                "physical_abs": [
                    abs(float(parts.phys[k].detach()) - s.values["phys"][k])
                    for k in range(2)
                ],
            }
        )
    success = np.asarray([bool(s.result.success) for s in solved])
    nit = np.asarray([int(s.result.nit or 0) for s in solved])
    slope = (
        -np.gradient(f1m, eps_grid)
        if len(eps_grid) > 1 and abs(range2) >= 1e-9
        else np.zeros_like(eps_grid)
    )
    result = ParetoResult(
        eps=eps_grid,
        f1=f1,
        f2=f2,
        f1_min=f1m,
        f2_min=f2m,
        theta=theta,
        success=success,
        nit=nit,
        pareto_mask=_pareto_mask(f1m, f2m),
        slope=slope,
        ideal=ideal,
        nadir=nadir,
        labels=tuple(f"{c.id}.{p} ({t})" for c, p, t in opt._objectives),
        method=tuple(method),
        capture={
            "scope": "composed one-step collocation callbacks",
            **problem.stats,
            "captured_bundle": problem.stats["bundle_captured"],
            "enabled": problem.capture_derivatives,
            "requested": problem.capture_requested,
            "captured_jacobian": problem.stats["jacobian_captured"],
            "batched_prepass": dict(opt._pareto_prepass_stats),
        },
        hessian={
            "kind": problem.hessian_mode,
            "nnz": len(problem.hess_rows) if problem.uses_hessian else 0,
            "direct_cuda_graph": problem.stats["hessian_captured"],
            "calls": problem.stats["hessian_calls"],
            "replays": problem.stats["hessian_replays"],
        },
        boundary_states=boundary,
        max_defect=defects,
        rollout_parity=rollout_parity,
        callback_shapes={
            "z": problem.n_z,
            "controls": problem.n_u,
            "boundary_states": problem.n_y,
            "constraints": problem.n_links * problem.Da + 1,
            "jacobian_nnz": len(problem.jac_rows),
            "hessian_nnz": len(problem.hess_rows) if problem.uses_hessian else 0,
            "state_width": problem.Da,
            "replicas": problem.n_replicas,
            "replica_state_width": problem.d_zone,
            "global_state_width": problem.d_global,
            "compiled_shapes": list(problem.fast.layout.shapes),
        },
        _optimizer=opt,
    )
    problem.close()
    return result
