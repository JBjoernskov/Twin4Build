"""Replica-aware sparsity metadata for functional augmented states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.func import jvp, vjp, vmap


@dataclass(frozen=True)
class ReplicaLayout:
    """Partition a flattened augmented state into global and replica blocks."""

    width: int
    n_replicas: int
    global_indices: np.ndarray
    replica_indices: tuple[np.ndarray, ...]

    @property
    def local_width(self) -> int:
        return int(self.replica_indices[0].size)

    @property
    def shared_width(self) -> int:
        return int(self.global_indices.size)

    @classmethod
    def from_functional_model(cls, functional_model) -> "ReplicaLayout":
        shapes = [(int(n), int(d)) for n, d in functional_model.state_shapes]
        feedback_widths = [int(w) for w in functional_model._feedback_widths]
        batch_sizes = [n for n, _ in shapes if n > 1]
        batch_sizes.extend(w for w in feedback_widths if w > 1)
        n_replicas = max(batch_sizes, default=1)
        unsupported = sorted(
            {n for n, _ in shapes if n not in (1, n_replicas)}
            | {w for w in feedback_widths if w not in (1, n_replicas)}
        )
        if unsupported:
            raise NotImplementedError(
                "Replica-colored collocation requires every augmented-state "
                f"batch width to be 1 or {n_replicas}; found {unsupported}."
            )

        global_indices: list[int] = []
        replicas: list[list[int]] = [[] for _ in range(n_replicas)]
        for start, (n_c, state_size) in zip(functional_model.state_offsets, shapes):
            if n_replicas > 1 and n_c == 1:
                global_indices.extend(range(start, start + state_size))
            else:
                for replica in range(n_replicas):
                    base = start + replica * state_size
                    replicas[replica].extend(range(base, base + state_size))

        offset = int(functional_model.D)
        for width in feedback_widths:
            if n_replicas > 1 and width == 1:
                global_indices.append(offset)
            else:
                for replica in range(n_replicas):
                    replicas[replica].append(offset + replica)
            offset += width

        if offset != functional_model.D_aug:
            raise RuntimeError(
                "Augmented-state replica mapping does not cover StateLayout "
                f"(mapped {offset}, expected {functional_model.D_aug})."
            )
        local_widths = {len(indices) for indices in replicas}
        if len(local_widths) != 1:
            raise NotImplementedError(
                "Replica-colored collocation requires equal local state widths; "
                f"found {sorted(local_widths)}."
            )
        covered = global_indices + [i for block in replicas for i in block]
        if sorted(covered) != list(range(functional_model.D_aug)):
            raise RuntimeError(
                "Replica mapping contains duplicate or missing state slots."
            )
        return cls(
            width=int(functional_model.D_aug),
            n_replicas=n_replicas,
            global_indices=np.asarray(global_indices, dtype=np.int64),
            replica_indices=tuple(
                np.asarray(indices, dtype=np.int64) for indices in replicas
            ),
        )

    def device_indices(self, device) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return (
            torch.as_tensor(self.global_indices, dtype=torch.long, device=device),
            tuple(
                torch.as_tensor(indices, dtype=torch.long, device=device)
                for indices in self.replica_indices
            ),
        )

    def colored_local_basis(self, *, dtype, device) -> torch.Tensor:
        basis = torch.zeros((self.local_width, self.width), dtype=dtype, device=device)
        for indices in self.replica_indices:
            idx = torch.as_tensor(indices, dtype=torch.long, device=device)
            basis[torch.arange(self.local_width, device=device), idx] = 1
        return basis

    def validate_step(
        self,
        fn,
        y: torch.Tensor,
        *,
        atol: float = 1e-9,
        exhaustive_limit: int = 128,
    ) -> None:
        """Verify that local inputs cannot affect other replicas or globals."""
        if self.n_replicas == 1:
            return
        global_idx, replica_idx = self.device_indices(y.device)
        checks: list[tuple[int, int]] = []
        if self.width <= exhaustive_limit:
            checks = [
                (replica, int(column))
                for replica, indices in enumerate(self.replica_indices)
                for column in indices
            ]
        else:
            for replica, indices in enumerate(self.replica_indices):
                if indices.size:
                    sample = np.linspace(
                        0, indices.size - 1, min(3, indices.size), dtype=int
                    )
                    checks.extend((replica, int(indices[k])) for k in sample)
        zero = torch.zeros_like(y)
        for replica, column in checks:
            tangent = zero.clone()
            tangent[column] = 1
            _, directional = jvp(fn, (y,), (tangent,))
            forbidden = [global_idx]
            forbidden.extend(
                indices for other, indices in enumerate(replica_idx) if other != replica
            )
            forbidden_idx = torch.cat(forbidden)
            if forbidden_idx.numel():
                magnitude = float(directional[forbidden_idx].abs().max())
                if magnitude > atol:
                    raise NotImplementedError(
                        "Replica-colored collocation detected unsupported "
                        "cross-replica/global dynamics: state column "
                        f"{column} affects another block by {magnitude:.3e} "
                        f"(tolerance {atol:.1e})."
                    )


@dataclass(frozen=True)
class ThetaReplicaLayout:
    """Partition normalized estimator theta into shared and replica-private slots.

    A batched parameter with one branch per model replica contributes one
    private slot to each replica. A scalar selector is shared: the functional model
    broadcasts that same public decision variable wherever the selector is
    routed.  Other widths are ambiguous and are rejected rather than silently
    assigning an incorrect sparse structure.
    """

    width: int
    n_replicas: int
    global_indices: np.ndarray
    replica_indices: tuple[np.ndarray, ...]

    @property
    def local_width(self) -> int:
        return int(self.replica_indices[0].size)

    @property
    def shared_width(self) -> int:
        return int(self.global_indices.size)

    @classmethod
    def from_estimator(cls, estimator, functional_model) -> "ThetaReplicaLayout":
        state_layout = ReplicaLayout.from_functional_model(functional_model)
        n_replicas = state_layout.n_replicas
        width = len(estimator._x0_norm)
        global_indices: list[int] = []
        replicas: list[list[int]] = [[] for _ in range(n_replicas)]

        if len(estimator._theta_slices) != len(
            getattr(estimator, "_unique_param_n_c", estimator._theta_slices)
        ):
            raise RuntimeError("Estimator theta metadata is internally inconsistent.")

        for unique, (start, stop) in enumerate(estimator._theta_slices):
            start, stop = int(start), int(stop)
            selector_width = stop - start
            flat = np.flatnonzero(np.asarray(estimator._theta_mask) == unique)
            if not flat.size:
                raise RuntimeError(
                    f"Unique theta group {unique} has no component/attribute route."
                )
            # Verify that the functional theta spec routes every occurrence to the
            # same normalized selector represented by this unique group.
            for flat_index in flat:
                expected = start if selector_width == 1 else slice(start, stop)
                spec = functional_model.theta_spec[int(flat_index)]
                actual = spec[2] if len(spec) > 2 else int(flat_index)
                same = (
                    actual == expected
                    if isinstance(expected, int)
                    else isinstance(actual, slice)
                    and actual.start == expected.start
                    and actual.stop == expected.stop
                )
                if not same:
                    raise RuntimeError(
                        "Composer theta selector disagrees with Estimator "
                        f"metadata for unique group {unique}: {actual!r}."
                    )

            if n_replicas == 1:
                replicas[0].extend(range(start, stop))
            elif selector_width == n_replicas:
                for replica in range(n_replicas):
                    replicas[replica].append(start + replica)
            elif selector_width == 1:
                global_indices.append(start)
            else:
                labels = [
                    (
                        estimator._flat_components[int(i)].id,
                        estimator._parameter_names[int(i)],
                    )
                    for i in flat
                ]
                raise NotImplementedError(
                    "Replica-colored estimator collocation requires each theta "
                    f"selector width to be 1 (shared) or {n_replicas} "
                    f"(private); group {unique} {labels} has width "
                    f"{selector_width}."
                )

        local_widths = {len(block) for block in replicas}
        if len(local_widths) != 1:
            raise NotImplementedError(
                "Replica-colored estimator collocation requires equal private "
                f"theta widths; found {sorted(local_widths)}."
            )
        covered = global_indices + [i for block in replicas for i in block]
        if sorted(covered) != list(range(width)):
            raise RuntimeError(
                "Theta replica mapping contains duplicate or missing slots."
            )
        return cls(
            width=width,
            n_replicas=n_replicas,
            global_indices=np.asarray(global_indices, dtype=np.int64),
            replica_indices=tuple(
                np.asarray(block, dtype=np.int64) for block in replicas
            ),
        )

    def device_indices(self, device):
        return (
            torch.as_tensor(self.global_indices, dtype=torch.long, device=device),
            tuple(
                torch.as_tensor(block, dtype=torch.long, device=device)
                for block in self.replica_indices
            ),
        )

    def colored_local_basis(self, *, dtype, device):
        basis = torch.zeros((self.local_width, self.width), dtype=dtype, device=device)
        rows = torch.arange(self.local_width, device=device)
        for block in self.replica_indices:
            basis[rows, torch.as_tensor(block, dtype=torch.long, device=device)] = 1
        return basis


def colored_jacobian(
    fn,
    value,
    layout,
    *,
    dtype=None,
    global_basis=None,
    local_basis=None,
):
    """Return global columns and simultaneous replica-local colored columns.

    Precomputed bases avoid host-to-device index construction when this helper
    executes inside a CUDA Graph capture.
    """
    dtype = value.dtype if dtype is None else dtype
    device = value.device
    if global_basis is None:
        global_idx, _ = layout.device_indices(device)
        global_basis = torch.zeros(
            (layout.shared_width, layout.width), dtype=dtype, device=device
        )
        if layout.shared_width:
            global_basis[
                torch.arange(layout.shared_width, device=device), global_idx
            ] = 1
    if layout.shared_width:
        global_jac = vmap(lambda d: jvp(fn, (value,), (d,))[1])(global_basis)
    else:
        output = fn(value)
        global_jac = output.new_zeros((0,) + output.shape)
    if local_basis is None:
        local_basis = layout.colored_local_basis(dtype=dtype, device=device)
    if layout.local_width:
        local_jac = vmap(lambda d: jvp(fn, (value,), (d,))[1])(local_basis)
    else:
        output = fn(value)
        local_jac = output.new_zeros((0,) + output.shape)
    return global_jac.movedim(0, -1), local_jac.movedim(0, -1)


def colored_hessian_rows(
    grad_fn,
    x,
    y,
    x_layout,
    y_layout,
    *rest,
    x_global_basis=None,
    x_local_basis=None,
    y_global_basis=None,
    y_local_basis=None,
):
    """Evaluate arrowhead Hessian rows with shared and local color bases.

    The returned tensors concatenate derivatives with respect to ``x`` and
    ``y`` along the last axis.  Shared rows are ordered ``x_global,y_global``;
    local colored rows are ordered ``x_local,y_local``.
    """
    _, pullback = vjp(grad_fn, x, y, *rest)
    n_shared = x_layout.shared_width + y_layout.shared_width
    bx = torch.zeros((n_shared, x.numel()), dtype=x.dtype, device=x.device)
    by = torch.zeros((n_shared, y.numel()), dtype=y.dtype, device=y.device)
    if x_layout.shared_width:
        if x_global_basis is None:
            xg, _ = x_layout.device_indices(x.device)
            bx[torch.arange(x_layout.shared_width, device=x.device), xg] = 1
        else:
            bx[: x_layout.shared_width] = x_global_basis
    if y_layout.shared_width:
        rows = x_layout.shared_width + torch.arange(
            y_layout.shared_width, device=y.device
        )
        if y_global_basis is None:
            yg, _ = y_layout.device_indices(y.device)
            by[rows, yg] = 1
        else:
            by[rows] = y_global_basis
    if n_shared:
        shared = vmap(lambda dx, dy: torch.cat(pullback((dx, dy))[:2], dim=0))(bx, by)
    else:
        shared = x.new_zeros((0, x.numel() + y.numel()))

    n_local = x_layout.local_width + y_layout.local_width
    bx = torch.zeros((n_local, x.numel()), dtype=x.dtype, device=x.device)
    by = torch.zeros((n_local, y.numel()), dtype=y.dtype, device=y.device)
    if x_layout.local_width:
        bx[: x_layout.local_width] = (
            x_layout.colored_local_basis(dtype=x.dtype, device=x.device)
            if x_local_basis is None
            else x_local_basis
        )
    if y_layout.local_width:
        by[x_layout.local_width :] = (
            y_layout.colored_local_basis(dtype=y.dtype, device=y.device)
            if y_local_basis is None
            else y_local_basis
        )
    if n_local:
        local = vmap(lambda dx, dy: torch.cat(pullback((dx, dy))[:2], dim=0))(bx, by)
    else:
        local = x.new_zeros((0, x.numel() + y.numel()))
    return shared, local


def validate_replica_influence(fn, value, input_layout, output_layout, *, atol=1e-9):
    """Reject a local input direction that reaches another state replica."""
    if input_layout.n_replicas == 1:
        return
    output_global, output_local = output_layout.device_indices(value.device)
    zero = torch.zeros_like(value)
    for replica, columns in enumerate(input_layout.replica_indices):
        for column in columns:
            tangent = zero.clone()
            tangent[int(column)] = 1
            _, directional = jvp(fn, (value,), (tangent,))
            forbidden = [output_global]
            forbidden.extend(
                block for other, block in enumerate(output_local) if other != replica
            )
            forbidden = torch.cat(forbidden)
            if forbidden.numel():
                magnitude = float(directional[forbidden].abs().max())
                if magnitude > atol:
                    raise NotImplementedError(
                        "Replica-colored collocation detected unsupported "
                        "cross-replica/global dynamics: input column "
                        f"{int(column)} in replica {replica} affects another "
                        f"state block by {magnitude:.3e} (tolerance {atol:.1e})."
                    )
