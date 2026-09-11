from types import SimpleNamespace

import numpy as np
import pytest
import torch

from twin4build.simulator._replica_layout import (
    ReplicaLayout,
    ThetaReplicaLayout,
    colored_hessian_rows,
    colored_jacobian,
)


def _functional_model(n_c):
    return SimpleNamespace(
        state_shapes=[(n_c, 2), (n_c, 1), (1, 1)] if n_c > 1 else [(1, 4)],
        state_offsets=[0, 2 * n_c, 3 * n_c] if n_c > 1 else [0],
        D=3 * n_c + 1 if n_c > 1 else 4,
        _feedback_widths=[n_c, 1] if n_c > 1 else [1],
        D_aug=4 * n_c + 2 if n_c > 1 else 5,
    )


@pytest.mark.parametrize("n_c", [1, 2, 3])
def test_augmented_layout_assigns_feedback_to_matching_replica(n_c):
    layout = ReplicaLayout.from_functional_model(_functional_model(n_c))
    assert layout.n_replicas == n_c
    if n_c == 1:
        np.testing.assert_array_equal(layout.replica_indices[0], np.arange(5))
        assert layout.global_indices.size == 0
        return
    assert layout.local_width == 4
    np.testing.assert_array_equal(layout.global_indices, [3 * n_c, 4 * n_c + 1])
    for replica, indices in enumerate(layout.replica_indices):
        np.testing.assert_array_equal(
            indices,
            [
                2 * replica,
                2 * replica + 1,
                2 * n_c + replica,
                3 * n_c + 1 + replica,
            ],
        )


def test_structural_validation_rejects_cross_replica_dynamics():
    layout = ReplicaLayout.from_functional_model(_functional_model(2))
    y = torch.arange(layout.width, dtype=torch.float64)

    def coupled(value):
        result = value.clone()
        result[layout.replica_indices[1][0]] += value[layout.replica_indices[0][0]]
        return result

    with pytest.raises(NotImplementedError, match="cross-replica"):
        layout.validate_step(coupled, y)


@pytest.mark.parametrize("n_c", [2, 3])
def test_theta_layout_and_colored_derivatives_match_dense(n_c):
    functional_model = _functional_model(n_c)
    components = [SimpleNamespace(id=f"p{i}") for i in range(3)]
    functional_model.theta_spec = [
        (components[0], "shared", 0),
        (components[1], "a", slice(1, 1 + n_c)),
        (components[2], "b", slice(1 + n_c, 1 + 2 * n_c)),
    ]
    estimator = SimpleNamespace(
        _x0_norm=np.zeros(1 + 2 * n_c),
        _theta_slices=[(0, 1), (1, 1 + n_c), (1 + n_c, 1 + 2 * n_c)],
        _unique_param_n_c=[1, n_c, n_c],
        _theta_mask=np.arange(3),
        _flat_components=components,
        _parameter_names=["shared", "a", "b"],
    )
    theta_layout = ThetaReplicaLayout.from_estimator(estimator, functional_model)
    state_layout = ReplicaLayout.from_functional_model(functional_model)
    assert theta_layout.shared_width == 1
    assert theta_layout.local_width == 2
    np.testing.assert_array_equal(theta_layout.global_indices, [0])

    theta = torch.linspace(-0.3, 0.4, theta_layout.width, dtype=torch.float64)
    state = torch.linspace(-0.2, 0.5, state_layout.width, dtype=torch.float64)

    def outputs(th):
        result = state.clone()
        for replica, rows in enumerate(state_layout.replica_indices):
            private = theta[theta_layout.replica_indices[replica]]
            result[rows] = torch.sin(state[rows] + theta[0]) + private.sum()
        result[state_layout.global_indices] += theta[0].square()
        return result

    global_jac, local_jac = colored_jacobian(outputs, theta, theta_layout)
    dense = torch.func.jacrev(outputs)(theta)
    torch.testing.assert_close(global_jac, dense[:, theta_layout.global_indices])
    for replica, rows in enumerate(state_layout.replica_indices):
        torch.testing.assert_close(
            local_jac[rows], dense[rows][:, theta_layout.replica_indices[replica]]
        )

    def scalar(th, y):
        value = th[0].square() + y[state_layout.global_indices].square().sum()
        for replica, rows in enumerate(state_layout.replica_indices):
            private = th[theta_layout.replica_indices[replica]]
            value = value + (torch.sin(private.sum() + y[rows].sum() + th[0]))
        return value

    grad_fn = torch.func.grad(scalar, argnums=(0, 1))
    shared, local = colored_hessian_rows(
        grad_fn, theta, state, theta_layout, state_layout
    )
    dense_h = torch.func.hessian(scalar, argnums=(0, 1))(theta, state)
    dense_rows = torch.cat(
        [
            torch.cat([dense_h[0][0], dense_h[0][1]], dim=1),
            torch.cat([dense_h[1][0], dense_h[1][1]], dim=1),
        ],
        dim=0,
    )
    shared_actual = np.concatenate(
        [theta_layout.global_indices, theta_layout.width + state_layout.global_indices]
    )
    torch.testing.assert_close(shared, dense_rows[shared_actual])
    for replica in range(n_c):
        local_actual = np.concatenate(
            [
                theta_layout.replica_indices[replica],
                theta_layout.width + state_layout.replica_indices[replica],
            ]
        )
        torch.testing.assert_close(
            local[:, local_actual], dense_rows[local_actual][:, local_actual]
        )
