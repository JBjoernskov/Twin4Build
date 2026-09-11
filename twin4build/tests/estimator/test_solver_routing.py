from unittest.mock import Mock

import pytest

from twin4build.estimator.estimator import Estimator


@pytest.mark.parametrize(
    ("method", "owner"),
    [
        (("scipy", "SLSQP", "ad"), "_solve_scipy"),
        (("casadi", "ipopt", "ad"), "_solve_ipopt"),
        (("custom", "batched-sqp", "ad"), "_solve_custom"),
    ],
)
def test_dispatch_routes_to_backend_owner(method, owner):
    estimator = object.__new__(Estimator)
    estimator._solve_scipy = Mock(return_value="scipy")
    estimator._solve_ipopt = Mock(return_value="ipopt")
    estimator._solve_custom = Mock(return_value="custom")

    result = estimator._dispatch_solve(method, 2, {"maxiter": 3})

    selected = getattr(estimator, owner)
    selected.assert_called_once()
    assert result == owner.removeprefix("_solve_")
    for name in ("_solve_scipy", "_solve_ipopt", "_solve_custom"):
        if name != owner:
            getattr(estimator, name).assert_not_called()


def test_dispatch_rejects_unknown_backend():
    estimator = object.__new__(Estimator)
    with pytest.raises(ValueError, match="Unsupported estimator backend"):
        estimator._dispatch_solve(("other", "solver", "ad"), None, {})


@pytest.mark.parametrize(
    ("owner", "method"),
    [
        ("_solve_scipy", ("scipy", "SLSQP", "ad")),
        ("_solve_ipopt", ("casadi", "ipopt", "ad")),
        ("_solve_custom", ("custom", "batched-sqp", "ad")),
    ],
)
def test_cross_backend_hessian_option_is_rejected(owner, method):
    estimator = object.__new__(Estimator)
    estimator._transcription = "single_shooting"
    estimator._functional_objective = object()
    args = (
        (method, None, {"hessian": "exact"})
        if owner == "_solve_scipy"
        else (
            method,
            {"hessian": "exact"},
        )
    )
    with pytest.raises(TypeError, match="hessian"):
        getattr(estimator, owner)(*args)
