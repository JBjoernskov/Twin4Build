from unittest.mock import Mock

import pytest

from twin4build.estimator.estimator import Estimator
from twin4build.solvers.registry import register_solver, unregister_solver


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


class _PlugIn:
    method = ("plugin", "step", "ad")

    def __init__(self):
        self.calls = []

    def solve(self, problem, options):
        self.calls.append((problem, options))
        return "plugin"


def _bare_estimator():
    estimator = object.__new__(Estimator)
    estimator._solve_scipy = Mock(return_value="scipy")
    estimator._solve_ipopt = Mock(return_value="ipopt")
    estimator._solve_custom = Mock(return_value="custom")
    estimator.estimation_problem = Mock(return_value="problem")
    return estimator


def test_dispatch_prefers_a_solver_instance():
    estimator = _bare_estimator()
    estimator._solver_instance = _PlugIn()
    assert estimator._dispatch_solve(("custom", "batched-sqp", "ad"), None, {"maxiter": 3}) == "plugin"
    assert estimator._solver_instance.calls == [("problem", {"maxiter": 3})]
    estimator._solve_custom.assert_not_called()


def test_dispatch_finds_a_registered_solver_and_it_shadows_a_built_in():
    estimator = _bare_estimator()
    estimator._solver_instance = None
    plugin = _PlugIn()
    plugin.method = ("custom", "batched-sqp", "ad")
    register_solver(plugin)
    try:
        assert estimator._dispatch_solve(plugin.method, None, {}) == "plugin"
        estimator._solve_custom.assert_not_called()
    finally:
        unregister_solver(plugin.method)
    assert estimator._dispatch_solve(plugin.method, None, {}) == "custom"
