# Standard library imports
import unittest
from unittest.mock import Mock

# Local application imports
from twin4build.estimator.estimator import Estimator
from twin4build.solvers.registry import register_solver, unregister_solver


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


class TestSolverRouting(unittest.TestCase):
    def test_dispatch_routes_to_backend_owner(self):
        for method, owner in [
            (("scipy", "SLSQP", "ad"), "_solve_scipy"),
            (("casadi", "ipopt", "ad"), "_solve_ipopt"),
            (("custom", "batched-sqp", "ad"), "_solve_custom"),
        ]:
            with self.subTest(method=method, owner=owner):
                estimator = object.__new__(Estimator)
                estimator._solve_scipy = Mock(return_value="scipy")
                estimator._solve_ipopt = Mock(return_value="ipopt")
                estimator._solve_custom = Mock(return_value="custom")

                result = estimator._dispatch_solve(method, {"maxiter": 3})

                selected = getattr(estimator, owner)
                selected.assert_called_once()
                self.assertEqual(result, owner.removeprefix("_solve_"))
                for name in ("_solve_scipy", "_solve_ipopt", "_solve_custom"):
                    if name != owner:
                        getattr(estimator, name).assert_not_called()

    def test_dispatch_rejects_unknown_backend(self):
        estimator = object.__new__(Estimator)
        with self.assertRaisesRegex(ValueError, "Unsupported estimator backend"):
            estimator._dispatch_solve(("other", "solver", "ad"), {})

    def test_cross_backend_hessian_option_is_rejected(self):
        for owner, method in [
            ("_solve_scipy", ("scipy", "SLSQP", "ad")),
            ("_solve_ipopt", ("casadi", "ipopt", "ad")),
            ("_solve_custom", ("custom", "batched-sqp", "ad")),
        ]:
            with self.subTest(owner=owner, method=method):
                estimator = object.__new__(Estimator)
                estimator._transcription = "single_shooting"
                estimator._functional_objective = object()
                with self.assertRaisesRegex(TypeError, "hessian"):
                    getattr(estimator, owner)(method, {"hessian": "exact"})

    def test_dispatch_prefers_a_solver_instance(self):
        estimator = _bare_estimator()
        estimator._solver_instance = _PlugIn()
        self.assertEqual(
            estimator._dispatch_solve(("custom", "batched-sqp", "ad"), {"maxiter": 3}),
            "plugin",
        )
        self.assertEqual(
            estimator._solver_instance.calls, [("problem", {"maxiter": 3})]
        )
        estimator._solve_custom.assert_not_called()

    def test_dispatch_finds_a_registered_solver_and_it_shadows_a_built_in(self):
        estimator = _bare_estimator()
        estimator._solver_instance = None
        plugin = _PlugIn()
        plugin.method = ("custom", "batched-sqp", "ad")
        register_solver(plugin)
        try:
            self.assertEqual(estimator._dispatch_solve(plugin.method, {}), "plugin")
            estimator._solve_custom.assert_not_called()
        finally:
            unregister_solver(plugin.method)
        self.assertEqual(estimator._dispatch_solve(plugin.method, {}), "custom")


if __name__ == "__main__":
    unittest.main()
