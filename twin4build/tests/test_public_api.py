"""Contract tests for the small, preferred top-level API."""

import inspect
import unittest

import twin4build as tb


class TestPublicApi(unittest.TestCase):
    def test_primary_workflow_is_exported(self):
        expected = {
            "Model",
            "SemanticModel",
            "Simulator",
            "Estimator",
            "Translator",
            "Optimizer",
            "System",
            "Scalar",
            "Vector",
            "Parameter",
            "State",
            "EstimationResult",
            "OptimizationResult",
            "__version__",
        }
        self.assertTrue(expected.issubset(set(tb.__all__)))
        self.assertTrue(tb.__version__)

    def test_results_share_dict_and_attribute_access(self):
        estimation = tb.EstimationResult(success=True, backend="slsqp")
        optimization = tb.OptimizationResult(success=True, backend="slsqp")

        for result in (estimation, optimization):
            self.assertTrue(result["success"])
            self.assertTrue(result.success)
            result["message"] = "done"
            self.assertEqual(result.message, "done")
            result.nit = 3
            self.assertEqual(result["nit"], 3)
            copied = result.copy()
            self.assertIs(type(copied), type(result))
            self.assertEqual(copied, result)

    def test_core_workflows_reject_unknown_keywords(self):
        simulator = tb.Simulator(None)
        calls = (
            lambda: simulator.simulate(unknown_option=True),
            lambda: tb.Estimator(simulator).estimate(unknown_option=True),
            lambda: tb.Optimizer(simulator).optimize(unknown_option=True),
            lambda: tb.Translator().translate(None, unknown_option=True),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
                    call()

    def test_translator_signature_only_shows_preferred_arguments(self):
        parameters = inspect.signature(tb.Translator.translate).parameters
        self.assertIn("systems", parameters)
        self.assertNotIn("systems_", parameters)
        self.assertNotIn("verbose", parameters)


if __name__ == "__main__":
    unittest.main()
