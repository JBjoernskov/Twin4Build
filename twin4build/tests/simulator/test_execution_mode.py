import datetime
import unittest
from unittest.mock import Mock

import numpy as np

import twin4build as tb
from twin4build.tests.estimator.example_fixture import EXAMPLE_START, load_model


class TestSimulatorExecutionMode(unittest.TestCase):
    def test_execution_api_is_hard_renamed(self):
        simulator = tb.Simulator(load_model())
        self.assertTrue(callable(simulator.build_functional_model))
        self.assertTrue(callable(simulator.record_exogenous_inputs))
        self.assertTrue(callable(simulator.rollout_functional))
        self.assertFalse(hasattr(simulator, "compose"))
        self.assertFalse(hasattr(simulator, "capture_rollout"))
        self.assertFalse(hasattr(simulator, "rollout_composed"))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            tb.Simulator(load_model(), execution_mode="invalid")

    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            tb.Simulator(load_model(), execution_backend="invalid")

    def test_cuda_graph_requires_functional_mode(self):
        with self.assertRaisesRegex(ValueError, "requires.*functional"):
            tb.Simulator(load_model(), execution_backend="cuda_graph")

    def test_clear_execution_cache_closes_session_idempotently(self):
        simulator = tb.Simulator(load_model())
        session = Mock()
        simulator._functional_session = session

        simulator.clear_execution_cache()
        simulator.clear_execution_cache()

        session.close.assert_called_once_with()
        self.assertIsNone(simulator._functional_session)

    def test_per_call_override_preserves_histories(self):
        start = EXAMPLE_START[0]
        end = start + datetime.timedelta(hours=2)
        model = load_model()
        simulator = tb.Simulator(model, execution_mode="functional")

        simulator.simulate(
            start_time=start,
            end_time=end,
            step_size=1200,
            show_progress_bar=False,
            execution_mode="object",
        )
        sensor = model.components["office_temperature_sensor"]
        reference = sensor.output["measuredValue"].history().detach().cpu().numpy()
        self.assertEqual(simulator._last_execution_mode, "object")
        self.assertEqual(simulator._last_execution_backend, "eager")

        simulator.simulate(
            start_time=start,
            end_time=end,
            step_size=1200,
            show_progress_bar=False,
        )
        candidate = sensor.output["measuredValue"].history().detach().cpu().numpy()
        self.assertEqual(simulator._last_execution_mode, "functional")
        self.assertEqual(simulator._last_execution_backend, "eager")
        np.testing.assert_allclose(candidate, reference, rtol=1e-5, atol=1e-4)

    def test_functional_never_runs_full_timestep_loop(self):
        start = EXAMPLE_START[0]
        model = load_model()
        simulator = tb.Simulator(model, execution_mode="functional")

        def forbidden(*args, **kwargs):
            raise AssertionError("full object-graph timestep loop was called")

        simulator._do_system_time_step = forbidden
        simulator.simulate(
            start_time=start,
            end_time=start + datetime.timedelta(hours=1),
            step_size=1200,
            show_progress_bar=False,
        )
        self.assertEqual(simulator._last_execution_mode, "functional")

    def test_repeated_functional_simulation_reuses_exogenous_recording(self):
        start = EXAMPLE_START[0]
        model = load_model()
        simulator = tb.Simulator(model, execution_mode="functional")
        kwargs = {
            "start_time": start,
            "end_time": start + datetime.timedelta(hours=1),
            "step_size": 1200,
            "show_progress_bar": False,
        }

        simulator.simulate(**kwargs)
        self.assertFalse(simulator._exogenous_recording_cache_hit)
        first = model.components["office_temperature_sensor"].output[
            "measuredValue"
        ].history().detach().clone()

        simulator.simulate(**kwargs)
        self.assertTrue(simulator._exogenous_recording_cache_hit)
        second = model.components["office_temperature_sensor"].output[
            "measuredValue"
        ].history().detach().clone()
        np.testing.assert_allclose(second.cpu(), first.cpu(), rtol=1e-5, atol=1e-4)

    def test_changed_period_invalidates_exogenous_recording(self):
        start = EXAMPLE_START[0]
        simulator = tb.Simulator(load_model(), execution_mode="functional")
        kwargs = {
            "end_time": start + datetime.timedelta(hours=1),
            "step_size": 1200,
            "show_progress_bar": False,
        }
        simulator.simulate(start_time=start, **kwargs)
        self.assertFalse(simulator._exogenous_recording_cache_hit)

        shifted = start + datetime.timedelta(hours=1)
        simulator.simulate(
            start_time=shifted,
            end_time=shifted + datetime.timedelta(hours=1),
            step_size=1200,
            show_progress_bar=False,
        )
        self.assertFalse(simulator._exogenous_recording_cache_hit)

    def test_estimation_recording_api_reuses_exogenous_tape(self):
        start = EXAMPLE_START[0]
        end = start + datetime.timedelta(hours=1)
        model = load_model()
        simulator = tb.Simulator(model, execution_mode="functional")
        model.initialize([start], [end], [1200])
        layout, functional_model = simulator.build_functional_model(
            step_size=[1200]
        )

        first = simulator.record_exogenous_inputs(
            functional_model,
            [start],
            [end],
            [1200],
            layout=layout,
        )
        self.assertFalse(simulator._exogenous_recording_cache_hit)
        second = simulator.record_exogenous_inputs(
            functional_model,
            [start],
            [end],
            [1200],
            layout=layout,
        )
        self.assertTrue(simulator._exogenous_recording_cache_hit)
        np.testing.assert_allclose(
            second.exogenous_tape[0].cpu(),
            first.exogenous_tape[0].cpu(),
            rtol=0,
            atol=0,
        )

    def test_cuda_graph_rejects_cpu_model(self):
        start = EXAMPLE_START[0]
        simulator = tb.Simulator(
            load_model(),
            execution_mode="functional",
            execution_backend="cuda_graph",
        )
        with self.assertRaisesRegex(RuntimeError, "requires a functional CUDA model"):
            simulator.simulate(
                start_time=start,
                end_time=start + datetime.timedelta(hours=1),
                step_size=1200,
                show_progress_bar=False,
            )


if __name__ == "__main__":
    unittest.main()
