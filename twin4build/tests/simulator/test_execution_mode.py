import datetime
import unittest

import numpy as np

import twin4build as tb
from twin4build.examples.collocation_comparison import EXAMPLE_START, load_model


class TestSimulatorExecutionMode(unittest.TestCase):
    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            tb.Simulator(load_model(), execution_mode="invalid")

    def test_per_call_override_preserves_histories(self):
        start = EXAMPLE_START[0]
        end = start + datetime.timedelta(hours=2)
        model = load_model()
        simulator = tb.Simulator(model, execution_mode="composed")

        simulator.simulate(
            start_time=start,
            end_time=end,
            step_size=1200,
            show_progress_bar=False,
            execution_mode="object_graph",
        )
        sensor = model.components["office_temperature_sensor"]
        reference = sensor.output["measuredValue"].history().detach().cpu().numpy()
        self.assertEqual(simulator._last_execution_mode, "object_graph")

        simulator.simulate(
            start_time=start,
            end_time=end,
            step_size=1200,
            show_progress_bar=False,
        )
        candidate = sensor.output["measuredValue"].history().detach().cpu().numpy()
        self.assertEqual(simulator._last_execution_mode, "composed")
        np.testing.assert_allclose(candidate, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
