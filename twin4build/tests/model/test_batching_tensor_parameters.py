"""``Model.batch_components()`` stacks fixed parameters.

``set_parameters`` turns a parameter that is set but not estimated into a
:class:`~twin4build.utils.types.TensorParameter` (an identified controller's
gains after a controllers stage, say).  Its batch is one ``TensorParameter``
whose ``n_c`` comes from the stacked values; passing ``n_c`` to the
constructor (which infers it) left every such class unbatched.
"""

import datetime
import unittest

import torch
from dateutil import tz

import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter


class GainSystem(core.System):
    """``y = gain * x``, ``gain`` a parameter."""

    x = Input()
    y = Output()
    gain = Parameter(1.0, lb=0.0, ub=10.0)

    def forward(self, x, inputs, p, step_size):
        return x, {"y": inputs["x"] * p["gain"]}


class TestBatchingStacksTensorParameters(unittest.TestCase):
    def test_fixed_parameters_batch(self):
        model = tb.Model(id="batching_tensor_parameters")
        schedule = tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": 1.0}, id="schedule")
        gains = [GainSystem(id=f"gain{i}") for i in range(3)]
        for i, component in enumerate(gains):
            # what set_parameters leaves on a parameter that is set but not estimated
            component.gain = tps.TensorParameter(torch.tensor(float(i + 1)), min_value=0.0, max_value=10.0, normalized=False)
            model.add_connection(schedule, component, "scheduleValue", "x")
        model.load(draw_semantic_model=False, draw_simulation_model=False)

        batched = model.batch_components()

        metas = [c for c in batched.components.values() if isinstance(c, GainSystem)]
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0]._n_c_batched, 3)
        self.assertIsInstance(metas[0].gain, tps.TensorParameter)
        torch.testing.assert_close(metas[0].gain.get().reshape(-1), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        start = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)
        batched.initialize(start_time=[start], end_time=[start + datetime.timedelta(hours=1)], step_size=600)


if __name__ == "__main__":
    unittest.main()
