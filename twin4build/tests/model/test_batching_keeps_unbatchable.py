"""``Model.batch_components()`` keeps a class it cannot batch.

A component whose parameters live on sub-models that only exist after
construction (built at rewire, say) cannot be rebuilt from its constructor
and stacked; before, one such class aborted the batching of the whole
model.  Now its instances join the batched model as shallow copies with
fresh wiring, sharing their sub-models, and every other class is still
batched.
"""

import datetime
import unittest

import torch
from dateutil import tz

import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps


class LateParameterSystem(core.System):
    """Its estimable parameter sits on a sub-model created after ``__init__``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"x": tps.Scalar()}
        self.output = {"y": tps.Scalar()}
        self.parameter = {"sub.gain": {"lb": 0.0, "ub": 10.0}}
        self._config = {"parameters": ["sub.gain"]}

    @property
    def config(self):
        return self._config

    def build(self):
        class Sub:
            pass

        self.sub = Sub()
        self.sub.gain = tps.Parameter(torch.tensor(2.0), min_value=0.0, max_value=10.0)

    def initialize(self, start_time, end_time, step_size):
        _, _, n_t, _ = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        for port in (*self.input.values(), *self.output.values()):
            port.initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)

    def do_step(self, second_time, date_time, step_size, step_index):
        self.output["y"]._set(self.input["x"].get() * self.sub.gain.get(), i_t=step_index)


class TestBatchingKeepsUnbatchableClass(unittest.TestCase):
    def test_late_parameter_class_stays_separate_and_the_rest_is_batched(self):
        model = tb.Model(id="batching_fallback")
        schedule = tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": 1.0}, id="schedule")
        late = [LateParameterSystem(id=f"late{i}") for i in range(2)]
        for component in late:
            component.build()
        products = [tb.ScalarProductSystem(scale_factor=1.0, id=f"product{i}") for i in range(3)]
        for i, product in enumerate(products):
            model.add_connection(schedule, product, "scheduleValue", "input_1")
            model.add_connection(late[i % 2], product, "y", "input_2")
        for component in late:
            model.add_connection(schedule, component, "scheduleValue", "x")
        model.load(draw_semantic_model=False, draw_simulation_model=False)

        batched = model.batch_components()

        kept = [c for c in batched.components.values() if isinstance(c, LateParameterSystem)]
        self.assertEqual({c.id for c in kept}, {"late0", "late1"})
        self.assertTrue(all(getattr(c, "_n_c_batched", 1) == 1 for c in kept))
        metas = [c for c in batched.components.values() if isinstance(c, tb.ScalarProductSystem)]
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0]._n_c_batched, 3)
        # A copy with fresh wiring that shares the source's sub-models.
        kept0 = model._component_to_meta["late0"][0]
        self.assertIsNot(kept0, late[0])
        self.assertIs(kept0.sub, late[0].sub)
        self.assertEqual(kept0.id, "late0")

        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        start = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)
        batched.initialize(start_time=[start], end_time=[start + datetime.timedelta(hours=1)], step_size=600)


if __name__ == "__main__":
    unittest.main()
