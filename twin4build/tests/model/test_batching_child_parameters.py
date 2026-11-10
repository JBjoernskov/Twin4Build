"""``Model.batch_components()`` stacks a composite's children's parameters.

A System that is not declared names its parameters in ``PARAM_NAMES`` (and
its config); its ``parameter`` map holds the bounds of its own parameters
only.  Batching took the map's keys as the parameters whenever the map was
not empty, so a composite with one bounded parameter of its own lost every
child's: the occupancy model's ``mass.V`` stayed at the constructor default
for every copy once it gained ``exhaustFlowRatio`` (#199), and the batched
zones' temperatures parted from the unbatched ones.
"""

import datetime
import unittest

import torch
from dateutil import tz

import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps


class _Child:
    """A sub-model holding a parameter, as ``OccupancySystem.mass`` does."""

    def __init__(self, V: float):
        self.V = tps.Parameter(torch.tensor(float(V)), min_value=1.0, max_value=1000.0)


class CompositeSystem(core.System):
    """``y = x * child.V * ratio``: a child's parameter and a bounded one of its own."""

    PARAM_NAMES = ("child.V", "ratio")

    def __init__(self, V: float = 100.0, ratio: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.child = _Child(V)
        self.ratio = tps.Parameter(torch.tensor(float(ratio)), min_value=0.1, max_value=10.0)
        self.parameter = {"ratio": {"lb": 0.1, "ub": 10.0}}  # its own bounds only
        self._config = {"parameters": ["child.V", "ratio"]}
        self.input = {"x": tps.Scalar()}
        self.output = {"y": tps.Scalar()}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size):
        _, _, n_t, _ = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        for port in (*self.input.values(), *self.output.values()):
            port.initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)

    def do_step(self, second_time, date_time, step_size, step_index):
        y = self.input["x"].get() * self.child.V.get() * self.ratio.get()
        self.output["y"]._set(y, i_t=step_index)


START = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)


def _build(model_id):
    model = tb.Model(id=model_id)
    schedule = tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": 2.0}, id="schedule")
    composites = [CompositeSystem(V=10.0 * (i + 1), ratio=0.5 + i, id=f"composite{i}") for i in range(3)]
    for component in composites:
        model.add_connection(schedule, component, "scheduleValue", "x")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, composites


def _simulate(model):
    tb.Simulator(model).simulate(
        start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=600, show_progress_bar=False
    )


class TestBatchingStacksChildParameters(unittest.TestCase):
    def test_each_copy_keeps_its_child_parameter(self):
        model, _ = _build("batching_child_parameters")
        batched = model.batch_components()
        metas = [c for c in batched.components.values() if isinstance(c, CompositeSystem)]
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0]._n_c_batched, 3)
        torch.testing.assert_close(
            metas[0].child.V.get().reshape(-1), torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            metas[0].ratio.get().reshape(-1), torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        )

    def test_the_batched_model_computes_what_the_copies_do(self):
        model, composites = _build("batching_child_parameters_parity")
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        _simulate(batched)
        for component in composites:
            meta, i_c = model.get_batched_component_info(component.id)
            torch.testing.assert_close(
                meta.output["y"]._history[..., i_c], component.output["y"]._history[..., 0]
            )


if __name__ == "__main__":
    unittest.main()
