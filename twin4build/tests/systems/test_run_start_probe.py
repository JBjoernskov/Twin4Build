"""The probe of every System, on request (ADR 0001, decision 10):
``Model.check(...)`` and ``Simulator.simulate(check=True)``.

Covered:

1. a ``forward`` that leaves out an output, gives one the System does not
   declare, gives an output at another size, or gives a state of another
   shape, and a ``derive`` that gives no dict, raise when probed, naming the
   System and what is wrong; every problem of the model at once;
2. a data source's ``series`` that gives an output the System does not
   declare, or a vector output at another size, raises where it fills the
   outputs, at every run start;
3. the probe runs on request only: a plain run does not probe, ``check``
   and ``simulate(check=True)`` do;
4. a correct System runs as before (the probe writes nothing).
"""

# Standard library imports
import datetime
import os
import shutil
import unittest
from unittest import mock

# Third party imports
import numpy as np
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter, Setting, State
from twin4build.systems.saref4syst.system import System

tb._IS_TESTING = True

START = datetime.datetime(2023, 1, 2, tzinfo=tz.UTC)
STEP = 600


class TwoOutputs(core.System):
    """``a = u``, ``b = 2 u``; ``drop`` leaves ``b`` out, ``add`` gives a third."""

    drop = Setting(False, structural=True)
    add = Setting(False, structural=True)
    u = Input()
    a = Output()
    b = Output()

    def forward(self, x, inputs, p, step_size):
        outs = {"a": inputs["u"], "b": 2.0 * inputs["u"]}
        if self.drop:
            del outs["b"]
        if self.add:
            outs["c"] = inputs["u"]
        return x, outs


class Widths(core.System):
    """``y`` has three values a copy; ``wrong`` gives two."""

    wrong = Setting(False, structural=True)
    u = Input()
    y = Output(size=3)

    def forward(self, x, inputs, p, step_size):
        n = 2 if self.wrong else 3
        return x, {"y": inputs["u"].unsqueeze(-1).expand(-1, n)}


class Lag(core.System):
    """A state of two; ``wrong`` gives back one."""

    wrong = Setting(False, structural=True)
    u = Input()
    y = Output()
    level = State(size=2, init=0.0)

    def forward(self, x, inputs, p, step_size):
        x_next = 0.5 * x + 0.5 * inputs["u"].unsqueeze(-1)
        if self.wrong:
            x_next = x_next[..., :1]
        return x_next, {"y": x_next[..., 0]}


class Derived(core.System):
    """``y = gain2 u`` with ``gain2`` derived; ``wrong`` derives a tuple."""

    wrong = Setting(False, structural=True)
    u = Input()
    y = Output()
    gain = Parameter(2.0, lb=0.0, ub=10.0)

    def derive(self, p, step_size):
        return (p["gain"] ** 2,) if self.wrong else {"gain2": p["gain"] ** 2}

    def forward(self, x, inputs, p, step_size):
        return x, {"y": p["gain2"] * inputs["u"]}


class Recorded(core.System):
    """A data source of one scalar and one vector of two; ``extra`` gives a
    third series, ``wrong`` the vector with three values a step."""

    extra = Setting(False)
    wrong = Setting(False)
    level = Output()
    profile = Output(size=2)

    def series(self, timestamps, step_size):
        n_s, n_t = timestamps.shape
        out = {"level": np.ones((n_s, n_t)), "profile": np.ones((n_s, n_t, 3 if self.wrong else 2))}
        if self.extra:
            out["other"] = np.ones((n_s, n_t))
        return out


def _signal(sid):
    return tb.ScheduleSystem(
        id=sid,
        weekday_ruleset={
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
            "ruleset_default_value": 1.0,
        },
    )


def _model(model_id, *systems):
    model = tb.Model(id=model_id)
    for k, system in enumerate(systems):
        if "u" in system.input:
            model.add_connection(_signal(f"u{k}"), system, "scheduleValue", "u")
        else:
            model.add_component(system)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


END = START + datetime.timedelta(seconds=3 * STEP)


def _simulate(model, mode="object", check=True):
    tb.Simulator(model, execution_mode=mode).simulate(
        start_time=START, end_time=END, step_size=STEP, show_progress_bar=False, check=check,
    )


class TestRunStartProbe(unittest.TestCase):
    MODEL_IDS = ("probe_outputs", "probe_widths", "probe_state", "probe_derive", "probe_all",
                 "probe_series_extra", "probe_series_width", "probe_on_request", "probe_ok")

    def tearDown(self):
        for model_id in self.MODEL_IDS:
            for suffix in ("", "_simulation_model"):
                shutil.rmtree(os.path.join("generated_files", "models", model_id + suffix), ignore_errors=True)

    def test_forward_gives_exactly_the_declared_outputs(self):
        model = _model("probe_outputs", TwoOutputs(id="dropped", drop=True), TwoOutputs(id="added", add=True))
        with self.assertRaises(ValueError) as caught:
            _simulate(model)
        message = str(caught.exception)
        self.assertIn("TwoOutputs 'dropped': forward gave no value for the output(s) b", message)
        self.assertIn("TwoOutputs 'added': forward gave c, which the System does not declare", message)

    def test_forward_gives_each_output_at_its_size(self):
        model = _model("probe_widths", Widths(id="widths", wrong=True))
        with self.assertRaisesRegex(ValueError, r"Widths 'widths': forward gave the output 'y' as \(1, 2\), not 1 copies of 3"):
            _simulate(model)

    def test_forward_gives_a_state_of_the_states_shape(self):
        model = _model("probe_state", Lag(id="lag", wrong=True))
        with self.assertRaisesRegex(ValueError, r"Lag 'lag': forward gave a state of \(1, 1\), not \(1, 2\)"):
            _simulate(model)

    def test_derive_gives_a_dict(self):
        model = _model("probe_derive", Derived(id="derived", wrong=True))
        with self.assertRaisesRegex(ValueError, "Derived 'derived': derive returned a tuple, not a dict"):
            _simulate(model)

    def test_every_problem_of_the_model_at_once_in_either_engine(self):
        for mode in ("object", "functional"):
            with self.subTest(mode=mode):
                model = _model("probe_all", Lag(id="lag", wrong=True), Widths(id="widths", wrong=True))
                with self.assertRaises(ValueError) as caught:
                    _simulate(model, mode)
                message = str(caught.exception)
                self.assertIn("Lag 'lag'", message)
                self.assertIn("Widths 'widths'", message)

    def test_series_gives_exactly_the_declared_outputs_at_their_sizes(self):
        with self.assertRaisesRegex(ValueError, "series\\(\\) gave other, which the System does not declare"):
            _simulate(_model("probe_series_extra", Recorded(id="recorded", extra=True)), check=False)
        with self.assertRaisesRegex(ValueError, "series\\(\\) gave 'profile' with 3 values a step, its size is 2"):
            _simulate(_model("probe_series_width", Recorded(id="recorded", wrong=True)), check=False)

    def test_the_probe_runs_on_request_only(self):
        model = _model("probe_on_request", TwoOutputs(id="added", add=True))
        with mock.patch.object(System, "_probe", autospec=True, side_effect=System._probe) as probe:
            _simulate(model, check=False)  # an output it does not declare, unprobed: the run goes on
            self.assertEqual(probe.call_count, 0)
        with self.assertRaisesRegex(ValueError, "TwoOutputs 'added': forward gave c"):
            model.check(START, END, STEP)
        with self.assertRaisesRegex(ValueError, "TwoOutputs 'added': forward gave c"):
            _simulate(model, check=True)

    def test_a_correct_system_runs_as_before(self):
        model = _model("probe_ok", Lag(id="lag"), Derived(id="derived"), Widths(id="widths"), Recorded(id="recorded"))
        _simulate(model)
        y = model.components["derived"].output["y"].history().reshape(-1)
        torch.testing.assert_close(y, torch.full_like(y, 4.0))
        level = model.components["lag"].output["y"].history().reshape(-1)
        torch.testing.assert_close(level, torch.tensor([0.5, 0.75, 0.875], dtype=level.dtype))


if __name__ == "__main__":
    unittest.main()
