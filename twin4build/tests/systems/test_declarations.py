"""A System written as class-level declarations plus one ``forward`` (ADR 0001).

Covered:

1. class definition: one input and one output may share a name, ports and
   parameters are not class attributes, any other repeated name raises, a
   ``size=`` must name a declaration of the class, a subclass inherits and
   redeclares;
2. construction: settings (default, required), parameter values by keyword,
   unknown keywords, the constructor keywords, ``PARAM_NAMES`` in declaration
   order, the bounds table and the saved config, attribute access to ports,
   a renamed port's alias;
3. sizes: ``"wired"``, another port, a setting, and wiring beyond a size;
4. stepping a stateful declared System: object mode against the closed form
   and against the functional engine;
5. two copies with different parameters, batched, against the copies stepped
   separately;
6. a saved model reloads with the same parameters and settings;
7. a composite's children: overrides, dotted parameter paths, a child and a
   parameter per slot of an input;
8. the converted valve keeps its constructor keywords, bounds and flow.
"""

# Standard library imports
import datetime
import math
import os
import shutil
import unittest
import warnings

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.registry import SystemRegistry
from twin4build.systems.saref4syst.declarations import (
    Child,
    Input,
    Output,
    Parameter,
    Setting,
    State,
)

tb._IS_TESTING = True

START = datetime.datetime(2023, 1, 2, tzinfo=tz.UTC)
STEP = 600
HOURS = 3


class LagSystem(core.System):
    """``level`` follows ``u`` with time constant ``tau`` [s]; ``y`` reads it."""

    u = Input(doc="Signal")
    y = Output(doc="Lagged signal")
    level = State(init=0.0)
    tau = Parameter(1800.0, lb=60.0, ub=36000.0, scaling="log", doc="Time constant [s]")

    def forward(self, x, inputs, p, step_size):
        a = torch.exp(-step_size / p["tau"]).unsqueeze(-1)
        x_next = a * x + (1.0 - a) * inputs["u"].unsqueeze(-1)
        return x_next, {"y": x_next[..., 0]}


class GainSystem(core.System):
    """``y = gain * x``; the gain's default is a setting."""

    x = Input()
    x = Output()
    y = Output(alias="out")
    offset = Setting(0.0)
    gain = Parameter(2.0, lb=0.0, ub=10.0)

    def forward(self, x, inputs, p, step_size):
        return x, {"x": inputs["x"], "y": p["gain"] * inputs["x"] + self.offset}


class SizedSystem(core.System):
    """Vector ports sized three ways."""

    n_zones = Setting()
    flows = Input(size="wired")
    temperatures = Input(size=flows)
    weights = Input(size=n_zones, default=1.0)
    total = Output()

    def forward(self, x, inputs, p, step_size):
        return x, {"total": (inputs["flows"] * inputs["temperatures"]).sum(-1) + inputs["weights"].sum(-1)}


class PairSystem(core.System):
    """A composite of two gains."""

    x = Input()
    y = Output()
    first = Child(GainSystem, gain=3.0)
    second = Child(GainSystem)

    def forward(self, x, inputs, p, step_size):
        _, a = self.first.forward(None, {"x": inputs["x"]}, {"gain": p["first.gain"]}, step_size)
        _, b = self.second.forward(None, {"x": a["y"]}, {"gain": p["second.gain"]}, step_size)
        return x, {"y": b["y"]}


class BranchesSystem(core.System):
    """A gain per branch (a child stepped once per slot of ``u``), and a scale
    per branch only with ``scale_per_branch``."""

    scale_per_branch = Setting(False)
    u = Input(size="wired")
    y = Output(size=u)
    scale = Parameter(1.0, lb=0.1, ub=10.0, per_slot=u, per_slot_if=scale_per_branch)
    branch = Child(GainSystem, per_slot=u)

    def forward(self, x, inputs, p, step_size):
        return x, {"y": inputs["u"]}


def _signal(sid="signal"):
    return tb.ScheduleSystem(
        id=sid,
        weekday_ruleset={
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [1],
            "ruleset_end_hour": [2],
            "ruleset_value": [3.0],
            "ruleset_default_value": 1.0,
        },
    )


def _remove(*model_ids):
    for model_id in model_ids:
        shutil.rmtree(os.path.join("generated_files", "models", model_id), ignore_errors=True)


def _lag_model(model_id, taus=(1800.0,), registry=None):
    model = tb.Model(id=model_id) if registry is None else tb.Model(id=model_id, system_registry=registry)
    signal = _signal()
    for k, tau in enumerate(taus):
        model.add_connection(signal, LagSystem(id=f"lag{k}", tau=tau), "scheduleValue", "u")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def _simulate(model, mode="object"):
    tb.Simulator(model, execution_mode=mode).simulate(
        start_time=START,
        end_time=START + datetime.timedelta(hours=HOURS),
        step_size=STEP,
        show_progress_bar=False,
    )


def _history(model, cid, port="y"):
    return model.components[cid].output[port].history().reshape(-1).detach().cpu()


def _closed_form(model, tau):
    """The lag of the schedule's own output."""
    a = math.exp(-STEP / tau)
    level, out = 0.0, []
    for u in _history(model, "signal", "scheduleValue").tolist():
        level = a * level + (1 - a) * u
        out.append(level)
    return torch.tensor(out, dtype=torch.float64)


class TestClassDefinition(unittest.TestCase):
    def test_one_input_and_one_output_share_a_name(self):
        self.assertEqual(list(GainSystem._declarations["input"]), ["x"])
        self.assertEqual(list(GainSystem._declarations["output"]), ["x", "y"])
        self.assertFalse(hasattr(GainSystem, "x"))
        self.assertFalse(hasattr(GainSystem, "gain"))

    def test_any_other_repeat_raises(self):
        bodies = {
            "two inputs": "class X(core.System):\n    v = Input()\n    v = Input()\n",
            "a parameter and a port": "class X(core.System):\n    v = Parameter(1.0)\n    v = Input()\n",
            "a port and a plain value": "class X(core.System):\n    v = Output()\n    v = 3\n",
            "a method and a port": "class X(core.System):\n    def v(self):\n        pass\n    v = Input()\n",
        }
        for label, body in bodies.items():
            with self.subTest(label), self.assertRaises(TypeError):
                exec(body, {"core": core, "Input": Input, "Output": Output, "Parameter": Parameter})

    def test_size_must_name_a_declaration_of_the_class(self):
        with self.assertRaisesRegex(TypeError, "size="):
            Input(size="five")
        with self.assertRaisesRegex(TypeError, "does not declare"):

            class Other(core.System):
                flows = Input(size=SizedSystem._declarations["input"]["flows"])

    def test_a_subclass_inherits_and_redeclares(self):
        class Wider(GainSystem):
            gain = Parameter(4.0, lb=0.0, ub=20.0)
            bias = Input(default=0.0)

        self.assertEqual(list(Wider._declarations["input"]), ["x", "bias"])
        self.assertEqual(Wider._declarations["parameter"]["gain"].value, 4.0)
        self.assertEqual(Wider.PARAM_NAMES, ("gain",))
        self.assertEqual(float(Wider(id="w").gain.get()), 4.0)


class TestConstruction(unittest.TestCase):
    def test_the_declarations_come_from_core(self):
        self.assertIs(core.Input, Input)
        self.assertIs(core.Parameter, Parameter)
        self.assertIsNot(core.Parameter, tps.Parameter)

    def test_settings_and_parameters_by_keyword(self):
        gain = GainSystem(id="g", gain=5.0, offset=1.5)
        self.assertEqual(float(gain.gain.get()), 5.0)
        self.assertEqual(gain.offset, 1.5)
        self.assertIsInstance(gain.gain, tps.Parameter)
        self.assertIn("gain", dict(gain.named_parameters()))

    def test_a_required_setting_and_an_unknown_keyword_raise(self):
        with self.assertRaisesRegex(TypeError, "n_zones"):
            SizedSystem(id="s")
        with self.assertRaisesRegex(TypeError, "gian"):
            GainSystem(id="g", gian=1.0)

    def test_what_the_base_derives_from_the_declarations(self):
        self.assertEqual(
            GainSystem.constructor_keywords() & {"gain", "offset", "id"}, {"gain", "offset", "id"}
        )
        gain = GainSystem(id="g")
        self.assertEqual(GainSystem.PARAM_NAMES, ("gain",))
        self.assertEqual(gain.parameter, {"gain": {"lb": 0.0, "ub": 10.0}})
        self.assertEqual(gain.config, {"parameters": ["offset", "gain"]})
        self.assertEqual(gain._batch_init_kwargs, {"offset": 0.0})

    def test_ports_by_attribute_and_by_old_name(self):
        gain = GainSystem(id="g")
        self.assertIs(gain.input.x, gain.input["x"])
        self.assertIs(gain.output.x, gain.output["x"])
        self.assertIsNot(gain.input.x, gain.output.x)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(gain.output["out"], gain.output.y)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        with self.assertRaises(AttributeError):
            gain.output.nothing


class TestSizes(unittest.TestCase):
    MODEL_ID = "test_declared_sizes"

    def tearDown(self):
        _remove(self.MODEL_ID)

    def test_wired_another_port_and_a_setting(self):
        model = tb.Model(id=self.MODEL_ID)
        sized = SizedSystem(id="sized", n_zones=4)
        for k in range(3):
            signal = _signal(f"signal{k}")
            model.add_connection(signal, sized, "scheduleValue", "flows", input_port_index=k)
            model.add_connection(signal, sized, "scheduleValue", "temperatures", input_port_index=k)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        self.assertEqual(sized.input["flows"].n_v, 3)
        self.assertEqual(sized.input["temperatures"].n_v, 3)
        self.assertEqual(sized.input["weights"].n_v, 4)
        torch.testing.assert_close(
            sized.output["total"].history().reshape(-1)[:2].cpu(), torch.tensor([7.0, 7.0], dtype=torch.float64)
        )

    def test_wiring_beyond_a_size_raises(self):
        model = tb.Model(id=self.MODEL_ID)
        sized = SizedSystem(id="sized", n_zones=2)
        model.add_connection(_signal(), sized, "scheduleValue", "flows", input_port_index=0)
        model.add_connection(_signal("w"), sized, "scheduleValue", "weights", input_port_index=2)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        with self.assertRaisesRegex(ValueError, "weights"):
            _simulate(model)


class TestStepping(unittest.TestCase):
    MODEL_IDS = ("test_declared_object", "test_declared_functional", "test_declared_batch",
                 "test_declared_separate")

    def tearDown(self):
        _remove(*self.MODEL_IDS)

    def test_object_mode_matches_the_closed_form_and_the_functional_engine(self):
        model = _lag_model("test_declared_object")
        _simulate(model, "object")
        torch.testing.assert_close(_history(model, "lag0"), _closed_form(model, 1800.0), rtol=1e-12, atol=1e-12)
        functional = _lag_model("test_declared_functional")
        _simulate(functional, "functional")
        torch.testing.assert_close(_history(functional, "lag0"), _history(model, "lag0"), rtol=1e-10, atol=1e-10)

    def test_batched_copies_keep_their_parameters(self):
        taus = (600.0, 1800.0, 7200.0)
        separate = _lag_model("test_declared_separate", taus)
        _simulate(separate)
        model = _lag_model("test_declared_batch", taus)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(batched)
        metas = {model._component_to_meta[f"lag{k}"][0].id for k in range(3)}
        self.assertEqual(len(metas), 1)
        for k, tau in enumerate(taus):
            meta, i_c = model._component_to_meta[f"lag{k}"]
            got = batched.components[meta.id].output["y"].history(i_c=i_c).reshape(-1).detach().cpu()
            torch.testing.assert_close(got, _history(separate, f"lag{k}"), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(got, _closed_form(separate, tau), rtol=1e-12, atol=1e-12)


class TestSaveAndReload(unittest.TestCase):
    MODEL_ID = "test_declared_saved"

    def tearDown(self):
        _remove(self.MODEL_ID, self.MODEL_ID + "_reloaded")

    def test_a_saved_model_reloads_with_its_values(self):
        registry = SystemRegistry()
        registry.register(LagSystem, type_id="test:LagSystem@1", provider="tests", version="1")
        registry.register(GainSystem, type_id="test:GainSystem@1", provider="tests", version="1")
        model = tb.Model(id=self.MODEL_ID, system_registry=registry)
        lag = LagSystem(id="lag0", tau=900.0)
        gain = GainSystem(id="gain", gain=4.0, offset=0.5)
        model.add_connection(_signal(), lag, "scheduleValue", "u")
        model.add_connection(lag, gain, "y", "x")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        model.serialize()
        path, _ = model.simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")

        reloaded = tb.Model(id=self.MODEL_ID + "_reloaded", system_registry=registry)
        reloaded.load(filename=path, draw_semantic_model=False, draw_simulation_model=False)
        self.assertAlmostEqual(float(reloaded.components["lag0"].tau.get()), 900.0, places=9)
        self.assertAlmostEqual(float(reloaded.components["gain"].gain.get()), 4.0, places=12)
        self.assertEqual(reloaded.components["gain"].offset, 0.5)
        _simulate(model)
        _simulate(reloaded)
        torch.testing.assert_close(_history(reloaded, "gain"), _history(model, "gain"))


class TestChildren(unittest.TestCase):
    def test_children_take_overrides_and_keep_dotted_paths(self):
        pair = PairSystem(id="pair", second=dict(gain=0.5))
        self.assertEqual(float(pair.first.gain.get()), 3.0)
        self.assertEqual(float(pair.second.gain.get()), 0.5)
        self.assertEqual(PairSystem.PARAM_NAMES, ("first.gain", "second.gain"))
        self.assertEqual(
            pair.config["parameters"], ["first.offset", "first.gain", "second.offset", "second.gain"]
        )
        self.assertEqual(pair._batch_init_kwargs, {"second": {"gain": 0.5}})
        names = dict(pair.named_parameters())
        self.assertIn("first.gain", names)
        _, outs = pair.forward(None, {"x": torch.tensor([2.0])}, {"first.gain": torch.tensor([3.0]),
                                                                 "second.gain": torch.tensor([0.5])}, STEP)
        torch.testing.assert_close(outs["y"], torch.tensor([3.0]))


class TestPerSlot(unittest.TestCase):
    def _initialized(self, **values):
        branches = BranchesSystem(id="branches", **values)
        branches.input["u"]._n_v = 3  # stepped on its own: the width by hand
        branches.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=[STEP])
        return branches

    def test_a_child_per_slot_holds_a_value_per_branch(self):
        branches = self._initialized()
        self.assertEqual(branches.branch.n_c, 3)
        self.assertEqual(tuple(branches.branch.gain.get().shape), (3,))
        self.assertEqual(branches.scale.get().numel(), 1)

    def test_a_parameter_per_slot_only_while_its_setting_holds(self):
        branches = self._initialized(scale_per_branch=True)
        self.assertEqual(tuple(branches.scale.get().shape), (3,))

    def test_given_values_per_branch_are_kept(self):
        branches = self._initialized(branch=dict(gain=[1.0, 2.0, 3.0]))
        torch.testing.assert_close(
            branches.branch.gain.get(), torch.tensor([1.0, 2.0, 3.0], dtype=branches.branch.gain.get().dtype)
        )

    def test_per_slot_must_name_an_input_of_the_class(self):
        other = Input(size="wired")
        with self.assertRaisesRegex(TypeError, "per_slot"):
            class Stray(core.System):
                u = Input(size="wired")
                part = Child(GainSystem, per_slot=other)
        with self.assertRaisesRegex(TypeError, "per_slot"):
            Parameter(1.0, per_slot=Setting(3))
        with self.assertRaisesRegex(TypeError, "per_slot_if"):
            Parameter(1.0, per_slot_if=Setting(False))


class TestConvertedValve(unittest.TestCase):
    def test_the_valve_keeps_its_interface_and_flow(self):
        valve = tb.ValveSystem(id="valve", waterFlowRateMax=0.02, valveAuthority=0.5)
        self.assertTrue(
            {"waterFlowRateMax", "valveAuthority", "id"} <= tb.ValveSystem.constructor_keywords()
        )
        self.assertEqual(tb.ValveSystem.PARAM_NAMES, ("waterFlowRateMax", "valveAuthority"))
        self.assertEqual(valve.parameter["waterFlowRateMax"], {"lb": 5e-3, "ub": 1.0, "relative": (0.25, 4.0)})
        self.assertEqual(valve.waterFlowRateMax.scaling, "log")
        u = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64)
        _, outs = valve.forward(None, {"valvePosition": u},
                                {"waterFlowRateMax": torch.tensor(0.02), "valveAuthority": torch.tensor(0.5)}, STEP)
        expected = u / torch.sqrt(u**2 * 0.5 + 0.5) * 0.02
        torch.testing.assert_close(outs["waterFlowRate"], expected)


if __name__ == "__main__":
    unittest.main()
