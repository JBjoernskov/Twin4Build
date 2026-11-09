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
9. initial values: a state and an output start at a setting, an output at its
   state's start;
10. a parameter's default computed from the System, an old keyword that moved
    into a child, a removed keyword that names its replacement, and the alias
    tables a renamed port fills;
11. an input that reads another input while unwired, in both engines; a child
    that takes its composite's wired width; a child parameter shared with
    another; a child's own id;
12. settings batch per copy: a value setting may differ between batched
    copies and reaches forward through p; a structural one splits batches.
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


class TankSystem(core.System):
    """A level that holds: the state and both outputs start at ``start``."""

    start = Setting(5.0)
    stored = State(init=start)
    level = Output(init=start)
    reading = Output(init=stored)

    def forward(self, x, inputs, p, step_size):
        return x, {"level": x[..., 0], "reading": x[..., 0]}


class PlateSystem(core.System):
    """A parameter computed from a setting; an old keyword now a child's."""

    DEPRECATED_KEYWORDS = {"part_gain": "part.gain"}
    REMOVED_ATTRIBUTES = {"legacy": "give width= instead"}

    width = Setting(2.0)
    area = Parameter(lambda plate: plate.width ** 2, lb=0.1, ub=100.0)
    part = Child(GainSystem)


class MixSystem(core.System):
    """``b`` reads ``a`` while unwired."""

    a = Input()
    b = Input(default=a)
    y = Output()

    def forward(self, x, inputs, p, step_size):
        return x, {"y": inputs["a"] + 10.0 * inputs["b"]}


class FlowSumSystem(core.System):
    flows = Input(size="wired")
    total = Output()

    def forward(self, x, inputs, p, step_size):
        return x, {"total": inputs["flows"].sum(dim=-1)}


class OuterSystem(core.System):
    """A child sums the composite's wired vector of the same name; its gain is the other child's."""

    SHARED_UNIT_PARAMS = {"second.gain": "first.gain"}

    flows = Input(size="wired")
    total = Output()
    inner = Child(FlowSumSystem)
    first = Child(GainSystem)
    second = Child(GainSystem)

    def forward(self, x, inputs, p, step_size):
        _, out = self.inner.forward(None, {"flows": inputs["flows"]}, {}, step_size)
        return x, out


class BiasSystem(core.System):
    """A value setting (``bias``), a structural one (``negate``) and one that
    sizes a port (``width``)."""

    bias = Setting(0.0)
    negate = Setting(False, structural=True)
    width = Setting(1)
    u = Input()
    extra = Input(size=width, default=0.0)
    y = Output()

    def forward(self, x, inputs, p, step_size):
        y = inputs["u"] + p["bias"]
        return x, {"y": -y if self.negate else y}


def _bias_model(model_id, biases, negate=(False, False)):
    model = tb.Model(id=model_id)
    for k, (bias, neg) in enumerate(zip(biases, negate)):
        system = BiasSystem(id=f"bias{k}", bias=bias, negate=neg)
        model.add_connection(_signal(f"signal{k}"), system, "scheduleValue", "u")
        model.add_connection(system, LagSystem(id=f"lag{k}"), "y", "u")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


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

    def test_batching_stacks_every_declared_parameter(self):
        """A fan's flow range has no bounds (it is not estimated) and is
        still one value per copy once batched."""
        model = tb.Model(id="test_declared_batch_fans")
        fans = []
        for k, (lo, hi) in enumerate(((0.0, 1.0), (0.5, 3.0))):
            fan = tb.FanSystem(id=f"fan{k}")
            fan.set_flow_range(lo, hi)
            fans.append(fan)
            model.add_connection(_signal(f"flow{k}"), fan, "scheduleValue", "airFlowRate")
            model.add_connection(_signal(f"temperature{k}"), fan, "scheduleValue", "inletAirTemperature")
            model.add_connection(fan, LagSystem(id=f"lag{k}"), "Power", "u")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        batched = model.batch_components()
        meta, _ = model._component_to_meta["fan0"]
        self.assertIs(model._component_to_meta["fan1"][0], meta)
        torch.testing.assert_close(meta.flowRangeLower.get().reshape(-1).cpu(), torch.tensor([0.0, 0.5], dtype=torch.float64))
        torch.testing.assert_close(meta.flowRangeUpper.get().reshape(-1).cpu(), torch.tensor([1.0, 3.0], dtype=torch.float64))
        torch.testing.assert_close(
            meta.beta3.get().reshape(-1).cpu(),
            torch.stack([f.beta3.get().reshape(-1)[0] for f in fans]).cpu(),
        )
        _remove("test_declared_batch_fans", batched.id)


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


class TestInitialValues(unittest.TestCase):
    def test_a_state_and_its_outputs_start_at_a_setting(self):
        tank = TankSystem(id="tank", start=7.0)
        tank.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=[STEP])
        self.assertEqual(float(tank.get_state().reshape(-1)[0]), 7.0)
        self.assertEqual(tank.output["level"].init_value, 7.0)
        self.assertEqual(tank.output["reading"].init_value, 7.0)
        self.assertEqual(float(tank.output["reading"].get().reshape(-1)[0]), 7.0)
        tank.start = 3.0  # the next run starts at the new value
        tank.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=[STEP])
        self.assertEqual(float(tank.get_state().reshape(-1)[0]), 3.0)

    def test_init_takes_a_number_a_setting_or_a_state(self):
        with self.assertRaisesRegex(TypeError, "init="):
            State(init="warm")
        with self.assertRaisesRegex(TypeError, "init="):
            State(init=State())
        with self.assertRaisesRegex(TypeError, "init="):
            class Stray(core.System):
                level = Output(init=Setting(1.0))


class TestBuildRules(unittest.TestCase):
    def test_a_default_computed_from_the_system_and_a_given_value(self):
        self.assertEqual(float(PlateSystem(id="p", width=3.0).area.get()), 9.0)
        self.assertEqual(float(PlateSystem(id="q", width=3.0, area=5.0).area.get()), 5.0)

    def test_an_old_keyword_moves_into_its_child(self):
        with self.assertWarns(DeprecationWarning):
            plate = PlateSystem(id="p", part_gain=5.0)
        self.assertEqual(float(plate.part.gain.get()), 5.0)
        self.assertEqual(plate._batch_init_kwargs["part"], {"gain": 5.0})

    def test_a_removed_keyword_names_its_replacement(self):
        with self.assertRaisesRegex(TypeError, "give width= instead"):
            PlateSystem(id="p", legacy=1)

    def test_a_renamed_port_fills_the_alias_table(self):
        self.assertEqual(GainSystem.OUTPUT_PORT_ALIASES, {"out": "y"})
        self.assertEqual(GainSystem(id="g").resolve_output_port("y"), "y")
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(GainSystem(id="g").resolve_output_port("out"), "y")


class TestCompositionRules(unittest.TestCase):
    MODEL_ID = "test_declared_composition"

    def tearDown(self):
        _remove(self.MODEL_ID)

    def test_an_unwired_input_reads_another_in_both_engines(self):
        model = tb.Model(id=self.MODEL_ID)
        mix = MixSystem(id="mix")
        model.add_connection(_signal(), mix, "scheduleValue", "a")
        model.add_connection(mix, LagSystem(id="lag"), "y", "u")  # the functional engine steps a state
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        self.assertEqual(mix.input_defaults(), {"b": "a"})
        _simulate(model)
        a = model.components["signal"].output["scheduleValue"].history().reshape(-1).cpu()
        y_object = mix.output["y"].history().reshape(-1).cpu().clone()
        torch.testing.assert_close(y_object, 11.0 * a)
        _simulate(model, mode="functional")
        torch.testing.assert_close(mix.output["y"].history().reshape(-1).cpu(), y_object)

    def test_with_defaults_fills_what_an_engine_fills(self):
        mix = MixSystem(id="mix")
        filled = mix.with_defaults({"a": torch.tensor([2.0])})
        torch.testing.assert_close(filled["b"], torch.tensor([2.0]))
        thermal = tb.BuildingSpaceThermalSystem(id="zone")
        filled = thermal.with_defaults({"outdoorTemperature": torch.tensor([5.0])})
        torch.testing.assert_close(filled["makeUpAirTemperature"], torch.tensor([5.0]))
        torch.testing.assert_close(filled["infiltrationAirFlowRate"], torch.tensor([0.0]))
        with self.assertRaisesRegex(TypeError, "default="):
            class Stray(core.System):
                a = Input(default=Input())

    def test_a_child_takes_the_wired_width_and_a_shared_parameter(self):
        model = tb.Model(id=self.MODEL_ID)
        outer = OuterSystem(id="outer")
        for k in range(3):
            model.add_connection(_signal(f"signal{k}"), outer, "scheduleValue", "flows", input_port_index=k)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        self.assertEqual(outer.inner.input["flows"].n_v, 3)
        self.assertNotIn("second.gain", OuterSystem.PARAM_NAMES)
        self.assertNotIn("second.gain", outer.config["parameters"])
        self.assertIs(outer.second.gain, outer.first.gain)

    def test_a_child_may_give_its_own_id(self):
        self.assertEqual(PairSystem(id="pair").first.id, "pair_first")
        self.assertEqual(PairSystem(id="pair", first=dict(id="custom")).first.id, "custom")


class TestSettingsPerCopy(unittest.TestCase):
    def tearDown(self):
        _remove("test_settings_separate", "test_settings_batched", "test_settings_split", "test_walls_t_init")

    def test_which_settings_are_values(self):
        self.assertEqual(BiasSystem.value_settings(), ("bias",))
        self.assertEqual(tb.WallSystem.value_settings(), ("T_init",))
        self.assertEqual(tb.ScheduleSystem.value_settings(), ())  # a data source
        self.assertEqual(tb.AirHandlingUnitSystem.value_settings(), ())

    def test_copies_with_different_values_batch_and_keep_them(self):
        separate = _bias_model("test_settings_separate", (1.0, 5.0))
        _simulate(separate)
        model = _bias_model("test_settings_batched", (1.0, 5.0))
        batched = model.batch_components()
        meta, _ = model._component_to_meta["bias0"]
        self.assertIs(model._component_to_meta["bias1"][0], meta)
        torch.testing.assert_close(meta.bias, torch.tensor([1.0, 5.0], dtype=torch.float64))
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        for mode in ("object", "functional"):
            _simulate(batched, mode=mode)
            for k in range(2):
                meta, i_c = model._component_to_meta[f"bias{k}"]
                got = batched.components[meta.id].output["y"].history(i_c=i_c).reshape(-1).detach().cpu()
                torch.testing.assert_close(got, separate.components[f"bias{k}"].output["y"].history().reshape(-1).cpu(),
                                           msg=mode)
        _remove(batched.id)

    def test_a_structural_setting_splits_the_batch(self):
        model = _bias_model("test_settings_split", (1.0, 1.0), negate=(False, True))
        batched = model.batch_components()
        self.assertIsNot(model._component_to_meta["bias0"][0], model._component_to_meta["bias1"][0])
        _remove(batched.id)

    def test_walls_start_at_their_own_value_in_one_batch(self):
        model = tb.Model(id="test_walls_t_init")
        node = tb.ThermalMassNodeSystem(C=5e7, T_init=22.0, id="node")
        for k, t_init in enumerate((18.0, 26.0)):
            wall = tb.WallSystem(T_init=t_init, id=f"wall{k}")
            model.add_connection(node, wall, "temperature", "temperatureA")
            model.add_connection(node, wall, "temperature", "temperatureB")
            model.add_connection(wall, node, "heatFlowRateB", "heatFlowRate", input_port_index=k)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        batched = model.batch_components()
        meta, _ = model._component_to_meta["wall0"]
        self.assertIs(model._component_to_meta["wall1"][0], meta)
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        meta = batched.components[meta.id]
        meta.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=[STEP])
        torch.testing.assert_close(meta.get_state().reshape(-1).cpu(), torch.tensor([18.0, 26.0], dtype=torch.float64))
        torch.testing.assert_close(meta.output["wallTemperature"].get().reshape(-1).cpu(),
                                   torch.tensor([18.0, 26.0], dtype=torch.float64))
        _remove(batched.id)


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
