"""The conformance check every System passes, in twin4build and in the
packages that build on it (ADR 0001, decision 11).

::

    import unittest
    from twin4build.testing import check_system

    class TestValve(unittest.TestCase):
        def test_the_valve_conforms(self):
            check_system(ValveSystem(id="valve"), inputs={"valvePosition": 0.4})

The example inputs live in the test, not in the System class.
"""

# Standard library imports
import copy
import datetime
import os
import shutil
from typing import Any, Dict, Optional

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.systems.registry import SystemRegistry
from twin4build.systems.saref4syst.declarations import Input, Output, State
from twin4build.systems.saref4syst.system import System
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.utils.data_loaders.sources import Source
from twin4build.utils.rgetattr import rgetattr

__all__ = ["check_system"]

START = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)


class _Probe(System):
    """Holds every output of the System checked, one slot each: the state
    that makes the functional engine step a model of a stateless System."""

    u = Input(size="wired")
    level = State(size=u)
    y = Output(size=u)

    def forward(self, x, inputs, p, step_size):
        return inputs["u"], {"y": inputs["u"]}


def check_system(
    system: System,
    inputs: Optional[Dict[str, Any]] = None,
    *,
    steps: int = 6,
    step_size: int = 600,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    """Check that ``system`` behaves as every System must.

    ``system`` is stepped in a model of its own, each input wired from a
    constant: ``inputs`` maps an input to a number, or, for a vector input,
    to a list of numbers, one per slot.  An input left out keeps its default.
    The checks:

    - save and reload give the same settings and parameters, and the same
      outputs;
    - a deepcopy steps the same as the original and shares no tensor with
      it;
    - object mode and the functional engine give the same outputs;
    - two copies with different parameter values, batched, give the
      outputs of the same two copies stepped separately;
    - the gradient of each output with respect to each parameter is finite
      (a data source's series is known before the run: not checked);
    - on a machine with CUDA, the System steps there as on the CPU.

    Args:
        system: An example of the System, not part of a model.
        inputs: The constant each input is wired from.
        steps: How many steps each run takes.
        step_size: The step size [s].
        rtol: The relative tolerance outputs are compared with.
        atol: The absolute tolerance outputs are compared with.

    Raises:
        AssertionError: Naming the check that failed.
        ValueError: When ``system`` is part of a model.
    """
    if system.connects_at or system.connected_through:
        raise ValueError(f"check_system takes a System that is not part of a model; '{system.id}' is wired")
    inputs = dict(inputs or {})
    copies = [copy.deepcopy(system) for _ in range(5)]
    check = _Check(system, inputs, steps, step_size, rtol, atol)
    try:
        check.deepcopy_shares_nothing(system, copies[0])
        original = check.model(system, "original")
        reference = check.run(original, system)
        check.same(check.run(check.model(copies[0], "copy"), copies[0]), reference, "a deepcopy steps differently")
        check.same(check.run(original, system, mode="functional"), reference,
                   "the functional engine and object mode differ")
        check.save_and_reload(original, system, reference)
        check.batched(copies[1], copies[2])
        check.gradients(copies[3])
        if torch.cuda.is_available():
            check.cuda(copies[4], reference)
    finally:
        check.clean_up()


class _Check:
    def __init__(self, system, inputs, steps, step_size, rtol, atol):
        self.cls = type(system)
        self.name = f"{self.cls.__name__} '{system.id}'"
        self.inputs = inputs
        self.steps = steps
        self.step_size = step_size
        self.rtol = rtol
        self.atol = atol
        self.model_ids = []
        self.registry = SystemRegistry()
        for cls in (self.cls, _Probe):
            if getattr(systems, cls.__name__, None) is not cls:
                self.registry.register(cls, type_id=f"check_system:{cls.__module__}.{cls.__qualname__}@1",
                                       provider="twin4build.testing", version="1")

    # -- building and stepping ----------------------------------------------
    def model(self, system, label, others=()):
        model_id = f"check_system_{self.cls.__name__}_{label}"
        self.model_ids.append(model_id)
        model = Model(id=model_id, system_registry=self.registry)
        for k, component in enumerate((system, *others)):
            self._wire(model, component, f"c{k}")
        model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        return model

    def _wire(self, model, system, tag):
        model.add_component(system)
        for name, value in self.inputs.items():
            if isinstance(value, (list, tuple)):
                for k, v in enumerate(value):
                    model.add_connection(_constant(v, f"{tag}_{name}_{k}"), system, "scheduleValue", name,
                                         input_port_index=k)
            else:
                model.add_connection(_constant(value, f"{tag}_{name}"), system, "scheduleValue", name)
        if not system.collect_states():
            if type(system)._is_declared:
                system.size_declared_ports()
            probe, slot = _Probe(id=f"{tag}_probe"), 0
            for name, port in system.output.items():
                if isinstance(port, tps.Vector):
                    for k in range(int(port._n_v or 0)):
                        model.add_connection(system, probe, name, "u", output_port_index=k, input_port_index=slot)
                        slot += 1
                else:
                    model.add_connection(system, probe, name, "u", input_port_index=slot)
                    slot += 1

    def run(self, model, system, mode="object", device=None):
        if device is not None:
            model.to(device=device)
        Simulator(model, execution_mode=mode).simulate(
            start_time=START,
            end_time=START + datetime.timedelta(seconds=self.steps * self.step_size),
            step_size=self.step_size,
            show_progress_bar=False,
        )
        return {name: _values(port) for name, port in system.output.items()}

    def same(self, got, expected, what):
        assert got.keys() == expected.keys(), f"{self.name}: {what}: outputs {sorted(got)} != {sorted(expected)}"
        for name in expected:
            try:
                torch.testing.assert_close(got[name], expected[name], rtol=self.rtol, atol=self.atol)
            except AssertionError as error:
                raise AssertionError(f"{self.name}: {what}: output {name!r}: {error}") from None

    # -- the checks -----------------------------------------------------------
    def deepcopy_shares_nothing(self, original, duplicate):
        own = {t.untyped_storage().data_ptr() for t in _tensors(original)}
        shared = own & {t.untyped_storage().data_ptr() for t in _tensors(duplicate)}
        assert not shared, f"{self.name}: a deepcopy shares {len(shared)} tensor(s) with the original"

    def save_and_reload(self, model, system, reference):
        model.serialize()
        path, _ = model.simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
        model_id = f"{model.id}_reloaded"
        self.model_ids.append(model_id)
        reloaded = Model(id=model_id, system_registry=self.registry)
        reloaded.load(filename=path, draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        again = reloaded.components[system.id]
        assert type(again) is type(system), f"{self.name}: reloads as {type(again).__name__}"
        for path_ in system.config.get("parameters", ()):
            before, after = _value(system, path_), _value(again, path_)
            assert _equal(before, after), f"{self.name}: {path_} saves as {before!r} and reloads as {after!r}"
        self.same(self.run(reloaded, again), reference, "the reloaded System steps differently")

    def batched(self, first, second):
        second.id = f"{second.id}_second"
        for path in _parameters(second):
            value = rgetattr(second, path).get().detach()
            changed = torch.where(value == 0, value, value * 1.05)
            bounds = second.parameter.get(path, {}) if isinstance(getattr(second, "parameter", None), dict) else {}
            if bounds.get("ub") is not None and bool((changed > bounds["ub"]).any()):
                changed = value * 0.95
            rgetattr(second, path).set(changed.clone(), normalized=False)
        separate = self.model(first, "separate", others=(second,))
        expected = [self.run(separate, c) for c in (first, second)]
        batched = separate.batch_components()
        self.model_ids.append(batched.id)
        batched.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        Simulator(batched, execution_mode="object").simulate(
            start_time=START,
            end_time=START + datetime.timedelta(seconds=self.steps * self.step_size),
            step_size=self.step_size,
            show_progress_bar=False,
        )
        for component, outputs in zip((first, second), expected):
            meta, i_c = separate._component_to_meta[component.id]
            got = {name: _values(port, i_c) for name, port in meta.output.items()}
            self.same(got, outputs, "two copies batched differ from the copies stepped separately")

    def gradients(self, system):
        paths = _parameters(system)
        if not paths or self.cls._is_data_source():
            return  # a data source's series is known before the run: nothing steps it
        model = self.model(system, "gradients")
        end = START + datetime.timedelta(seconds=self.steps * self.step_size)
        model.initialize([START], [end], [self.step_size])
        simulator = Simulator(model, execution_mode="functional")
        # theta: every value of every parameter (a curve's points, one each)
        spec, values, names = [], [], []
        for path in paths:
            value = rgetattr(system, path).get().detach().reshape(-1).to(tps.float_dtype())
            n = system.values_per_copy(path)
            spec.append((system, path, len(values) if n == 1 else list(range(len(values), len(values) + n))))
            values.extend(value[:n])
            names.extend([path] * n)
        layout, functional = simulator.build_functional_model(
            theta_spec=spec,
            outputs=[(system, name) for name in system.output],
            step_size=[self.step_size],
        )
        tape = simulator.record_exogenous_inputs(functional, [START], [end], [self.step_size], layout=layout)
        theta = torch.stack(values).requires_grad_(True)
        out = simulator.rollout_functional(functional, tape.Y0[0], theta, tape.exogenous_tape[0], transform_mode=True)
        columns = out.reshape(out.shape[0], -1)
        for j in range(columns.shape[1]):
            (grad,) = torch.autograd.grad(columns[:, j].sum(), theta, retain_graph=True, allow_unused=True)
            if grad is not None and not bool(torch.isfinite(grad).all()):
                bad = sorted({p for p, g in zip(names, grad) if not bool(torch.isfinite(g))})
                raise AssertionError(f"{self.name}: output column {j} has a gradient that is not finite in {bad}")

    def cuda(self, system, reference):
        got = self.run(self.model(system, "cuda"), system, device=torch.device("cuda"))
        self.same(got, reference, "the System steps differently on CUDA")

    def clean_up(self):
        for model_id in self.model_ids:
            for suffix in ("", "_simulation_model"):
                shutil.rmtree(os.path.join("generated_files", "models", model_id + suffix), ignore_errors=True)


def _constant(value, sid):
    return ScheduleSystem(
        id=sid,
        weekday_ruleset={
            "ruleset_default_value": float(value),
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
    )


def _values(port, i_c=None):
    history = port.history() if i_c is None else port.history(i_c=i_c)
    return history.detach().cpu().double().reshape(history.shape[0], -1).clone()


def _parameters(system):
    """The paths of the System's parameters that hold a value."""
    return [p for p in getattr(system, "PARAM_NAMES", ()) if isinstance(rgetattr(system, p), tps.Parameter)]


def _tensors(system):
    """The tensors the System holds: its parameters (the stored, normalized
    tensors) and an nn.Module's parameters and buffers."""
    out = [rgetattr(system, p) for p in _parameters(system)]
    if isinstance(system, torch.nn.Module):
        out.extend(system.parameters())
        out.extend(system.buffers())
    return [t for t in out if isinstance(t, torch.Tensor) and t.numel() > 0]


def _value(system, path):
    try:
        value = rgetattr(system, path)
    except AttributeError:
        return AttributeError
    return value.get() if isinstance(value, (tps.Parameter, tps.TensorParameter)) else value


def _equal(a, b) -> bool:
    if isinstance(a, Source) and isinstance(b, Source):
        return type(a) is type(b) and a.signature() == b.signature()
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        a, b = torch.as_tensor(a).detach().cpu().double(), torch.as_tensor(b).detach().cpu().double()
        return a.shape == b.shape and bool(torch.allclose(a, b, rtol=1e-12, atol=0.0, equal_nan=True))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if type(a) is type(b) and hasattr(a, "__dict__") and not isinstance(a, type):
        return _equal(vars(a), vars(b))
    return a == b
