"""The two powers of ``SpaceHeaterSystem``, and the deprecated ``Power``.

``toRoomPower`` is the heat the radiator gives to the room,
``UA/n * sum_i (T_i - T_zone)``; ``toRadiatorPower`` is the heat it takes
from the heating circuit, ``waterFlowRate * c_p * (supplyWaterTemperature -
outletWaterTemperature)``.  They differ by the rate of change of the heat
stored in the radiator.

Covered:

1. steady state: the two powers are equal;
2. energy balance over a transient: the integral of their difference is the
   change of the stored heat, within the error of the discrete sum;
3. zero flow: ``toRadiatorPower`` is zero while ``toRoomPower`` decays;
4. the deprecated ``Power`` connects, reads, is an objective and a pattern
   output, warns, and the model holds ``toRoomPower``;
5. a model saved with ``Power`` loads, simulates as before and saves
   ``toRoomPower``;
6. fused against unfused;
7. batched against unbatched, two radiators with different parameters;
8. the functional rollout against the object path, and the gradient of the
   summed water-side power with respect to UA through the rollout;
9. a ``WeightedSumSystem`` over ``toRadiatorPower`` feeding a
   ``SensorSystem``, fused, batched and as a measurement with a gradient.
"""

# Standard library imports
import datetime
import os
import re
import shutil
import tempfile
import unittest
import warnings

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.examples.utils as example_utils
from twin4build.examples import patterns as example_patterns
from twin4build.optimizer.optimizer import _resolve_port_names
from twin4build.utils import constants

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.UTC)
CP = float(constants.CP_WATER)
FLOW = 0.02  # kg/s
T_SUPPLY = 60.0
T_ROOM = 21.0


def _schedule(value, sid):
    return tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": value,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        id=sid,
    )


def _window(base, value, start_hour, end_hour, sid):
    """``value`` from ``start_hour`` to ``end_hour``, ``base`` otherwise."""
    return tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": base,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [start_hour],
            "ruleset_end_hour": [end_hour],
            "ruleset_value": [value],
        },
        id=sid,
    )


def _radiator(sid, C=3e5, UA=60.0):
    return tb.SpaceHeaterSystem(thermalMassHeatCapacity=C, UA=UA, nelements=3, id=sid)


def _simulate(model, hours, step, **kwargs):
    sim = tb.Simulator(model, **kwargs)
    sim.simulate(
        start_time=START,
        end_time=START + datetime.timedelta(hours=hours),
        step_size=step,
        show_progress_bar=False,
    )
    return sim


def _series(port, i_c=0):
    hist = port.history()
    return hist.reshape(hist.shape[0], -1)[:, i_c].detach().cpu().clone()


def _water_side(radiator, i_c=0):
    """``waterFlowRate * c_p * (supply - outlet)`` from the radiator's ports."""
    return (
        _series(radiator.input["waterFlowRate"], i_c)
        * CP
        * (
            _series(radiator.input["supplyWaterTemperature"], i_c)
            - _series(radiator.output["outletWaterTemperature"], i_c)
        )
    )


def _standalone(flow, model_id):
    """One radiator in a room held at ``T_ROOM`` by a schedule."""
    model = tb.Model(id=model_id)
    radiator = _radiator("Rad")
    model.add_connection(_schedule(T_SUPPLY, "Tsup"), radiator, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(flow, radiator, "scheduleValue", "waterFlowRate")
    model.add_connection(_schedule(T_ROOM, "Troom"), radiator, "scheduleValue", "indoorTemperature")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, radiator


def _zone(sid):
    return tb.BuildingSpaceThermalSystem(
        C_air=1e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0, id=sid
    )


def _rooms(n, fuse=True, model_id="rooms", total=False, room_port="toRoomPower"):
    """``n`` zones with a radiator each (different parameters); the valves
    open at hour 1.  With ``total`` a ``WeightedSumSystem`` sums the
    radiators' ``toRadiatorPower`` into a ``SensorSystem`` (a heat meter on
    the circuit)."""
    model = tb.Model(id=model_id)
    outdoor, zero, supply_air = _schedule(5.0, "Outdoor"), _schedule(0.0, "Zero"), _schedule(20.0, "SupplyAirTemp")
    water = _schedule(T_SUPPLY, "WaterTemp")
    flow = _window(0.0, FLOW, 1, 23, "WaterFlow")
    total_sum = tb.WeightedSumSystem(id="CircuitSum") if total else None
    for k in range(n):
        zone = _zone(f"Zone{k}")
        for port, src in (
            ("outdoorTemperature", outdoor), ("supplyAirFlowRate", zero), ("exhaustAirFlowRate", zero),
            ("supplyAirTemperature", supply_air), ("globalIrradiation", zero), ("numberOfPeople", zero),
        ):
            model.add_connection(src, zone, "scheduleValue", port)
        radiator = _radiator(f"Radiator{k}", C=2e5 * (1 + 0.5 * k), UA=50.0 + 15.0 * k)
        model.add_connection(water, radiator, "scheduleValue", "supplyWaterTemperature")
        model.add_connection(flow, radiator, "scheduleValue", "waterFlowRate")
        model.add_connection(zone, radiator, "indoorTemperature", "indoorTemperature")
        model.add_connection(radiator, zone, room_port, "heatGain")
        if total:
            model.add_connection(radiator, total_sum, "toRadiatorPower", "inputs", input_port_index=k)
    if total:
        model.add_connection(total_sum, tb.SensorSystem(id="HeatMeter"), "value", "measuredValue")
    model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=fuse)
    return model


def _batched(model, fuse=True):
    batched = model.batch_components()
    batched.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=fuse)
    return batched


def _instance_series(model, cid, port, batched=None):
    if batched is None:
        return _series(model.components[cid].output[port])
    meta, i_c = model._component_to_meta[cid]
    return _series(meta.output[port], i_c)


def _theta_entry(model, radiator, name="UA"):
    """``(component, parameter)`` of the functional map: a fused radiator's
    parameter lives on its fused block under a prefixed name."""
    fused = model.simulation_model._fusion_member_to_fused.get(radiator.id)
    if fused is None:
        return (radiator, name)
    return (fused, f"{fused._member_keys[radiator.id]}.{name}")


def _rollout(model, step, hours, theta_spec, outputs=None, measurements=None):
    end = START + datetime.timedelta(hours=hours)
    model.initialize([START], [end], [step])
    simulator = tb.Simulator(model, execution_mode="functional")
    layout, functional_model = simulator.build_functional_model(
        theta_spec=theta_spec, outputs=outputs, measurements=measurements, step_size=[step]
    )
    recording = simulator.record_exogenous_inputs(functional_model, [START], [end], [step], layout=layout)

    def run(value):
        return simulator.rollout_functional(
            functional_model, recording.Y0[0], value, recording.exogenous_tape[0], transform_mode=True
        )

    return run


class TestSteadyState(unittest.TestCase):
    def test_powers_are_equal_in_steady_state(self):
        model, radiator = _standalone(_schedule(FLOW, "Flow"), "powers_steady")
        _simulate(model, hours=12, step=600)
        to_room = _series(radiator.output["toRoomPower"])
        to_radiator = _series(radiator.output["toRadiatorPower"])
        self.assertGreater(float(to_room[-1]), 100.0)
        # The slowest time constant is ~1e3 s: after 12 h the stored heat no
        # longer changes, so the two powers agree to rounding.
        torch.testing.assert_close(to_radiator[-1], to_room[-1], rtol=1e-9, atol=1e-6)
        torch.testing.assert_close(to_radiator, _water_side(radiator), rtol=1e-12, atol=1e-9)


class TestEnergyBalance(unittest.TestCase):
    def test_integrated_difference_is_the_change_of_stored_heat(self):
        """The valve opens at hour 1 on a radiator at room temperature.

        Summing the element equations gives ``d/dt sum_i C_i T_i =
        toRadiatorPower - toRoomPower =: f``.  The discrete model is exact
        for inputs held over a step, so the change of stored heat is the
        exact integral of ``f``; the outputs are ``f`` at the end of each
        step, so ``dt * sum_k f_k`` is its right Riemann sum.  After the
        valve opens every element warms monotonically (a cascade of
        first-order elements driven from equilibrium), so ``f`` decreases
        monotonically and the right sum underestimates the integral by at
        most ``dt * (f(0+) - f(end))``: the tolerance, 60 s times the drop
        of ``f``, about 2 % of the stored heat here.  Before the valve opens
        both powers are zero (to rounding).
        """
        step = 60.0
        model, radiator = _standalone(_window(0.0, FLOW, 1, 23, "Flow"), "powers_energy")
        _simulate(model, hours=4, step=int(step))
        f = (_series(radiator.output["toRadiatorPower"]) - _series(radiator.output["toRoomPower"])).double()
        closed = _series(radiator.input["waterFlowRate"]) == 0.0
        self.assertTrue(bool(closed.any()) and bool((~closed).any()))
        self.assertLess(float(f[closed].abs().max()), 1e-6)  # rounding only

        n = radiator.nelements
        C_i = float(radiator.thermalMassHeatCapacity.get().reshape(-1)[0]) / n
        x_end = radiator.get_state().detach().reshape(-1).double()
        stored = C_i * float((x_end - T_ROOM).sum())  # the start is T_ROOM everywhere
        integral = step * float(f.sum())

        f_open = FLOW * CP * (T_SUPPLY - T_ROOM)  # f(0+): all elements at room temperature
        bound = step * (f_open - float(f[-1]))
        self.assertGreater(stored, 1e6)
        self.assertLessEqual(integral - stored, 1e-6 * stored)
        self.assertGreaterEqual(integral - stored, -bound - 1e-6 * stored)
        self.assertLess(abs(integral - stored) / stored, 0.03)


class TestZeroFlow(unittest.TestCase):
    def test_no_heat_from_the_circuit_while_the_radiator_cools(self):
        model, radiator = _standalone(_window(0.0, FLOW, 0, 3, "Flow"), "powers_zero_flow")
        _simulate(model, hours=8, step=600)
        closed = _series(radiator.input["waterFlowRate"]) == 0.0
        self.assertGreater(int(closed.sum()), 10)
        to_radiator = _series(radiator.output["toRadiatorPower"])[closed]
        to_room = _series(radiator.output["toRoomPower"])[closed]
        self.assertTrue(bool((to_radiator == 0.0).all()))
        self.assertTrue(bool((to_room > 0.0).all()))
        self.assertTrue(bool((to_room[1:] < to_room[:-1]).all()))
        self.assertLess(float(to_room[-1]), 0.5 * float(to_room[0]))


class TestDeprecatedPower(unittest.TestCase):
    def test_connection_through_power_holds_to_room_power(self):
        model = tb.Model(id="powers_alias")
        radiator, zone = _radiator("Rad"), _zone("Zone")
        with self.assertWarnsRegex(DeprecationWarning, "SpaceHeaterSystem.toRoomPower") as caught:
            model.add_connection(radiator, zone, "Power", "heatGain")
        self.assertEqual(caught.filename, __file__)  # points at the caller
        ports = [c.output_port for c in radiator.connected_through]
        self.assertEqual(ports, ["toRoomPower"])
        (point,) = zone.connects_at
        self.assertEqual([c.output_port for c in point.connects_system_through], ["toRoomPower"])
        with self.assertWarns(DeprecationWarning):
            model.remove_connection(radiator, zone, "Power", "heatGain")
        self.assertEqual(radiator.connected_through, [])

    def test_output_mapping_answers_to_power(self):
        radiator = _radiator("Rad")
        with self.assertWarnsRegex(DeprecationWarning, "toRoomPower") as caught:
            port = radiator.output["Power"]
        self.assertEqual(caught.filename, __file__)
        self.assertIs(port, radiator.output["toRoomPower"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIn("Power", radiator.output)  # no warning
            self.assertNotIn("Power", list(radiator.output))
            self.assertEqual(
                set(radiator.output), {"outletWaterTemperature", "toRoomPower", "toRadiatorPower"}
            )
        with self.assertWarns(DeprecationWarning):
            self.assertIs(radiator.output.get("Power"), port)
        self.assertIsNone(radiator.output.get("NoSuchPort"))
        with self.assertRaises(KeyError):
            radiator.output["NoSuchPort"]
        # Only the radiator renamed ``Power``: a fan's is its own.
        fan = tb.FanSystem(id="Fan")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(fan.resolve_output_port("Power"), "Power")
            self.assertIn("Power", fan.output)

    def test_optimizer_tuples_outputs_and_history_through_power(self):
        with self.assertWarnsRegex(DeprecationWarning, "toRoomPower"):
            model = _rooms(1, model_id="powers_alias_rooms", room_port="Power")
        radiator = model.components["Radiator0"]
        self.assertEqual([c.output_port for c in radiator.connected_through], ["toRoomPower"])
        self.assertEqual(len(model.simulation_model._fused_components), 1)
        with self.assertWarnsRegex(DeprecationWarning, "toRoomPower"):
            entries = _resolve_port_names([(radiator, "Power", "min"), [radiator, "toRadiatorPower", "max"]])
        self.assertEqual(entries, [(radiator, "toRoomPower", "min"), (radiator, "toRadiatorPower", "max")])
        _simulate(model, hours=3, step=600)
        with self.assertWarns(DeprecationWarning):
            old = radiator.output["Power"].history()
        reference = _series(radiator.output["toRoomPower"])
        torch.testing.assert_close(old, radiator.output["toRoomPower"].history())
        # A functional map asked for the fused radiator's ``Power``.
        with self.assertWarnsRegex(DeprecationWarning, "toRoomPower"):
            run = _rollout(model, 600, 3, [], outputs=[(radiator, "Power")])
        out = run(torch.zeros(0, dtype=torch.float64))
        torch.testing.assert_close(out[:, 0], reference.double(), rtol=1e-9, atol=1e-7)

    def test_pattern_written_with_power_connects_to_room_power(self):
        """A signature pattern that reads the radiator's ``Power`` (an add-on
        package's pattern) still translates, into the same connections."""
        filename = example_utils.get_path(["estimator_example", "one_room_example_model.xlsm"])

        def connections(model):
            ids = lambda c: re.sub(r"\[N[0-9a-f]{32}\]", "[bnode]", c.id)
            return sorted(
                (ids(c), connection.output_port, ids(point.connection_point_of), point.input_port)
                for c in model.components.values()
                for connection in c.connected_through
                for point in connection.connects_system_at
            )

        current = tb.Translator().translate(
            core.SemanticModel(rdf_file=filename, id="powers_pattern_new"),
            patterns=example_patterns.default_patterns(),
            id="powers_pattern_new",
        )
        old_patterns = example_patterns.default_patterns()
        renamed = 0
        for sp in old_patterns:
            if sp.system is tb.BuildingSpaceSystem and "heatGain" in sp.inputs:
                node, ports, out_index, in_index = sp.inputs["heatGain"]
                sp.inputs["heatGain"] = (node, {c: "Power" for c in ports}, out_index, in_index)
                renamed += 1
        self.assertGreater(renamed, 0)
        with self.assertWarnsRegex(DeprecationWarning, "SpaceHeaterSystem.toRoomPower"):
            old = tb.Translator().translate(
                core.SemanticModel(rdf_file=filename, id="powers_pattern_old"),
                patterns=old_patterns,
                id="powers_pattern_old",
            )
        self.assertIn("toRoomPower", {c[1] for c in connections(old)})
        self.assertNotIn("Power", {c[1] for c in connections(old) if "heater" in c[0]})
        self.assertEqual(connections(old), connections(current))


class TestModelSavedWithPower(unittest.TestCase):
    MODEL_ID = "powers_saved"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for suffix in ("", "_old", "_resaved"):
            shutil.rmtree(os.path.join("generated_files", "models", self.MODEL_ID + suffix), ignore_errors=True)

    def _build(self):
        model = tb.Model(id=self.MODEL_ID)
        room = tb.BuildingSpaceSystem(id="room", C_wall=1e6, C_air=1e5, R_out=0.01, R_in=0.02, airVolume=100.0)
        heater = _radiator("heater", C=2e5, UA=45.0)
        for port, value in (
            ("outdoorTemperature", 5.0), ("numberOfPeople", 0.0), ("globalIrradiation", 0.0),
            ("supplyAirFlowRate", 0.0), ("supplyAirTemperature", 18.0),
        ):
            model.add_connection(_schedule(value, f"s_{port}"), room, "scheduleValue", port)
        model.add_connection(_schedule(FLOW, "water"), heater, "scheduleValue", "waterFlowRate")
        model.add_connection(_schedule(T_SUPPLY, "t_water"), heater, "scheduleValue", "supplyWaterTemperature")
        model.add_connection(room, heater, "indoorTemperature", "indoorTemperature")
        model.add_connection(heater, room, "toRoomPower", "heatGain")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        return model

    @staticmethod
    def _run(model):
        _simulate(model, hours=3, step=600, execution_mode="object")
        return {
            port: _series(model.components[cid].output[port])
            for cid, port in (("room", "indoorTemperature"), ("heater", "toRoomPower"), ("heater", "toRadiatorPower"))
        }

    @staticmethod
    def _output_ports(path):
        with open(path, encoding="utf-8") as handle:
            return re.findall(r't4b:output_port "([^"]+)"', handle.read())

    def test_old_file_loads_simulates_and_resaves_the_new_name(self):
        model = self._build()
        reference = self._run(model)
        model.serialize()
        path, _ = model._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
        self.assertIn("toRoomPower", self._output_ports(path))
        self.assertNotIn("toRadiatorPower", self._output_ports(path))  # nothing reads it here

        # A file written before the rename records the radiator's ``Power``.
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        old_text = text.replace('t4b:output_port "toRoomPower"', 't4b:output_port "Power"')
        self.assertNotEqual(old_text, text)
        old_path = os.path.join(self.tmp, "instance_graph.ttl")
        with open(old_path, "w", encoding="utf-8") as handle:
            handle.write(old_text)

        loaded = tb.Model(id=self.MODEL_ID + "_old")
        with self.assertWarnsRegex(DeprecationWarning, "SpaceHeaterSystem.toRoomPower"):
            loaded.load(filename=old_path, draw_semantic_model=False, draw_simulation_model=False)
        heater = loaded.components["heater"]
        self.assertEqual([c.output_port for c in heater.connected_through], ["toRoomPower"])
        self.assertEqual(len(loaded.simulation_model._fused_components), 1)
        result = self._run(loaded)
        for port, series in reference.items():
            torch.testing.assert_close(result[port], series, msg=port)

        loaded.serialize()
        resaved, _ = loaded._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
        ports = self._output_ports(resaved)
        self.assertIn("toRoomPower", ports)
        self.assertNotIn("Power", ports)


class TestFusedAgainstUnfused(unittest.TestCase):
    def test_both_powers_agree(self):
        """Tolerance as in ``test_fusion.TestFusionConsistency``: 0.05 K at
        30-second steps over 6 hours (the unfused path lags the zone by one
        step), carried to the powers through their gains: ``m c_p`` for
        ``toRadiatorPower`` and ``UA`` for ``toRoomPower``."""
        fused = _rooms(1, fuse=True, model_id="powers_fused")
        unfused = _rooms(1, fuse=False, model_id="powers_unfused")
        self.assertEqual(len(fused.simulation_model._fused_components), 1)
        self.assertEqual(len(unfused.simulation_model._fused_components), 0)
        _simulate(fused, hours=6, step=30)
        _simulate(unfused, hours=6, step=30)
        radiator = fused.components["Radiator0"]
        ua = float(radiator.UA.get().reshape(-1)[0])
        for cid, port, tolerance in (
            ("Zone0", "indoorTemperature", 0.05),
            ("Radiator0", "outletWaterTemperature", 0.05),
            ("Radiator0", "toRadiatorPower", 0.05 * FLOW * CP),
            ("Radiator0", "toRoomPower", 0.05 * ua),
        ):
            a = _instance_series(fused, cid, port)
            b = _instance_series(unfused, cid, port)
            err = float((a - b).abs().max())
            self.assertLess(err, tolerance, f"{cid}.{port}: fused vs unfused max err {err}")
        to_radiator = _series(radiator.output["toRadiatorPower"])
        self.assertGreater(float(to_radiator.max()), 1000.0)
        # Within the fused block the water side is exactly the formula of
        # its own ports.
        torch.testing.assert_close(to_radiator, _water_side(radiator), rtol=1e-12, atol=1e-9)


class TestBatchedAgainstUnbatched(unittest.TestCase):
    def test_both_powers_agree_per_instance(self):
        for fuse in (True, False):
            with self.subTest(fuse=fuse):
                reference = _rooms(2, fuse=fuse, model_id=f"powers_ref_{fuse}")
                _simulate(reference, hours=4, step=600)
                model = _rooms(2, fuse=fuse, model_id=f"powers_src_{fuse}")
                batched = _batched(model, fuse=fuse)
                rad_meta, _ = model._component_to_meta["Radiator0"]
                self.assertIs(model._component_to_meta["Radiator1"][0], rad_meta)
                self.assertEqual(rad_meta.n_c, 2)
                self.assertEqual(len(batched.simulation_model._fused_components), 1 if fuse else 0)
                for mode in ("object", "functional"):
                    kwargs = {"execution_mode": mode}
                    if mode == "functional":
                        kwargs["execution_backend"] = "eager"
                    _simulate(batched, hours=4, step=600, **kwargs)
                    for k in range(2):
                        for port in ("toRoomPower", "toRadiatorPower"):
                            torch.testing.assert_close(
                                _instance_series(model, f"Radiator{k}", port, batched),
                                _instance_series(reference, f"Radiator{k}", port),
                                msg=f"{mode} Radiator{k}.{port}",
                            )
                # the two instances differ, so the comparison is per instance
                self.assertGreater(
                    float(
                        (
                            _instance_series(reference, "Radiator0", "toRadiatorPower")
                            - _instance_series(reference, "Radiator1", "toRadiatorPower")
                        ).abs().max()
                    ),
                    10.0,
                )


class TestFunctionalPath(unittest.TestCase):
    def test_functional_equals_object_and_the_gradient_flows(self):
        for fuse in (True, False):
            with self.subTest(fuse=fuse):
                model = _rooms(1, fuse=fuse, model_id=f"powers_functional_{fuse}")
                radiator = model.components["Radiator0"]
                _simulate(model, hours=4, step=600)
                reference = {port: _series(radiator.output[port]) for port in ("toRoomPower", "toRadiatorPower")}
                _simulate(model, hours=4, step=600, execution_mode="functional", execution_backend="eager")
                for port, series in reference.items():
                    torch.testing.assert_close(_series(radiator.output[port]), series, rtol=1e-9, atol=1e-7, msg=port)

                ua = float(radiator.UA.get().reshape(-1)[0])
                run = _rollout(
                    model, 600, 4, [_theta_entry(model, radiator)],
                    outputs=[(radiator, "toRadiatorPower"), (radiator, "toRoomPower")],
                )
                theta = torch.tensor([ua], dtype=torch.float64, requires_grad=True)
                out = run(theta)
                torch.testing.assert_close(out[:, 0].detach(), reference["toRadiatorPower"].double(), rtol=1e-9, atol=1e-7)
                torch.testing.assert_close(out[:, 1].detach(), reference["toRoomPower"].double(), rtol=1e-9, atol=1e-7)
                (grad,) = torch.autograd.grad(out[:, 0].sum(), theta)
                self.assertTrue(bool(torch.isfinite(grad).all()))
                self.assertNotEqual(float(grad), 0.0)
                h = 1e-4 * ua
                with torch.no_grad():
                    plus = run(torch.tensor([ua + h], dtype=torch.float64))[:, 0].sum()
                    minus = run(torch.tensor([ua - h], dtype=torch.float64))[:, 0].sum()
                torch.testing.assert_close(grad[0], (plus - minus) / (2 * h), rtol=1e-5, atol=1e-8)


class TestHeatMeterSum(unittest.TestCase):
    def test_summed_water_side_power_feeds_a_sensor(self):
        model = _rooms(2, fuse=True, model_id="powers_meter", total=True)
        self.assertEqual(len(model.simulation_model._fused_components), 2)
        meter = model.components["HeatMeter"]
        radiators = [model.components[f"Radiator{k}"] for k in range(2)]
        for mode in ("object", "functional"):
            kwargs = {"execution_mode": mode}
            if mode == "functional":
                kwargs["execution_backend"] = "eager"
            _simulate(model, hours=4, step=600, **kwargs)
            expected = sum(_water_side(r) for r in radiators)
            self.assertGreater(float(expected.max()), 1000.0)
            torch.testing.assert_close(_series(meter.output["measuredValue"]), expected, rtol=1e-12, atol=1e-8, msg=mode)
        reference = _series(meter.output["measuredValue"])

        batched_source = _rooms(2, fuse=True, model_id="powers_meter_src", total=True)
        batched = _batched(batched_source, fuse=True)
        self.assertEqual(len(batched.simulation_model._fused_components), 1)
        for mode in ("object", "functional"):
            kwargs = {"execution_mode": mode}
            if mode == "functional":
                kwargs["execution_backend"] = "eager"
            _simulate(batched, hours=4, step=600, **kwargs)
            torch.testing.assert_close(
                _series(batched.components["HeatMeter"].output["measuredValue"]), reference, msg=f"batched {mode}"
            )

        # The sum as a measurement of an estimation: d(meter)/d(UA_0).
        ua = float(radiators[0].UA.get().reshape(-1)[0])
        run = _rollout(model, 600, 4, [_theta_entry(model, radiators[0])], measurements=[meter])
        theta = torch.tensor([ua], dtype=torch.float64, requires_grad=True)
        out = run(theta)
        torch.testing.assert_close(out[:, 0].detach(), reference.double(), rtol=1e-9, atol=1e-7)
        (grad,) = torch.autograd.grad(out.sum(), theta)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertNotEqual(float(grad), 0.0)


if __name__ == "__main__":
    unittest.main()
