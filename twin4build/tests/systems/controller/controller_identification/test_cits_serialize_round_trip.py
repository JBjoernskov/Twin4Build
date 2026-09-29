"""An identified controller survives ``Model.serialize()`` ->
``Model.load(filename=...)`` (#191).

``ControllerIdentificationPISystem.config`` used to list the sizes and the
candidate's parameters only.  What the rewire derives (the candidate
structure, the selection weights, the gate, the ``onOffSignal``
normalisation, the playback flag) was not serialized, and the candidates
are built once the sizes are known, so the reloaded controller was a bare
default: a radiator loop reloaded with a constant command.

The model here is one radiator loop: the controller's command scales the
heat a radiator gives to a room, and the room temperature is the
controller's feedback.  It is rewired from data (with a decoy feedback
sensor and a decoy setpoint to prune), given "fitted" values, simulated,
serialized and loaded into a fresh model that is simulated WITHOUT another
rewire.
"""

# Standard library imports
import copy
import datetime
import json
import os
import shutil
import tempfile
import unittest
import warnings

# Third party imports
import numpy as np
import pandas as pd
import rdflib
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.building_space.building_space_system import BuildingSpaceSystem
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.controller.controller_identification.controller_identification_system import (
    ControllerIdentificationSystem,
)
from twin4build.systems.controller.setpoint_controller.cascade_controller.cascade_controller_system import (
    CascadeControllerSystem,
)
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.utils.scalar_product_system import ScalarProductSystem
from twin4build.utils.get_main_dir import get_main_dir

tb._IS_TESTING = True

STEP = 600
START = datetime.datetime(2024, 3, 4, 0, 0, tzinfo=tz.UTC)
N_DAYS = 3
# Simulated window: crosses the 06:00 gate transition of the second day.
SIM_START = START + datetime.timedelta(days=1, hours=4)
SIM_END = SIM_START + datetime.timedelta(hours=6)

PID_REF = (
    "twin4build.systems.controller.setpoint_controller.pid_controller."
    "pid_controller_system:PIDControllerSystem"
)
CASCADE_REF = (
    "twin4build.systems.controller.setpoint_controller.cascade_controller."
    "cascade_controller_system:CascadeControllerSystem"
)


def _values(parameter):
    """Physical value(s) of a parameter as a flat list."""
    return parameter.get().detach().cpu().reshape(-1).tolist()


def _write_loop_data(folder):
    """Three days of 10-minute data of a gated reverse-acting heating loop."""
    n = N_DAYS * 144
    index = pd.date_range(START, periods=n, freq="10min")
    t = np.arange(n)
    hour = (t % 144) / 6.0
    day = (hour >= 6.0) & (hour < 18.0)
    fb = 22.0 + 1.5 * np.sin(2 * np.pi * t / 144) + 0.2 * np.sin(2 * np.pi * t / 23)
    sp = np.full(n, 22.0)
    # The flow setpoint is the schedule the BMS gates the valve on.
    gate = np.where(day, 300.0, 0.0)
    u = np.where(day, np.clip(0.5 + 0.4 * (sp - fb), 0.0, 1.0), 0.0)
    series = {
        "zone_t": fb,
        "sp_heat": sp,
        "sp_cool": np.full(n, 25.0),
        "supply_flow": 300.0 + 20.0 * np.sin(2 * np.pi * t / 37),
        "flow_sp": gate,
        "cmd": u,
    }
    paths = {}
    for name, values in series.items():
        paths[name] = os.path.join(folder, f"{name}.csv")
        pd.DataFrame({"time": index, "value": values}).to_csv(paths[name], index=False)
    return paths


def _sensor(id_, paths):
    return SensorSystem(
        id=id_, filename=paths[id_], datecolumn=0, valuecolumn=1, use_spreadsheet=True
    )


def _schedule(id_, value):
    return ScheduleSystem(
        id=id_,
        weekday_ruleset={
            "ruleset_start_minute": [], "ruleset_end_minute": [], "ruleset_start_hour": [],
            "ruleset_end_hour": [], "ruleset_value": [], "ruleset_default_value": value,
        },
    )


def build_radiator_loop(model_id, paths):
    """controller -> radiator -> room -> zone sensor -> controller."""
    model = tb.Model(id=model_id)
    cits = ControllerIdentificationPISystem(id="cits")
    room = BuildingSpaceSystem(
        id="room", C_wall=1e6, C_air=1e4, C_boundary=5e5, R_out=0.01, R_in=0.02, R_boundary=0.03,
        f_wall=0.5, f_air=0.3, Q_occ_gain=100.0, CO2_occ_gain=0.004, CO2_start=400.0, airVolume=100.0,
        T_wall_start=20.0, T_air_start=20.0, T_int_start=20.0, T_boundary_start=18.0,
    )
    # Heat to the room: command x 1 x 1500 W.
    radiator = ScalarProductSystem(id="radiator", scale_factor=1500.0)
    zone_t, sp_heat, sp_cool, supply_flow, flow_sp, cmd = (
        _sensor(k, paths) for k in ("zone_t", "sp_heat", "sp_cool", "supply_flow", "flow_sp", "cmd")
    )
    # The controller as the translator leaves it: every candidate signal wired.
    model.add_connection(zone_t, cits, "measuredValue", "sensorValue", input_port_index=0)
    model.add_connection(supply_flow, cits, "measuredValue", "sensorValue", input_port_index=1)
    model.add_connection(sp_cool, cits, "measuredValue", "setpointValue", input_port_index=0)
    model.add_connection(sp_heat, cits, "measuredValue", "setpointValue", input_port_index=1)
    model.add_connection(flow_sp, cits, "measuredValue", "onOffSignal", input_port_index=0)
    model.add_connection(cits, cmd, "inputSignal", "measuredValue", output_port_index=0)
    model.add_connection(cits, radiator, "inputSignal", "input_1", output_port_index=0)
    # The plant.
    model.add_connection(_schedule("one", 1.0), radiator, "scheduleValue", "input_2")
    model.add_connection(radiator, room, "output", "heatGain")
    model.add_connection(_schedule("t_out", 5.0), room, "scheduleValue", "outdoorTemperature")
    model.add_connection(_schedule("people", 0.0), room, "scheduleValue", "numberOfPeople")
    model.add_connection(_schedule("sun", 0.0), room, "scheduleValue", "globalIrradiation")
    model.add_connection(_schedule("flow", 0.05), room, "scheduleValue", "supplyAirFlowRate")
    model.add_connection(_schedule("t_sup", 18.0), room, "scheduleValue", "supplyAirTemperature")
    model.add_connection(room, zone_t, "indoorTemperature", "measuredValue")
    return model


def rewire(model, mode):
    model.rewire(
        start_time=[START], end_time=[START + datetime.timedelta(days=N_DAYS)], step_size=STEP, mode=mode
    )


def set_fitted_values(model):
    """Values as Stage 1 leaves them: the gains written the way the
    estimator writes them, the others in place."""
    cits = model.components["cits"]
    model.simulation_model.set_parameters(
        [0.8, 2456.3], [cits] * 2, ["candidate_0_0.kp", "candidate_0_0.Ti"], overwrite=True
    )
    model.simulation_model.set_parameters(
        [0.1, 0.9, 0.05, 0.4, 0.7],
        [cits] * 5,
        ["candidate_0_0.output_min", "candidate_0_0.output_max", "default_output_0", "gate_0.threshold", "gate_0.band"],
    )


def simulate(model):
    """Controller output and room temperature over the simulated window."""
    sim = tb.Simulator(model, execution_mode="object")
    sim.simulate(start_time=SIM_START, end_time=SIM_END, step_size=STEP, show_progress_bar=False)
    command = model.components["radiator"].input["input_1"]._history.detach().clone()
    temperature = model.components["room"].output["indoorTemperature"]._history.detach().clone()
    return command, temperature


def controller_state(cits):
    """Everything that defines the controller, as plain Python values."""
    state = {
        "sizes": (cits.n_sensors, cits.n_setpoints, cits.n_on_off_signals, cits.n_actuators),
        "candidate_structure": cits.candidate_structure,
        "candidate_classes": [type(cits.candidate_0_0).__name__],
        "playback": cits.playback,
        "rewire_mode": cits.rewire_mode,
        "norm_min": cits.on_off_signal_norm_min.tolist(),
        "norm_max": cits.on_off_signal_norm_max.tolist(),
        "is_reverse": cits.candidate_0_0.is_reverse,
        "isReverse": cits.candidate_0_0.isReverse,
    }
    for name in (
        "alpha_0", "beta_0", "gamma_0", "gamma_gate_0", "alpha_gate_0", "default_output_0",
        "gate_0.threshold", "gate_0.band", "gate_0.steepness", "gate_0.polarity",
        "candidate_0_0.kp", "candidate_0_0.Ti", "candidate_0_0.Td",
        "candidate_0_0.output_min", "candidate_0_0.output_max",
    ):
        state[name] = _values(_resolve(cits, name))
    for name in ("kp", "Ti", "output_min", "output_max"):
        parameter = getattr(cits.candidate_0_0, name)
        state[f"{name}.bounds"] = (
            parameter.min_value.reshape(-1).tolist(), parameter.max_value.reshape(-1).tolist()
        )
    return state


def wiring(cits):
    """What feeds which slot of the controller: ``{(port, slot): sender id}``."""
    feeds = {}
    for point in cits.connects_at:
        for connection in point.connects_system_through:
            slot = point.input_port_index.get(connection)
            slot = int(slot.item()) if hasattr(slot, "item") else slot
            feeds[(point.input_port, slot)] = connection.connects_system.id
    return feeds


def _resolve(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def instance_graph(model):
    path, _ = model._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
    return path


def remove_model_folder(model_id):
    """Generated files of a model (graphs, parameter files)."""
    shutil.rmtree(os.path.join(get_main_dir(), "generated_files", "models", model_id), ignore_errors=True)


class TestCitsSerializeRoundTrip(unittest.TestCase):
    MODEL_ID = "test_cits_serialize_round_trip"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.paths = _write_loop_data(cls.tmp)
        cls.model_ids = []

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        for model_id in cls.model_ids:
            remove_model_folder(model_id)

    def _model_id(self, suffix):
        model_id = f"{self.MODEL_ID}_{suffix}"
        remove_model_folder(model_id)
        self.model_ids.append(model_id)
        return model_id

    def _identified(self, mode):
        """A rewired model with fitted values, loaded and ready to simulate."""
        model = build_radiator_loop(self._model_id(mode), self.paths)
        rewire(model, mode)
        set_fitted_values(model)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        return model

    def _reload(self, model, suffix):
        model.serialize()
        reloaded = tb.Model(id=self._model_id(suffix))
        reloaded.load(filename=instance_graph(model), draw_semantic_model=False, draw_simulation_model=False)
        return reloaded

    def _assert_state_equal(self, state, reference):
        self.assertEqual(set(state), set(reference))
        for key, expected in reference.items():
            value = state[key]
            if isinstance(expected, list) and expected and isinstance(expected[0], float):
                np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12, err_msg=key)
            elif isinstance(expected, tuple) and isinstance(expected[0], list):
                for v, e in zip(value, expected):
                    np.testing.assert_allclose(v, e, rtol=1e-12, atol=1e-12, err_msg=key)
            else:
                self.assertEqual(value, expected, key)

    def _round_trip(self, mode):
        model = self._identified(mode)
        cits = model.components["cits"]
        reference_state = controller_state(cits)
        command, temperature = simulate(model)

        reloaded = self._reload(model, f"{mode}_reloaded")
        self.assertEqual(set(reloaded.components), set(model.components))
        cits2 = reloaded.components["cits"]
        self.assertIsNot(cits2, cits)
        self.assertIsInstance(cits2, ControllerIdentificationPISystem)
        # Built by the constructor: the structure is in the literals.
        self.assertTrue(cits2._built)
        self._assert_state_equal(controller_state(cits2), reference_state)
        self.assertEqual(
            [(t[1], t[3], t[4]) for t in cits2.get_estimable_parameters()],
            [(t[1], t[3], t[4]) for t in cits.get_estimable_parameters()],
        )

        command2, temperature2 = simulate(reloaded)
        torch.testing.assert_close(command2, command, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(temperature2, temperature, rtol=1e-10, atol=1e-10)
        return model, reloaded, command, temperature

    def test_simulate_mode_round_trip(self):
        model, reloaded, command, temperature = self._round_trip("simulate")
        cits = model.components["cits"]
        # The rewire pruned the decoys, identified a reverse-acting loop with
        # an active gate on the flow setpoint, and the fitted values are in.
        state = controller_state(reloaded.components["cits"])
        self.assertEqual(state["sizes"], (1, 1, 1, 1))
        self.assertEqual(state["candidate_structure"], [{"type": "setpoint", "ref": PID_REF}])
        self.assertEqual(state["rewire_mode"], "simulate")
        self.assertFalse(state["playback"])
        self.assertTrue(state["is_reverse"])
        self.assertTrue(state["isReverse"])
        self.assertEqual(state["alpha_gate_0"], [1.0])
        self.assertEqual(state["norm_min"], [0.0])
        self.assertEqual(state["norm_max"], [300.0])
        self.assertAlmostEqual(state["candidate_0_0.kp"][0], 0.8, places=12)
        self.assertAlmostEqual(state["candidate_0_0.Ti"][0], 2456.3, places=9)
        self.assertAlmostEqual(state["candidate_0_0.output_min"][0], 0.1, places=12)
        self.assertAlmostEqual(state["candidate_0_0.output_max"][0], 0.9, places=12)
        self.assertAlmostEqual(state["default_output_0"][0], 0.05, places=12)
        self.assertEqual(
            wiring(reloaded.components["cits"]),
            {("sensorValue", 0): "zone_t", ("setpointValue", 0): "sp_heat", ("onOffSignal", 0): "flow_sp"},
        )
        self.assertEqual(wiring(reloaded.components["cits"]), wiring(cits))
        # The loop is exercised on both sides of the gate: parked at the
        # default output before 06:00, modulating after.
        command = command.flatten()
        self.assertAlmostEqual(float(command[6]), 0.05, delta=0.01)
        self.assertGreater(float(command.max()), 0.2)
        self.assertGreater(float(temperature.flatten().std()), 1e-3)

    def test_train_mode_round_trip(self):
        _, reloaded, _, _ = self._round_trip("train")
        self.assertEqual(reloaded.components["cits"].rewire_mode, "train")
        self.assertFalse(reloaded.components["cits"].playback)

    def test_playback_mode_round_trip(self):
        model, reloaded, command, _ = self._round_trip("playback")
        cits2 = reloaded.components["cits"]
        self.assertEqual(cits2.rewire_mode, "playback")
        self.assertTrue(cits2.playback)
        self.assertEqual(wiring(cits2), wiring(model.components["cits"]))
        self.assertEqual(wiring(cits2)[("actuatorMeasured", 0)], "cmd")
        # The radiator is driven by the historised command (one step later:
        # the controller runs after the room it reads).
        first = int((SIM_START - START).total_seconds()) // STEP
        n_steps = int((SIM_END - SIM_START).total_seconds()) // STEP
        expected = pd.read_csv(self.paths["cmd"])["value"].to_numpy()[first:first + n_steps]
        self.assertGreater(expected.max(), 0.1)
        np.testing.assert_allclose(command.flatten().numpy()[1:], expected[:-1], atol=1e-12)

    def test_parameter_files_leave_the_controller_as_it_is(self):
        """``load`` keeps a parameter file per component and applies it on
        the next load: every entry of the config has to be writable, and
        writing the controller's own values back changes nothing."""
        model = self._identified("simulate")
        cits = model.components["cits"]
        reference_state = controller_state(cits)
        candidate = cits.candidate_0_0
        command, temperature = simulate(model)
        reloaded = self._reload(model, "files_reloaded")
        for m, kwargs in ((model, {}), (reloaded, {}), (reloaded, {"force_config_overwrite": True})):
            m.load(draw_semantic_model=False, draw_simulation_model=False, **kwargs)
            self._assert_state_equal(controller_state(m.components["cits"]), reference_state)
        self.assertIs(cits.candidate_0_0, candidate)
        command2, temperature2 = simulate(reloaded)
        torch.testing.assert_close(command2, command, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(temperature2, temperature, rtol=1e-10, atol=1e-10)

    def test_reloaded_model_accepts_a_later_rewire(self):
        """A rewire after the reload derives the controller again: the mode
        of the last rewire decides, also whether the loop is open."""
        model = self._identified("playback")
        reloaded = self._reload(model, "rewired_reloaded")
        cits2 = reloaded.components["cits"]
        self.assertTrue(cits2.playback)

        rewire(reloaded, "simulate")
        self.assertEqual(cits2.rewire_mode, "simulate")
        self.assertFalse(cits2.playback)
        self.assertEqual(
            (cits2.n_sensors, cits2.n_setpoints, cits2.n_on_off_signals, cits2.n_actuators), (1, 1, 1, 1)
        )
        # Seeds again, not the fitted values.
        self.assertNotAlmostEqual(_values(cits2.candidate_0_0.Ti)[0], 2456.3, places=3)
        self.assertTrue(cits2.candidate_0_0.is_reverse)
        self.assertEqual(cits2.on_off_signal_norm_max.tolist(), [300.0])
        reloaded.load(draw_semantic_model=False, draw_simulation_model=False)
        command, _ = simulate(reloaded)
        self.assertTrue(bool(torch.isfinite(command).all()))

        rewire(reloaded, "playback")
        self.assertEqual(cits2.rewire_mode, "playback")
        self.assertTrue(cits2.playback)

    def test_model_serialized_before_the_change_still_loads(self):
        """A graph without the new literals (and with the sizes as they used
        to be written) loads to the controller it loaded to before: built
        when it is initialized, from its connections."""
        model = self._identified("simulate")
        model.serialize()
        graph = rdflib.Graph()
        graph.parse(instance_graph(model), format="turtle")
        t4b = core.namespace.T4B
        subject = t4b["cits"]
        old_literals = {"n_sensors", "n_setpoints", "n_actuators"} | {
            f"candidate_0_0.{n}" for n in ("kp", "Ti", "Td", "output_min", "output_max", "is_reverse")
        }
        kept = 0
        for _, predicate, obj in list(graph.triples((subject, None, None))):
            if not isinstance(obj, rdflib.Literal):
                continue
            name = str(predicate)[len(str(t4b)):]
            if name in old_literals:
                kept += 1
                continue
            graph.remove((subject, predicate, obj))
        self.assertEqual(kept, len(old_literals))
        graph.add((subject, t4b["n_on_off_signals"], rdflib.Literal("None")))
        legacy = os.path.join(self.tmp, "legacy_instance_graph.ttl")
        graph.serialize(destination=legacy, format="turtle")

        reloaded = tb.Model(id=self._model_id("legacy_reloaded"))
        reloaded.load(filename=legacy, draw_semantic_model=False, draw_simulation_model=False)
        cits2 = reloaded.components["cits"]
        self.assertFalse(cits2._built)
        self.assertFalse(cits2.playback)
        self.assertIsNone(cits2.rewire_mode)
        command, _ = simulate(reloaded)
        self.assertTrue(cits2._built)
        self.assertEqual(
            (cits2.n_sensors, cits2.n_setpoints, cits2.n_on_off_signals, cits2.n_actuators), (1, 1, 1, 1)
        )
        self.assertTrue(bool(torch.isfinite(command).all()))


class TestCitsStructureLiterals(unittest.TestCase):
    """The candidate structure and the normalisation bounds as literals."""

    def test_generic_candidate_structure_survives(self):
        """A generic controller with its own candidate list reloads with
        that list, not with the class default (two PIDs and a cascade)."""
        model_ids = ["test_cits_structure", "test_cits_structure_reloaded"]
        for model_id in model_ids:
            remove_model_folder(model_id)
        try:
            model = tb.Model(id=model_ids[0])
            cits = ControllerIdentificationSystem(
                id="cits", n_sensors=2, n_setpoints=1, n_on_off_signals=2, n_actuators=1,
                setpoint_controllers=[PIDControllerSystem],
                setpoint_controller_kwargs=[{"kp": 0.4, "Ti": 600.0, "is_reverse": True}],
                cascade_controllers=[CascadeControllerSystem],
                cascade_controller_kwargs=[{"kp_a": 0.2, "Ti_a": 300.0}],
            )
            cits.gamma_gate_0.set(torch.tensor([0.0, 1.0], dtype=tps.float_dtype()), normalized=False)
            cits.beta_0.set(torch.tensor([1.0, 0.0], dtype=tps.float_dtype()), normalized=False)
            cits.on_off_signal_norm_min = [0.0, 15.0]
            cits.on_off_signal_norm_max = [300.0, 25.0]
            for k in range(2):
                model.add_connection(_schedule(f"s{k}", 21.0), cits, "scheduleValue", "sensorValue", input_port_index=k)
                model.add_connection(_schedule(f"o{k}", 1.0), cits, "scheduleValue", "onOffSignal", input_port_index=k)
            model.add_connection(_schedule("sp", 22.0), cits, "scheduleValue", "setpointValue", input_port_index=0)
            model.add_connection(cits, SensorSystem(id="out"), "inputSignal", "measuredValue", output_port_index=0)
            model.load(draw_semantic_model=False, draw_simulation_model=False)
            model.serialize()

            reloaded = tb.Model(id=model_ids[1])
            reloaded.load(filename=instance_graph(model), draw_semantic_model=False, draw_simulation_model=False)
            cits2 = reloaded.components["cits"]
            self.assertEqual(
                cits2.candidate_structure,
                [{"type": "setpoint", "ref": PID_REF}, {"type": "cascade", "ref": CASCADE_REF}],
            )
            self.assertEqual(cits2.n_candidates, 2)
            self.assertIsInstance(cits2.candidate_0_0, PIDControllerSystem)
            self.assertIsInstance(cits2.candidate_0_1, CascadeControllerSystem)
            self.assertFalse(hasattr(cits2, "candidate_0_2"))
            self.assertTrue(cits2.candidate_0_0.is_reverse)
            self.assertAlmostEqual(_values(cits2.candidate_0_0.kp)[0], 0.4, places=12)
            self.assertAlmostEqual(_values(cits2.candidate_0_1.ctrl_a.Ti)[0], 300.0, places=9)
            self.assertEqual(_values(cits2.gamma_gate_0), [0.0, 1.0])
            self.assertEqual(_values(cits2.beta_0), [1.0, 0.0])
            self.assertEqual(_values(cits2.beta_b_0), _values(cits.beta_b_0))
            self.assertEqual(cits2.on_off_signal_norm_min.tolist(), [0.0, 15.0])
            self.assertEqual(cits2.on_off_signal_norm_max.tolist(), [300.0, 25.0])
        finally:
            for model_id in model_ids:
                remove_model_folder(model_id)

    def test_assigning_a_structure_rebuilds_the_candidates(self):
        cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=1, n_on_off_signals=1)
        first = cits.candidate_0_0
        # The same structure (a one-element list reads back unwrapped) is a no-op.
        cits.candidate_structure = {"type": "setpoint", "ref": PID_REF}
        self.assertIs(cits.candidate_0_0, first)
        cits.candidate_structure = [
            {"type": "setpoint", "ref": PID_REF},
            {"type": "cascade", "ref": CASCADE_REF},
        ]
        self.assertEqual(cits.n_candidates, 2)
        self.assertIsNot(cits.candidate_0_0, first)
        self.assertIsInstance(cits.candidate_0_1, CascadeControllerSystem)
        self.assertEqual(_values(cits.alpha_0), [0.5, 0.5])
        self.assertIn("beta_b_0", cits.config["parameters"])

    def test_candidate_class_without_an_import_path(self):
        """A class that cannot be imported by name is not serialized (with a
        warning); its entry stands for the candidate declared at that
        position."""

        class LocalPID(PIDControllerSystem):
            pass

        sizes = dict(n_sensors=1, n_setpoints=1, n_on_off_signals=1)
        cits = ControllerIdentificationSystem(id="cits", setpoint_controllers=[LocalPID], **sizes)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            structure = cits.candidate_structure
        self.assertEqual(structure, [{"type": "setpoint", "ref": None}])
        self.assertTrue(any("not importable" in str(w.message) for w in caught))
        rebuilt = ControllerIdentificationSystem(
            id="cits", setpoint_controllers=[LocalPID], candidate_structure=structure, **sizes
        )
        self.assertIsInstance(rebuilt.candidate_0_0, LocalPID)
        with self.assertRaises(ValueError):
            ControllerIdentificationSystem(
                id="cits", setpoint_controllers=[LocalPID], candidate_structure=structure * 2, **sizes
            )

    def test_config_is_json_and_bounds_come_before_values(self):
        cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=1, n_on_off_signals=2)
        self.assertNotIn("rewire_mode", cits.config["parameters"])
        cits.rewire_mode = "train"
        names = cits.config["parameters"]
        self.assertIn("rewire_mode", names)
        self.assertLess(names.index("candidate_0_0.kp.max_value"), names.index("candidate_0_0.kp"))
        populated = cits.populate_config()["parameters"]
        self.assertEqual(json.loads(json.dumps(populated)), populated)
        self.assertEqual(populated["on_off_signal_norm_max"], [1.0, 1.0])
        # A number stands for every slot; the bounds follow a copy.
        cits.on_off_signal_norm_max = 300.0
        self.assertEqual(copy.deepcopy(cits).on_off_signal_norm_max.tolist(), [300.0, 300.0])
        self.assertEqual(populated["candidate_0_0.kp.min_value"], [0.001])
        # Not built: the structure only.
        unbuilt = ControllerIdentificationPISystem(id="unbuilt")
        self.assertEqual(
            unbuilt.config["parameters"],
            ["n_sensors", "n_setpoints", "n_on_off_signals", "n_actuators", "candidate_structure", "playback"],
        )
        self.assertFalse(hasattr(unbuilt, "on_off_signal_norm_min"))


if __name__ == "__main__":
    unittest.main()
