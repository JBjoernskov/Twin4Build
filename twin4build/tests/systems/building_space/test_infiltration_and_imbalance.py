"""Infiltration heat and the room's air-flow imbalance outputs.

1. the thermal model's infiltration is outdoor air through the envelope: one
   step from rest is the exact exponential towards the outdoor temperature;
2. a ``BuildingSpaceSystem`` has one infiltration parameter, ``mass.m_inf``,
   which its thermal model reads on the object path, also after an estimator
   replaced the parameter object;
3. ``surplusAirFlowRate`` / ``makeUpAirFlowRate`` are the max of the
   imbalance either way (hard mode) and close the room's mass balance in the
   smooth mode, fused and unfused;
4. object and functional paths agree, fused and unfused, and the room
   temperature's gradient with respect to ``mass.m_inf`` is the finite
   difference (the thermal model reads the estimated value, not a default).
"""

# Standard library imports
import datetime
import functools
import math
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.utils.smooth_saturation import saturation_mode
from twin4build.utils import constants

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.UTC)
DT = 600.0
HOURS = 4


def _exact(test):
    """The exact max (the hard saturation mode)."""

    @functools.wraps(test)
    def run(self):
        with saturation_mode("hard"):
            return test(self)

    return run


def _t(v):
    return torch.tensor([float(v)], dtype=torch.float64)


def _schedule(value, sid):
    return tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": value}, id=sid)


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


def _room_model(fuse=True, model_id="infiltration", m_inf=0.004, people=0.0, interior=None):
    """A room on two supply branches (0.15 kg/s) and an exhaust of 0.05 kg/s,
    0.25 kg/s from hour 1 to 3 (surplus, then make-up), with a wall to a
    neighbour held at 15 C (the room and the wall fuse).

    ``interior`` (``True`` / ``False``) adds a zone ``Interior`` supplied by
    the room's surplus at the room's temperature, and (``True``) at the
    room's CO2 on ``supplyAirCO2``."""
    model = tb.Model(id=model_id)
    room = tb.BuildingSpaceSystem(
        thermal_kwargs=dict(C_air=2e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0),
        mass_kwargs=dict(V=100.0, G_occ=5e-6, m_inf=m_inf),
        id="Room",
    )
    for port, value in (
        ("outdoorTemperature", 0.0),
        ("supplyAirTemperature", 20.0),
        ("globalIrradiation", 0.0),
        ("numberOfPeople", people),
        ("heatGain", 500.0),
        ("outdoorCO2", 400.0),
    ):
        model.add_connection(_schedule(value, f"S_{port}"), room, "scheduleValue", port)
    if interior is not None:
        zone = tb.BuildingSpaceSystem(
            thermal_kwargs=dict(C_air=5e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0),
            mass_kwargs=dict(V=400.0, G_occ=5e-6, m_inf=0.002),
            id="Interior",
        )
        for port, value in (
            ("outdoorTemperature", 0.0),
            ("globalIrradiation", 0.0),
            ("numberOfPeople", 0.0),
            ("heatGain", 0.0),
            ("outdoorCO2", 400.0),
        ):
            model.add_connection(_schedule(value, f"I_{port}"), zone, "scheduleValue", port)
        model.add_connection(_schedule(0.0, "I_exhaust"), zone, "scheduleValue", "exhaustAirFlowRate", input_port_index=0)
        model.add_connection(room, zone, "surplusAirFlowRate", "supplyAirFlowRate", input_port_index=0)
        model.add_connection(room, zone, "indoorTemperature", "supplyAirTemperature")
        if interior:
            model.add_connection(room, zone, "indoorCO2", "supplyAirCO2")
    model.add_connection(_schedule(0.10, "Supply0"), room, "scheduleValue", "supplyAirFlowRate", input_port_index=0)
    model.add_connection(_schedule(0.05, "Supply1"), room, "scheduleValue", "supplyAirFlowRate", input_port_index=1)
    model.add_connection(
        _window(0.05, 0.25, 1, 3, "Exhaust"), room, "scheduleValue", "exhaustAirFlowRate", input_port_index=0
    )
    wall = tb.WallSystem(C=2e5, R_a=0.02, R_b=0.02, id="Wall")
    model.add_connection(room, wall, "indoorTemperature", "temperatureA")
    model.add_connection(_schedule(15.0, "Neighbour"), wall, "scheduleValue", "temperatureB")
    model.add_connection(wall, room, "heatFlowRateA", "wallHeatGain", input_port_index=0)
    model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=fuse)
    return model, model.components["Room"]


def _simulate(model, **kwargs):
    sim = tb.Simulator(model, **kwargs)
    sim.simulate(
        start_time=START,
        end_time=START + datetime.timedelta(hours=HOURS),
        step_size=int(DT),
        show_progress_bar=False,
    )
    return sim


def _series(port):
    hist = port.history()
    return hist.reshape(hist.shape[0], -1)[:, 0].detach().cpu().clone().double()


def _flows(model):
    """The schedules' supply total and exhaust, per step."""
    supply = sum(_series(model.components[s].output["scheduleValue"]) for s in ("Supply0", "Supply1"))
    return supply, _series(model.components["Exhaust"].output["scheduleValue"])


def _theta_entry(model, room, name):
    """A fused room's parameter lives on its fused block under a prefixed name."""
    fused = model.simulation_model._fusion_member_to_fused.get(room.id)
    if fused is None:
        return (room, name)
    return (fused, f"{fused._member_keys[room.id]}.{name}")


class TestInfiltrationHeat(unittest.TestCase):
    @_exact
    def test_one_step_is_the_exact_exponential_to_outdoor(self):
        m_inf, C_air = 0.01, 1e6
        zone = tb.BuildingSpaceThermalSystem(
            C_air=C_air, C_wall=1e6, R_out=0.05, R_in=1e9, f_wall=0.0, f_air=0.0, Q_occ_gain=0.0, m_inf=m_inf, id="z"
        )
        zone.n_c = 1
        params = {name: getattr(zone, name).get().reshape(-1) for name in zone.PARAM_NAMES}
        inputs = {
            "outdoorTemperature": _t(0.0),
            "supplyAirFlowRate": _t(0.0),
            "exhaustAirFlowRate": _t(0.0),
            "supplyAirTemperature": _t(20.0),
            "globalIrradiation": _t(0.0),
            "numberOfPeople": _t(0.0),
            "heatGain": _t(0.0),
        }
        x0 = torch.tensor([[20.0, 20.0]], dtype=torch.float64)
        x1, out = zone.forward(x0, inputs, params, DT)
        expected = 20.0 * math.exp(-float(constants.CP_AIR) * m_inf / C_air * DT)
        self.assertAlmostEqual(float(x1[0, 0]), expected, places=9)
        self.assertEqual(float(out["surplusAirFlowRate"]), 0.0)
        self.assertEqual(float(out["makeUpAirFlowRate"]), 0.0)
        # without infiltration the decoupled air node holds its temperature
        zone._fwd_mat_cache = None
        x1, _ = zone.forward(x0, inputs, {**params, "m_inf": _t(0.0)}, DT)
        self.assertAlmostEqual(float(x1[0, 0]), 20.0, places=9)


class TestOneInfiltrationParameter(unittest.TestCase):
    def test_the_composite_lists_only_the_co2_models_flow(self):
        room = tb.BuildingSpaceSystem(id="R")
        self.assertIn("mass.m_inf", room.config["parameters"])
        self.assertNotIn("thermal.m_inf", room.config["parameters"])
        self.assertNotIn("thermal.m_inf", room.parameter)
        p = room._resolve_sub_params(room.thermal, "thermal", {"mass.m_inf": _t(0.007)})
        self.assertEqual(float(p["m_inf"]), 0.007)
        # not estimated: the CO2 model's value, not the thermal default
        p = room._resolve_sub_params(room.thermal, "thermal", {})
        self.assertEqual(float(p["m_inf"].reshape(-1)[0]), float(room.mass.m_inf.get().reshape(-1)[0]))

    def test_the_object_path_follows_a_replaced_parameter(self):
        model, room = _room_model(fuse=False, model_id="inf_replaced")
        _simulate(model)
        self.assertIs(room.thermal.m_inf, room.mass.m_inf)
        before = _series(room.output["indoorTemperature"])
        model._write_parameters([0.012], [room], ["mass.m_inf"], overwrite=True)
        _simulate(model)
        self.assertIs(room.thermal.m_inf, room.mass.m_inf)
        after = _series(room.output["indoorTemperature"])
        # more outdoor air at 0 C: a colder room
        self.assertLess(float(after[-1]), float(before[-1]) - 0.1)


class TestImbalanceOutputs(unittest.TestCase):
    @_exact
    def test_the_max_of_the_imbalance_either_way(self):
        for fuse in (False, True):
            with self.subTest(fuse=fuse):
                model, room = _room_model(fuse=fuse, model_id=f"imbalance_hard_{fuse}")
                self.assertEqual(room.id in model.simulation_model._fusion_member_to_fused, fuse)
                _simulate(model)
                supply, exhaust = _flows(model)
                surplus = _series(room.output["surplusAirFlowRate"])
                make_up = _series(room.output["makeUpAirFlowRate"])
                torch.testing.assert_close(surplus, (supply - exhaust).clamp(min=0.0), rtol=0, atol=1e-12)
                torch.testing.assert_close(make_up, (exhaust - supply).clamp(min=0.0), rtol=0, atol=1e-12)
                self.assertGreater(float(surplus.max()), 0.09)
                self.assertGreater(float(make_up.max()), 0.09)

    def test_the_streams_balance_in_the_smooth_mode(self):
        for fuse in (False, True):
            with self.subTest(fuse=fuse):
                model, room = _room_model(fuse=fuse, model_id=f"imbalance_smooth_{fuse}")
                _simulate(model)
                supply, exhaust = _flows(model)
                surplus = _series(room.output["surplusAirFlowRate"])
                make_up = _series(room.output["makeUpAirFlowRate"])
                torch.testing.assert_close(supply + make_up, exhaust + surplus, rtol=0, atol=1e-12)
                self.assertTrue(bool((surplus >= 0).all() and (make_up >= 0).all()))


class TestNeighbourTakesTheSurplus(unittest.TestCase):
    def test_the_surplus_carries_the_rooms_co2(self):
        """Ten people in the room; its surplus supplies the interior.  With
        ``supplyAirCO2`` wired the interior receives the room's CO2, unwired
        outdoor air; object and functional paths agree, fused and unfused."""
        for fuse in (False, True):
            co2 = {}
            for wired in (False, True):
                with self.subTest(fuse=fuse, wired=wired):
                    model, room = _room_model(
                        fuse=fuse, model_id=f"surplus_{fuse}_{wired}", people=10.0, interior=wired
                    )
                    zone = model.components["Interior"]
                    _simulate(model)
                    self.assertEqual(zone.mass._supply_wired, wired)
                    reference = {
                        port: _series(zone.output[port]) for port in ("indoorCO2", "indoorTemperature")
                    }
                    torch.testing.assert_close(
                        _series(zone.input["supplyAirFlowRate"]), _series(room.output["surplusAirFlowRate"])
                    )
                    _simulate(model, execution_mode="functional", execution_backend="eager")
                    for port, series in reference.items():
                        torch.testing.assert_close(_series(zone.output[port]), series, rtol=1e-9, atol=1e-9, msg=port)
                    co2[wired] = reference["indoorCO2"]
            room_co2 = _series(room.output["indoorCO2"])
            self.assertGreater(float(room_co2[-1]), 550.0)
            self.assertGreater(float(co2[True][-1]), float(co2[False][-1]) + 50.0)


class TestPaths(unittest.TestCase):
    def test_functional_equals_object_and_the_gradient_reaches_the_temperature(self):
        for fuse in (False, True):
            with self.subTest(fuse=fuse):
                model, room = _room_model(fuse=fuse, model_id=f"inf_paths_{fuse}")
                ports = ("indoorTemperature", "surplusAirFlowRate", "makeUpAirFlowRate")
                _simulate(model)
                reference = {port: _series(room.output[port]) for port in ports}
                _simulate(model, execution_mode="functional", execution_backend="eager")
                for port in ports:
                    torch.testing.assert_close(_series(room.output[port]), reference[port], rtol=1e-9, atol=1e-9, msg=port)

                if fuse:
                    fused = model.simulation_model._fusion_member_to_fused[room.id]
                    key = fused._member_keys[room.id]
                    self.assertIn(f"{key}.mass.m_inf", fused.PARAM_NAMES)
                    self.assertNotIn(f"{key}.thermal.m_inf", fused.PARAM_NAMES)

                end = START + datetime.timedelta(hours=HOURS)
                model.initialize([START], [end], [int(DT)])
                simulator = tb.Simulator(model, execution_mode="functional")
                layout, functional_model = simulator.build_functional_model(
                    theta_spec=[_theta_entry(model, room, "mass.m_inf")],
                    outputs=[(room, "indoorTemperature")],
                    step_size=[int(DT)],
                )
                recording = simulator.record_exogenous_inputs(
                    functional_model, [START], [end], [int(DT)], layout=layout
                )

                def run(value):
                    return simulator.rollout_functional(
                        functional_model, recording.Y0[0], value, recording.exogenous_tape[0], transform_mode=True
                    )

                m_inf = float(room.mass.m_inf.get().reshape(-1)[0])
                theta = torch.tensor([m_inf], dtype=torch.float64, requires_grad=True)
                out = run(theta)
                torch.testing.assert_close(out[:, 0].detach(), reference["indoorTemperature"], rtol=1e-9, atol=1e-9)
                (grad,) = torch.autograd.grad(out[:, 0].sum(), theta)
                # more 0 C outdoor air: colder
                self.assertLess(float(grad), 0.0)
                h = 1e-3 * m_inf
                with torch.no_grad():
                    plus = run(torch.tensor([m_inf + h], dtype=torch.float64))[:, 0].sum()
                    minus = run(torch.tensor([m_inf - h], dtype=torch.float64))[:, 0].sum()
                torch.testing.assert_close(grad[0], (plus - minus) / (2 * h), rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
