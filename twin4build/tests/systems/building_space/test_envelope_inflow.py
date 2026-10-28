"""The envelope's inflow (``infiltrationAirFlowRate``, 28 Oct 2026): air an envelope-leakage component lets into a
room at outdoor state, counted as supply in the room's air balance and entering with the make-up stream.

1. the transform: the deficit the inflow leaves plus the inflow on the make-up slot, at their mixed state; a zero
   inflow leaves the inputs bit for bit;
2. without a deficit, an inflow ``f`` is exactly the constant infiltration ``m_inf = f`` (outdoor air in, room air
   out), fused and unfused;
3. in a deficit the inflow replaces make-up air instead of adding to it (sub-additive), and the make-up output
   drops by it;
4. with a wired make-up temperature (a hallway) the inflow mixes in: the same as one make-up stream at the
   flow-weighted temperature;
5. object and functional paths agree with an inflow, fused and unfused (smooth mode).
"""
import datetime
import functools
import unittest

import torch
from dateutil import tz

import twin4build as tb
from twin4build.systems.building_space import air_balance
from twin4build.systems.utils.smooth_saturation import saturation_mode

tb._IS_TESTING = True
START = datetime.datetime(2024, 1, 4, tzinfo=tz.UTC)
DT = 600
HOURS = 4


def _exact(test):
    @functools.wraps(test)
    def run(self):
        with saturation_mode("hard"):
            return test(self)
    return run


def _t(v):
    return torch.tensor([float(v)], dtype=torch.float64)


def _schedule(value, sid):
    return tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": value}, id=sid)


def _model(fuse=True, m_inf=0.0, inflow=None, supply=0.15, exhaust=0.05, make_up_temperature=None, model_id="inflow"):
    """A room (fused to a wall to a neighbour at 15 C) on ``supply`` and ``exhaust`` [kg/s], outdoor air at 0 C and
    400 ppm, ``m_inf`` [kg/s], an envelope ``inflow`` [kg/s] when given, and a wired make-up temperature when given."""
    model = tb.Model(id=model_id)
    room = tb.BuildingSpaceSystem(
        thermal_kwargs=dict(C_air=2e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0),
        mass_kwargs=dict(V=100.0, G_occ=5e-6, m_inf=m_inf),
        id="Room",
    )
    for port, value in (("outdoorTemperature", 0.0), ("supplyAirTemperature", 20.0), ("globalIrradiation", 0.0),
                        ("numberOfPeople", 2.0), ("heatGain", 500.0), ("outdoorCO2", 400.0)):
        model.add_connection(_schedule(value, f"S_{port}"), room, "scheduleValue", port)
    model.add_connection(_schedule(supply, "Supply"), room, "scheduleValue", "supplyAirFlowRate", input_port_index=0)
    model.add_connection(_schedule(exhaust, "Exhaust"), room, "scheduleValue", "exhaustAirFlowRate", input_port_index=0)
    if inflow is not None:
        model.add_connection(_schedule(inflow, "Inflow"), room, "scheduleValue", "infiltrationAirFlowRate")
    if make_up_temperature is not None:
        model.add_connection(_schedule(make_up_temperature, "MakeUpT"), room, "scheduleValue", "makeUpAirTemperature")
    wall = tb.WallSystem(C=2e5, R_a=0.02, R_b=0.02, id="Wall")
    model.add_connection(room, wall, "indoorTemperature", "temperatureA")
    model.add_connection(_schedule(15.0, "Neighbour"), wall, "scheduleValue", "temperatureB")
    model.add_connection(wall, room, "heatFlowRateA", "wallHeatGain", input_port_index=0)
    model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=fuse)
    return model


def _run(execution_mode=None, execution_backend=None, **kwargs):
    model = _model(**kwargs)
    sim_kwargs = {}
    if execution_mode is not None:
        sim_kwargs = dict(execution_mode=execution_mode, execution_backend=execution_backend)
    sim = tb.Simulator(model, **sim_kwargs)
    sim.simulate(start_time=START, end_time=START + datetime.timedelta(hours=HOURS), step_size=DT, show_progress_bar=False)
    room = model.components["Room"]

    def series(port):
        h = room.output[port].history()
        return h.reshape(h.shape[0], -1)[:, 0].detach().cpu().clone().double()

    return {p: series(p) for p in ("indoorTemperature", "indoorCO2", "makeUpAirFlowRate", "surplusAirFlowRate")}


class TestTransform(unittest.TestCase):
    @_exact
    def test_deficit(self):
        out = air_balance.balanced_flow_inputs(
            {"supplyAirFlowRate": _t(0.1), "exhaustAirFlowRate": _t(0.3), "infiltrationAirFlowRate": _t(0.05),
             "makeUpAirTemperature": _t(20.0), "outdoorTemperature": _t(0.0)},
            "makeUpAirTemperature", "outdoorTemperature")
        self.assertAlmostEqual(float(out["exhaustAirFlowRate"]), 0.2, places=12)  # deficit 0.15 + inflow 0.05
        self.assertAlmostEqual(float(out["makeUpAirTemperature"]), (0.15 * 20.0 + 0.05 * 0.0) / 0.2, places=12)

    @_exact
    def test_surplus(self):
        out = air_balance.balanced_flow_inputs(
            {"supplyAirFlowRate": _t(0.3), "exhaustAirFlowRate": _t(0.1), "infiltrationAirFlowRate": _t(0.05),
             "makeUpAirTemperature": _t(20.0), "outdoorTemperature": _t(0.0)},
            "makeUpAirTemperature", "outdoorTemperature")
        self.assertAlmostEqual(float(out["exhaustAirFlowRate"]), 0.05, places=12)  # the inflow alone
        self.assertAlmostEqual(float(out["makeUpAirTemperature"]), 0.0, places=12)  # outdoor air
        surplus, make_up = air_balance.imbalance_flows(_t(0.3), _t(0.1), _t(0.05))
        self.assertAlmostEqual(float(surplus), 0.25, places=12)
        self.assertAlmostEqual(float(make_up), 0.0, places=12)

    def test_zero_inflow_is_bit_for_bit(self):
        base = {"supplyAirFlowRate": _t(0.1), "exhaustAirFlowRate": _t(0.3), "makeUpAirTemperature": _t(19.3),
                "outdoorTemperature": _t(-2.0)}
        without = air_balance.balanced_flow_inputs(base, "makeUpAirTemperature", "outdoorTemperature")
        with_zero = air_balance.balanced_flow_inputs({**base, "infiltrationAirFlowRate": _t(0.0)},
                                                     "makeUpAirTemperature", "outdoorTemperature")
        self.assertTrue(torch.equal(without["exhaustAirFlowRate"], with_zero["exhaustAirFlowRate"]))
        self.assertTrue(torch.equal(with_zero["makeUpAirTemperature"], base["makeUpAirTemperature"]))


class TestRoom(unittest.TestCase):
    @_exact
    def test_without_a_deficit_the_inflow_is_the_constant_infiltration(self):
        for fuse in (True, False):
            a = _run(fuse=fuse, m_inf=0.02, model_id=f"m_inf_{fuse}")
            b = _run(fuse=fuse, m_inf=0.0, inflow=0.02, model_id=f"inflow_{fuse}")
            for port in ("indoorTemperature", "indoorCO2"):
                torch.testing.assert_close(a[port], b[port], rtol=0, atol=1e-8, msg=f"{port}, fused {fuse}")
            torch.testing.assert_close(b["surplusAirFlowRate"], a["surplusAirFlowRate"] + 0.02, rtol=0, atol=1e-12)

    @_exact
    def test_in_a_deficit_the_inflow_replaces_make_up_air(self):
        # supply 0.15, exhaust 0.25: a deficit of 0.10 from outdoors either way; the inflow is part of it
        a = _run(exhaust=0.25, model_id="deficit")
        b = _run(exhaust=0.25, inflow=0.04, model_id="deficit_inflow")
        for port in ("indoorTemperature", "indoorCO2"):
            torch.testing.assert_close(a[port], b[port], rtol=0, atol=1e-8, msg=port)
        torch.testing.assert_close(a["makeUpAirFlowRate"], b["makeUpAirFlowRate"] + 0.04, rtol=0, atol=1e-12)

    @_exact
    def test_a_wired_make_up_temperature_mixes_with_the_inflow(self):
        # 0.06 from the hallway at 20 C and 0.04 from outdoors at 0 C = 0.10 at 12 C
        a = _run(exhaust=0.25, make_up_temperature=12.0, model_id="mixed_reference")
        b = _run(exhaust=0.25, make_up_temperature=20.0, inflow=0.04, model_id="mixed_inflow")
        torch.testing.assert_close(a["indoorTemperature"], b["indoorTemperature"], rtol=0, atol=1e-8)

    def test_object_and_functional_paths_agree_fused_and_unfused(self):
        for fuse in (False, True):
            with self.subTest(fuse=fuse):
                a = _run(fuse=fuse, exhaust=0.25, make_up_temperature=18.0, inflow=0.03, model_id=f"paths_{fuse}")
                b = _run(fuse=fuse, exhaust=0.25, make_up_temperature=18.0, inflow=0.03, model_id=f"paths_f_{fuse}",
                         execution_mode="functional", execution_backend="eager")
                for port in ("indoorTemperature", "indoorCO2", "makeUpAirFlowRate", "surplusAirFlowRate"):
                    torch.testing.assert_close(a[port], b[port], rtol=1e-9, atol=1e-9, msg=port)


if __name__ == "__main__":
    unittest.main()
