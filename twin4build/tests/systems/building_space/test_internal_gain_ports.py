"""``plugLoad`` and ``exchangeHeatGain``: heat inputs into the room's air node.

* ``plugLoad`` [W]: electrical plug load, all of it heat.  A scalar that reads
  0 while unwired (``Input(default=0.0)``), so an unwired room runs as before.
* ``exchangeHeatGain`` [W]: the heat flows of openings to other zones, a vector
  port apart from ``wallHeatGain``, so the walls between rooms can stay out of
  fusion (``fusable=False``) while the openings between zones of one room fuse.

Both enter the air node exactly as ``heatGain`` does (``1 / C_air``).
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb

tb._IS_TESTING = True

DT = 600.0
START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
BASE_U = [
    "outdoorTemperature", "supplyAirFlowRate", "exhaustAirFlowRate", "supplyAirTemperature", "globalIrradiation",
    "numberOfPeople", "heatGain", "makeUpAirTemperature", "plugLoad", "infiltrationAirFlowRate",
]


def _t(v):
    return torch.tensor([float(v)], dtype=torch.float64)


class TestThermalPorts(unittest.TestCase):
    """The one-step map of a thermal zone: both inputs are heat in the air node."""

    def setUp(self):
        start = START.replace(tzinfo=None)
        self.zone = tb.BuildingSpaceThermalSystem(
            C_air=1e5, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=0.0, id="zone",
        )
        self.zone.initialize([start], [start + datetime.timedelta(hours=1)], [DT])
        self.params = {name: self.zone._forward_params()[name] for name in self.zone.PARAM_NAMES}
        self.x0 = self.zone.get_state()

    def _step(self, heat=0.0, plug=None, exchange=None):
        zone = self.zone
        zone.n_exchanges = 0 if exchange is None else len(exchange)
        inputs = {
            "outdoorTemperature": _t(5.0),
            "supplyAirFlowRate": _t(0.1),
            "exhaustAirFlowRate": _t(0.1),
            "supplyAirTemperature": _t(18.0),
            "globalIrradiation": _t(0.0),
            "numberOfPeople": _t(0.0),
            "heatGain": _t(heat),
        }
        if plug is not None:
            inputs["plugLoad"] = _t(plug)
        if exchange is not None:
            inputs["exchangeHeatGain"] = torch.tensor([exchange], dtype=torch.float64)
        # the matrices follow the openings' slots: derived afresh
        x1, _ = zone.forward(self.x0, zone.with_defaults(inputs), zone.full_params(dict(self.params), DT), DT)
        return x1.reshape(-1)

    def test_unwired_layout(self):
        self.assertEqual(self.zone.n_exchanges, 0)
        self.assertEqual([name for name, _ in self.zone._ss_layout()["u"]], BASE_U)
        self.assertEqual(self.zone._build_matrices()[1].shape[-1], len(BASE_U))

    def test_the_plug_load_is_heat_in_the_air(self):
        torch.testing.assert_close(self._step(plug=500.0), self._step(heat=500.0), rtol=0, atol=1e-12)

    def test_a_zero_plug_load_changes_nothing(self):
        torch.testing.assert_close(self._step(plug=0.0), self._step(), rtol=0, atol=1e-12)
        self.assertGreater(float(self._step(plug=500.0)[0]), float(self._step()[0]))

    def test_exchange_slots_are_heat_in_the_air(self):
        torch.testing.assert_close(self._step(exchange=[300.0, -100.0]), self._step(heat=200.0), rtol=0, atol=1e-12)

    def test_wired_layout_keeps_the_existing_slots(self):
        self.zone.n_exchanges = 2
        u = self.zone._ss_layout()["u"]
        names = [name for name, _ in u]
        self.assertEqual(names[: len(BASE_U) - 2], BASE_U[:-2])
        self.assertEqual(names[len(BASE_U) - 2:], ["exchangeHeatGain", "plugLoad", "infiltrationAirFlowRate"])
        self.assertEqual(dict(u)["exchangeHeatGain"], 2)
        self.assertEqual(names[self.zone.MAKE_UP_TEMPERATURE_SLOT], "makeUpAirTemperature")
        self.assertEqual(self.zone._build_matrices()[1].shape[-1], len(BASE_U) + 2)

    def test_fusable_ports(self):
        for cls in (tb.BuildingSpaceThermalSystem, tb.BuildingSpaceSystem):
            self.assertIn("exchangeHeatGain", cls.FUSABLE_INPUT_PORTS)
            self.assertNotIn("plugLoad", cls.FUSABLE_INPUT_PORTS)


def _schedule(value, sid):
    return tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": float(value)}, id=sid)


def _simulate(model, hours=6, step=int(DT)):
    tb.Simulator(model).simulate(
        start_time=START, end_time=START + datetime.timedelta(hours=hours), step_size=step, show_progress_bar=False
    )


def _history(component, port="indoorTemperature"):
    return component.output[port].history().detach().reshape(-1).clone()


class TestCompositePlugLoad(unittest.TestCase):
    """A ``BuildingSpaceSystem`` hands the wired ``plugLoad`` down to its thermal model."""

    def _room(self, tag, heat=0.0, plug=None):
        model = tb.Model(id=f"plug_load_{tag}")
        room = tb.BuildingSpaceSystem(id="Room")
        ports = [("outdoorTemperature", 5.0), ("supplyAirTemperature", 18.0), ("globalIrradiation", 0.0),
                 ("numberOfPeople", 0.0), ("heatGain", heat), ("outdoorCo2Concentration", 400.0)]
        if plug is not None:
            ports.append(("plugLoad", plug))
        for port, value in ports:
            model.add_connection(_schedule(value, f"S_{port}"), room, "scheduleValue", port)
        for port in ("supplyAirFlowRate", "exhaustAirFlowRate"):
            model.add_connection(_schedule(0.1, f"S_{port}"), room, "scheduleValue", port, input_port_index=0)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        return room

    def test_the_plug_load_heats_like_a_heater(self):
        plugged = self._room("plug", plug=800.0)
        heated = self._room("heat", heat=800.0)
        torch.testing.assert_close(_history(plugged), _history(heated), rtol=0, atol=1e-9)

    def test_a_zero_plug_load_is_no_plug_load(self):
        torch.testing.assert_close(
            _history(self._room("zero", plug=0.0)), _history(self._room("none")), rtol=0, atol=1e-9
        )


def build_zones(fuse=True, model_id="exchange", r_opening=0.02, composite=False, pairs=1):
    """``pairs`` x two zones, zone A heated, coupled by a wall on ``wallHeatGain`` that opts out of fusion
    (``fusable=False``) and by an opening (a fusable ``WallSystem`` standing in for an opening's heat side) on
    ``exchangeHeatGain``."""
    model = tb.Model(id=model_id)
    outdoor = _schedule(5.0, "Outdoor")
    zero = _schedule(0.0, "Zero")
    supply_t = _schedule(20.0, "SupplyT")
    heater = _schedule(1500.0, "Heater")
    co2 = _schedule(400.0, "OutdoorCO2")
    zones = {}
    for k in range(pairs):
        for side in ("A", "B"):
            zid = f"Zone{side}{k}"
            if composite:
                zone = tb.BuildingSpaceSystem(
                    thermal=dict(C_air=1e6, C_wall=5e6, R_out=0.01, R_in=0.01, Q_occ_gain=100.0), id=zid,
                )
                model.add_connection(co2, zone, "scheduleValue", "outdoorCo2Concentration")
                for port in ("supplyAirFlowRate", "exhaustAirFlowRate"):
                    model.add_connection(zero, zone, "scheduleValue", port, input_port_index=0)
            else:
                zone = tb.BuildingSpaceThermalSystem(
                    C_air=1e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0, id=zid
                )
                for port in ("supplyAirFlowRate", "exhaustAirFlowRate"):
                    model.add_connection(zero, zone, "scheduleValue", port)
            model.add_connection(outdoor, zone, "scheduleValue", "outdoorTemperature")
            model.add_connection(supply_t, zone, "scheduleValue", "supplyAirTemperature")
            model.add_connection(zero, zone, "scheduleValue", "globalIrradiation")
            model.add_connection(zero, zone, "scheduleValue", "numberOfPeople")
            model.add_connection(heater if side == "A" else zero, zone, "scheduleValue", "heatGain")
            zones[zid] = zone
        wall = tb.WallSystem(C=2e5, R_a=0.05, R_b=0.05, fusable=False, id=f"Wall{k}")
        opening = tb.WallSystem(C=1e4, R_a=r_opening, R_b=r_opening, id=f"Opening{k}")
        a, b = zones[f"ZoneA{k}"], zones[f"ZoneB{k}"]
        for element, port in ((wall, "wallHeatGain"), (opening, "exchangeHeatGain")):
            model.add_connection(a, element, "indoorTemperature", "temperatureA")
            model.add_connection(b, element, "indoorTemperature", "temperatureB")
            model.add_connection(element, a, "heatFlowRateA", port, input_port_index=0)
            model.add_connection(element, b, "heatFlowRateB", port, input_port_index=0)
    model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=fuse)
    return model, zones


class TestOpeningsFuseWallsDont(unittest.TestCase):
    """Walls between rooms couple one step late, openings between zones of one room are fused (exact, stable
    however wide)."""

    def test_the_block_holds_the_zones_and_the_opening_only(self):
        model, zones = build_zones(model_id="exchange_fused")
        fused = list(model.simulation_model._fused_components.values())
        self.assertEqual(len(fused), 1)
        self.assertEqual({m.id for m in fused[0].members}, {"ZoneA0", "ZoneB0", "Opening0"})
        _simulate(model)
        self.assertEqual(zones["ZoneA0"].n_exchanges, 1)
        self.assertEqual(zones["ZoneA0"].n_walls, 1)

    def test_fused_matches_unfused_at_a_small_step(self):
        fused, zf = build_zones(model_id="exchange_on")
        unfused, zu = build_zones(fuse=False, model_id="exchange_off")
        _simulate(fused, step=30)
        _simulate(unfused, step=30)
        for zid in ("ZoneA0", "ZoneB0"):
            err = float((_history(zf[zid]) - _history(zu[zid])).abs().max())
            self.assertLess(err, 0.05, f"{zid}: fused vs unfused max err {err}")
        # the opening carries heat from the heated zone
        self.assertGreater(float(_history(zf["ZoneB0"])[-1]), float(_history(zu["ZoneB0"])[0]) - 1.0)

    def test_a_wide_opening_stays_stable(self):
        model, zones = build_zones(model_id="exchange_wide", r_opening=1e-5)
        _simulate(model)
        a, b = _history(zones["ZoneA0"]), _history(zones["ZoneB0"])
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
        self.assertLess(float((a - b).abs().max()), 0.5)


class TestBatchedExchangeSlots(unittest.TestCase):
    """Batched composite rooms keep their openings' slots: the children take them from the room's wiring."""

    def test_batched_rooms_match_the_unbatched_model(self):
        # both fused (the rooms and their openings in one block)
        model, zones = build_zones(model_id="exchange_batched", composite=True, pairs=2)
        _simulate(model)
        reference = {zid: _history(z) for zid, z in zones.items()}
        batched = model.batch_components()
        meta, _ = model._component_to_meta["ZoneA0"]
        self.assertGreater(meta.n_c, 1)
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        self.assertEqual(len(batched.simulation_model._fused_components), 1)
        _simulate(batched)
        self.assertEqual(meta.thermal.n_exchanges, 1)
        for zid in zones:
            meta, i_c = model._component_to_meta[zid]
            hist = meta.output["indoorTemperature"].history().detach()
            got = hist.reshape(hist.shape[0], -1)[:, i_c]
            torch.testing.assert_close(got, reference[zid], rtol=0, atol=1e-9, msg=zid)


if __name__ == "__main__":
    unittest.main()
