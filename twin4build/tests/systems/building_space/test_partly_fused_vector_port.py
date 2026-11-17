"""A fused block's vector input fed partly inside the block.

A room's ``wallHeatGain`` slots can come from walls of both kinds: a wall fused
with the room (toward a space that never fuses, a hallway) and a wall kept out
of fusion (``fusable=False``, toward another room).  The block keeps the port
as an external input, substitutes the fused wall's slot exactly, and sums only
the outside slots into the port's column -- the port object still holds the
fused slot's value (written after each step), which must not count twice.
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

DT = 600
START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))


def _schedule(value, sid):
    return tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": float(value)}, id=sid)


def _zone(model, zid, heat, schedules):
    zone = tb.BuildingSpaceThermalSystem(
        C_air=2e5, C_wall=5e6, R_out=0.05, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=0.0, id=zid,
    )
    for port in ("outdoorTemperature", "supplyAirFlowRate", "exhaustAirFlowRate", "supplyAirTemperature",
                 "globalIrradiation", "numberOfPeople"):
        model.add_connection(schedules[port], zone, "scheduleValue", port)
    model.add_connection(_schedule(heat, f"Heat_{zid}"), zone, "scheduleValue", "heatGain")
    return zone


def _fused_wall(model, zone, hallway, slot):
    """A wall from ``zone`` (side A) to a constant hallway temperature (side B): it fuses with the zone."""
    wall = tb.WallSystem(C=1e5, R_a=0.01, R_b=0.01, id=f"HallWall_{zone.id}")
    model.add_connection(zone, wall, "indoorTemperature", "temperatureA")
    model.add_connection(hallway, wall, "scheduleValue", "temperatureB")
    model.add_connection(wall, zone, "heatFlowRateA", "wallHeatGain", input_port_index=slot)
    return wall


def _schedules():
    values = {"outdoorTemperature": 5.0, "supplyAirFlowRate": 0.0, "exhaustAirFlowRate": 0.0,
              "supplyAirTemperature": 20.0, "globalIrradiation": 0.0, "numberOfPeople": 0.0}
    return {port: _schedule(value, f"S_{port}") for port, value in values.items()}


def _simulate(model, mode="object", hours=12):
    kwargs = {} if mode == "object" else {"execution_mode": "functional", "execution_backend": "eager"}
    tb.Simulator(model, **kwargs).simulate(start_time=START, end_time=START + datetime.timedelta(hours=hours),
                                           step_size=DT, show_progress_bar=False)


def _history(component):
    return component.output["indoorTemperature"].history().detach().reshape(-1).clone()


def one_room(outside_on_wall_port: bool, model_id: str):
    """One room with a fused wall to a 15 C hallway on ``wallHeatGain`` slot 0, plus 800 W from outside: on
    ``wallHeatGain`` slot 1 (the port fed partly inside) or on ``heatGain`` (the reference)."""
    model = tb.Model(id=model_id)
    schedules = _schedules()
    room = _zone(model, "Room", 0.0 if outside_on_wall_port else 800.0, schedules)
    _fused_wall(model, room, _schedule(15.0, "Hallway"), slot=0)
    if outside_on_wall_port:
        model.add_connection(_schedule(800.0, "Outside"), room, "scheduleValue", "wallHeatGain", input_port_index=1)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, room


def two_rooms(model_id: str, fuse: bool = True, pairs: int = 1, swap_b: bool = False):
    """``pairs`` x two rooms, each with a fused wall to a 15 C hallway, joined by a wall kept out of fusion; room A
    heated.  The hallway wall on slot 0 and the wall between on slot 1, or the other way round in room B
    (``swap_b``)."""
    model = tb.Model(id=model_id)
    schedules = _schedules()
    hallway = _schedule(15.0, "Hallway")
    rooms = []
    for k in range(pairs):
        a = _zone(model, f"RoomA{k}", 1500.0, schedules)
        b = _zone(model, f"RoomB{k}", 0.0, schedules)
        hall_slot = {a: 0, b: 1 if swap_b else 0}
        for room in (a, b):
            _fused_wall(model, room, hallway, slot=hall_slot[room])
        between = tb.WallSystem(C=1e5, R_a=0.05, R_b=0.05, fusable=False, id=f"Between{k}")
        model.add_connection(a, between, "indoorTemperature", "temperatureA")
        model.add_connection(b, between, "indoorTemperature", "temperatureB")
        model.add_connection(between, a, "heatFlowRateA", "wallHeatGain", input_port_index=1 - hall_slot[a])
        model.add_connection(between, b, "heatFlowRateB", "wallHeatGain", input_port_index=1 - hall_slot[b])
        rooms += [a, b]
    model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=fuse)
    return (model, *rooms) if pairs == 1 else (model, rooms)


class TestPartlyFusedVectorPort(unittest.TestCase):
    def test_the_block_keeps_the_port_and_counts_the_outside_slot_once(self):
        partly, room = one_room(True, "partly_fused_port")
        reference, ref_room = one_room(False, "partly_fused_reference")
        blocks = list(partly.simulation_model._fused_components.values())
        self.assertEqual(len(blocks), 1)
        self.assertEqual({m.id for m in blocks[0].members}, {"Room", "HallWall_Room"})
        self.assertIn("Room.wallHeatGain", blocks[0].input)
        _simulate(partly)
        _simulate(reference)
        torch.testing.assert_close(_history(room), _history(ref_room), rtol=0, atol=1e-9)

    def test_two_rooms_each_their_own_block(self):
        model, a, b = two_rooms("partly_fused_two_rooms")
        blocks = list(model.simulation_model._fused_components.values())
        self.assertEqual(sorted(sorted(m.id for m in blk.members) for blk in blocks),
                         [["HallWall_RoomA0", "RoomA0"], ["HallWall_RoomB0", "RoomB0"]])
        _simulate(model)
        ta, tb_ = _history(a), _history(b)
        self.assertTrue(bool(torch.isfinite(ta).all() and torch.isfinite(tb_).all()))
        self.assertGreater(float(tb_[-1]), 15.0)  # the heat of room A reaches room B through the wall between

    def test_object_and_functional_agree(self):
        obj, a_o, b_o = two_rooms("partly_fused_object")
        fun, a_f, b_f = two_rooms("partly_fused_functional")
        _simulate(obj, "object")
        _simulate(fun, "functional")
        torch.testing.assert_close(_history(a_f), _history(a_o), rtol=0, atol=1e-9)
        torch.testing.assert_close(_history(b_f), _history(b_o), rtol=0, atol=1e-9)

    def test_matches_the_unfused_model_at_a_small_step(self):
        fused, a_f, b_f = two_rooms("partly_fused_small_step")
        unfused, a_u, b_u = two_rooms("partly_unfused_small_step", fuse=False)
        for model in (fused, unfused):
            tb.Simulator(model).simulate(start_time=START, end_time=START + datetime.timedelta(hours=6),
                                         step_size=20, show_progress_bar=False)
        for f, u in ((a_f, a_u), (b_f, b_u)):
            self.assertLess(float((_history(f) - _history(u)).abs().max()), 0.05)

    def test_swapped_slots_match_the_unswapped_model(self):
        """Room B's hallway wall on slot 1 and the wall between on slot 0: the same physics."""
        plain, a_p, b_p = two_rooms("partly_fused_plain")
        swapped, a_s, b_s = two_rooms("partly_fused_swapped", swap_b=True)
        _simulate(plain)
        _simulate(swapped)
        torch.testing.assert_close(_history(b_s), _history(b_p), rtol=0, atol=1e-9)
        torch.testing.assert_close(_history(a_s), _history(a_p), rtol=0, atol=1e-9)

    def test_batched_rooms_match_the_unbatched_model(self):
        """Batched, the inside slot at different positions among the rooms (``swap_b``)."""
        for swap in (False, True):
            model, rooms = two_rooms(f"partly_fused_batched_{swap}", pairs=2, swap_b=swap)
            _simulate(model)
            reference = {r.id: _history(r) for r in rooms}
            batched = model.batch_components()
            batched.load(draw_semantic_model=False, draw_simulation_model=False)
            _simulate(batched)
            for r in rooms:
                meta, i_c = model._component_to_meta[r.id]
                hist = meta.output["indoorTemperature"].history().detach()
                got = hist.reshape(hist.shape[0], -1)[:, i_c]
                torch.testing.assert_close(got, reference[r.id], rtol=0, atol=1e-9, msg=f"{r.id} swap={swap}")


if __name__ == "__main__":
    unittest.main()
