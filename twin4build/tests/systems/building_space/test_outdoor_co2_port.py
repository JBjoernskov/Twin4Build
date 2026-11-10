"""The rooms' outdoor CO2 port is ``outdoorCo2Concentration``, the name the
occupancy inversion and the outdoor environment use (#195); its old name
``outdoorCO2`` is gone in 2.0.

Covered:

1. the ports read 400 ppmv while unwired (a 0 ppmv default made the
   inversion book the whole indoor concentration as people);
2. ``outdoorCO2`` is no port of the mass balance or the composite room, and
   connecting it raises.
"""

# Standard library imports
import unittest
import warnings

# Third party imports
import torch

# Local application imports
import twin4build as tb

tb._IS_TESTING = True

OLD, NEW = "outdoorCO2", "outdoorCo2Concentration"


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


def _room(sid="room"):
    return tb.BuildingSpaceSystem(
        id=sid,
        thermal=dict(C_wall=1e6, C_air=1e5, R_out=0.01, R_in=0.02),
        mass=dict(V=100.0),
    )


class TestDefaults(unittest.TestCase):
    def test_unwired_ports_read_400_ppmv(self):
        for component in (
            tb.BuildingSpaceMassSystem(V=100.0, G_occ=5e-6, m_inf=0.004, id="mass"),
            _room(),
            tb.OccupancySystem(id="occupancy"),
        ):
            port = component.input[NEW]
            port.initialize(n_t=2)
            torch.testing.assert_close(port.get(), torch.full_like(port.get(), 400.0), msg=component.id)


class TestOldNameIsGone(unittest.TestCase):
    def test_outdoor_co2_is_no_port_of_the_room(self):
        for component in (tb.BuildingSpaceMassSystem(V=100.0, G_occ=5e-6, m_inf=0.004, id="mass"), _room()):
            model = tb.Model(id=f"outdoor_co2_gone_{component.id}")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                self.assertEqual(component.resolve_input_port(OLD), OLD)
                self.assertNotIn(OLD, component.input)
                self.assertEqual(type(component).INPUT_PORT_ALIASES, {})
                with self.assertRaises(AssertionError):
                    model.add_connection(_schedule(400.0, "c_out"), component, "scheduleValue", OLD)
            self.assertEqual(component.connects_at, [])
            self.assertIn(NEW, component.input)


if __name__ == "__main__":
    unittest.main()
