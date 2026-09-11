"""Occupancy inferred from a room's CO2 balance, translated from the graph.

The graph-driven form of the CSV-driven construction in
``full_workflow_example``: an :class:`OccupancySystem` per room, fed by the
room's CO2 sensor and the supply-air-flow sensors of the VAVs serving it,
and its ``scheduleValue`` wired into the zone's ``numberOfPeople``.
"""

# Standard library imports
import os
import shutil
import unittest

# Local application imports
import twin4build
import twin4build.core as core
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.air_handling_unit.air_handling_unit_system import (
    AirHandlingUnitSystem,
)
from twin4build.systems.building_space.building_space_system import BuildingSpaceSystem
from twin4build.systems.outdoor_environment.outdoor_environment_system import (
    OutdoorEnvironmentSystem,
)
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.utils.occupancy_system import OccupancySystem
from twin4build.tests.translator.test_brick14_bms_patterns import EX, build_graph
from twin4build.translator.translator import Translator

twin4build._IS_TESTING = True

SYSTEMS = [
    BuildingSpaceSystem,
    AirHandlingUnitSystem,
    OccupancySystem,
    OutdoorEnvironmentSystem,
    SensorSystem,
]


def incoming(component, port):
    for cp in component.connects_at:
        if cp.input_port == port:
            return [conn.connects_system for conn in cp.connects_system_through]
    return []


def incoming_ports(component, port):
    for cp in component.connects_at:
        if cp.input_port == port:
            return {conn.output_port for conn in cp.connects_system_through}
    return set()


class TestOccupancyFromCO2(unittest.TestCase):
    MODEL_ID = "test_occupancy_from_co2"

    @classmethod
    def setUpClass(cls):
        sm = SemanticModel(id=cls.MODEL_ID, namespaces={"ex": str(EX)})
        build_graph(sm)  # two rooms, one CO2 sensor each; R02 has two VAVs
        cls.model = Translator().translate(sm, systems=SYSTEMS, id=cls.MODEL_ID)

    @classmethod
    def tearDownClass(cls):
        path = os.path.join("generated_files", "models", cls.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_one_occupancy_per_room_fed_by_its_sensors(self):
        occ = sorted(self.model.get_components_by_class(OccupancySystem), key=lambda c: c.id)
        self.assertEqual(len(occ), 2, [c.id for c in occ])
        for o in occ:
            (co2,) = incoming(o, "indoorCo2Measured")
            self.assertIsInstance(co2, SensorSystem)
            self.assertTrue(co2.uuid.endswith("CO201"), co2.uuid)
            flows = incoming(o, "supplyAirFlowRateMeasured")
            self.assertTrue(flows and all(isinstance(f, SensorSystem) for f in flows))
            self.assertTrue(all("FCI" in f.uuid for f in flows), [f.uuid for f in flows])
            # The MEASUREMENT, not the modelled value: the CO2 sensor is a
            # virtual sensor fed by the zone, and reading its measuredValue
            # would close a loop zone -> sensor -> occupancy -> zone.
            self.assertEqual(incoming_ports(o, "indoorCo2Measured"), {"measuredData"})
            self.assertEqual(incoming_ports(o, "supplyAirFlowRateMeasured"), {"measuredData"})
            # outdoor CO2 is left for fill_missing_inputs
            self.assertEqual(incoming(o, "outdoorCo2Concentration"), [])

    def test_room_with_two_vavs_gets_both_flows(self):
        occ = [c for c in self.model.get_components_by_class(OccupancySystem) if "R02" in c.id]
        self.assertEqual(len(occ), 1)
        flows = sorted(f.uuid for f in incoming(occ[0], "supplyAirFlowRateMeasured"))
        self.assertEqual(flows, ["R02_FCI01", "R02_FCI02"])

    def test_zone_takes_number_of_people_from_occupancy(self):
        rooms = self.model.get_components_by_class(BuildingSpaceSystem)
        self.assertEqual(len(rooms), 2)
        for room in rooms:
            src = incoming(room, "numberOfPeople")
            self.assertEqual([type(s).__name__ for s in src], ["OccupancySystem"])


if __name__ == "__main__":
    unittest.main()
