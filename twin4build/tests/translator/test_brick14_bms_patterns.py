"""End-to-end translation of a small Brick 1.4 / BMS-style graph.

Mirrors the topology of the Hoeje-Taastrup Raadhus graph: rooms are
``rec:Room``, the VAV carries its ``Damper_Position_Command`` as a direct
point (no ``brick:Damper`` equipment), the zone temperature / CO2 sensors
hang off the room, and the weather station has a
``brick:Solar_Irradiance_Sensor``.  Before the pattern additions none of
BuildingSpace / AirHandlingUnit / OutdoorEnvironment matched.
"""

# Standard library imports
import os
import shutil
import unittest

# Third party imports
from rdflib import RDF, Literal, Namespace, URIRef

# Local application imports
import twin4build
import twin4build.core as core
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.air_handling_unit.air_handling_unit_system import (
    AirHandlingUnitSystem,
)
from twin4build.systems.building_space.building_space_system import (
    BuildingSpaceSystem,
)
from twin4build.systems.outdoor_environment.outdoor_environment_system import (
    OutdoorEnvironmentSystem,
)
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.translator.translator import Translator

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REC = core.namespace.REC
REF = core.namespace.BRICKREF
EX = Namespace("http://example.org/htr#")


def _point(g, owner, name, cls, tsid=True):
    p = EX[name]
    g.add((owner, BRICK.hasPoint, p))
    g.add((p, RDF.type, cls))
    if tsid:
        ref = EX[name + "_ref"]
        g.add((p, REF.hasExternalReference, ref))
        g.add((ref, RDF.type, REF.TimeseriesReference))
        g.add((ref, REF.hasTimeseriesId, Literal(name)))
    return p


def build_graph(sm):
    g = sm.instance_graph
    ahu, ws = EX.AHU01, EX.WS01
    g.add((ahu, RDF.type, BRICK.AHU))
    _point(g, ahu, "AHU01_SAT", BRICK.Supply_Air_Temperature_Sensor)
    _point(g, ahu, "AHU01_SAT_SP", BRICK.Supply_Air_Temperature_Setpoint)
    g.add((ws, RDF.type, BRICK.Weather_Station))
    _point(g, ws, "WS01_TOUT", BRICK.Outside_Air_Temperature_Sensor)
    _point(g, ws, "WS01_SOLAR", BRICK.Solar_Irradiance_Sensor)
    for i in (1, 2):
        room, vav = EX[f"R0{i}"], EX[f"R0{i}_VAV01"]
        g.add((room, RDF.type, REC.Room))
        g.add((vav, RDF.type, BRICK.VAV))
        g.add((ahu, BRICK.feeds, vav))
        g.add((vav, BRICK.feeds, room))
        _point(g, vav, f"R0{i}_VAV01_CMD", BRICK.Damper_Position_Command)
        _point(g, vav, f"R0{i}_FCI01", BRICK.Supply_Air_Flow_Sensor)
        _point(g, room, f"R0{i}_TRU01", BRICK.Zone_Air_Temperature_Sensor)
        _point(g, room, f"R0{i}_CO201", BRICK.Zone_CO2_Level_Sensor)
    # ``isPointOf`` is only materialised by the reasoner from ``hasPoint``
    # (owl:inverseOf) -- the patterns rely on that, as for real graphs.


class TestBrick14BmsPatterns(unittest.TestCase):
    MODEL_ID = "test_brick14_bms_patterns"

    def tearDown(self):
        path = os.path.join("generated_files", "models", self.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_translation_matches_physics_and_sensors(self):
        sm = SemanticModel(id=self.MODEL_ID, namespaces={"ex": str(EX)})
        build_graph(sm)
        model = Translator().translate(
            sm,
            systems=[
                BuildingSpaceSystem,
                AirHandlingUnitSystem,
                OutdoorEnvironmentSystem,
                SensorSystem,
            ],
            id=self.MODEL_ID,
        )
        by_cls = {}
        for c in model.components.values():
            by_cls.setdefault(type(c).__name__, []).append(c)

        self.assertEqual(sorted(c.id for c in by_cls["BuildingSpaceSystem"]), ["R01", "R02"])
        self.assertEqual(len(by_cls["AirHandlingUnitSystem"]), 1)
        self.assertEqual(len(by_cls["OutdoorEnvironmentSystem"]), 1)

        def incoming(comp, port):
            for cp in comp.connects_at:
                if cp.input_port == port:
                    return [conn.connects_system for conn in cp.connects_system_through]
            return []

        ahu = by_cls["AirHandlingUnitSystem"][0]
        rooms = {c.id: c for c in by_cls["BuildingSpaceSystem"]}
        for room in rooms.values():
            self.assertEqual(incoming(room, "supplyAirFlowRate"), [ahu])
            self.assertEqual(
                [type(c).__name__ for c in incoming(room, "outdoorTemperature")],
                ["OutdoorEnvironmentSystem"],
            )
        self.assertEqual(len(incoming(ahu, "exhaustTemperature")), 2)
        self.assertEqual(
            [c.uuid for c in incoming(ahu, "supplyAirTemperatureSetpoint")],
            ["AHU01_SAT_SP"],
        )

        # Room-attached zone sensors are wired to the modelled room state.
        sensors = {c.uuid: c for c in by_cls["SensorSystem"] if c.uuid}
        for i in (1, 2):
            self.assertEqual(incoming(sensors[f"R0{i}_TRU01"], "measuredValue"), [rooms[f"R0{i}"]])
            self.assertEqual(incoming(sensors[f"R0{i}_CO201"], "measuredValue"), [rooms[f"R0{i}"]])
            self.assertEqual(incoming(sensors[f"R0{i}_FCI01"], "measuredValue"), [ahu])
        self.assertEqual(incoming(sensors["AHU01_SAT"], "measuredValue"), [ahu])

        outdoor = by_cls["OutdoorEnvironmentSystem"][0]
        self.assertEqual(outdoor.uuid_outdoorTemperature, "WS01_TOUT")
        self.assertEqual(outdoor.uuid_globalIrradiation, "WS01_SOLAR")
        # The vendored Brick ``ref`` schema parses offline.
        self.assertNotIn(str(REF), sm.error_namespaces)


if __name__ == "__main__":
    unittest.main()
