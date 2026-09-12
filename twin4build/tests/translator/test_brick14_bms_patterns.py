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
from rdflib import RDF, BNode, Literal, Namespace, URIRef

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
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
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
    _point(g, ahu, "AHU01_FCI01", BRICK.Supply_Air_Flow_Sensor)  # total supply flow
    _point(g, ahu, "AHU01_FCU01", BRICK.Return_Air_Flow_Sensor)  # total return flow
    g.add((ws, RDF.type, BRICK.Weather_Station))
    _point(g, ws, "WS01_TOUT", BRICK.Outside_Air_Temperature_Sensor)
    _point(g, ws, "WS01_SOLAR", BRICK.Solar_Irradiance_Sensor)
    for i in (1, 2):
        room, vav = EX[f"R0{i}"], EX[f"R0{i}_VAV01"]
        g.add((room, RDF.type, REC.Room))
        vol = BNode()
        g.add((room, BRICK.volume, vol))
        g.add((vol, BRICK.value, Literal(20.0 + i)))
        g.add((vav, RDF.type, BRICK.VAV))
        g.add((ahu, BRICK.feeds, vav))
        g.add((vav, BRICK.feeds, room))
        _point(g, vav, f"R0{i}_VAV01_CMD", BRICK.Damper_Position_Command)
        _point(g, vav, f"R0{i}_FCI01", BRICK.Supply_Air_Flow_Sensor)
        _point(g, vav, f"R0{i}_VAV01_POS", BRICK.Damper_Position_Sensor)
        _point(g, vav, f"R0{i}_SpFCI01_C", BRICK.Supply_Air_Flow_Setpoint)
        _point(g, room, f"R0{i}_TRU01", BRICK.Zone_Air_Temperature_Sensor)
        _point(g, room, f"R0{i}_SpTRU01", BRICK.Zone_Air_Temperature_Setpoint)
        _point(g, room, f"R0{i}_SpTRU01_K", BRICK.Zone_Air_Cooling_Temperature_Setpoint)
        _point(g, room, f"R0{i}_CO201", BRICK.Zone_CO2_Level_Sensor)
    # A second VAV on room R02 (HTR rooms have up to four): each VAV must
    # get its own controller although they share the room's sensor and
    # setpoints (regression: the walker used to keep one match per room).
    vav2 = EX["R02_VAV02"]
    g.add((vav2, RDF.type, BRICK.VAV))
    g.add((ahu, BRICK.feeds, vav2))
    g.add((vav2, BRICK.feeds, EX["R02"]))
    _point(g, vav2, "R02_VAV02_CMD", BRICK.Damper_Position_Command)
    _point(g, vav2, "R02_FCI02", BRICK.Supply_Air_Flow_Sensor)
    _point(g, vav2, "R02_VAV02_POS", BRICK.Damper_Position_Sensor)
    _point(g, vav2, "R02_SpFCI02_C", BRICK.Supply_Air_Flow_Setpoint)
    # ``isPointOf`` is only materialised by the reasoner from ``hasPoint``
    # (owl:inverseOf) -- the patterns rely on that, as for real graphs.


def _outgoing_components(component, port):
    out = []
    for conn in component.connected_through:
        if conn.output_port == port:
            out.extend(cp.connection_point_of for cp in conn.connects_system_at)
    return out


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
                ControllerIdentificationPISystem,
            ],
            id=self.MODEL_ID,
        )
        by_cls = {}
        for c in model.components.values():
            by_cls.setdefault(type(c).__name__, []).append(c)

        # Rooms model the ``rec:Room`` node plus the ``brick:volume`` blank
        # node, so the id is the composite ``[<bnode>][R0i]`` form; resolve
        # them through the translator's sim -> sem map instead.
        translator = model._translator
        rooms = {}
        for c in by_cls["BuildingSpaceSystem"]:
            uris = {str(n.uri) for n in translator.sim2sem_map[c]}
            (room_uri,) = [u for u in uris if u.startswith(str(EX))]
            rooms[room_uri.split("#")[-1]] = c
        self.assertEqual(sorted(rooms), ["R01", "R02"])
        self.assertEqual(len(by_cls["AirHandlingUnitSystem"]), 1)
        self.assertEqual(len(by_cls["OutdoorEnvironmentSystem"]), 1)

        def incoming(comp, port):
            for cp in comp.connects_at:
                if cp.input_port == port:
                    return [conn.connects_system for conn in cp.connects_system_through]
            return []

        ahu = by_cls["AirHandlingUnitSystem"][0]
        for room in rooms.values():
            self.assertEqual(incoming(room, "supplyAirFlowRate"), [ahu])
            # Coil-less VAV: the room breathes AHU supply air directly.
            self.assertEqual(incoming(room, "supplyAirTemperature"), [ahu])
            self.assertEqual(
                [type(c).__name__ for c in incoming(room, "outdoorTemperature")],
                ["OutdoorEnvironmentSystem"],
            )
        # Room volume read from ``brick:volume [brick:value x]``.
        self.assertAlmostEqual(float(rooms["R01"].mass.V.get()), 21.0)
        self.assertAlmostEqual(float(rooms["R02"].mass.V.get()), 22.0)
        self.assertEqual(len(incoming(ahu, "exhaustTemperature")), 2)  # one slot per room
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
        # The AHU's own flow meters read the totals over the branches.
        for uuid, port in (("AHU01_FCI01", "totalSupplyAirFlowRate"), ("AHU01_FCU01", "totalExhaustAirFlowRate")):
            self.assertEqual(incoming(sensors[uuid], "measuredValue"), [ahu])
            self.assertEqual(
                {conn.output_port for cp in sensors[uuid].connects_at for conn in cp.connects_system_through},
                {port},
            )

        # One PI-CITS per VAV, with the loop variables taken from the room:
        # zone temperature sensor -> sensorValue, zone temperature setpoints
        # -> setpointValue, and the VAV flow setpoint on the gate bus.
        cits_list = by_cls["ControllerIdentificationPISystem"]
        self.assertEqual(len(cits_list), 3)  # R01_VAV01, R02_VAV01, R02_VAV02
        gates = []
        for cits in cits_list:
            uuids = lambda port: sorted(c.uuid for c in incoming(cits, port))  # noqa: E731
            (sensor_uuid,) = uuids("sensorValue")
            i = sensor_uuid[2]  # R0<i>_TRU01
            self.assertEqual(sensor_uuid, f"R0{i}_TRU01")
            self.assertEqual(uuids("setpointValue"), [f"R0{i}_SpTRU01", f"R0{i}_SpTRU01_K"])
            (gate,) = uuids("onOffSignal")
            gates.append(gate)
            # Every controller feeds its historised command sensor ...
            downstream = _outgoing_components(cits, "inputSignal")
            self.assertTrue(any(isinstance(c, SensorSystem) for c in downstream))
        self.assertEqual(sorted(gates), ["R01_SpFCI01_C", "R02_SpFCI01_C", "R02_SpFCI02_C"])
        # ... and every controller drives its own AHU branch: one branch per
        # VAV (issue #179), both damper ports of a branch driven by the SAME
        # controller.
        drivers = [c for c in cits_list if ahu in _outgoing_components(c, "inputSignal")]
        self.assertEqual(len(drivers), 3)
        driver_by_slot = {}
        for cp in ahu.connects_at:
            if cp.input_port not in ("supplyDamperPosition", "exhaustDamperPosition"):
                continue
            for conn in cp.connects_system_through:
                slot = int(cp.input_port_index[conn])
                driver_by_slot.setdefault(slot, {})[cp.input_port] = conn.connects_system
        self.assertEqual(len(driver_by_slot), 3)  # one branch per VAV
        for slot, ports in driver_by_slot.items():
            self.assertEqual(
                set(ports), {"supplyDamperPosition", "exhaustDamperPosition"}, slot
            )
            self.assertIs(ports["supplyDamperPosition"], ports["exhaustDamperPosition"])
        # The exhaust temperature stays one slot per room; the AHU maps
        # branch -> room itself.
        self.assertEqual(len(incoming(ahu, "exhaustTemperature")), 2)
        # A two-VAV room reads both of its branches into its Vector flow
        # port (one connection, two slot pairs) ...
        (cp,) = [cp for cp in rooms["R02"].connects_at if cp.input_port == "supplyAirFlowRate"]
        (conn,) = cp.connects_system_through
        self.assertEqual(sorted(cp.input_port_index[conn].tolist()), [0, 1])
        self.assertEqual(len(set(cp.output_port_index[conn].tolist())), 2)
        # ... and its flow sensors read those same branches.
        branch_of = dict(zip(cp.input_port_index[conn].tolist(), cp.output_port_index[conn].tolist()))
        for uuid in ("R02_FCI01", "R02_FCI02"):
            (scp,) = [scp for scp in sensors[uuid].connects_at if scp.input_port == "measuredValue"]
            (sconn,) = scp.connects_system_through
            self.assertIn(int(scp.output_port_index[sconn]), branch_of.values())

        outdoor = by_cls["OutdoorEnvironmentSystem"][0]
        self.assertEqual(outdoor.uuid_outdoorTemperature, "WS01_TOUT")
        self.assertEqual(outdoor.uuid_globalIrradiation, "WS01_SOLAR")
        # Per-feed transformation dispatch: the irradiance rule must reach
        # the irradiation feed, the temperature rule the temperature feed.
        clip = lambda x: min(x, 1500.0)  # noqa: E731
        f2c = lambda x: (x - 32) * 5 / 9  # noqa: E731
        model.set_transformations(
            {BRICK.Solar_Irradiance_Sensor: clip, BRICK.Temperature_Sensor: f2c}
        )
        self.assertIs(outdoor._transformation_globalIrradiation, clip)
        self.assertIs(outdoor._transformation_outdoorTemperature, f2c)
        # The vendored Brick ``ref`` schema parses offline.
        self.assertNotIn(str(REF), sm.error_namespaces)


if __name__ == "__main__":
    unittest.main()
