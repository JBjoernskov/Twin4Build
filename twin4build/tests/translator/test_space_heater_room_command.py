"""Room-level radiator from a BRICK ``Heating_Command`` (no equipment node).

BMS-derived graphs often carry no radiator equipment, no water-flow sensor
and no supply-temperature point: the only heating information on a room is a
valve command (Hoeje-Taastrup Raadhus: ``R08_01_MVV01``).  The pattern under
test turns that into one :class:`SpaceHeaterSystem` per room, fed by the
command's historised sensor, and the delivered ``Power`` becomes the room's
``heatGain``.
"""

# Standard library imports
import os
import shutil
import unittest

# Third party imports
from rdflib import RDF, BNode, Literal, Namespace

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
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.space_heater.space_heater_system import SpaceHeaterSystem
from twin4build.systems.valve.valve_system import ValveSystem
from twin4build.translator.translator import Translator

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REC = core.namespace.REC
REF = core.namespace.BRICKREF
EX = Namespace("http://example.org/heating#")


def _point(g, owner, name, cls):
    p = EX[name]
    g.add((owner, BRICK.hasPoint, p))
    g.add((p, RDF.type, cls))
    ref = BNode()
    g.add((p, REF.hasExternalReference, ref))
    g.add((ref, RDF.type, REF.TimeseriesReference))
    g.add((ref, REF.hasTimeseriesId, Literal(name)))
    return p


def _graph(sm):
    """Two rooms served by one AHU; only R01 has a radiator valve command."""
    g = sm.instance_graph
    g.add((EX.AHU01, RDF.type, BRICK.AHU))
    for i in (1, 2):
        room, vav = EX[f"R0{i}"], EX[f"R0{i}_VAV01"]
        g.add((room, RDF.type, REC.Room))
        g.add((vav, RDF.type, BRICK.VAV))
        g.add((EX.AHU01, BRICK.feeds, vav))
        g.add((vav, BRICK.feeds, room))
        _point(g, vav, f"R0{i}_VAV01_CMD", BRICK.Damper_Position_Command)
        _point(g, vav, f"R0{i}_SpFCI01", BRICK.Supply_Air_Flow_Setpoint)
        _point(g, room, f"R0{i}_TRU01", BRICK.Zone_Air_Temperature_Sensor)
        _point(g, room, f"R0{i}_SpTRU01", BRICK.Zone_Air_Temperature_Setpoint)
    _point(g, EX["R01"], "R01_MVV01", BRICK.Heating_Command)


def _incoming(component, port):
    for cp in component.connects_at:
        if cp.input_port == port:
            return [conn.connects_system for conn in cp.connects_system_through]
    return []


class TestSpaceHeaterFromRoomHeatingCommand(unittest.TestCase):
    MODEL_ID = "test_space_heater_room_command"

    def tearDown(self):
        path = os.path.join("generated_files", "models", self.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_radiator_is_wired_to_room_and_command(self):
        sm = SemanticModel(id=self.MODEL_ID, namespaces={"ex": str(EX)})
        _graph(sm)
        model = Translator().translate(
            sm,
            systems=[
                BuildingSpaceSystem,
                AirHandlingUnitSystem,
                SpaceHeaterSystem,
                ValveSystem,
                ControllerIdentificationPISystem,
                SensorSystem,
            ],
            id=self.MODEL_ID,
        )
        heaters = model.get_components_by_class(SpaceHeaterSystem)
        rooms = model.get_components_by_class(BuildingSpaceSystem)
        self.assertEqual(len(rooms), 2)
        # Only the room carrying a Heating_Command gets a radiator.
        self.assertEqual(len(heaters), 1)
        heater = heaters[0]

        # Water side: a valve modelled on the command turns the 0-1 opening
        # into kg/s (waterFlowRateMax is estimated).  The opening comes from
        # the controller identified at the command URI, as the AHU's damper
        # positions do -- a translated radiator needs its loop, like the AHU.
        flow_sources = _incoming(heater, "waterFlowRate")
        self.assertEqual([type(c).__name__ for c in flow_sources], ["ValveSystem"])
        (opening,) = _incoming(flow_sources[0], "valvePosition")
        self.assertIsInstance(opening, ControllerIdentificationPISystem)

        # Air side: the room it heats, and the heat goes back into that room.
        (room_in,) = _incoming(heater, "indoorTemperature")
        self.assertIsInstance(room_in, BuildingSpaceSystem)
        heated = [r for r in rooms if heater in _incoming(r, "heatGain")]
        self.assertEqual(heated, [room_in])

        # The room without a command keeps an unwired heatGain, so
        # ``fill_missing_inputs`` can still supply a constant there.
        unheated = [r for r in rooms if r is not room_in]
        self.assertEqual(_incoming(unheated[0], "heatGain"), [])

        # ``supplyWaterTemperature`` is not in the graph; it stays unwired for
        # ``fill_missing_inputs``.
        self.assertEqual(_incoming(heater, "supplyWaterTemperature"), [])


if __name__ == "__main__":
    unittest.main()
