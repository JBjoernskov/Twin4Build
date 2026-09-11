"""The space-heater control chain, mirroring the VAV damper chain.

A BMS graph with explicit radiator equipment::

    Space_Heater  feeds        Room
    Space_Heater  hasPoint     Heating_Command

must translate into the same closed loop the dampers get::

    CITS(zone temperature vs setpoint, gated by the flow setpoint)
        -> Heating_Command (inputSignal)
        -> ValveSystem.valvePosition -> waterFlowRate
        -> SpaceHeaterSystem.waterFlowRate
        -> Power -> BuildingSpaceSystem.heatGain
"""

# Standard library imports
import os
import shutil
import unittest

# Third party imports
from rdflib import RDF

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
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.space_heater.space_heater_system import SpaceHeaterSystem
from twin4build.systems.valve.valve_system import ValveSystem
from twin4build.tests.translator.test_brick14_bms_patterns import EX, _point, build_graph
from twin4build.translator.translator import Translator

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REC = core.namespace.REC

SYSTEMS = [
    BuildingSpaceSystem,
    AirHandlingUnitSystem,
    SpaceHeaterSystem,
    ValveSystem,
    ControllerIdentificationPISystem,
    OutdoorEnvironmentSystem,
    SensorSystem,
]


def _graph(sm):
    """The Brick 1.4 BMS graph, plus a radiator located in room R01.

    R02 keeps no radiator, so it doubles as the control case: no valve, no
    space heater, no heating loop there.
    """
    build_graph(sm)
    g = sm.instance_graph
    g.add((EX.R01_RAD01, RDF.type, BRICK.Space_Heater))
    g.add((EX.R01_RAD01, BRICK.feeds, EX.R01))
    _point(g, EX.R01_RAD01, "R01_MVV01", BRICK.Heating_Command)


def incoming(component, port):
    for cp in component.connects_at:
        if cp.input_port == port:
            return [conn.connects_system for conn in cp.connects_system_through]
    return []


def outgoing(component, port):
    return [
        cp.connection_point_of
        for conn in component.connected_through
        if conn.output_port == port
        for cp in conn.connects_system_at
    ]


class TestSpaceHeaterControlLoop(unittest.TestCase):
    MODEL_ID = "test_space_heater_control_loop"

    @classmethod
    def setUpClass(cls):
        sm = SemanticModel(id=cls.MODEL_ID, namespaces={"ex": str(EX)})
        _graph(sm)
        cls.model = Translator().translate(sm, systems=SYSTEMS, id=cls.MODEL_ID)

    @classmethod
    def tearDownClass(cls):
        path = os.path.join("generated_files", "models", cls.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_one_controller_per_actuator(self):
        """The radiator valve gets its own PI loop, like the damper."""
        cits = self.model.get_components_by_class(ControllerIdentificationPISystem)
        # Three VAVs in the shared graph plus the one radiator valve.
        self.assertEqual(len(cits), 4, [c.id for c in cits])
        valve_loops = [c for c in cits if "MVV" in c.id]
        self.assertEqual(len(valve_loops), 1, [c.id for c in cits])

    def test_heating_loop_reads_the_room(self):
        cits = [
            c
            for c in self.model.get_components_by_class(ControllerIdentificationPISystem)
            if "MVV" in c.id
        ][0]
        uuids = lambda port: sorted(
            s.uuid for s in incoming(cits, port) if isinstance(s, SensorSystem)
        )
        self.assertEqual(uuids("sensorValue"), ["R01_TRU01"])
        # Every zone temperature setpoint on the room is offered to the
        # tracked-setpoint bus; the gamma weights pick between them.
        self.assertEqual(uuids("setpointValue"), ["R01_SpTRU01", "R01_SpTRU01_K"])
        # Gated on the flow setpoint of the VAV serving the room -- the same
        # signal the damper loop is gated on.
        self.assertEqual(uuids("onOffSignal"), ["R01_SpFCI01_C"])

    def test_radiator_is_fed_by_the_valve_and_heats_the_room(self):
        heaters = self.model.get_components_by_class(SpaceHeaterSystem)
        self.assertEqual(len(heaters), 1, [h.id for h in heaters])
        heater = heaters[0]
        (water,) = incoming(heater, "waterFlowRate")
        # Through the valve modelled on the command, which in turn reads
        # the controller identified at that URI -- controller -> valve ->
        # radiator, the water-side mirror of controller -> damper -> AHU.
        self.assertIsInstance(water, ValveSystem)
        (opening,) = incoming(water, "valvePosition")
        self.assertIsInstance(opening, ControllerIdentificationPISystem)
        self.assertIn("MVV", opening.id)
        (room_in,) = incoming(heater, "indoorTemperature")
        self.assertIsInstance(room_in, BuildingSpaceSystem)
        rooms = self.model.get_components_by_class(BuildingSpaceSystem)
        self.assertEqual(len(rooms), 2)
        heated = [r for r in rooms if incoming(r, "heatGain") == [heater]]
        self.assertEqual(heated, [room_in])
        # The room without a radiator keeps heatGain unwired for
        # fill_missing_inputs.
        other = [r for r in rooms if r is not room_in]
        self.assertEqual(incoming(other[0], "heatGain"), [])
        # Supply water temperature is not in the graph: left for
        # fill_missing_inputs.
        self.assertEqual(incoming(heater, "supplyWaterTemperature"), [])

    def test_command_sensor_is_driven_by_its_controller(self):
        """The historised command point becomes controller-driven, so the
        loop closes through it exactly as the damper command does."""
        # A command point yields two sensors, as the damper commands do: a
        # historised leaf and the controller-driven one the loop runs
        # through.
        sensors = [
            s
            for s in self.model.get_components_by_class(SensorSystem)
            if s.uuid == "R01_MVV01"
        ]
        driven = [
            s
            for s in sensors
            if [type(d).__name__ for d in incoming(s, "measuredValue")]
            == ["ControllerIdentificationPISystem"]
        ]
        self.assertEqual(len(driven), 1, [s.id for s in sensors])
        # It carries the identification residual (measured vs modelled
        # command); the radiator itself takes the controller's output
        # directly, so the loop does not depend on which sensor the MILP
        # picked.
        heater = self.model.get_components_by_class(SpaceHeaterSystem)[0]
        self.assertEqual(
            [type(x).__name__ for x in incoming(heater, "waterFlowRate")], ["ValveSystem"]
        )


if __name__ == "__main__":
    unittest.main()
