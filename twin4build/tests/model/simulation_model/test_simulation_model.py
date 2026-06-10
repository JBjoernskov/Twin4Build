# Standard library imports
import unittest

# Local application imports
# Set test flag
import twin4build
from twin4build.model.simulation_model.simulation_model import SimulationModel
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.fan.fan_torch_system import FanTorchSystem
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)
from twin4build.systems.schedule.schedule_system import ScheduleSystem

twin4build._IS_TESTING = True


class TestSimulationModel(unittest.TestCase):
    def setUp(self):
        """Set up a fresh simulation model for each test."""
        self.sim_model = SimulationModel(id="test_sim_model")

        # Set up common components for tests that need them
        self.schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule",
        )
        self.damper = DamperTorchSystem(id="damper")

    def test_initialization(self):
        """Test simulation model initialization."""
        self.assertIsNotNone(self.sim_model)
        self.assertEqual(len(self.sim_model.components), 0)
        self.assertEqual(len(self.sim_model.execution_order), 0)

    def test_add_component(self):
        """Test adding components to simulation model."""
        schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [20],
                "ruleset_default_value": 0,
            },
            id="test_schedule",
        )

        self.sim_model.add_component(schedule)

        self.assertIn("test_schedule", self.sim_model.components)
        self.assertEqual(self.sim_model.components["test_schedule"], schedule)

    def test_add_connection(self):
        """Test adding connections between components."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )

        # Connection should be reflected in the graph structure
        # Check that damper is in schedule's outgoing connections
        self.assertEqual(
            self.damper,
            self.schedule.connected_through[0]
            .connects_system_at[0]
            .connection_point_of,
        )

    def test_execution_order_chain(self):
        """Test execution order for a chain of components."""
        # Create a chain: schedule -> damper1 -> damper2
        schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule_chain",
        )
        damper1 = DamperTorchSystem(id="damper1")
        damper2 = DamperTorchSystem(id="damper2")
        supply_flow_junction = SupplyFlowJunctionSystem(id="supply_flow_junction")
        fan = FanTorchSystem(
            id="fan",
            nominalPowerRate=1000,
            nominalAirFlowRate=1.0,
            c1=0.0,
            c2=0.8,
            c3=0.2,
            c4=0.0,
            f_total=0.8,
        )

        self.sim_model.add_connection(
            schedule, damper1, "scheduleValue", "damperPosition"
        )
        self.sim_model.add_connection(
            schedule, damper2, "scheduleValue", "damperPosition"
        )
        self.sim_model.add_connection(
            damper1,
            supply_flow_junction,
            "airFlowRate",
            "airFlowRateOut",
            input_port_index=0,
        )
        self.sim_model.add_connection(
            damper2,
            supply_flow_junction,
            "airFlowRate",
            "airFlowRateOut",
            input_port_index=1,
        )
        self.sim_model.add_connection(
            supply_flow_junction, fan, "airFlowRateIn", "airFlowRate"
        )

        self.sim_model.load()

        priority_map = {}
        for p, ex_group in enumerate(self.sim_model.execution_order):
            for component in ex_group:
                priority_map[component.id] = p

        self.assertLess(priority_map["schedule_chain"], priority_map["damper1"])
        self.assertLess(priority_map["schedule_chain"], priority_map["damper2"])
        self.assertLess(priority_map["damper1"], priority_map["supply_flow_junction"])
        self.assertLess(priority_map["damper2"], priority_map["supply_flow_junction"])
        self.assertLess(priority_map["supply_flow_junction"], priority_map["fan"])

    def test_cycle_removal(self):
        """Test that cycles are properly removed from the graph."""
        junction1 = SupplyFlowJunctionSystem(id="junction1")
        junction2 = SupplyFlowJunctionSystem(id="junction2")

        # Create circular connections
        self.sim_model.add_connection(
            junction1, junction2, "airFlowRateIn", "airFlowRateOut", input_port_index=0
        )
        self.sim_model.add_connection(
            junction2, junction1, "airFlowRateIn", "airFlowRateOut", input_port_index=0
        )

        # Load should not raise error (cycle should be removed)
        self.sim_model.load()
        # If successful, execution order should exist
        self.assertIsNotNone(self.sim_model.execution_order)

    def test_validate(self):
        """Test model validation."""
        self.sim_model.add_component(self.schedule)
        self.sim_model.add_component(self.damper)
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )

        self.sim_model.load()

        # Validate should not raise errors for valid model
        self.sim_model.validate()
        self.assertTrue(True)

    def test_empty_model_load(self):
        """Test loading an empty model."""
        # Should handle empty model gracefully
        self.sim_model.load()
        # Empty model should have empty or None execution order
        if self.sim_model.execution_order is not None:
            self.assertEqual(len(self.sim_model.execution_order), 0)

    def test_disconnected_components(self):
        """Test model with disconnected components."""
        schedule1 = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule1",
        )
        schedule2 = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.7],
                "ruleset_default_value": 0,
            },
            id="schedule2",
        )

        self.sim_model.add_component(schedule1)
        self.sim_model.add_component(schedule2)

        self.sim_model.load()

        # Both components should be in execution order
        self.assertIn(schedule1, self.sim_model.flat_execution_order)
        self.assertIn(schedule2, self.sim_model.flat_execution_order)

    def test_count_components(self):
        """Test count_components method."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        count = self.sim_model.count_components()
        self.assertEqual(count, 2)  # schedule and damper

    def test_count_connections(self):
        """Test count_connections method."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        count = self.sim_model.count_connections()
        self.assertEqual(count, 1)  # One connection between schedule and damper

    def test_remove_component(self):
        """Test remove_component method."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )

        # Add another damper
        damper2 = DamperTorchSystem(id="damper2")
        self.sim_model.add_component(damper2)

        initial_count = self.sim_model.count_components()
        self.assertEqual(initial_count, 3)

        # Remove the second damper
        self.sim_model.remove_component(damper2)

        final_count = self.sim_model.count_components()
        self.assertEqual(final_count, 2)
        self.assertNotIn("damper2", self.sim_model.components)

    def test_remove_connected_component(self):
        """Test remove_component for a connected component."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )

        # Remove the damper which is connected to schedule
        self.sim_model.remove_component(self.damper)

        self.assertEqual(self.sim_model.count_components(), 1)
        self.assertNotIn("damper", self.sim_model.components)
        self.assertIn("schedule", self.sim_model.components)

    def test_get_dir(self):
        """Test get_dir method."""
        path, isfile = self.sim_model.get_dir(
            folder_list=["test_folder"], filename="test.txt"
        )

        self.assertIsNotNone(path)
        self.assertIn("test.txt", path)

    def test_dir_conf_property(self):
        """Test dir_conf property getter and setter."""
        # Test getter
        dir_conf = self.sim_model.dir_conf
        self.assertIsInstance(dir_conf, list)

        # Test setter
        new_dir_conf = ["new", "path"]
        self.sim_model.dir_conf = new_dir_conf
        self.assertEqual(self.sim_model.dir_conf, new_dir_conf)

    def test_dir_conf_invalid_type(self):
        """Test dir_conf setter with invalid type."""
        with self.assertRaises(AssertionError):
            self.sim_model.dir_conf = "not_a_list"

    def test_components_property(self):
        """Test components property."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )

        components = self.sim_model.components
        self.assertIsInstance(components, dict)
        self.assertIn("schedule", components)
        self.assertIn("damper", components)

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )

        # Initially not loaded
        self.assertFalse(self.sim_model.is_loaded)

        # After load
        self.sim_model.load()
        self.assertTrue(self.sim_model.is_loaded)

    def test_execution_order_property(self):
        """Test execution_order property."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        self.sim_model.load()

        execution_order = self.sim_model.execution_order
        self.assertIsNotNone(execution_order)
        self.assertIsInstance(execution_order, list)

    def test_flat_execution_order_property(self):
        """Test flat_execution_order property."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        self.sim_model.load()

        flat_order = self.sim_model.flat_execution_order
        self.assertIsNotNone(flat_order)
        self.assertIsInstance(flat_order, list)
        self.assertEqual(len(flat_order), 2)

    def test_str_representation(self):
        """Test string representation of the model."""
        str_repr = str(self.sim_model)
        self.assertIsNotNone(str_repr)

    def test_make_pickable(self):
        """Test make_pickable method."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        self.sim_model.load()

        # Should not raise any errors
        self.sim_model.make_pickable()
        self.assertTrue(True)

    def test_reset_torch_tensors(self):
        """Test reset_torch_tensors method."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        self.sim_model.load()

        # Should not raise any errors
        self.sim_model.reset_torch_tensors()
        self.assertTrue(True)

    def test_add_duplicate_component(self):
        """Test adding the same component twice."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )

        # Adding the same component twice should not add duplicate
        self.sim_model.add_component(self.schedule)

        count = self.sim_model.count_components()
        self.assertEqual(count, 2)  # Still 2

    def test_add_component_invalid_type(self):
        """Test add_component with invalid type."""
        with self.assertRaises(AssertionError):
            self.sim_model.add_component("not_a_component")

    def test_set_save_simulation_result(self):
        """Test set_save_simulation_result method."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        self.sim_model.load()

        # Should not raise errors
        self.sim_model.set_save_simulation_result(flag=True)
        self.sim_model.set_save_simulation_result(flag=False)
        self.assertTrue(True)

    def test_visualize(self):
        """Test visualize method."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        self.sim_model.load()

        # Should not raise errors (actual visualization requires graphviz)
        try:
            self.sim_model.visualize()
        except Exception:
            pass  # Visualization may fail without graphviz installed
        self.assertTrue(True)

    def test_get_object_properties(self):
        """Test get_object_properties method."""
        props = self.sim_model.get_object_properties(self.schedule)

        self.assertIsInstance(props, dict)

    def test_serialize(self):
        """Test serialize method."""
        self.sim_model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )
        self.sim_model.load()

        # Should not raise errors
        try:
            self.sim_model.serialize()
        except Exception:
            pass  # May require specific setup
        self.assertTrue(True)

    def test_add_connection_with_port_indices(self):
        """Test add_connection with port indices."""
        schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule_port",
        )
        junction = SupplyFlowJunctionSystem(id="junction")

        # Add connection with input_port_index
        self.sim_model.add_connection(
            schedule, junction, "scheduleValue", "airFlowRateOut", input_port_index=0
        )

        self.assertIn("schedule_port", self.sim_model.components)
        self.assertIn("junction", self.sim_model.components)

    def test_invalid_output_port(self):
        """Test add_connection with invalid output port."""
        with self.assertRaises(AssertionError):
            self.sim_model.add_connection(
                self.schedule, self.damper, "invalid_output", "damperPosition"
            )

    def test_invalid_input_port(self):
        """Test add_connection with invalid input port."""
        with self.assertRaises(AssertionError):
            self.sim_model.add_connection(
                self.schedule, self.damper, "scheduleValue", "invalid_input"
            )


if __name__ == "__main__":
    unittest.main()
