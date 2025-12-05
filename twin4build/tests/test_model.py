# Standard library imports
import unittest

# Local application imports
from twin4build.model.model import Model
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.utils.uppath import uppath


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model(id="test_model")

    def test_initialization(self):
        self.assertEqual(self.model.id, "test_model")
        self.assertIsNotNone(self.model.simulation_model)
        self.assertIsNotNone(self.model.semantic_model)

    def test_invalid_id(self):
        with self.assertRaises(AssertionError):
            Model(id="invalid@id")

    def test_add_component(self):
        component = ScheduleSystem(
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
        self.model.add_component(component)
        self.assertIn("test_schedule", self.model.components)
        self.assertEqual(self.model.components["test_schedule"], component)

    def test_add_connection(self):
        schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [20],
                "ruleset_default_value": 0,
            },
            id="schedule",
        )
        damper = DamperTorchSystem(id="damper")

        self.model.add_component(schedule)
        self.model.add_component(damper)

        self.model.add_connection(schedule, damper, "scheduleValue", "damperPosition")

        # Verify connection in simulation model
        # Note: Model.add_connection calls simulation_model.add_connection
        # We might need to check the graph or connections list if available
        # For now, we check if it runs without error and if validate passes later

    def test_load(self):
        schedule = ScheduleSystem(
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
        damper = DamperTorchSystem(id="damper")

        self.model.add_component(schedule)
        self.model.add_component(damper)
        self.model.add_connection(schedule, damper, "scheduleValue", "damperPosition")

        self.model.load()

        self.assertTrue(self.model.is_loaded)
        self.assertTrue(len(self.model.simulation_model.execution_order) > 0)

    def test_model_str(self):
        """Test model string representation."""
        schedule = ScheduleSystem(
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
        self.model.add_component(schedule)

        str_repr = str(self.model)
        self.assertIsNotNone(str_repr)
        self.assertIn("schedule", str_repr)

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        self.assertFalse(self.model.is_loaded)

        schedule = ScheduleSystem(
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
        damper = DamperTorchSystem(id="damper")

        self.model.add_component(schedule)
        self.model.add_component(damper)
        self.model.add_connection(schedule, damper, "scheduleValue", "damperPosition")

        self.model.load()
        self.assertTrue(self.model.is_loaded)

        new_component = DamperTorchSystem(id="damper2")

        self.model.add_connection(
            schedule, new_component, "scheduleValue", "damperPosition"
        )

        self.assertFalse(self.model.is_loaded)

    def test_components_property(self):
        """Test components property."""
        self.assertEqual(len(self.model.components), 0)

        schedule = ScheduleSystem(
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
        self.model.add_component(schedule)

        self.assertEqual(len(self.model.components), 1)
        self.assertIn("schedule", self.model.components)

    def test_execution_order_property(self):
        """Test execution_order property after loading."""
        schedule = ScheduleSystem(
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
        damper = DamperTorchSystem(id="damper")

        self.model.add_component(schedule)
        self.model.add_component(damper)
        self.model.add_connection(schedule, damper, "scheduleValue", "damperPosition")

        self.model.load()

        # Execution order should be a list
        exec_order = self.model.execution_order
        self.assertIsNotNone(exec_order)
        self.assertTrue(len(exec_order) > 0)

    def test_get_dir(self):
        """Test get_dir method."""
        dir_path, exists = self.model.get_dir()
        self.assertIsNotNone(dir_path)

    def test_dir_conf_property(self):
        """Test dir_conf property."""
        dir_conf = self.model.dir_conf
        self.assertIsNotNone(dir_conf)
        self.assertIsInstance(dir_conf, list)

    def test_simulation_model_property(self):
        """Test simulation_model property."""
        sim_model = self.model.simulation_model
        self.assertIsNotNone(sim_model)

    def test_semantic_model_property(self):
        """Test semantic_model property."""
        sem_model = self.model.semantic_model
        self.assertIsNotNone(sem_model)


class TestModelAdvanced(unittest.TestCase):
    def setUp(self):
        self.model = Model(id="test_advanced_model")

    def test_set_parameters_from_array(self):
        """Test set_parameters_from_array method."""
        # Local application imports
        from twin4build.systems.damper.damper_torch_system import DamperTorchSystem

        damper = DamperTorchSystem(id="damper", nominalAirFlowRate=1.0)
        self.model.add_component(damper)
        self.model.load()

        # Set parameter from array
        self.model.set_parameters_from_array(
            values=[0.8],
            components=[damper],
            parameter_names=["nominalAirFlowRate"],
            normalized=[False],
            save_original=True,
        )

        # Check parameter was set
        self.assertAlmostEqual(damper.nominalAirFlowRate.get().item(), 0.8, places=2)

    def test_restore_parameters(self):
        """Test restore_parameters method."""
        # Local application imports
        from twin4build.systems.damper.damper_torch_system import DamperTorchSystem

        damper = DamperTorchSystem(id="damper", nominalAirFlowRate=1.0)
        self.model.add_component(damper)
        self.model.load()

        # Save original and set new value
        self.model.set_parameters_from_array(
            values=[0.5],
            components=[damper],
            parameter_names=["nominalAirFlowRate"],
            normalized=[False],
            save_original=True,
        )

        # Restore
        self.model.restore_parameters(keep_values=False)


if __name__ == "__main__":
    unittest.main()
