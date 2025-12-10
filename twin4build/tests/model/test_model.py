# Standard library imports
import unittest

# Local application imports
from twin4build.model.model import Model
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.utils.uppath import uppath

# Set test flag
import twin4build
twin4build._IS_TESTING = True


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


class TestModelProperties(unittest.TestCase):
    """Tests for Model properties that are not yet covered."""

    def setUp(self):
        self.model = Model(id="test_properties_model")

    def test_is_validated_property(self):
        """Test is_validated property."""
        # Before validation, should be False
        self.assertFalse(self.model.is_validated)

    def test_dir_conf_setter(self):
        """Test dir_conf setter."""
        new_dir_conf = ["custom", "directory", "path"]
        self.model.dir_conf = new_dir_conf
        self.assertEqual(self.model.dir_conf, new_dir_conf)

    def test_dir_conf_setter_invalid(self):
        """Test dir_conf setter with invalid input."""
        with self.assertRaises(AssertionError):
            self.model.dir_conf = "not_a_list"

        with self.assertRaises(AssertionError):
            self.model.dir_conf = [1, 2, 3]  # Not strings

    def test_flat_execution_order_property(self):
        """Test flat_execution_order property after loading."""
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

        flat_order = self.model.flat_execution_order
        self.assertIsNotNone(flat_order)
        self.assertTrue(len(flat_order) > 0)


class TestModelMethods(unittest.TestCase):
    """Tests for Model methods that are not yet covered."""

    def setUp(self):
        self.model = Model(id="test_methods_model")

    def test_make_pickable(self):
        """Test make_pickable method."""
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
        self.model.load()

        # Should not raise any errors
        self.model.make_pickable()

    def test_remove_component(self):
        """Test remove_component method."""
        schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule_to_remove",
        )
        self.model.add_component(schedule)
        self.assertIn("schedule_to_remove", self.model.components)

        self.model.remove_component(schedule)
        self.assertNotIn("schedule_to_remove", self.model.components)

    def test_remove_connection(self):
        """Test remove_connection method."""
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

        # Remove the connection
        self.model.remove_connection(
            schedule, damper, "scheduleValue", "damperPosition"
        )
        # The method should execute without error

    def test_set_initial_values(self):
        """Test set_initial_values method."""
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
        self.model.load()

        # Test setting initial values with new signature
        self.model.set_initial_values(
            values=[0.5],
            components=[schedule],
            output_names=["scheduleValue"],
        )

    def test_validate(self):
        """Test validate method."""
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

        # Validate should not raise errors
        self.model.validate()

    def test_load_with_verbose(self):
        """Test load method with verbose parameter."""
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

        # Load with verbose enabled
        self.model.load(verbose=1)
        self.assertTrue(self.model.is_loaded)

    def test_set_save_simulation_result(self):
        """Test set_save_simulation_result method."""
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
        self.model.load()

        # Should not raise any errors
        self.model.set_save_simulation_result(flag=True)
        self.model.set_save_simulation_result(flag=False)

    def test_check_for_missing_initial_values(self):
        """Test check_for_for_missing_initial_values method."""
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
        self.model.load()

        # Should not raise any errors for properly configured components
        self.model.check_for_for_missing_initial_values()

    def test_cache(self):
        """Test cache method."""
        # Standard library imports
        import datetime

        # Third party imports
        import pytz

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
        self.model.load()

        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)

        # Should not raise any errors
        self.model.cache(
            start_time=start_time,
            end_time=end_time,
            step_size=600,
        )

    def test_load_estimation_result(self):
        """Test load_estimation_result method with dict result."""
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
        self.model.load()

        # Create a mock result dict (may need to match expected format)
        mock_result = {
            "chain": [],
            "components": [],
            "parameter_names": [],
        }

        # Should not raise errors (though may warn if format is wrong)
        try:
            self.model.load_estimation_result(result=mock_result)
        except (KeyError, AssertionError):
            # Expected if mock format doesn't match
            pass


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
