import unittest
from twin4build.model.model import Model
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
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
                "ruleset_default_value": 0
            },
            id="test_schedule"
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
                "ruleset_default_value": 0
            },
            id="schedule"
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
                "ruleset_default_value": 0
            },
            id="schedule"
        )
        damper = DamperTorchSystem(id="damper")
        
        self.model.add_component(schedule)
        self.model.add_component(damper)
        self.model.add_connection(schedule, damper, "scheduleValue", "damperPosition")
        
        self.model.load()
        
        self.assertTrue(self.model.is_loaded)
        self.assertTrue(len(self.model.simulation_model.execution_order) > 0)

if __name__ == '__main__':
    unittest.main()

