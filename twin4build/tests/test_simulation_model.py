import unittest
from twin4build.model.simulation_model.simulation_model import SimulationModel
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.junction.supply_flow_junction_system import SupplyFlowJunctionSystem
from twin4build.systems.fan.fan_torch_system import FanTorchSystem


class TestSimulationModel(unittest.TestCase):
    def setUp(self):
        """Set up a fresh simulation model for each test."""
        self.sim_model = SimulationModel(id="test_sim_model")

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
                "ruleset_default_value": 0
            },
            id="test_schedule"
        )
        
        self.sim_model.add_component(schedule)
        
        self.assertIn("test_schedule", self.sim_model.components)
        self.assertEqual(self.sim_model.components["test_schedule"], schedule)

    def test_add_connection(self):
        """Test adding connections between components."""
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
        self.sim_model.add_connection(schedule, damper, "scheduleValue", "damperPosition")
        
        # Connection should be reflected in the graph structure
        # Check that damper is in schedule's outgoing connections
        self.assertEqual(damper, schedule.connected_through[0].connects_system_at[0].connection_point_of)

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
                "ruleset_default_value": 0
            },
            id="schedule"
        )
        damper1 = DamperTorchSystem(id="damper1")
        damper2 = DamperTorchSystem(id="damper2")
        supply_flow_junction = SupplyFlowJunctionSystem(id="supply_flow_junction")
        fan = FanTorchSystem(id="fan", nominalPowerRate=1000, nominalAirFlowRate=1.0, c1=0.0, c2=0.8, c3=0.2, c4=0.0, f_total=0.8)
        
        self.sim_model.add_connection(schedule, damper1, "scheduleValue", "damperPosition")
        self.sim_model.add_connection(schedule, damper2, "scheduleValue", "damperPosition")
        self.sim_model.add_connection(damper1, supply_flow_junction, "airFlowRate", "airFlowRateOut", input_port_index=0)
        self.sim_model.add_connection(damper2, supply_flow_junction, "airFlowRate", "airFlowRateOut", input_port_index=1)
        self.sim_model.add_connection(supply_flow_junction, fan, "airFlowRateIn", "airFlowRate")

        
        self.sim_model.load()

        priority_map = {}
        for p, ex_group in enumerate(self.sim_model.execution_order):
            for component in ex_group:
                priority_map[component.id] = p

        self.assertLess(priority_map["schedule"], priority_map["damper1"])
        self.assertLess(priority_map["schedule"], priority_map["damper2"])
        self.assertLess(priority_map["damper1"], priority_map["supply_flow_junction"])
        self.assertLess(priority_map["damper2"], priority_map["supply_flow_junction"])
        self.assertLess(priority_map["supply_flow_junction"], priority_map["fan"])

    def test_cycle_removal(self):
        """Test that cycles are properly removed from the graph."""
        # Create a cycle: component1 -> component2 -> component1
        # Note: This requires components that can form a feedback loop
        # We'll use SupplyFlowJunction which can have circular dependencies
        
        junction1 = SupplyFlowJunctionSystem(id="junction1")
        junction2 = SupplyFlowJunctionSystem(id="junction2")
        
        # Create circular connections
        self.sim_model.add_connection(junction1, junction2, "airFlowRateIn", "airFlowRateOut", input_port_index=0)
        self.sim_model.add_connection(junction2, junction1, "airFlowRateIn", "airFlowRateOut", input_port_index=0)
        
        # Load should not raise error (cycle should be removed)
        self.sim_model.load()
        # If successful, execution order should exist
        self.assertIsNotNone(self.sim_model.execution_order)

    def test_validate(self):
        """Test model validation."""
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
        
        self.sim_model.add_component(schedule)
        self.sim_model.add_component(damper)
        self.sim_model.add_connection(schedule, damper, "scheduleValue", "damperPosition")
        
        self.sim_model.load()
        
        # Validate should not raise errors for valid model
        self.sim_model.validate()
        # If validate method exists and succeeds
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
        # Create two independent components
        schedule1 = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0
            },
            id="schedule1"
        )
        schedule2 = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.7],
                "ruleset_default_value": 0
            },
            id="schedule2"
        )
        
        self.sim_model.add_component(schedule1)
        self.sim_model.add_component(schedule2)
        
        self.sim_model.load()
        
        # Both components should be in execution order
        self.assertIn(schedule1, self.sim_model.flat_execution_order)
        self.assertIn(schedule2, self.sim_model.flat_execution_order)


if __name__ == '__main__':
    unittest.main()

