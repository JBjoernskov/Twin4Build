# Standard library imports
import os
import unittest
import datetime
from dateutil import tz

# Local application imports
import twin4build as tb


class TestExamples(unittest.TestCase):
    def test_fcn(self):
        """
        This test creates a model using a function and simulates it.
        It includes the supply flow junction system because this model has a Vector input.
        This setup was used to reproduce https://github.com/JBjoernskov/Twin4Build/issues/76. 
        """

        def fcn(self):
            # 1. Create a schedule
            position_schedule = tb.ScheduleSystem(
                weekDayRulesetDict = {
                    "ruleset_default_value": 0,
                    "ruleset_start_minute": [0,0,0,0,0,0,0],
                    "ruleset_end_minute": [0,0,0,0,0,0,0],
                    "ruleset_start_hour": [6,7,8,12,14,16,18],
                    "ruleset_end_hour": [7,8,12,14,16,18,22],
                    "ruleset_value": [0,0.1,1,0,0,0.5,0.7]
                },
                id="PositionSchedule"
            )

            # 2. Create a damper
            damper = tb.DamperTorchSystem(
                nominalAirFlowRate=1.6,
                a=5,
                id="Damper"
            )

            supply_flow_junction = tb.SupplyFlowJunctionSystem(id="SupplyFlowJunction")

            self.add_connection(position_schedule, damper, "scheduleValue", "damperPosition")
            self.add_connection(damper, supply_flow_junction, "airFlowRate", "airFlowRateOut")


        model = tb.Model(id="test_model_fcn")
        model.load(fcn=fcn)

        simulator = tb.Simulator(model)
        step_size = 600  # Seconds
        start_time = datetime.datetime(year=2024, month=1, day=10, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        end_time = datetime.datetime(year=2024, month=1, day=12, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

        # Simulate the model
        simulator.simulate(step_size=step_size,
                            start_time=start_time,
                            end_time=end_time)


if __name__ == "__main__":
    unittest.main()
