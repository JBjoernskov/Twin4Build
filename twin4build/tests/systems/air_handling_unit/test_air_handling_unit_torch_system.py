# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
from twin4build.systems.air_handling_unit.air_handling_unit_torch_system import (
    AirHandlingUnitTorchSystem,
)

# Set test flag
import twin4build

twin4build._IS_TESTING = True


class TestAirHandlingUnitTorchSystem(unittest.TestCase):
    def setUp(self):
        damper_kwargs = {"a": 1.0, "nominalAirFlowRate": 0.5}
        heat_recovery_kwargs = {
            "eps_75_h": 0.7,
            "eps_100_h": 0.75,
            "eps_75_c": 0.65,
            "eps_100_c": 0.7,
            "primaryAirFlowRateMax": 1.0,
            "secondaryAirFlowRateMax": 1.0,
        }
        fan_kwargs = {
            "nominalPowerRate": 1000.0,
            "nominalAirFlowRate": 1,
            "c1": 0,
            "c2": 0.2,
            "c3": 0.8,
            "c4": 0,
            "f_total": 1.0,
        }
        self.ahu = AirHandlingUnitTorchSystem(
            id="test_ahu",
            damper_kwargs=damper_kwargs,
            heat_recovery_kwargs=heat_recovery_kwargs,
            supply_fan_kwargs=fan_kwargs,
            exhaust_fan_kwargs=fan_kwargs,
            n_branches=2,
        )

    def test_initialization(self):
        """AHU initializes without error."""
        self.assertIsNotNone(self.ahu)
        self.assertEqual(self.ahu.n_branches, 2)

    def test_do_step(self):
        """AHU runs one step and produces expected outputs."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.ahu.initialize(start_time=start_time, end_time=end_time, step_size=step_size)

        # Set vector inputs for two branches
        self.ahu.input["supplyDamperPosition"].set(
            torch.tensor([[1, 1]]), i_t=0
        )
        self.ahu.input["exhaustDamperPosition"].set(
            torch.tensor([[0.9, 0.9]]), i_t=0
        )
        self.ahu.input["exhaustTemperature"].set(
            torch.tensor([[22.0, 22.0]]), i_t=0
        )
        # Scalars
        self.ahu.input["supplyAirTemperatureSetpoint"].set(
            torch.tensor([18.0]), i_t=0
        )
        self.ahu.input["outdoorAirTemperature"].set(
            torch.tensor([10.0]), i_t=0
        )

        self.ahu.do_step(
            second_time=0,
            date_time=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            step_size=step_size,
            step_index=0,
        )

        supply_flow = self.ahu.output["supplyAirFlowRate"].get()
        supply_temp = self.ahu.output["supplyAirTemperature"].get()
        supply_fan_power = self.ahu.output["supplyFanPower"].get()
        exhaust_fan_power = self.ahu.output["exhaustFanPower"].get()

        print(supply_flow, supply_temp, supply_fan_power, exhaust_fan_power)

        self.assertIsNotNone(supply_flow)
        self.assertTrue(torch.all(supply_flow > 0.0))
        self.assertIsNotNone(supply_temp)
        self.assertTrue(torch.isfinite(supply_temp).all())
        self.assertIsNotNone(supply_fan_power)
        self.assertGreaterEqual(supply_fan_power.item(), 0.0)
        self.assertIsNotNone(exhaust_fan_power)
        self.assertGreaterEqual(exhaust_fan_power.item(), 0.0)


if __name__ == "__main__":
    unittest.main()


