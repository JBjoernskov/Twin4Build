# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build

twin4build._IS_TESTING = True

from twin4build.systems.fan_coil_unit.fan_coil_unit_torch_system import (
    FanCoilUnitTorchSystem,
)


class TestFanCoilUnitTorchSystem(unittest.TestCase):
    def setUp(self):
        self.fcu = FanCoilUnitTorchSystem(
            id="test_fcu",
            Q_flow_nominal=2000.0,
            T_w_supply_nominal=60.0,
            T_w_return_nominal=45.0,
            T_air_in_nominal=21.0,
            thermalMassHeatCapacity=50000.0,
            nelements=3,
        )

    def test_initialization(self):
        """Test fan coil unit system initialization."""
        self.assertIsNotNone(self.fcu)
        self.assertEqual(self.fcu.id, "test_fcu")
        self.assertEqual(self.fcu.nelements, 3)

    def test_input_output_ports(self):
        """Test that input and output ports are correctly defined."""
        self.assertIn("supplyWaterTemperature", self.fcu.input)
        self.assertIn("waterFlowRate", self.fcu.input)
        self.assertIn("airFlowRate", self.fcu.input)
        self.assertIn("inletAirTemperature", self.fcu.input)
        self.assertIn("outletWaterTemperature", self.fcu.output)
        self.assertIn("outletAirTemperature", self.fcu.output)
        self.assertIn("Power", self.fcu.output)

    def test_do_step_heating(self):
        """Test FCU in heating mode (hot water, cool air)."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.fcu.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        self.fcu.input["supplyWaterTemperature"].set(torch.tensor([60.0]), i_t=0)
        self.fcu.input["waterFlowRate"].set(torch.tensor([0.1]), i_t=0)
        self.fcu.input["airFlowRate"].set(torch.tensor([0.5]), i_t=0)
        self.fcu.input["inletAirTemperature"].set(torch.tensor([21.0]), i_t=0)

        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.fcu.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        outlet_water_temp = self.fcu.output["outletWaterTemperature"].get()
        outlet_air_temp = self.fcu.output["outletAirTemperature"].get()
        power = self.fcu.output["Power"].get()

        self.assertIsNotNone(outlet_water_temp)
        self.assertIsNotNone(outlet_air_temp)
        self.assertIsNotNone(power)
        # In heating mode, power should be positive (water hotter than air)
        self.assertGreater(power.item(), 0.0)
        # Outlet air should be warmer than inlet
        self.assertGreater(outlet_air_temp.item(), 21.0)

    def test_do_step_cooling(self):
        """Test FCU in cooling mode (chilled water, warm air)."""
        fcu_cool = FanCoilUnitTorchSystem(
            id="test_fcu_cooling",
            Q_flow_nominal=-3000.0,
            T_w_supply_nominal=7.0,
            T_w_return_nominal=12.0,
            T_air_in_nominal=26.0,
            thermalMassHeatCapacity=30000.0,
            nelements=3,
        )

        start_time = [datetime.datetime(2023, 7, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 7, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        fcu_cool.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        fcu_cool.input["supplyWaterTemperature"].set(torch.tensor([7.0]), i_t=0)
        fcu_cool.input["waterFlowRate"].set(torch.tensor([0.15]), i_t=0)
        fcu_cool.input["airFlowRate"].set(torch.tensor([0.6]), i_t=0)
        fcu_cool.input["inletAirTemperature"].set(torch.tensor([26.0]), i_t=0)

        datetime_val = datetime.datetime(2023, 7, 1, 0, 0, 0, tzinfo=tz.UTC)
        fcu_cool.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        power = fcu_cool.output["Power"].get()
        outlet_air_temp = fcu_cool.output["outletAirTemperature"].get()

        # In cooling mode, power should be negative (water colder than air)
        self.assertLess(power.item(), 0.0)
        # Outlet air should be cooler than inlet
        self.assertLess(outlet_air_temp.item(), 26.0)

    def test_do_step_batch(self):
        """Test FCU with batch size > 1."""
        fcu_batch = FanCoilUnitTorchSystem(
            id="test_fcu_batch",
            Q_flow_nominal=2000.0,
            T_w_supply_nominal=60.0,
            T_w_return_nominal=45.0,
            T_air_in_nominal=21.0,
            thermalMassHeatCapacity=50000.0,
            nelements=2,
        )

        batch_size = 3
        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        ] * batch_size
        end_time = [
            datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)
        ] * batch_size
        step_size = [600] * batch_size
        fcu_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        fcu_batch.input["supplyWaterTemperature"].set(
            torch.tensor([60.0, 65.0, 55.0]), i_t=0
        )
        fcu_batch.input["waterFlowRate"].set(
            torch.tensor([0.1, 0.15, 0.08]), i_t=0
        )
        fcu_batch.input["airFlowRate"].set(
            torch.tensor([0.5, 0.6, 0.4]), i_t=0
        )
        fcu_batch.input["inletAirTemperature"].set(
            torch.tensor([21.0, 22.0, 20.0]), i_t=0
        )

        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        fcu_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        outlet_water_temp = fcu_batch.output["outletWaterTemperature"].get()
        outlet_air_temp = fcu_batch.output["outletAirTemperature"].get()
        power = fcu_batch.output["Power"].get()

        self.assertIsNotNone(outlet_water_temp)
        self.assertIsNotNone(outlet_air_temp)
        self.assertIsNotNone(power)
        self.assertEqual(outlet_water_temp.shape[0], batch_size)
        self.assertEqual(outlet_air_temp.shape[0], batch_size)
        self.assertEqual(power.shape[0], batch_size)

    def test_multiple_timesteps(self):
        """Test FCU over multiple simulation steps to verify dynamic behavior."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.fcu.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        powers = []
        for step_idx in range(5):
            self.fcu.input["supplyWaterTemperature"].set(
                torch.tensor([60.0]), i_t=step_idx
            )
            self.fcu.input["waterFlowRate"].set(torch.tensor([0.1]), i_t=step_idx)
            self.fcu.input["airFlowRate"].set(torch.tensor([0.5]), i_t=step_idx)
            self.fcu.input["inletAirTemperature"].set(
                torch.tensor([21.0]), i_t=step_idx
            )

            datetime_val = datetime.datetime(
                2023, 1, 1, 0, step_idx * 10, 0, tzinfo=tz.UTC
            )
            self.fcu.do_step(
                second_time=step_idx * 600,
                date_time=datetime_val,
                step_size=step_size,
                step_index=step_idx,
            )
            powers.append(self.fcu.output["Power"].get().item())

        # With constant inputs and hot water, power should be positive throughout
        for p in powers:
            self.assertGreater(p, 0.0)

    def test_ua_initialization(self):
        """Test that UA initialization produces a reasonable value."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.fcu.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        ua_val = self.fcu.UA.get().item()
        self.assertGreater(ua_val, 0.0)

    def test_no_ua_initialization(self):
        """Test that UA is preserved when initialize_UA is False."""
        fcu_no_init = FanCoilUnitTorchSystem(
            id="test_fcu_no_ua_init",
            Q_flow_nominal=2000.0,
            T_w_supply_nominal=60.0,
            T_w_return_nominal=45.0,
            T_air_in_nominal=21.0,
            thermalMassHeatCapacity=50000.0,
            nelements=2,
            initialize_UA=False,
        )
        # UA should remain at its placeholder value
        initial_ua = fcu_no_init.UA.get().item()

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        fcu_no_init.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        self.assertAlmostEqual(fcu_no_init.UA.get().item(), initial_ua, places=5)


if __name__ == "__main__":
    unittest.main()
