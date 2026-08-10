# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.space_heater.space_heater_system import (
    SpaceHeaterSystem,
)

twin4build._IS_TESTING = True


class TestSpaceHeaterTorchSystem(unittest.TestCase):
    def setUp(self):
        self.heater = SpaceHeaterSystem(
            id="test_heater",
            Q_flow_nominal_sh=1000.0,
            T_a_nominal_sh=60.0,
            T_b_nominal_sh=40.0,
            TAir_nominal_sh=20.0,
            thermalMassHeatCapacity=5000.0,
            nelements=2,
        )

    def test_initialization(self):
        """Test space heater system initialization."""
        self.assertIsNotNone(self.heater)
        self.assertEqual(self.heater.id, "test_heater")
        self.assertEqual(self.heater.nelements, 2)

    def test_initialize_UA_sets_physical_value(self):
        """initialize_UA must store a physical UA readable via Parameter.get()."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.heater.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        ua = float(self.heater.UA.get().reshape(-1)[0])
        # Rough LMTD-style scale: Q / (T_b - T_air) ≈ 50; fsolve is nearby.
        self.assertGreater(ua, 10.0)
        self.assertLess(ua, 200.0)
        # Must not be the placeholder (10) accidentally denormalized as 10*(UA).
        self.assertNotAlmostEqual(ua, 10.0, places=3)

    def test_do_step(self):
        """Test space heater system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.heater.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.heater.input["supplyWaterTemperature"].set(torch.tensor([60.0]), i_t=0)
        self.heater.input["waterFlowRate"].set(torch.tensor([0.1]), i_t=0)
        self.heater.input["indoorTemperature"].set(torch.tensor([20.0]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.heater.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs
        outlet_temp = self.heater.output["outletWaterTemperature"].get()
        radiator_power = self.heater.output["Power"].get()

        self.assertIsNotNone(outlet_temp)
        self.assertIsNotNone(radiator_power)

    def test_do_step_batch(self):
        """Test space heater system do_step method with batch size > 1."""
        heater_batch = SpaceHeaterSystem(
            id="test_heater_batch",
            Q_flow_nominal_sh=1000.0,
            T_a_nominal_sh=60.0,
            T_b_nominal_sh=40.0,
            TAir_nominal_sh=20.0,
            thermalMassHeatCapacity=5000.0,
            nelements=2,
        )

        batch_size = 3

        # Batch size is determined by the length of start_time/end_time/step_size lists
        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        ] * batch_size
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)] * batch_size
        step_size = [600] * batch_size
        heater_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 3 (inputs already initialized with correct batch_size by initialize())
        heater_batch.input["supplyWaterTemperature"].set(
            torch.tensor([60.0, 65.0, 55.0]), i_t=0
        )
        heater_batch.input["waterFlowRate"].set(torch.tensor([0.1, 0.15, 0.08]), i_t=0)
        heater_batch.input["indoorTemperature"].set(
            torch.tensor([20.0, 22.0, 18.0]), i_t=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        heater_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        outlet_temp = heater_batch.output["outletWaterTemperature"].get()
        radiator_power = heater_batch.output["Power"].get()

        self.assertIsNotNone(outlet_temp)
        self.assertIsNotNone(radiator_power)
        self.assertEqual(
            outlet_temp.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(
            radiator_power.shape[0], batch_size
        )  # Output batch matches input batch


if __name__ == "__main__":
    unittest.main()
