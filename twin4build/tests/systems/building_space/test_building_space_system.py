# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.building_space.building_space_system import (
    BuildingSpaceSystem,
)

twin4build._IS_TESTING = True


class TestBuildingSpaceTorchSystem(unittest.TestCase):
    def setUp(self):
        self.space = BuildingSpaceSystem(
            id="test_space",
            C_wall=1000000.0,
            C_air=10000.0,
            C_boundary=500000.0,
            R_out=0.01,
            R_in=0.02,
            R_boundary=0.03,
            f_wall=0.5,
            f_air=0.3,
            Q_occ_gain=100.0,
            CO2_occ_gain=0.004,
            CO2_start=400.0,
            airVolume=100.0,
            T_wall_start=20.0,
            T_air_start=20.0,
            T_int_start=20.0,
            T_boundary_start=18.0,
        )

    def test_initialization(self):
        """Test building space system initialization."""
        self.assertIsNotNone(self.space)
        self.assertEqual(self.space.id, "test_space")

    def test_do_step(self):
        """Test building space system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.space.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.space.input["outdoorTemperature"].set(torch.tensor([5.0]), i_t=0)
        self.space.input["globalIrradiation"].set(torch.tensor([200.0]), i_t=0)
        self.space.input["supplyAirFlowRate"].set(torch.tensor([0.05]), i_t=0)
        self.space.input["exhaustAirFlowRate"].set(torch.tensor([0.02]), i_t=0)
        self.space.input["supplyAirTemperature"].set(torch.tensor([22.0]), i_t=0)
        self.space.input["numberOfPeople"].set(torch.tensor([3.0]), i_t=0)
        self.space.input["heatGain"].set(torch.tensor([0.0]), i_t=0)
        self.space.input["outdoorCO2"].set(torch.tensor([400.0]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.space.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs
        temp = self.space.output["indoorTemperature"].get()
        co2 = self.space.output["indoorCO2"].get()

        self.assertIsNotNone(temp)
        self.assertIsNotNone(co2)

    def test_do_step_batch(self):
        """Test building space system do_step method with batch size > 1."""
        space_batch = BuildingSpaceSystem(
            id="test_space_batch",
            C_wall=1000000.0,
            C_air=10000.0,
            C_boundary=500000.0,
            R_out=0.01,
            R_in=0.02,
            R_boundary=0.03,
            f_wall=0.5,
            f_air=0.3,
            Q_occ_gain=100.0,
            CO2_occ_gain=0.004,
            CO2_start=400.0,
            airVolume=100.0,
            T_wall_start=20.0,
            T_air_start=20.0,
            T_int_start=20.0,
            T_boundary_start=18.0,
        )

        batch_size = 2

        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        ] * batch_size
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)] * batch_size
        step_size = [600] * batch_size
        space_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 2
        space_batch.input["outdoorTemperature"].set(torch.tensor([5.0, 8.0]), i_t=0)
        space_batch.input["globalIrradiation"].set(torch.tensor([200.0, 250.0]), i_t=0)
        space_batch.input["supplyAirFlowRate"].set(torch.tensor([0.05, 0.06]), i_t=0)
        space_batch.input["exhaustAirFlowRate"].set(torch.tensor([0.02, 0.03]), i_t=0)
        space_batch.input["supplyAirTemperature"].set(torch.tensor([22.0, 23.0]), i_t=0)
        space_batch.input["numberOfPeople"].set(torch.tensor([3.0, 5.0]), i_t=0)
        space_batch.input["heatGain"].set(torch.tensor([0.0, 0.0]), i_t=0)
        space_batch.input["outdoorCO2"].set(torch.tensor([400.0, 410.0]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        space_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        temp = space_batch.output["indoorTemperature"].get()
        co2 = space_batch.output["indoorCO2"].get()

        self.assertIsNotNone(temp)
        self.assertIsNotNone(co2)
        self.assertEqual(temp.shape[0], batch_size)
        self.assertEqual(co2.shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()
