# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.damper.damper_system import DamperSystem

twin4build._IS_TESTING = True


class TestDamperTorchSystem(unittest.TestCase):
    def setUp(self):
        self.damper = DamperSystem(id="test_damper", a=1.0, nominalAirFlowRate=0.5)

    def test_initialization(self):
        """Test damper system initialization."""
        self.assertIsNotNone(self.damper)
        self.assertEqual(self.damper.id, "test_damper")

    def test_do_step(self):
        """Test damper system do_step method."""
        # Initialize
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.damper.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set input
        self.damper.input["damperPosition"].set(torch.tensor([0.5]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.damper.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check that output was calculated
        airflow = self.damper.output["airFlowRate"].get()
        self.assertIsNotNone(airflow)
        self.assertGreater(airflow.item(), 0)

    def test_airflow_calculation(self):
        """Test that damper calculates airflow correctly."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.damper.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # At 100% position, should give nominal airflow
        self.damper.input["damperPosition"].set(torch.tensor([1.0]), i_t=0)
        self.damper.do_step(
            second_time=0,
            date_time=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            step_size=600,
            step_index=0,
        )

        airflow = self.damper.output["airFlowRate"].get().item()
        self.assertAlmostEqual(airflow, 0.5, places=2)

    def test_offset_c_is_the_closed_damper_flow(self):
        """``c`` tied (default): zero flow when closed, ``c`` not estimable,
        and it follows a re-estimated ``a``.  Given / ``set_c``: ``a + c``
        is the closed-damper flow (a VAV's minimum flow), ``c`` is
        estimable, the flow never goes negative and full opening still
        gives the nominal flow."""
        kw = dict(
            start_time=[datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)],
            end_time=[datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)],
            step_size=[600],
        )
        when = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)

        def flow(damper, position):
            damper.initialize(**kw)
            damper.input["damperPosition"].set(torch.tensor([position]), i_t=0)
            damper.do_step(second_time=0, date_time=when, step_size=600, step_index=0)
            return damper.output["airFlowRate"].get().item()

        tied = DamperSystem(id="tied", a=1.0, nominalAirFlowRate=0.5)
        self.assertTrue(tied.c_tied)
        self.assertNotIn("c", [e[1] for e in tied.get_estimable_parameters()])
        self.assertAlmostEqual(flow(tied, 0.0), 0.0)
        tied.a = twin4build.utils.types.Parameter(torch.tensor(2.0), requires_grad=False, scaling="log")
        self.assertAlmostEqual(flow(tied, 0.0), 0.0)  # c followed a
        self.assertAlmostEqual(flow(tied, 1.0), 0.5, places=6)

        free = DamperSystem(id="free", a=1.0, nominalAirFlowRate=0.5, c=-0.9)
        self.assertFalse(free.c_tied)
        self.assertIn("c", [e[1] for e in free.get_estimable_parameters()])
        self.assertAlmostEqual(flow(free, 0.0), 0.1, places=6)   # minimum flow a + c
        self.assertAlmostEqual(flow(free, 1.0), 0.5, places=6)   # still the nominal flow

        untied = DamperSystem(id="untied", a=1.0, nominalAirFlowRate=0.5)
        untied.set_c(-1.2)  # below -a: a dead band, never a negative flow
        self.assertFalse(untied.c_tied)
        self.assertAlmostEqual(flow(untied, 0.0), 0.0)
        self.assertAlmostEqual(flow(untied, 1.0), 0.5, places=6)

    def test_do_step_batch(self):
        """Test damper system do_step method with batch size > 1."""
        damper_batch = DamperSystem(
            id="test_damper_batch", a=1.0, nominalAirFlowRate=0.5
        )

        batch_size = 3

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        damper_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        damper_batch.input["damperPosition"].initialize(n_t=1, n_s=batch_size)
        damper_batch.output["damperPosition"].initialize(n_t=1, n_s=batch_size)
        damper_batch.output["airFlowRate"].initialize(n_t=1, n_s=batch_size)

        # Set input with batch size 3
        damper_batch.input["damperPosition"].set(torch.tensor([0.5, 0.7, 0.3]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        damper_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check that output was calculated - verify batch shape consistency
        airflow = damper_batch.output["airFlowRate"].get()
        self.assertIsNotNone(airflow)
        self.assertEqual(
            airflow.shape[0], batch_size
        )  # Output batch matches input batch


if __name__ == "__main__":
    unittest.main()
