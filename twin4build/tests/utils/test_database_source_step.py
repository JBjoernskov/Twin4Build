"""Regression test: a sensor that reads the database starts a run.

A data source System's run start hands its reader the step size as a float
(``System.run_timestamps`` gives ``_scalar_step_size``), and
``load_from_database`` asserts an int (``step_size must be an integer``), so
every ``SensorSystem`` with a ``DatabaseSource`` raised at its first run
start.  ``DatabaseSource.window`` passes whole seconds as an int.
"""

# Standard library imports
import datetime
import unittest
from unittest import mock

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
import twin4build as tb
from twin4build.utils.data_loaders import sources

START = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
END = START + datetime.timedelta(hours=1)
STEP = 600


def _loader(**kwargs):
    """``load_from_database``'s contract: an int step, the window's values on its grid."""
    assert isinstance(kwargs["step_size"], int), "step_size must be an integer"
    index = pd.date_range(kwargs["start_time"], kwargs["end_time"], freq=f"{kwargs['step_size']}s", inclusive="left")
    return pd.Series(np.arange(len(index), dtype=float), index=index, name="value")


class TestDatabaseSourceStep(unittest.TestCase):
    def test_a_database_sensor_starts_a_run(self):
        model = tb.Model(id="database_sensor")
        sensor = tb.SensorSystem(source=sources.DatabaseSource("point", dbconfig={"db_host": "h"}), id="sensor")
        model.add_component(sensor)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        with mock.patch.object(sources, "load_from_database", side_effect=_loader) as load:
            model.initialize(start_time=[START], end_time=[END], step_size=[STEP])
        self.assertTrue(load.called)
        self.assertTrue(all(isinstance(c.kwargs["step_size"], int) for c in load.call_args_list))
        values = sensor.time_series_input.values.detach().numpy().reshape(-1)
        np.testing.assert_allclose(values[:6], np.arange(6, dtype=float))


if __name__ == "__main__":
    unittest.main()
