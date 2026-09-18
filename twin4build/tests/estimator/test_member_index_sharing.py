"""Element-mapped shared parameters (``member_index``).

A shared group takes its width from its first member; another member may
map its own ``n_c`` elements onto a subset of the group's elements.  The
case: an occupancy's per-VAV dampers are the AHU's per-branch dampers,
``([ahu, occ], "supply_damper.a", x0, lb, ub, "shared", [None, [1, 2]])``.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.utils.occupancy_system import OccupancySystem
from twin4build.tests.systems.building_space.test_vector_flow_ports import build_model

tb._IS_TESTING = True


class TestMemberIndexSharing(unittest.TestCase):
    START = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)

    def setUp(self):
        self.model, self.ahu, self.zones = build_model("member_index_sharing")
        # Initialize so the AHU's dampers carry one element per branch (3).
        sim = tb.Simulator(self.model, execution_mode="object")
        sim.simulate(
            start_time=self.START, end_time=self.START + datetime.timedelta(minutes=30),
            step_size=600, show_progress_bar=False,
        )
        self.estimator = tb.Estimator(sim)
        # A stand-alone occupancy whose two damper slots are branches 1 and 2.
        self.occ = OccupancySystem(id="occ", co2_filename="unused.csv", damper_filename="unused.csv")
        self.occ.supply_damper.expand_to_n_c(2)

    def _process(self, params):
        est = self.estimator
        est._process_parameters_list(est._validate_list_format(params))
        est._x0_norm = np.asarray(est._x0, dtype=float)  # set by estimate() normally
        return est

    def test_mapped_member_reads_and_writes_its_group_elements(self):
        est = self._process([
            ([self.ahu, self.occ], "supply_damper.nominalAirFlowRate", [0.1, 0.2, 0.3], 0.01, 1.0, "shared", [None, [1, 2]]),
            (self.zones["A"], "thermal.C_air", 1e4, 1e3, 1e5),
        ])
        # One theta entry of width 3 for the group, one for C_air.
        self.assertEqual(est._theta_slices, [(0, 1), (1, 4)])
        np.testing.assert_allclose(est._x0, [1e4, 0.1, 0.2, 0.3])
        # The occupancy member gathers elements 1 and 2 of the group.
        theta = np.array([5.0, 0.11, 0.22, 0.33])
        values = est._theta_to_param_values(theta)
        np.testing.assert_allclose(values[1], [0.11, 0.22, 0.33])  # AHU, whole group
        np.testing.assert_allclose(values[2], [0.22, 0.33])  # occupancy, mapped
        # ... and the composer sees the same gather as an index list.
        spec, unique = est._composer_theta_spec()
        selectors = {(c.id, attr): sel for c, attr, sel in spec}
        self.assertEqual(selectors[("ahu", "supply_damper.nominalAirFlowRate")], slice(1, 4))
        self.assertEqual(selectors[("occ", "supply_damper.nominalAirFlowRate")], [2, 3])
        self.assertEqual(len(unique), 2)

    def test_width_mismatch_without_index_is_rejected(self):
        with self.assertRaises(ValueError):
            self._process([
                ([self.ahu, self.occ], "supply_damper.nominalAirFlowRate", 0.1, 0.01, 1.0, "shared"),
            ])

    def test_index_must_match_the_member_width_and_the_group(self):
        with self.assertRaises(ValueError):
            self._process([
                ([self.ahu, self.occ], "supply_damper.nominalAirFlowRate", 0.1, 0.01, 1.0, "shared", [None, [1]]),
            ])
        with self.assertRaises(ValueError):
            self._process([
                ([self.ahu, self.occ], "supply_damper.nominalAirFlowRate", 0.1, 0.01, 1.0, "shared", [None, [2, 3]]),
            ])
        with self.assertRaises(ValueError):
            self._process([
                ([self.ahu, self.occ], "supply_damper.nominalAirFlowRate", 0.1, 0.01, 1.0, "shared", [[0, 1, 2], None]),
            ])


if __name__ == "__main__":
    unittest.main()
