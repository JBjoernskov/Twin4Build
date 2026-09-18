"""The rewire pins the gate ACTIVE in both modes: a controller identified
with its gate (Stage 1, ``mode="train"``) is simulated with it (Stage 2,
``mode="simulate"``).  Regression: ``simulate`` used to pin ``alpha_gate``
to 0 (gate bypassed), so VAV dampers identified on a flow-setpoint schedule
opened all weekend in the closed-loop physics simulation."""

# Standard library imports
import unittest

# Third party imports
import torch

# Local application imports
import twin4build as tb
import twin4build.utils.types as tps
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.controller.controller_identification.pi_loop_rewire import (
    _pin_frozen_cits_state,
)

tb._IS_TESTING = True


class TestGatePin(unittest.TestCase):
    def _cits(self):
        cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=1, n_on_off_signals=1)
        cits.n_actuators = 1
        cits._build_components()
        getattr(cits, "alpha_gate_0").set(torch.tensor(0.3, dtype=tps.float_dtype()))
        return cits

    def test_gate_active_in_both_modes(self):
        for mode in ("train", "simulate"):
            cits = self._cits()
            _pin_frozen_cits_state([cits], mode=mode)
            self.assertAlmostEqual(float(cits._get_alpha_gate(0).reshape(-1)[0]), 1.0, msg=mode)

    def test_gate_bypassed_when_no_slot_discriminates(self):
        cits = self._cits()
        _pin_frozen_cits_state([cits], mode="simulate", gate_active={"cits": False})
        self.assertAlmostEqual(float(cits._get_alpha_gate(0).reshape(-1)[0]), 0.0)
        cits = self._cits()
        _pin_frozen_cits_state([cits], mode="train", gate_active={"other": False})
        self.assertAlmostEqual(float(cits._get_alpha_gate(0).reshape(-1)[0]), 1.0)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            _pin_frozen_cits_state([self._cits()], mode="playback")


if __name__ == "__main__":
    unittest.main()
