"""Regression check: the collocation objective must score the SAME signal the
simulator produces.

A pass-through sensor that executes BEFORE its producer in the Gauss-Seidel
order reads the producer's PREVIOUS-step output, so ``do_step`` -- and the
single-shooting objective, which shifts to match (see
``FastSingleShooting.__init__``) -- scores a one-step-lagged signal for that
sensor, while ``F_aug`` returns the current step's.

Collocation originally omitted that shift, which made its objective a
different function from single-shooting's on the lagged sensor: the fit looked
better inside the NLP than the model actually was (measured on the
full-workflow example: CO2 RMSE 9.97 inside the NLP vs 14.51 when the
resulting model was simulated), and a collocation "refinement" of a
single-shooting optimum was not minimizing the same objective it was warm
started from.

The estimator's own post-solve audit already rolls the solution out BOTH ways,
so the invariant is directly assertable: a sequential ``F_aug`` rollout and a
``do_step`` rollout from the same initial state must produce the same signal.

Note this deliberately compares the two ROLLOUTS (``rollout_rmse`` vs
``do_step_rmse``) and not the NLP-internal fit.  The NLP-internal number only
equals a rollout when the continuity defects are satisfied; the data-informed
warm start intentionally plants measured data into the boundary states and so
starts infeasible, which would make an NLP-vs-rollout comparison fail for
reasons having nothing to do with the sensor lag.  The two rollouts are
genuine trajectories either way, which isolates exactly one thing: whether the
composed one-step map scores the same quantity ``do_step`` does.

They come from different code paths (composed map vs object graph), so exact
equality is not expected -- but they must agree far more tightly than the
one-step-offset error this guards against.
"""

# Standard library imports
import datetime
import unittest

# Local application imports
import twin4build as tb

tb._IS_TESTING = True

from twin4build.examples.collocation_comparison import (
    EXAMPLE_START,
    STEP_SIZE,
    example_measurements,
    example_parameters,
    load_model,
)
from twin4build.utils.rgetattr import rgetattr


def _x0_from_model(entry):
    """Re-read a parameter entry's x0 from the model's current (fitted) value.

    The estimator reorders and regroups parameters internally, so ``result_x``
    is not aligned with the input list; reading each group's representative off
    the model is the supported way to chain one estimate into the next.
    """
    comps, attr, _x0, lo, hi = entry[:5]
    comp = comps[0] if isinstance(comps, list) else comps
    v = float(rgetattr(comp, attr).get().reshape(-1)[0])
    eps = 1e-9 * (hi - lo)  # keep x0 strictly inside the bounds
    return (comps, attr, min(max(v, lo + eps), hi - eps), lo, hi, *entry[5:])


class TestCollocationSensorLag(unittest.TestCase):
    """The composed map must score the same signal the object graph produces."""

    #: Relative gap allowed between the composed-map rollout and the do_step
    #: rollout.  Calibrated against measurement on this fixture, NOT guessed:
    #: with the lag shift every sensor agrees to 0.00% (identical to four
    #: decimals); without it the CO2 sensor shows 2.2% while all others stay at
    #: 0.00%.  1e-3 sits an order of magnitude below the defect it must catch
    #: and two above the agreement it must tolerate.
    #:
    #: The size of the gap depends on how well the model fits: a one-step shift
    #: barely moves the RMSE when the residual is large, which is why the
    #: fixture is pre-fitted below.  An earlier version of this test skipped
    #: that and used a 10% threshold -- it passed on the broken code.
    MAX_REL_GAP = 1e-3

    #: SLSQP iterations run before the collocation call.  Required for
    #: sensitivity (see above), not for convergence.
    PREFIT_ITERS = 20

    @classmethod
    def setUpClass(cls):
        model = load_model()
        estimator = tb.Estimator(tb.Simulator(model))
        start = EXAMPLE_START[0]
        end = start + datetime.timedelta(hours=24)
        parameters = example_parameters(model)
        measurements = example_measurements(model)

        # Pre-fit so the residual is small enough for a one-step shift to move
        # the RMSE measurably (see MAX_REL_GAP).  Not about convergence.
        estimator.estimate(
            parameters=parameters,
            measurements=measurements,
            start_time=[start],
            end_time=[end],
            step_size=STEP_SIZE,
            n_warmup=5,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": cls.PREFIT_ITERS, "fast": True},
        )
        # estimate() leaves the model at the fitted values; carry them into the
        # collocation call as its x0.
        parameters = [_x0_from_model(e) for e in parameters]

        # ``maxiter=0`` returns the warm start: no solve, so the test is fast
        # and deterministic.  The assertion compares two ROLLOUTS of whatever
        # point comes back, which is valid regardless of that point's
        # feasibility or optimality -- so no solver progress is needed.
        cls.result = estimator.estimate(
            parameters=parameters,
            measurements=measurements,
            start_time=[start],
            end_time=[end],
            step_size=STEP_SIZE,
            n_warmup=5,
            method=("casadi", "ipopt", "ad", "collocation"),
            options={"maxiter": 0, "early_stopping": False},
        )

    def test_audit_is_reported(self):
        self.assertIn(
            "transcription_audit",
            self.result,
            "collocation result must carry the post-solve audit",
        )
        self.assertTrue(self.result["transcription_audit"]["per_sensor"])

    def test_composed_rollout_matches_do_step_rollout(self):
        """The composed map and the object graph must produce the same signal.

        Compares the audit's *sequential ``F_aug`` rollout* against its
        *``do_step`` rollout*, both started from the same estimated initial
        state.  Both are genuine rollouts, so -- unlike the NLP-internal fit --
        this comparison does not depend on the collocation defects being
        satisfied, and it isolates exactly one thing: whether the composed
        one-step map scores the same quantity ``do_step`` does.

        Without the sensor-lag shift the lagged sensor disagrees grossly
        (measured on the full-workflow example: CO2 7.40 via F_aug vs 11.50
        via do_step) while every other sensor matches to four digits.
        """
        per_sensor = self.result["transcription_audit"]["per_sensor"]
        offenders = []
        for sid, e in per_sensor.items():
            roll, step = float(e["rollout_rmse"]), float(e["do_step_rmse"])
            scale = max(abs(roll), abs(step), 1e-12)
            rel = abs(roll - step) / scale
            if rel > self.MAX_REL_GAP:
                offenders.append(
                    f"{sid}: F_aug rollout={roll:.4f} do_step={step:.4f} rel={rel:.1%}"
                )
        self.assertFalse(
            offenders,
            "the composed one-step map and the object graph disagree on the "
            "SAME trajectory -- the collocation objective is scoring a "
            "different signal than the simulator produces (check the one-step "
            "sensor-lag shift in _transcription.py):\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
