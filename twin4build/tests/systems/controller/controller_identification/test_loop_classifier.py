"""Synthetic-loop unit tests for :mod:`loop_classifier`.

Each test simulates an exact discrete PI controller (forward-Euler
integrator) and verifies that :func:`score_pair` recovers ``kp`` and
``Ti`` to the precision predicted by the joint regression analysis.

Edge cases covered:
    * Perfect P controller (``Ti = inf``)
    * PI with several values of ``Ti``
    * Reverse-acting loop (``slope < 0``)
    * Wrong (sensor, setpoint) pair (uncorrelated noise)
    * Schedule-only actuator (no closed-loop response)
    * Saturated actuator regime (most samples at the bounds)
    * Constant setpoint (``var(de) ~ 0``)
"""

# Standard library imports
import math
import unittest

# Third party imports
import numpy as np

# Local application imports
import twin4build

twin4build._IS_TESTING = True

from twin4build.systems.controller.controller_identification.loop_classifier import (  # noqa: E402
    confidence_label,
    derive_actuator_seeds,
    derive_actuator_seeds_gmm,
    derive_gate_seeds_from_on_mask,
    score_pair,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _simulate_pi_loop(
    *,
    h: float,
    n: int,
    kp: float,
    Ti: float,
    setpoint: np.ndarray,
    measurement: np.ndarray,
    is_reverse: bool = False,
    output_min: float = 0.0,
    output_max: float = 1.0,
    rng: np.random.Generator = None,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Run a discrete PI controller forward in time over the given signals.

    Returns the actuator timeseries ``u`` of length ``n``.
    """
    e = setpoint - measurement
    if is_reverse:
        e = -e

    integ = 0.0
    u = np.empty(n, dtype=np.float64)
    for k in range(n):
        if math.isfinite(Ti) and Ti > 0:
            integ += (h / Ti) * e[k]
        u_raw = kp * (e[k] + integ)
        if rng is not None and noise_std > 0.0:
            u_raw += rng.normal(0.0, noise_std)
        u[k] = max(output_min, min(output_max, u_raw))
    return u


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestScorePair(unittest.TestCase):
    """Tests for :func:`score_pair`."""

    def setUp(self) -> None:
        self.h = 600.0   # 10-minute steps, matching bldg1 config
        self.rng = np.random.default_rng(42)

    def _make_unsaturated_signals(self, *, n: int, e_amp: float = 0.05):
        """Build (sp, meas) with a smooth, zero-mean error signal.

        ``e = sp - meas`` follows a sum of bounded sinusoids so that
        ``cumsum(e)`` stays bounded too -- avoiding the integrator-windup
        artefacts that arise when a synthetic loop has no plant feedback.
        """
        t = np.arange(n)
        sp = np.full(n, 21.0)
        # Zero-mean periodic error so cumsum(e) is bounded.
        e = e_amp * (
            np.sin(2 * np.pi * t / 240)
            + 0.5 * np.cos(2 * np.pi * t / 90)
            + 0.3 * np.sin(2 * np.pi * t / 50)
        )
        meas = sp - e
        return sp, meas

    def _construct_pi_actuator(
        self,
        *,
        sp: np.ndarray,
        meas: np.ndarray,
        kp: float,
        Ti: float,
        h: float,
        u_center: float = 0.5,
    ) -> np.ndarray:
        """Construct ``u`` directly from the regression equation
        ``Δu = kp·Δe + (kp·h/Ti)·e_mid``.

        This is the *exact* generative model that ``score_pair`` tries to
        identify -- so the recovered coefficients should match within
        floating-point tolerance.  Unlike forward-simulating a PI loop
        without a plant, this avoids unbounded integrator-windup artefacts.

        ``u_center`` is the value the actuator hovers around: we shift the
        cumulative-sum trajectory so its mean equals ``u_center``, which
        keeps the resulting ``u`` away from the [0, 1] saturation bounds
        regardless of where the integrator drifts to.
        """
        e = sp - meas
        de = np.diff(e)
        e_mid = 0.5 * (e[:-1] + e[1:])
        du = kp * de + (kp * h / Ti) * e_mid
        u = np.empty_like(e)
        u[0] = 0.0
        u[1:] = np.cumsum(du)
        u = u - u.mean() + u_center
        return u

    def test_perfect_p_controller_recovers_kp(self) -> None:
        """Pure P (``Ti = inf``): joint slope on de is exactly kp.

        Uses noise-free smooth signals so the regression sees the model
        exactly and recovers kp to machine precision.
        """
        n = 1000
        sp, meas = self._make_unsaturated_signals(n=n, e_amp=0.02)
        # Bias the actuator into the middle of [0, 1] so saturation never bites.
        kp_true = 0.5
        offset = 0.5
        e = sp - meas
        u = np.clip(kp_true * e + offset, 0.0, 1.0)
        score = score_pair(u, sp, meas, self.h, n_min=50)
        self.assertIsNone(score.reason)
        self.assertAlmostEqual(score.slope, kp_true, places=5)
        self.assertAlmostEqual(score.kp, kp_true, places=5)
        self.assertGreater(score.r2, 0.99)

    def test_pi_recovers_kp_and_Ti_for_several_Ti(self) -> None:
        """PI with various Ti: kp/Ti recovered exactly from the generative model.

        Builds ``u`` directly from the regression equation ``Δu = kp·Δe +
        (kp·h/Ti)·e_mid``; this is the exact statistical model the joint
        OLS fit identifies, so recovery should be at floating-point
        precision modulo any saturation clipping.
        """
        n = 2000
        sp, meas = self._make_unsaturated_signals(n=n, e_amp=0.05)
        for kp_true, Ti_true in [(0.4, 1800.0), (0.6, 3600.0), (0.3, 900.0)]:
            with self.subTest(kp=kp_true, Ti=Ti_true):
                u = self._construct_pi_actuator(
                    sp=sp, meas=meas, kp=kp_true, Ti=Ti_true, h=self.h,
                )
                score = score_pair(u, sp, meas, self.h, n_min=50)
                self.assertIsNone(score.reason)
                self.assertGreater(score.r2, 0.95)
                # kp -> floating-point precision (the regression is exact).
                self.assertAlmostEqual(
                    score.kp, kp_true, delta=1e-6,
                    msg=f"kp recovered={score.kp} expected={kp_true}",
                )
                self.assertIsNotNone(score.Ti)
                self.assertAlmostEqual(
                    score.Ti, Ti_true, delta=1e-3 * Ti_true,
                    msg=f"Ti recovered={score.Ti} expected={Ti_true}",
                )

    def test_reverse_acting_loop_has_negative_slope(self) -> None:
        """Reverse-acting -> slope < 0, ``kp`` (= |slope|) > 0."""
        n = 1500
        sp, meas = self._make_unsaturated_signals(n=n, e_amp=0.05)
        # Reverse-acting: flip the sign of the constructed actuator response.
        # Equivalent to negating kp in the generative model.
        u_dir = self._construct_pi_actuator(
            sp=sp, meas=meas, kp=0.5, Ti=1800.0, h=self.h,
        )
        u = 1.0 - u_dir
        score = score_pair(u, sp, meas, self.h, n_min=50)
        self.assertIsNone(score.reason)
        self.assertLess(score.slope, 0.0)
        self.assertGreater(score.kp, 0.0)
        self.assertGreater(score.r2, 0.95)

    def test_wrong_pair_low_r2(self) -> None:
        """Independent (random) actuator and sensor: R^2 close to zero."""
        n = 1500
        sp = np.full(n, 21.0)
        sp[200:600] = 22.0
        meas = sp - 0.5 * self.rng.standard_normal(n)
        # u is independent of (sp, meas): white noise with realistic levels.
        u = np.clip(
            0.4 + 0.1 * self.rng.standard_normal(n), 0.05, 0.95,
        )
        score = score_pair(u, sp, meas, self.h, n_min=50)
        # The fit may not have an explicit "reason" (it is a real OLS), but
        # R^2 should be tiny.
        self.assertLess(score.r2, 0.10)

    def test_schedule_only_actuator_low_r2(self) -> None:
        """Actuator follows a schedule independent of (sp, meas) -> low R^2."""
        n = 1500
        sp = np.full(n, 21.0)
        meas = sp - 0.5 * self.rng.standard_normal(n)
        # Diurnal schedule, repeated every 144 steps (1 day at h=600s).
        t = np.arange(n)
        u = 0.5 + 0.4 * np.sin(2 * np.pi * t / 144)
        u = np.clip(u, 0.05, 0.95)
        score = score_pair(u, sp, meas, self.h, n_min=50)
        self.assertLess(score.r2, 0.10)

    def test_saturated_actuator_rejected(self) -> None:
        """Mostly-saturated u: too few active samples -> reason set."""
        n = 200
        sp = np.full(n, 21.0)
        meas = sp.copy()
        u = np.full(n, 0.99)  # fully saturated
        score = score_pair(u, sp, meas, self.h, n_min=50, sat_lo=0.03, sat_hi=0.97)
        self.assertEqual(score.reason, "too_few_active_samples")
        self.assertEqual(score.r2, 0.0)

    def test_constant_setpoint_and_constant_measurement(self) -> None:
        """When both signals are constant, var(de) ~ 0 -> reason set."""
        n = 500
        sp = np.full(n, 21.0)
        meas = np.full(n, 21.0)
        u = np.linspace(0.2, 0.8, n)  # u varies, but e is identically zero
        score = score_pair(u, sp, meas, self.h, n_min=50)
        self.assertEqual(score.reason, "constant_setpoint")

    def test_too_few_samples(self) -> None:
        """Below n_min after masking: reason set."""
        sp = np.array([21.0, 21.5, 22.0, 21.7])
        meas = np.array([20.5, 21.0, 21.7, 21.5])
        u = np.array([0.4, 0.5, 0.6, 0.5])
        score = score_pair(u, sp, meas, self.h, n_min=50)
        self.assertEqual(score.reason, "too_few_active_samples")

    def test_nan_handling(self) -> None:
        """NaNs in any signal are masked out and counted toward n_min."""
        n = 2000
        sp, meas = self._make_unsaturated_signals(n=n, e_amp=0.05)
        u = self._construct_pi_actuator(
            sp=sp, meas=meas, kp=0.4, Ti=1800.0, h=self.h,
        )
        # Inject NaNs into ~10% of u.
        u[::10] = np.nan
        score = score_pair(u, sp, meas, self.h, n_min=50)
        self.assertIsNone(score.reason)
        self.assertGreater(score.n_active, 1500)
        self.assertGreater(score.r2, 0.95)


class TestDeriveActuatorSeeds(unittest.TestCase):
    """Tests for :func:`derive_actuator_seeds`."""

    def test_typical_actuator(self) -> None:
        """Output range derived from the 1st/99th percentile."""
        rng = np.random.default_rng(1)
        u = np.clip(0.5 + 0.2 * rng.standard_normal(2000), 0.0, 1.0)
        seeds = derive_actuator_seeds(u, p_lo=1, p_hi=99, cushion=0.02)
        self.assertGreaterEqual(seeds.output_min_x0, 0.0)
        self.assertLessEqual(seeds.output_max_x0, 1.0)
        self.assertLess(seeds.output_min_x0, seeds.output_max_x0)
        self.assertGreater(seeds.default_output_x0, 0.3)
        self.assertLess(seeds.default_output_x0, 0.7)

    def test_parked_low_default(self) -> None:
        """Mostly-zero actuator -> default_output near 0 (e.g. reheat valve)."""
        u = np.zeros(2000)
        u[100:300] = 0.5  # brief active periods
        seeds = derive_actuator_seeds(u)
        self.assertLess(seeds.default_output_x0, 0.1)

    def test_parked_high_default(self) -> None:
        """Mostly-one actuator -> default_output near 1 (e.g. damper)."""
        u = np.ones(2000)
        u[100:300] = 0.4
        seeds = derive_actuator_seeds(u)
        self.assertGreater(seeds.default_output_x0, 0.9)

    def test_all_nan_falls_back_to_safe_defaults(self) -> None:
        u = np.full(100, np.nan)
        seeds = derive_actuator_seeds(u)
        self.assertEqual(seeds.output_min_x0, 0.0)
        self.assertEqual(seeds.output_max_x0, 1.0)
        self.assertEqual(seeds.default_output_x0, 0.5)


class TestDeriveActuatorSeedsGmm(unittest.TestCase):
    """Tests for :func:`derive_actuator_seeds_gmm`.

    Each case constructs a synthetic actuator timeseries with a known
    ground-truth (kind, default_output, active range) and verifies the
    GMM decomposition recovers it.  The non-trivial property under test
    is that the *active-mode percentile* seeds (``output_min_x0`` /
    ``output_max_x0``) are not contaminated by off-mode mass -- the
    motivating failure mode of the legacy full-trajectory extractor.
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)

    def _bimodal(
        self,
        *,
        n: int,
        off_mean: float,
        off_std: float,
        active_mean: float,
        active_std: float,
        active_frac: float,
    ) -> np.ndarray:
        """Sample a 2-mixture trajectory.

        ``active_frac`` fraction of samples come from N(active_mean, active_std);
        the remainder from N(off_mean, off_std).  Result clipped to ``[0, 1]``.
        """
        n_active = int(round(active_frac * n))
        n_off = n - n_active
        off = self.rng.normal(off_mean, off_std, size=n_off)
        act = self.rng.normal(active_mean, active_std, size=n_active)
        u = np.concatenate([off, act])
        self.rng.shuffle(u)
        return np.clip(u, 0.0, 1.0)

    def test_damper_off_at_one(self) -> None:
        """Damper-like actuator: off mode at ~1, active mode at ~0.55."""
        u = self._bimodal(
            n=2000, off_mean=0.97, off_std=0.01,
            active_mean=0.55, active_std=0.07, active_frac=0.4,
        )
        seeds = derive_actuator_seeds_gmm(u)
        self.assertEqual(seeds.kind, "damper")
        self.assertAlmostEqual(seeds.default_output_x0, 1.0, places=5)
        self.assertGreater(seeds.bimodality, 1.0)
        # Active-mode percentiles should be inside the modulating range,
        # NOT pulled toward the upper bound by the off-mode mass.
        self.assertLess(seeds.output_max_x0, 0.85)
        self.assertGreater(seeds.output_min_x0, 0.25)
        self.assertLess(seeds.output_min_x0, seeds.output_max_x0)
        # on_mask should pick up most of the active samples.
        self.assertGreater(int(seeds.on_mask.sum()), 600)
        self.assertLess(int(seeds.on_mask.sum()), 1100)

    def test_reheat_off_at_zero(self) -> None:
        """Reheat-like actuator: off mode at ~0, active mode at ~0.4."""
        u = self._bimodal(
            n=2000, off_mean=0.02, off_std=0.01,
            active_mean=0.4, active_std=0.08, active_frac=0.3,
        )
        seeds = derive_actuator_seeds_gmm(u)
        self.assertEqual(seeds.kind, "reheat")
        self.assertAlmostEqual(seeds.default_output_x0, 0.0, places=5)
        self.assertGreater(seeds.bimodality, 1.0)
        self.assertLess(seeds.output_max_x0, 0.7)
        self.assertGreater(seeds.output_min_x0, 0.1)
        # Cluster means in the seeds reflect actual cluster, not bound.
        self.assertLess(seeds.off_mean, 0.1)
        self.assertGreater(seeds.active_mean, 0.25)

    def test_always_on_modulating(self) -> None:
        """Bimodal but neither mode at a bound: always-on, gate inactive."""
        u = self._bimodal(
            n=2000, off_mean=0.30, off_std=0.04,
            active_mean=0.65, active_std=0.04, active_frac=0.5,
        )
        seeds = derive_actuator_seeds_gmm(u)
        self.assertEqual(seeds.kind, "always_on")
        # default_output falls back to active_mean (not a bound).
        self.assertGreater(seeds.default_output_x0, 0.4)
        self.assertLess(seeds.default_output_x0, 0.8)

    def test_unimodal_not_misclassified_as_damper_or_reheat(self) -> None:
        """Single-cluster trajectory centred away from bounds: never classifies
        as ``damper`` or ``reheat``.

        The GMM may still fit two pseudo-clusters on a unimodal distribution
        (the EM has no prior over component count), but neither pseudo-cluster
        will sit near a saturation bound, so ``kind`` falls through to
        ``always_on`` (gate inactive, percentile fallback) or ``ambiguous``
        (low bimodality fallback).  Either is safe -- the failure mode this
        test guards against is mistakenly tagging it ``damper`` (default=1)
        or ``reheat`` (default=0), which would lock the actuator at a bound.
        """
        u = np.clip(0.5 + 0.05 * self.rng.standard_normal(2000), 0.0, 1.0)
        seeds = derive_actuator_seeds_gmm(u)
        self.assertIn(seeds.kind, ("ambiguous", "always_on"))
        # Critical safety property: default_output must not be slammed to a bound.
        self.assertGreater(seeds.default_output_x0, 0.1)
        self.assertLess(seeds.default_output_x0, 0.9)
        # Fallback values must still be inside [0, 1].
        self.assertGreaterEqual(seeds.output_min_x0, 0.0)
        self.assertLessEqual(seeds.output_max_x0, 1.0)
        self.assertLess(seeds.output_min_x0, seeds.output_max_x0)

    def test_constant_signal_falls_back(self) -> None:
        """Constant actuator: GMM fails -> safe fallback values returned."""
        u = np.full(2000, 0.7)
        seeds = derive_actuator_seeds_gmm(u)
        # No exception raised; some kind of "ambiguous" decomposition.
        self.assertIn(seeds.kind, ("ambiguous", "always_off"))
        self.assertIsNotNone(seeds.reason)

    def test_all_nan_safe_defaults(self) -> None:
        """All-NaN: returns safe defaults without raising."""
        u = np.full(100, np.nan)
        seeds = derive_actuator_seeds_gmm(u)
        self.assertEqual(seeds.kind, "ambiguous")
        self.assertEqual(seeds.output_min_x0, 0.0)
        self.assertEqual(seeds.output_max_x0, 1.0)
        self.assertEqual(seeds.default_output_x0, 0.5)

    def test_on_mask_aligned_with_input(self) -> None:
        """on_mask length matches input length, including NaN positions."""
        u = self._bimodal(
            n=1000, off_mean=0.02, off_std=0.01,
            active_mean=0.5, active_std=0.08, active_frac=0.4,
        )
        u[::25] = np.nan  # inject NaNs
        seeds = derive_actuator_seeds_gmm(u)
        self.assertEqual(seeds.on_mask.shape, (1000,))
        # NaN positions must be flagged off (not used for the regression).
        nan_positions = np.where(~np.isfinite(u))[0]
        self.assertFalse(seeds.on_mask[nan_positions].any())

    def test_active_seeds_independent_of_off_mass(self) -> None:
        """output_min/max from active mode should not change when more
        off-mode samples are added.

        This is the core "active-mode percentile" property: the legacy
        full-trajectory extractor would shift output_min toward the bound
        as off-mode samples grow, biasing the saturation seed.
        """
        active_block = self.rng.normal(0.45, 0.05, size=400)
        off_short = self.rng.normal(0.99, 0.01, size=200)
        off_long = self.rng.normal(0.99, 0.01, size=2000)

        u_short = np.clip(np.concatenate([off_short, active_block]), 0.0, 1.0)
        u_long = np.clip(np.concatenate([off_long, active_block]), 0.0, 1.0)
        self.rng.shuffle(u_short)
        self.rng.shuffle(u_long)

        s_short = derive_actuator_seeds_gmm(u_short)
        s_long = derive_actuator_seeds_gmm(u_long)
        self.assertEqual(s_short.kind, "damper")
        self.assertEqual(s_long.kind, "damper")
        # output_min/max are computed from the SAME active block in both,
        # so the seeds should agree to within Monte-Carlo noise.
        self.assertAlmostEqual(
            s_short.output_min_x0, s_long.output_min_x0, delta=0.1,
        )
        self.assertAlmostEqual(
            s_short.output_max_x0, s_long.output_max_x0, delta=0.1,
        )


class TestDeriveGateSeedsFromOnMask(unittest.TestCase):
    """Tests for :func:`derive_gate_seeds_from_on_mask`.

    The function takes the GMM's ``on_mask`` (sample-level "PI is
    active" label) plus the per-slot ``onOffSignal`` candidates and
    ranks each slot by how well its values predict ``on_mask``.  The
    contract under test:

    1. A schedule signal that is high during ``on_mask=True`` -> AUC
       near 1, gamma_gate concentrated on that slot, high confidence.
    2. An anti-polarity signal (high during ``off``) -> AUC near 0,
       still selected (its ``|auc - 0.5|`` is large), with
       ``winner_polarity = -1``.
    3. Pure noise -> low AUC discrimination, ``confidence="low"``,
       gamma_gate uniform, threshold/band fall back to neutral.
    4. Multiple informative slots -> gamma_gate softmax keeps the
       second best non-trivial; the highest still wins.
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(123)
        self.n = 1000
        # Build an on/off pattern: 14h on / 10h off cycles.
        cycle = np.tile(
            np.concatenate([np.ones(140), np.zeros(100)]),
            self.n // 240 + 2,
        )[: self.n].astype(bool)
        self.on_mask = cycle

    def _binary_signal(self, off_val=15.0, on_val=22.0, noise=0.05):
        s = np.where(self.on_mask, on_val, off_val).astype(np.float64)
        s += self.rng.normal(0.0, noise, size=self.n)
        return s

    def test_clear_winner_high_auc(self) -> None:
        """One schedule slot perfectly predicts on_mask."""
        sched = self._binary_signal(off_val=15.0, on_val=22.0)
        # A second uninformative slot to make sure gamma is normalised.
        noise = self.rng.normal(20.0, 1.0, size=self.n)
        oo_min = np.array([15.0, 15.0])
        oo_max = np.array([22.0, 25.0])
        seeds = derive_gate_seeds_from_on_mask(
            self.on_mask, [sched, noise], oo_min, oo_max,
        )
        self.assertEqual(seeds.winner_slot, 0)
        self.assertEqual(seeds.winner_polarity, +1)
        self.assertGreater(seeds.auc_per_slot[0], 0.9)
        self.assertLess(abs(seeds.auc_per_slot[1] - 0.5), 0.15)
        self.assertEqual(seeds.confidence, "high")
        # gamma should be near one-hot on slot 0.
        self.assertGreater(seeds.gamma_gate_x0[0], 0.9)
        self.assertLess(seeds.gamma_gate_x0[1], 0.1)
        # threshold/band should bracket the on-state value (~1 normalised).
        # active range ~ [1, 1] (binary) widened by margin.
        self.assertGreater(seeds.gate_threshold_x0, 0.5)
        self.assertLess(seeds.gate_threshold_x0, 1.0)
        self.assertGreater(seeds.gate_band_x0, 0.1)

    def test_anti_polarity_signal_wins(self) -> None:
        """Signal high during off (low during on) still ranks top."""
        # Build signal: high (1.0) during off, low (0.0) during on.
        anti = np.where(self.on_mask, 0.0, 1.0).astype(np.float64)
        anti += self.rng.normal(0.0, 0.05, size=self.n)
        oo_min = np.array([0.0])
        oo_max = np.array([1.0])
        seeds = derive_gate_seeds_from_on_mask(
            self.on_mask, [anti], oo_min, oo_max,
        )
        # AUC ~ 0 because high signal predicts off (label=False).
        self.assertLess(seeds.auc_per_slot[0], 0.1)
        # But the discriminative score max(auc, 1-auc) is high, so it wins.
        self.assertEqual(seeds.confidence, "high")
        self.assertEqual(seeds.winner_polarity, -1)

    def test_pure_noise_falls_back_to_neutral(self) -> None:
        """All slots uninformative -> uniform gamma, neutral threshold/band."""
        noise1 = self.rng.normal(0.0, 1.0, size=self.n)
        noise2 = self.rng.normal(5.0, 1.0, size=self.n)
        oo_min = np.array([-3.0, 2.0])
        oo_max = np.array([3.0, 8.0])
        seeds = derive_gate_seeds_from_on_mask(
            self.on_mask, [noise1, noise2], oo_min, oo_max,
        )
        self.assertEqual(seeds.confidence, "low")
        self.assertIsNotNone(seeds.reason)
        # Uniform gamma.
        np.testing.assert_allclose(
            seeds.gamma_gate_x0, np.array([0.5, 0.5]), atol=1e-9,
        )
        # Neutral fallback values.
        self.assertAlmostEqual(seeds.gate_threshold_x0, 0.5, places=5)
        self.assertAlmostEqual(seeds.gate_band_x0, 1.0, places=5)

    def test_two_informative_slots_winner_clear(self) -> None:
        """When two slots are both informative, the better one still
        wins decisively because softmax is sharp by construction."""
        strong = self._binary_signal(off_val=15.0, on_val=22.0, noise=0.02)
        weak = self._binary_signal(off_val=15.0, on_val=22.0, noise=2.0)
        oo_min = np.array([15.0, 15.0])
        oo_max = np.array([22.0, 22.0])
        seeds = derive_gate_seeds_from_on_mask(
            self.on_mask, [strong, weak], oo_min, oo_max,
        )
        self.assertEqual(seeds.winner_slot, 0)
        self.assertGreater(seeds.gamma_gate_x0[0], seeds.gamma_gate_x0[1])
        # Both AUCs above 0.5; both are informative.
        self.assertGreater(seeds.auc_per_slot[0], 0.95)
        self.assertGreater(seeds.auc_per_slot[1], 0.7)

    def test_degenerate_mask_falls_back(self) -> None:
        """All-True or all-False on_mask -> low confidence, neutral seeds."""
        sched = self._binary_signal()
        oo_min = np.array([15.0])
        oo_max = np.array([22.0])
        for mask in (np.ones(self.n, dtype=bool), np.zeros(self.n, dtype=bool)):
            seeds = derive_gate_seeds_from_on_mask(
                mask, [sched], oo_min, oo_max,
            )
            self.assertEqual(seeds.confidence, "low")

    def test_threshold_band_brackets_active_range(self) -> None:
        """For a continuous schedule with active range ~[0.6, 1.0], the
        seeded ``[threshold, threshold+band]`` should bracket it."""
        # Continuous schedule: ramp up to 1 during on, drop to 0 during off.
        s = np.where(self.on_mask, 1.0, 0.0).astype(np.float64)
        # Add some smoothing so quantiles are non-degenerate.
        s += self.rng.normal(0.0, 0.05, size=self.n)
        oo_min = np.array([0.0])
        oo_max = np.array([1.0])
        seeds = derive_gate_seeds_from_on_mask(
            self.on_mask, [s], oo_min, oo_max,
        )
        self.assertEqual(seeds.confidence, "high")
        # Active samples are near 1.0; threshold should be < 1 and band > 0.
        upper = seeds.gate_threshold_x0 + seeds.gate_band_x0
        self.assertGreater(upper, 0.95)
        self.assertLess(seeds.gate_threshold_x0, 0.95)


class TestConfidenceLabel(unittest.TestCase):
    """Tests for :func:`confidence_label`."""

    def test_high(self) -> None:
        self.assertEqual(confidence_label(0.6, 200), "high")

    def test_medium(self) -> None:
        self.assertEqual(confidence_label(0.3, 200), "medium")

    def test_low(self) -> None:
        self.assertEqual(confidence_label(0.05, 200), "low")

    def test_failed_too_few_samples(self) -> None:
        self.assertEqual(confidence_label(0.9, 5), "failed")

    def test_failed_nan(self) -> None:
        self.assertEqual(confidence_label(float("nan"), 200), "failed")


if __name__ == "__main__":
    unittest.main()
