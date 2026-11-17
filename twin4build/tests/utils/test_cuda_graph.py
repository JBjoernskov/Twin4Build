"""Focused lifecycle and replay tests for direct CUDA Graph wrappers."""

# Standard library imports
import gc
import os
import unittest
import weakref
from unittest import mock

# Third party imports
import torch

# Local application imports
import twin4build.utils._cuda_graph as cuda_graph
from twin4build.estimator import _collocation


class _FakeGraph:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class TestCudaGraphWrappers(unittest.TestCase):
    def test_bound_method_does_not_retain_owner(self):
        class Owner:
            def call(self, value):
                return value

        owner = Owner()
        wrapper = cuda_graph.CudaGraphCallable(owner.call)
        owner_ref = weakref.ref(owner)
        del owner
        gc.collect()

        self.assertIsNone(owner_ref())
        with self.assertRaises(ReferenceError):
            wrapper.fn
        wrapper.close()
        wrapper.close()

    def test_close_defers_reset_during_capture_and_is_idempotent(self):
        cuda_graph._deferred_graphs.clear()
        wrapper = cuda_graph.CudaGraphCallable(lambda value: value)
        graph = _FakeGraph()
        wrapper.graph = graph

        with mock.patch.object(cuda_graph, "_is_capturing", return_value=True):
            wrapper.close()
            wrapper.close()
        self.assertEqual(graph.reset_calls, 0)
        self.assertIn(graph, cuda_graph._deferred_graphs)

        with (
            mock.patch.object(cuda_graph, "_is_capturing", return_value=False),
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            cuda_graph._drain_deferred_graphs()
        self.assertEqual(graph.reset_calls, 1)
        synchronize.assert_called_once()
        self.assertFalse(cuda_graph._deferred_graphs)

    def test_direct_bundle_never_falls_back_after_capture_invalidation(self):
        calls = {"eager": 0}

        def eager(value):
            calls["eager"] += 1
            return value

        class InvalidGraph:
            def __init__(self, fn):
                pass

            def __call__(self, *inputs):
                raise cuda_graph.CudaGraphCaptureInvalidated("invalidated")

            def close(self):
                pass

        bundle = _collocation._DirectCudaGraph(eager, True, "test")
        with (
            mock.patch.object(_collocation, "CudaGraphCallable", InvalidGraph),
            self.assertRaises(cuda_graph.CudaGraphCaptureInvalidated),
        ):
            bundle(torch.tensor(1.0))
        self.assertEqual(calls["eager"], 0)

    def test_direct_bundle_ordinary_setup_error_falls_back_once(self):
        calls = {"eager": 0}

        def eager(value):
            calls["eager"] += 1
            return value + 1

        class IncompatibleGraph:
            def __init__(self, fn):
                pass

            def __call__(self, *inputs):
                raise RuntimeError("capture unsupported")

            def close(self):
                pass

        bundle = _collocation._DirectCudaGraph(eager, True, "test")
        with mock.patch.object(_collocation, "CudaGraphCallable", IncompatibleGraph):
            self.assertEqual(bundle(torch.tensor(1.0)), 2.0)
        self.assertEqual(calls["eager"], 1)
        self.assertFalse(bundle.stats["enabled"])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_default_capture_is_one_warmup_and_one_record(self):
        calls = {"n": 0}

        def fn(value):
            calls["n"] += 1
            return value * 2

        wrapper = cuda_graph.CudaGraphCallable(fn)
        x = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.float64)
        with wrapper:
            torch.testing.assert_close(wrapper(x), x * 2)
        self.assertEqual(calls["n"], 2)
        wrapper_probe = cuda_graph.CudaGraphCallable(fn, verify_input_tracking=True)
        with wrapper_probe:
            wrapper_probe(x)
        self.assertEqual(calls["n"], 5)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_capture_replay_parity_and_owned_snapshot(self):
        wrapper = cuda_graph.CudaGraphCallable(
            lambda x, scale: (x.sin() * scale, x.square() + scale)
        )
        x1 = torch.linspace(-0.7, 0.9, 17, device="cuda", dtype=torch.float64)
        x2 = x1 + 0.13
        scale1 = torch.tensor(1.4, device="cuda", dtype=torch.float64)
        scale2 = torch.tensor(0.6, device="cuda", dtype=torch.float64)

        with wrapper:
            first = tuple(value.clone() for value in wrapper(x1, scale1))
            second = tuple(value.clone() for value in wrapper(x2, scale2))
            expected = (x2.sin() * scale2, x2.square() + scale2)
            torch.testing.assert_close(second, expected)
            torch.testing.assert_close(first, (x1.sin() * scale1, x1.square() + scale1))
            self.assertEqual(wrapper.capture_count, 1)
            self.assertEqual(wrapper.replay_count, 1)
            wrapper.reset()
            self.assertFalse(wrapper.closed)
            recaptured = tuple(value.clone() for value in wrapper(x1, scale1))
            torch.testing.assert_close(
                recaptured, (x1.sin() * scale1, x1.square() + scale1)
            )
            self.assertEqual(wrapper.capture_count, 2)
        self.assertTrue(wrapper.closed)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_fixed_shape_collocation_bundle_capture_replay_parity(self):
        dtype, device = torch.float64, torch.device("cuda")
        A = torch.tensor([[0.3, -0.2, 0.7], [0.5, 0.4, -0.1]], dtype=dtype, device=device)

        def callbacks(z, sigma, lam):
            az = A @ z
            objective = z.pow(4).sum()
            gradient = 4.0 * z.pow(3)
            defects = az.sin()
            jacobian = az.cos().unsqueeze(1) * A
            hessian = torch.diag(12.0 * sigma * z.square())
            hessian = hessian + A.T @ torch.diag(-lam * az.sin()) @ A
            return torch.cat(
                (
                    objective.reshape(1),
                    gradient,
                    defects,
                    jacobian.reshape(-1),
                    hessian.reshape(-1),
                )
            )

        bundle = _collocation._DirectCudaGraph(callbacks, True, "collocation test")
        z1 = torch.tensor([0.2, -0.4, 0.7], dtype=dtype, device=device)
        z2 = torch.tensor([-0.3, 0.1, 0.9], dtype=dtype, device=device)
        sigma1 = torch.tensor(0.8, dtype=dtype, device=device)
        sigma2 = torch.tensor(1.1, dtype=dtype, device=device)
        lam1 = torch.tensor([0.2, -0.5], dtype=dtype, device=device)
        lam2 = torch.tensor([-0.1, 0.6], dtype=dtype, device=device)

        first = bundle(z1, sigma1, lam1).clone()
        second = bundle(z2, sigma2, lam2).clone()
        torch.testing.assert_close(first, callbacks(z1, sigma1, lam1))
        torch.testing.assert_close(second, callbacks(z2, sigma2, lam2))
        self.assertTrue(bundle.stats["captured"])
        self.assertEqual(bundle.stats["replays"], 1)
        self.assertGreater(bundle.stats["capture_seconds"], 0)
        self.assertGreaterEqual(bundle.stats["replay_seconds"], 0)
        bundle.close()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_capture_records_phases_and_notes_the_failing_phase(self):
        with mock.patch.dict(os.environ, {"T4B_CUDA_GRAPH_SYNC_PHASES": "1"}):
            wrapper = cuda_graph.CudaGraphCallable(lambda x: x * 2.0)
            x = torch.ones(4, device="cuda", dtype=torch.float64)
            wrapper(x)
            self.assertEqual(wrapper.last_phase, "done")
            self.assertEqual(cuda_graph.LAST_PHASE, "capture:done")

            calls = {"n": 0}

            def flaky(x):
                # Warmup passes; the record phase raises, so the note must say "record".
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("boom during record")
                return x * 2.0

            wrapper = cuda_graph.CudaGraphCallable(flaky)
            with self.assertRaisesRegex(RuntimeError, "boom") as info:
                wrapper(x)
            self.assertEqual(wrapper.last_phase, "record")
            notes = getattr(info.exception, "__notes__", [])
            self.assertTrue(any("phase: capture:record" in note for note in notes))
            self.assertEqual(cuda_graph.LAST_PHASE, "capture:record")


class TestReplayParityBound(unittest.TestCase):
    """A replay may differ from the eager run by what reordered sums differ
    by, ``c * n * u * scale``, and by nothing like a wrong capture's error."""

    @staticmethod
    def _values(dtype, numel, seed=0):
        generator = torch.Generator().manual_seed(seed)
        values = torch.rand(numel, generator=generator, dtype=torch.float64)
        return ((values * 2 - 1) * 300).to(dtype)

    @staticmethod
    def _allowed(eager, n):
        u = torch.finfo(eager.dtype).eps / 2
        return cuda_graph.REPLAY_BOUND_FACTOR * n * u * float(eager.abs().max())

    @staticmethod
    def _up(values, steps=1):
        """``values`` moved ``steps`` ulp up."""
        for _ in range(steps):
            values = torch.nextafter(values, torch.full_like(values, float("inf")))
        return values

    def test_largest_dimension_of_the_tensors(self):
        tensors = (torch.zeros(3, 40), [torch.zeros(5), torch.tensor(2.0)], None)
        self.assertEqual(cuda_graph.largest_dimension(*tensors), 40)
        self.assertEqual(cuda_graph.largest_dimension(torch.tensor(1.0)), 1)
        self.assertEqual(cuda_graph.largest_dimension(), 1)

    def test_float32_differences_of_a_few_ulp_pass(self):
        eager = self._values(torch.float32, 1700)
        replay = eager.clone()
        index = torch.arange(11) * 150
        replay[index] = self._up(replay[index], steps=3)
        n = cuda_graph.largest_dimension(eager)
        observed, allowed = cuda_graph.check_replay_matches_eager(
            replay, eager, n=n, label="test"
        )
        self.assertGreater(observed, 0)
        self.assertLess(observed, 1e-3 * allowed)
        self.assertAlmostEqual(allowed, self._allowed(eager, 1700))
        # the fixed tolerance the bound replaces rejects them
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(replay, eager, rtol=1e-9, atol=1e-11)

    def test_float32_sums_in_another_order_pass(self):
        generator = torch.Generator().manual_seed(1)
        terms = torch.randn(64, 500, generator=generator, dtype=torch.float32)
        forward = torch.zeros(64, dtype=torch.float32)
        backward = torch.zeros(64, dtype=torch.float32)
        for j in range(500):
            forward += terms[:, j]
            backward += terms[:, -1 - j]
        self.assertFalse(torch.equal(forward, backward))
        cuda_graph.check_replay_matches_eager(
            backward, forward, n=cuda_graph.largest_dimension(terms), label="test"
        )

    def test_a_wrong_capture_fails(self):
        eager = self._values(torch.float32, 200)
        n = cuda_graph.largest_dimension(eager)
        wrong = {"1e-3 relative": eager * (1 + 1e-3), "stale": eager.roll(1)}
        for name, replay in wrong.items():
            with self.subTest(name), self.assertRaises(AssertionError):
                cuda_graph.check_replay_matches_eager(replay, eager, n=n, label="test")

    def test_float64_stays_strict(self):
        eager = self._values(torch.float64, 200)
        n = cuda_graph.largest_dimension(eager)
        self.assertLess(self._allowed(eager, n) / float(eager.abs().max()), 2e-13)
        replay = eager.clone()
        replay[:5] = self._up(replay[:5])
        cuda_graph.check_replay_matches_eager(replay, eager, n=n, label="test")
        with self.assertRaises(AssertionError):
            cuda_graph.check_replay_matches_eager(
                eager * (1 + 2e-7), eager, n=n, label="test"
            )

    def test_near_zero_outputs_pass_at_the_tensor_scale(self):
        eager = torch.full((100,), 1.0e3, dtype=torch.float32)
        eager[::10] = 1.0e-6
        replay = eager.clone()
        # two ulp of the tensor's largest value, a hundred times these
        # elements' own size: what a sum of terms of size 1e3 that cancels
        # to near zero may give in another order
        ulp = float(self._up(torch.tensor(1.0e3)) - 1.0e3)
        replay[::10] += 2 * ulp
        observed, allowed = cuda_graph.check_replay_matches_eager(
            replay, eager, n=100, label="test"
        )
        self.assertGreater(observed, 100 * 1.0e-6)
        self.assertLess(observed, allowed)

    def test_the_message_names_the_observed_and_the_allowed_difference(self):
        eager = self._values(torch.float32, 200)
        replay = eager.clone()
        replay[17] += 0.5
        observed = float((replay - eager).abs().max())
        allowed = self._allowed(eager, 200)
        with self.assertRaises(AssertionError) as info:
            cuda_graph.check_replay_matches_eager(
                (replay,), (eager,), n=200, label="step: CUDA graph replay"
            )
        message = str(info.exception)
        self.assertIn("step: CUDA graph replay differs from eager output", message)
        self.assertIn(f"max |replay - eager| = {observed:.3e}", message)
        self.assertIn(f"allowed {allowed:.3e}", message)
        self.assertIn("1 of 200 elements over", message)

    def test_a_pass_logs_the_observed_and_the_allowed_difference(self):
        eager = self._values(torch.float32, 200)
        replay = eager.clone()
        replay[3] = self._up(replay[3])
        with mock.patch.object(cuda_graph.LOGGER, "debug") as debug:
            observed, allowed = cuda_graph.check_replay_matches_eager(
                replay, eager, n=200, label="test"
            )
        debug.assert_called_once()
        args = debug.call_args.args
        self.assertIn("test", args)
        self.assertIn(observed, args)
        self.assertIn(allowed, args)
        self.assertGreater(observed, 0)

    def test_nans_infinities_and_exact_outputs(self):
        eager = self._values(torch.float64, 50)
        eager[4] = float("nan")
        eager[5] = float("inf")
        counts = torch.arange(6)
        cuda_graph.check_replay_matches_eager(
            (eager.clone(), counts.clone()), (eager, counts), n=50, label="test"
        )
        moved_nan = eager.clone()
        moved_nan[4], moved_nan[6] = 1.0, float("nan")
        lost_inf = eager.clone()
        lost_inf[5] = 1.0e300
        other_count = counts.clone()
        other_count[2] += 1
        wrong = {
            "NaN moved": (moved_nan, counts),
            "infinity lost": (lost_inf, counts),
            "integer differs": (eager, other_count),
            "output missing": (eager,),
        }
        for name, replay in wrong.items():
            with self.subTest(name), self.assertRaises(AssertionError):
                cuda_graph.check_replay_matches_eager(
                    replay, (eager, counts), n=50, label="test"
                )


if __name__ == "__main__":
    unittest.main()
