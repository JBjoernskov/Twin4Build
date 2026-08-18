"""Tests for the CUDA-graph path on the collocation exact Hessian (issue #122).

Two things can go wrong in that change and neither would raise:

1. **Packing order.** The Hessian values must be emitted in exactly the order
   ``hess_rows``/``hess_cols`` were built: ``Btt[triu]``, then per segment
   ``Bty[g].ravel()`` followed by ``Byy[g][triu]``.  The refactor replaced a
   360-iteration numpy loop with a vectorised device-side ``cat`` so the packing
   could live inside the graph.  Get the order wrong and IPOPT receives a
   correct-looking Hessian with its entries permuted -- it would converge
   badly rather than fail, which is the worst kind of bug.

2. **Replay fidelity.** A captured graph must reproduce eager exactly.

The packing test runs anywhere.  The runner tests exercise the CPU fallback,
which is the path CI takes; the capture path itself needs CUDA and is skipped
without it.
"""

# Standard library imports
import os
import unittest

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build

twin4build._IS_TESTING = True

from twin4build.estimator._cuda_graph import CudaGraphRunner, cuda_graphs_enabled


def _pack_reference(Btt, Bty, Byy, iu_t, iu_y, n_seg):
    """The ORIGINAL numpy packing, kept here as the reference to match."""
    Btt_np = Btt.cpu().numpy()
    Bty_np = Bty.cpu().numpy()
    Byy_np = Byy.cpu().numpy()
    vals = [Btt_np[iu_t]]
    for g in range(n_seg):
        vals.append(Bty_np[g].ravel())
        vals.append(Byy_np[g][iu_y])
    return np.concatenate(vals)


def _pack_vectorised(Btt, Bty, Byy, iu_t, iu_y, n_seg, n_theta, Da):
    """The replacement, as implemented in _transcription._hess_core."""
    dev = Btt.device
    t0 = torch.as_tensor(iu_t[0], dtype=torch.long, device=dev)
    t1 = torch.as_tensor(iu_t[1], dtype=torch.long, device=dev)
    y0 = torch.as_tensor(iu_y[0], dtype=torch.long, device=dev)
    y1 = torch.as_tensor(iu_y[1], dtype=torch.long, device=dev)
    return torch.cat([
        Btt[t0, t1],
        torch.cat([
            Bty.reshape(n_seg, n_theta * Da),
            Byy[:, y0, y1],
        ], dim=1).reshape(-1),
    ])


class TestHessianPackingOrder(unittest.TestCase):
    """The vectorised packing must reproduce the numpy loop exactly."""

    def test_packing_matches_reference(self):
        for n_theta, Da, n_seg in ((4, 3, 5), (28, 15, 7), (1, 2, 3)):
            with self.subTest(n_theta=n_theta, Da=Da, n_seg=n_seg):
                g = torch.Generator().manual_seed(n_theta * 100 + Da * 10 + n_seg)
                Btt = torch.randn(n_theta, n_theta, dtype=torch.float64, generator=g)
                Bty = torch.randn(n_seg, n_theta, Da, dtype=torch.float64, generator=g)
                Byy = torch.randn(n_seg, Da, Da, dtype=torch.float64, generator=g)
                iu_t = np.triu_indices(n_theta)
                iu_y = np.triu_indices(Da)

                ref = _pack_reference(Btt, Bty, Byy, iu_t, iu_y, n_seg)
                got = _pack_vectorised(
                    Btt, Bty, Byy, iu_t, iu_y, n_seg, n_theta, Da
                ).numpy()

                self.assertEqual(got.shape, ref.shape)
                # Element-for-element, not allclose: this is a reordering, so
                # any difference at all is a wrong permutation.
                np.testing.assert_array_equal(
                    got, ref,
                    "vectorised packing does not match the numpy loop -- IPOPT "
                    "would receive permuted Hessian entries",
                )

    def test_length_matches_the_sparsity_pattern(self):
        """The emitted length must equal the nnz the pattern declares."""
        n_theta, Da, n_seg = 6, 4, 9
        iu_t, iu_y = np.triu_indices(n_theta), np.triu_indices(Da)
        expected = len(iu_t[0]) + n_seg * (n_theta * Da + len(iu_y[0]))
        Btt = torch.zeros(n_theta, n_theta, dtype=torch.float64)
        Bty = torch.zeros(n_seg, n_theta, Da, dtype=torch.float64)
        Byy = torch.zeros(n_seg, Da, Da, dtype=torch.float64)
        got = _pack_vectorised(Btt, Bty, Byy, iu_t, iu_y, n_seg, n_theta, Da)
        self.assertEqual(got.numel(), expected)


class TestCudaGraphRunner(unittest.TestCase):
    """The runner must be transparent: same answers, graph or not."""

    @staticmethod
    def _fn(a, b):
        return (a * 2.0 + b).sum(dim=-1)

    def test_cpu_falls_back_and_is_correct(self):
        """On CPU there is nothing to capture -- it must still compute."""
        r = CudaGraphRunner(self._fn, name="test")
        a = torch.randn(4, 5, dtype=torch.float64)
        b = torch.randn(4, 5, dtype=torch.float64)
        out = r(a=a, b=b)
        torch.testing.assert_close(out, self._fn(a, b))
        self.assertTrue(r.disabled, "CPU inputs should disable capture")
        self.assertFalse(r.captured)

    def test_disabled_runner_still_computes(self):
        r = CudaGraphRunner(self._fn, name="test", enabled=False)
        a = torch.randn(3, 2, dtype=torch.float64)
        b = torch.randn(3, 2, dtype=torch.float64)
        torch.testing.assert_close(r(a=a, b=b), self._fn(a, b))

    def test_env_flag_respected(self):
        prev = os.environ.get("TWIN4BUILD_CUDA_GRAPH")
        os.environ["TWIN4BUILD_CUDA_GRAPH"] = "0"
        try:
            self.assertFalse(cuda_graphs_enabled())
        finally:
            if prev is None:
                os.environ.pop("TWIN4BUILD_CUDA_GRAPH", None)
            else:
                os.environ["TWIN4BUILD_CUDA_GRAPH"] = prev
        self.assertTrue(cuda_graphs_enabled())

    def test_a_failing_function_falls_back_rather_than_raising(self):
        """A function that cannot be captured must degrade, not explode."""
        calls = {"n": 0}

        def hostile(a):
            calls["n"] += 1
            # .item() forces a host sync -- illegal during capture.
            return a * float(a.sum().item())

        r = CudaGraphRunner(hostile, name="hostile")
        a = torch.randn(3, dtype=torch.float64)
        for _ in range(6):
            out = r(a=a)
            torch.testing.assert_close(out, hostile(a))
        self.assertGreater(calls["n"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_capture_replays_bit_identically(self):
        dev = torch.device("cuda")
        r = CudaGraphRunner(self._fn, name="test-cuda", warmup=2)
        a = torch.randn(8, 16, dtype=torch.float64, device=dev)
        b = torch.randn(8, 16, dtype=torch.float64, device=dev)
        for _ in range(4):
            out = r(a=a, b=b)
        self.assertTrue(r.captured, "expected capture after warm-up")
        # Bit-identical: replay runs the recorded kernels.
        self.assertTrue(torch.equal(out, self._fn(a, b)))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_new_input_values_are_reflected_in_the_replay(self):
        """The classic CUDA-graph bug: replaying with stale inputs."""
        dev = torch.device("cuda")
        r = CudaGraphRunner(self._fn, name="test-values", warmup=2)
        a = torch.randn(6, 4, dtype=torch.float64, device=dev)
        b = torch.randn(6, 4, dtype=torch.float64, device=dev)
        for _ in range(4):
            r(a=a, b=b)
        a2 = a + 3.0
        out = r(a=a2, b=b).clone()
        torch.testing.assert_close(out, self._fn(a2, b))


if __name__ == "__main__":
    unittest.main()
