"""A CUDA-graph capture of the eager step is allowed, and warned about when large.

Capturing the eager step is legitimate where nothing can compile it (a
Windows torch without Triton) and cheap on a small model.  On a large one
the record runs to tens of gigabytes of host memory (a 2000-component,
864-step pass recorded 43 GB), so both capture paths warn before it, with
the size in the message, and point at the compiled step.  No CUDA needed:
the size rule is a pure function and the constructor accepts every
combination.
"""

# Standard library imports
import unittest
import warnings

# Local application imports
import twin4build as tb
from twin4build.utils import _cuda_graph


class TestEagerCaptureWarning(unittest.TestCase):
    def test_constructor_accepts_capture_without_compile(self):
        for compile_step in (False, "auto"):
            tb.Simulator(None, execution_mode="functional", execution_backend="cuda_graph", compile_step=compile_step)

    def test_large_eager_capture_warns_with_the_size(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warned = _cuda_graph.warn_if_large_eager_capture(2000, 864, compiled=False)
        self.assertTrue(warned)
        self.assertEqual(len(caught), 1)
        self.assertIn("2000 functional components x 864 time steps", str(caught[0].message))
        self.assertIn("compile_step", str(caught[0].message))

    def test_small_or_compiled_captures_are_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(_cuda_graph.warn_if_large_eager_capture(100, 864, compiled=False))
            self.assertFalse(_cuda_graph.warn_if_large_eager_capture(2000, 864, compiled=True))
        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
