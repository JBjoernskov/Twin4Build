"""A compiled step of small systems calls no matrix-product routine.

A product of small matrices as a library call ends the kernel around it, so a
step of many small systems compiles to thousands of launches of a few numbers
each.  ``discrete_statespace_system.matmul`` writes a product whose left factor
is within ``small_matmul_max`` as a broadcast multiply and a sum while
``torch.compile`` traces; the ``combo_kernels`` setting compiles the step with
Inductor's combo kernels.
"""
import unittest
from unittest import mock

import torch

import twin4build as tb
from twin4build.systems.utils import discrete_statespace_system as dss
from twin4build.tests.simulator.test_step_kinds import _fused

tb._IS_TESTING = True

#: The matrix-product routines a traced graph can call.
PRODUCTS = frozenset({"matmul", "bmm", "mm", "mv", "dot", "baddbmm", "addbmm", "addmm", "addmv", "einsum", "tensordot", "linear"})


def _traced(fn, *args):
    """``fn(*args)`` under ``torch.compile`` with a backend that runs the traced
    graph as it is: (the result, the names of the product routines the graph calls)."""
    calls = []

    def backend(gm, example_inputs):
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method"):
                name = node.target if isinstance(node.target, str) else getattr(node.target, "__name__", "")
                if name in PRODUCTS:
                    calls.append(name)
        return gm.forward

    try:
        torch._dynamo.reset()
        out = torch.compile(fn, fullgraph=True, dynamic=False, backend=backend)(*args)
    finally:
        torch._dynamo.reset()
    return out, calls


class TestMatmul(unittest.TestCase):
    def test_a_small_left_factor_is_written_out_while_tracing_only(self):
        gen = torch.Generator().manual_seed(4)
        limit = tb.settings.value("small_matmul_max")
        cases = {
            "a state's next value": ((5, 3, 3), (5, 3, 1), True),
            "one output row": ((5, 1, 4), (4, 40), True),
            "a block's rows": ((5, 2, 3), (5, 3, 2 * limit), True),
            "a wide left factor": ((2, 3, limit + 1), (limit + 1, 2), False),
            "a tall left factor": ((2, limit + 1, 3), (3, 2), False),
        }
        for what, (shape_a, shape_b, written_out) in cases.items():
            with self.subTest(what):
                a = torch.randn(shape_a, generator=gen, dtype=torch.float64)
                b = torch.randn(shape_b, generator=gen, dtype=torch.float64)
                out, calls = _traced(dss.matmul, a, b)
                self.assertEqual(calls, [] if written_out else ["matmul"])
                torch.testing.assert_close(out, a @ b, rtol=1e-12, atol=1e-12)
                with mock.patch.object(dss, "_small_matmul", side_effect=AssertionError("eager code keeps the library's product")):
                    torch.testing.assert_close(dss.matmul(a, b), a @ b, rtol=0, atol=0)

    def test_the_setting_moves_the_threshold(self):
        a = torch.ones(2, 6, 6, dtype=torch.float64)
        b = torch.ones(2, 6, 1, dtype=torch.float64)
        for limit, written_out in ((16, True), (6, True), (5, False), (0, False)):
            with self.subTest(limit=limit), tb.settings.override(small_matmul_max=limit):
                self.assertEqual(_traced(dss.matmul, a, b)[1], [] if written_out else ["matmul"])


class TestStepProducts(unittest.TestCase):
    """Three heated zones with radiators, batched and fused: every system of
    the step is small."""

    @classmethod
    def setUpClass(cls):
        cls.fm, cls.y0, cls.tape, cls.theta0, cls.fused = _fused(model_id="step_products")
        cls.fm.prepare_trace()

    def test_the_traced_step_calls_no_product_routine(self):
        step = self.fm.step("scalar")
        eager = step(self.y0, self.theta0, self.tape[0])
        (y_next, meas), calls = _traced(step, self.y0, self.theta0, self.tape[0])
        self.assertEqual(calls, [])
        torch.testing.assert_close(y_next, eager[0], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(meas, eager[1], rtol=1e-10, atol=1e-10)

    def test_the_count_sees_the_products_when_none_is_written_out(self):
        # the same step with the threshold at zero: the library's products are in the graph
        with tb.settings.override(small_matmul_max=0):
            _, calls = _traced(self.fm.step("scalar"), self.y0, self.theta0, self.tape[0])
        self.assertGreater(len(calls), 0)

    def test_the_gradient_of_the_traced_step_calls_none_either(self):
        step = self.fm.step("scalar")

        def loss_and_gradient(y, theta, u):
            return torch.func.grad(lambda t: step(y, t, u)[0].sum())(theta)

        gradient, calls = _traced(loss_and_gradient, self.y0, self.theta0, self.tape[0])
        self.assertEqual(calls, [])
        torch.testing.assert_close(gradient, loss_and_gradient(self.y0, self.theta0, self.tape[0]), rtol=1e-8, atol=1e-10)


class TestComboKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fm, cls.y0, cls.tape, cls.theta0, _ = _fused(model_id="step_combo")

    def _compile_options(self):
        """The options the step factory hands ``torch.compile`` (the compile itself replaced)."""
        self.fm.__dict__.pop("_step_scalar_compiled", None)
        try:
            with mock.patch("torch.compile", side_effect=lambda fn, **kwargs: fn) as compile_:
                self.fm.step("scalar", compiled=True)
            return compile_.call_args.kwargs
        finally:
            self.fm.__dict__.pop("_step_scalar_compiled", None)

    def test_the_setting_is_an_option_of_the_steps_compile(self):
        self.assertFalse(tb.settings.value("combo_kernels"))
        self.assertEqual(self._compile_options(), {"fullgraph": True, "dynamic": False, "options": None})
        with tb.settings.override(combo_kernels=True):
            self.assertEqual(self._compile_options()["options"], {"combo_kernels": True})

    @unittest.skipUnless(torch.cuda.is_available(), "Inductor's combo kernels are Triton's: checked on the GPU")
    def test_a_step_compiled_with_combo_kernels_is_the_eager_step(self):
        fm, y0, tape, theta0, _ = _fused(device="cuda", model_id="step_combo_gpu")
        eager = fm.step("scalar")(y0, theta0, tape[0])
        try:
            torch._dynamo.reset()
            with tb.settings.override(combo_kernels=True):
                compiled = fm.step("scalar", compiled=True)
                out = compiled(y0, theta0, tape[0])
        finally:
            torch._dynamo.reset()
        torch.testing.assert_close(out[0], eager[0], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(out[1], eager[1], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
