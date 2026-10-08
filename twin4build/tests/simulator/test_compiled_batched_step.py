"""``compiled_batched_step`` (``torch.compile(vmap(F_aug))``) steps every row as ``F_aug`` steps it alone (#241).

torch 2.11 Inductor miscompiled ``F_aug``'s assembly of a vector input from slots assembled one by one (``index_put``
into an unbatched buffer under ``vmap``): every batch row wrote the same storage, or the CPU code generation failed.
A minimal model reproduces both: leaves (``v = p``) feed one vector port of a hub, sinks accumulate the hub's slots.
Before the fix, the leaves' direct slots failed in the CPU code generation, and two loops through one vector port
(as an office damper takes the max of its CO2 and temperature loops) gave every row the same values (450 off after
four steps).  The model compiles in about 13 s on the CPU; the example office's step took about 170 s, which the CI
runners did not finish in six hours.
"""
import datetime
import unittest

import torch
from dateutil import tz

import twin4build as tb
from twin4build.tests.model.test_batching_vector_slots import Hub, Leaf, Sink


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _has_cpp_compiler() -> bool:
    """Whether Inductor's CPU backend finds its C++ compiler (the Windows CI
    runners have no ``cl``: InvalidCxxCompiler)."""
    try:
        from torch._inductor.cpp_builder import get_compiler_version_info, get_cpp_compiler

        get_compiler_version_info(get_cpp_compiler())
    except Exception:  # noqa: BLE001
        return False
    return True


NEEDS_CPP = unittest.skipUnless(_has_cpp_compiler(), "Inductor's CPU backend finds no C++ compiler")
STEP = 600
START = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)


def _functional(device, loops: bool):
    """The minimal model on ``device`` as a functional map: ``(fm, y0, tape, n_theta)``.  ``loops``: two loops
    through the hub's one vector port (leaf i -> hub slot i -> sink i -> hub slot 2 + i), else four leaves into
    the hub's own slots, each read by a sink."""
    model = tb.Model(id=f"compiled_batched_step_{'loops' if loops else 'slots'}")
    hub = Hub(id="hub")
    n = 2 if loops else 4
    leaves = [Leaf(p=float(i + 1), id=f"leaf{i}") for i in range(n)]
    sinks = [Sink(id=f"sink{i}") for i in range(n)]
    for i in range(n):
        model.add_connection(leaves[i], hub, "v", "x", input_port_index=i)
        model.add_connection(hub, sinks[i], "y", "u", output_port_index=i, input_port_index=0)
        if loops:
            model.add_connection(sinks[i], hub, "w", "x", input_port_index=n + i)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    model.to(device=device, dtype=torch.float64)
    sim = tb.Simulator(model, execution_mode="functional", execution_backend="eager", compile_step=False)
    end = START + datetime.timedelta(seconds=8 * STEP)
    model.initialize(start_time=[START], end_time=[end], step_size=[STEP])
    layout, fm = sim.build_functional_model(
        theta_spec=[(leaf, "p") for leaf in leaves], outputs=[(sink, "w") for sink in sinks], step_size=STEP
    )
    recording = sim.record_exogenous_inputs(fm, [START], [end], [STEP], layout=layout)
    fm.prepare_routes(device)  # as a session does: the vector ports regrouped into the assembly #241 broke
    return fm, recording.Y0[0], recording.exogenous_tape[0], n


class TestCompiledBatchedStep(unittest.TestCase):
    STEPS, ROWS = 4, 3

    def _rows_agree(self, device, loops):
        fm, y0, tape, n = _functional(device, loops)
        theta0 = torch.arange(1.0, n + 1.0, dtype=torch.float64, device=device)
        Theta = torch.stack([theta0 * (1.0 + 0.5 * b) for b in range(self.ROWS)])  # distinct rows
        Y = y0.unsqueeze(0).expand(self.ROWS, -1).contiguous()
        rows = [Y[b] for b in range(self.ROWS)]
        with torch.no_grad():
            for t in range(self.STEPS):
                u = tape[t]
                Y, M = fm.compiled_batched_step(Y, Theta, u)
                for b in range(self.ROWS):
                    rows[b], m = fm.F_aug(rows[b], Theta[b], u, transform_mode=True)
                    torch.testing.assert_close(M[b], m, rtol=1e-9, atol=1e-9, msg=f"step {t}, row {b}: outputs")
                    torch.testing.assert_close(Y[b], rows[b], rtol=1e-9, atol=1e-9, msg=f"step {t}, row {b}: state")
        # the rows do differ, so a batch collapsed onto one row would show
        self.assertFalse(torch.allclose(rows[0], rows[-1]))

    @NEEDS_CPP
    def test_loops_through_one_vector_port_on_the_cpu(self):
        self._rows_agree("cpu", loops=True)

    @NEEDS_CPP
    def test_slots_of_one_vector_port_on_the_cpu(self):
        self._rows_agree("cpu", loops=False)

    @unittest.skipUnless(torch.cuda.is_available() and _has_triton(), "needs CUDA and Triton")
    def test_loops_through_one_vector_port_on_the_gpu(self):
        self._rows_agree("cuda", loops=True)


if __name__ == "__main__":
    unittest.main()
