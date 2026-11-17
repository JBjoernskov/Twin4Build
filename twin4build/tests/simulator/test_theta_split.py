"""A step reads its parameters as pieces of ONE split of theta.

Indexed one by one (``theta[index]`` per parameter), every read's backward is a
theta-sized vector of zeros with that read's entries filled in, and a step's
adjoint sums one such vector per parameter: on a model with 1,800 parameters
among 225,000 entries that was most of a fit's time.  One ``torch.split`` has
one concatenation as its backward.
"""
import unittest

import torch

from twin4build.simulator._functional import _contiguous, _theta_pieces
from twin4build.tests.simulator.test_compiled_batched_step import NEEDS_CPP, _functional


def _consumers_of(theta, *outputs):
    """The names of the autograd nodes that read ``theta`` directly."""
    names, seen, stack = [], set(), [o.grad_fn for o in outputs if o.grad_fn is not None]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for parent, _ in node.next_functions:
            if parent is None:
                continue
            if getattr(parent, "variable", None) is theta:
                names.append(type(node).__name__)
            stack.append(parent)
    return names


class TestThetaSplit(unittest.TestCase):
    def test_the_layout_takes_contiguous_disjoint_indices_only(self):
        self.assertEqual(_contiguous(3), (3, 4))
        self.assertEqual(_contiguous(slice(2, 5)), (2, 5))
        self.assertEqual(_contiguous(slice(None, 5)), (0, 5))
        for other in (-1, slice(2, None), slice(0, 6, 2), [1, 2], True, torch.tensor([1])):
            self.assertIsNone(_contiguous(other))
        sizes, end, piece_of = _theta_pieces(
            {
                "a": {"p": 0, "q": slice(1, 4), "t": slice(8, 10)},
                # shared with a.p; a list; a slice that overlaps a.q
                "b": {"p": 0, "r": [5, 7], "s": slice(3, 6)},
            }
        )
        # [0:1] and [8:10] are pieces, the gap between them is one nobody reads;
        # the two overlapping slices and the list are read from theta directly
        self.assertEqual((sizes, end), ([1, 7, 2], 10))
        self.assertEqual(piece_of, {"a": {"p": 0, "t": 2}, "b": {"p": 0}})

    def test_one_step_reads_theta_once_and_its_gradient_is_the_indexed_one(self):
        fm, y0, tape, n = _functional("cpu", loops=True)
        self.assertEqual(sum(len(v) for v in fm._theta_split[2].values()), n)  # every parameter is a piece

        def step(theta):
            y, m = fm.F_aug(y0, theta, tape[0], transform_mode=True)
            return y, m

        theta = torch.arange(1.0, n + 1.0, dtype=torch.float64).requires_grad_(True)
        y, m = step(theta)
        self.assertEqual(_consumers_of(theta, y, m), ["SplitWithSizesBackward0"])
        split = torch.autograd.grad(y.sum() + m.sum(), theta)[0]

        # the same step with every parameter indexed on its own
        layout = fm._theta_split
        fm._theta_split = ([], 0, {})
        try:
            theta_i = theta.detach().clone().requires_grad_(True)
            y_i, m_i = step(theta_i)
            self.assertEqual(len(_consumers_of(theta_i, y_i, m_i)), n)
            indexed = torch.autograd.grad(y_i.sum() + m_i.sum(), theta_i)[0]
        finally:
            fm._theta_split = layout
        torch.testing.assert_close(y, y_i, rtol=0, atol=0)
        torch.testing.assert_close(split, indexed, rtol=1e-14, atol=0)
        self.assertGreater(float(split.abs().max()), 0.0)

    def test_a_theta_with_entries_beyond_the_parameters_splits(self):
        fm, y0, tape, n = _functional("cpu", loops=False)
        theta = torch.arange(1.0, n + 4.0, dtype=torch.float64).requires_grad_(True)  # three entries nobody reads
        y, m = fm.F_aug(y0, theta, tape[0], transform_mode=True)
        grad = torch.autograd.grad(y.sum() + m.sum(), theta)[0]
        self.assertEqual(float(grad[n:].abs().max()), 0.0)

    @NEEDS_CPP
    def test_the_compiled_rows_step_differentiates_through_the_split(self):
        fm, y0, tape, n = _functional("cpu", loops=True)
        rows = 3
        Theta = torch.stack(
            [torch.arange(1.0, n + 1.0, dtype=torch.float64) * (1.0 + 0.5 * b) for b in range(rows)]
        ).requires_grad_(True)
        Y = y0.unsqueeze(0).expand(rows, -1).contiguous()
        U = tape[0].unsqueeze(0).expand(rows, -1).contiguous()
        y, m = fm.compiled_rows_step(Y, Theta, U)
        compiled = torch.autograd.grad(y.sum() + m.sum(), Theta)[0]
        for b in range(rows):
            theta = Theta[b].detach().clone().requires_grad_(True)
            y_b, m_b = fm.F_aug(y0, theta, tape[0], transform_mode=True)
            torch.testing.assert_close(compiled[b], torch.autograd.grad(y_b.sum() + m_b.sum(), theta)[0], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
