# Standard library imports
import contextlib
import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

# Third party imports
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn as nn
from scipy import signal

# Local application imports
import twin4build.utils.types as tps
from twin4build import core
from twin4build import settings as _settings


#: Set by the functional engine for the length of a step it traces, captures
#: or transforms (``FunctionalModel.F``): the discretisation then uses the
#: matmul-only matrix exponential and never asks torch whether a transform is
#: active (a query that has no place in a captured graph).
TRACEABLE = False


@contextlib.contextmanager
def traceable(flag: bool = True):
    """Within the block, discretise with the matmul-only matrix exponential
    (or, with ``flag=False``, leave the choice to detection), as the
    functional engine does for a step it traces, captures or transforms."""
    global TRACEABLE
    previous = TRACEABLE
    TRACEABLE = bool(flag)
    try:
        yield
    finally:
        TRACEABLE = previous


def _traceable() -> bool:
    """Whether the discretisation must use the matmul-only exponential: the
    engine says so, ``torch.compile`` is tracing, or a ``torch.func``
    transform is active."""
    if TRACEABLE:
        return True
    try:
        if torch.compiler.is_compiling():
            return True
    except AttributeError:  # pragma: no cover - very old torch
        pass
    return _functorch_active()


def _functorch_transform_active() -> bool:
    """Whether a functorch transform (vmap/grad) is on the stack."""
    try:
        return bool(torch._C._are_functorch_transforms_active())
    except AttributeError:  # pragma: no cover - very old torch
        return False


def _small_matmul(a, b):
    """``a @ b`` for tiny matrices as broadcast multiply + sum.

    Under ``torch.compile`` a ``bmm``/``baddbmm`` on a 3x3 stays a separate
    cuBLAS call; written as pointwise-multiply + reduction Inductor fuses it
    with the surrounding arithmetic into one Triton kernel.  Only used while
    Dynamo is tracing (see :func:`_expm_ss`); eager code keeps cuBLAS, which
    is faster when nothing fuses.
    """
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def matmul(a, b):
    """``a @ b`` as a step writes it (both at least two-dimensional).

    While ``torch.compile`` traces, a product whose left factor has at most
    ``small_matmul_max`` rows and columns is a broadcast multiply and a sum
    (:func:`_small_matmul`): a small system's next state and outputs, a fused
    block's rows.  As a library call each of them ends the kernel around it,
    and a step of many small systems is then mostly launches; written out,
    Inductor fuses it with the arithmetic on both sides, forward and
    backward.  A larger product, and every product of eager code, is the
    library's.
    """
    if torch.compiler.is_compiling():
        limit = _settings.value("small_matmul_max")
        if a.shape[-2] <= limit and a.shape[-1] <= limit:
            return _small_matmul(a, b)
    return a @ b


#: The scaled 1-norm above which the exponential refuses (NaN): the degree-8
#: series is not an exponential there, and squaring would amplify the error.
_EXPM_REFUSE_NORM = 2.0
#: Osborne sweeps of the balancing: fixed, so the step traces and vmaps.  Two
#: reach the norm more sweeps do to within a tenth (simultaneous sweeps
#: alternate; an odd count lands on the worse side).
_BALANCE_SWEEPS = 2


def _balance(M):
    """A diagonal ``d`` ``(..., N)`` for which ``D^-1 M D`` is balanced: every
    row's off-diagonal 1-norm near its column's (Osborne's iteration, a fixed
    number of simultaneous sweeps).

    A balance equation's matrix ``C^-1 K`` (capacities times conductances) is
    badly scaled, not stiff: its 1-norm is far above its spectral radius (a
    210-state block of thermal zones and their walls: 3,400 against 200).  The
    scaling and squaring of :func:`_expm_ss` is sized by the norm, so the
    balanced matrix needs fewer doublings for the same accuracy.  The
    exponential is the same for any ``d``, ``exp(M) = D exp(D^-1 M D) D^-1``,
    so ``d`` carries no gradient; it is rounded to powers of two (scaling by
    it is exact) and kept only where it lowers the 1-norm.  A zero row (the
    augmented block's last) or column is scaled to the mean norm.
    """
    M = M.detach()
    n = M.shape[-1]
    tiny = torch.finfo(M.dtype).tiny
    magnitude = M.abs()
    off = magnitude * (1.0 - torch.eye(n, dtype=M.dtype, device=M.device))
    d = torch.ones(M.shape[:-1], dtype=M.dtype, device=M.device)
    for _ in range(_BALANCE_SWEEPS):
        scaled = off * d.unsqueeze(-2) / d.unsqueeze(-1)
        c, r = scaled.sum(dim=-2), scaled.sum(dim=-1)
        mean = 0.5 * (c.sum(dim=-1, keepdim=True) + r.sum(dim=-1, keepdim=True)) / n
        f = torch.where((c > 0) & (r > 0), torch.sqrt(r / c.clamp_min(tiny)), torch.ones_like(c))
        f = torch.where((c > 0) & (r == 0), (mean / c.clamp_min(tiny)).clamp(max=1.0), f)
        f = torch.where((c == 0) & (r > 0), (r / mean.clamp_min(tiny)).clamp(min=1.0), f)
        d = (d * f).clamp(2.0**-40, 2.0**40)
    d = torch.exp2(torch.round(torch.log2(d)))
    before = magnitude.sum(dim=-2).amax(dim=-1, keepdim=True)
    after = (magnitude * d.unsqueeze(-2) / d.unsqueeze(-1)).sum(dim=-2).amax(dim=-1, keepdim=True)
    return torch.where(after < before, d, torch.ones_like(d))


def _expm_ss(M, squarings=None, *, phi1=False, fused=None):
    """Matrix exponential by scaling and squaring with a Taylor series:
    ``exp(M)``, or with ``phi1`` the pair ``(exp(M), phi1(M))``, ``phi1(M) =
    sum_k M^k / (k+1)!``.

    ``torch.matrix_exp`` has no ``vmap`` batching rule, so under ``vmap`` (the
    collocation fast path maps ``F`` over segments) it silently falls back to a
    sequential Python loop over the batch -- and its VJP, hit by ``jacrev``, is
    slower still.  This is *pure* matmul/add, so ``vmap`` batches it natively
    and autograd differentiates it cleanly.

    ``exp(M) = (exp(M / 2^s))^(2^s)``, in four steps:

    1. **Balance** (:func:`_balance`): the similar matrix ``D^-1 M D`` has the
       smaller norm.
    2. **Scale** by ``2^-s`` with ``s = squarings`` (the ``expm_squarings``
       setting).  ``s`` is a FIXED count (a data-dependent one is not
       ``vmap``-safe and cannot be traced), so it must cover the stiffest
       matrix a run reaches: the balanced norm over ``2^s`` should stay below
       about 0.5 for float32 accuracy and 0.05 for float64.  Too few doublings
       once made a bounded exponential overflow in the series and come out NaN
       at bound-hitting optima; a matrix whose scaled norm exceeds
       ``_EXPM_REFUSE_NORM`` now comes out NaN by design, never as a wrong
       finite value.
    3. **Series**: the degree-8 Taylor polynomial of ``exp(Ms) - I`` in four
       products (Paterson-Stockmeyer: ``Ms^2``, ``Ms^3``, ``Ms^4`` and one
       more) instead of Horner's eight.  Tracking ``exp(Ms) - I`` keeps the
       small non-identity part at full relative precision instead of rounding
       it against one, which matters through the doublings.  ``phi1`` shares
       the powers (one product more).
    4. **Square** ``s`` times: ``exp(2Y) - I = 2 D + D @ D`` for ``D = exp(Y) -
       I``, and ``phi1(2Y) = (I + exp(Y)) phi1(Y) / 2``.

    The ZOH discretisation with ``phi1`` is ``Ad = exp(A T)``, ``Bd = phi1(A T)
    B T``: no inverse of ``A`` (a zero-flow mass balance is singular) and no
    augmented block.

    ``fused`` writes the products as broadcast multiply and sum, which
    Inductor fuses into few kernels; cuBLAS calls would not fuse.  ``None``
    (the default) does so while ``torch.compile`` traces a matrix up to the
    ``small_matmul_max`` setting: the exponential's products were 79 % of the
    kernels left in a compiled rollout step (issue #134), but a broadcast
    product moves n times a product's memory, so a large matrix keeps ``bmm``
    (measured on a laptop GPU, float32, 960 matrices per call, forward and
    backward: the broadcast form wins up to 13 x 13 and loses from 17 x 17 on).
    """
    N = M.shape[-1]
    lead = M.shape[:-2]
    if squarings is None:
        squarings = _settings.value("expm_squarings")
    if fused is None:
        fused = torch.compiler.is_compiling() and N <= _settings.value("small_matmul_max")
    if fused:

        def mul(a, b):
            return _small_matmul(a, b)

        def addmul(base, a, b, alpha=1.0, beta=1.0):
            return beta * base + alpha * _small_matmul(a, b)

    else:
        # One fused ``baddbmm`` kernel per doubling on a (batch, N, N) view;
        # it has vmap, forward-AD and reverse-AD rules, so the collocation
        # ``vmap`` path and ``jacfwd``/``hessian`` keep working.
        mul = torch.bmm

        def addmul(base, a, b, alpha=1.0, beta=1.0):
            return torch.baddbmm(base, a, b, beta=beta, alpha=alpha)

    X = M.reshape(-1, N, N)
    d = _balance(X)
    row, col = d.unsqueeze(-1), d.unsqueeze(-2)
    Xb = X * col / row  # D^-1 X D
    refuse = Xb.detach().abs().sum(dim=-2).amax(dim=-1) / (2.0**squarings) > _EXPM_REFUSE_NORM
    Xs = Xb / (2.0**squarings)
    I = torch.eye(N, dtype=M.dtype, device=M.device)
    X2 = mul(Xs, Xs)
    X3 = mul(X2, Xs)
    X4 = mul(X2, X2)
    # exp(Xs) - I = Xs + Xs^2/2! + ... + Xs^8/8!
    delta = Xs + X2 / 2.0 + X3 / 6.0 + mul(X4, I / 24.0 + Xs / 120.0 + X2 / 720.0 + X3 / 5040.0 + X4 / 40320.0)
    if phi1:
        # phi1(Xs) = I + Xs/2! + ... + Xs^8/9!
        phi = (
            I
            + Xs / 2.0
            + X2 / 6.0
            + X3 / 24.0
            + mul(X4, I / 120.0 + Xs / 720.0 + X2 / 5040.0 + X3 / 40320.0 + X4 / 362880.0)
        )
    for _ in range(squarings):
        if phi1:
            phi = addmul(phi, delta, phi, alpha=0.5)
        delta = addmul(delta, delta, delta, beta=2.0)
    bad = refuse.reshape(-1, 1, 1)
    nan = torch.full((), float("nan"), dtype=M.dtype, device=M.device)
    expM = torch.where(bad, nan, (I + delta) * row / col).reshape(*lead, N, N)  # D exp(D^-1 X D) D^-1
    if not phi1:
        return expM
    return expM, torch.where(bad, nan, phi * row / col).reshape(*lead, N, N)


def _expm_ss_fused(M, squarings=None):
    """:func:`_expm_ss` with every product written for kernel fusion, whatever
    the matrix's size and whether or not ``torch.compile`` traces; for tests."""
    return _expm_ss(M, squarings, fused=True)


def _expm_phi1_ss(X, squarings=None):
    """``(exp(X), phi1(X))`` (:func:`_expm_ss` with ``phi1``)."""
    return _expm_ss(X, squarings, phi1=True)


def _expm_phi1_ss_fused(X, squarings=None):
    """:func:`_expm_phi1_ss` with every product written for kernel fusion; for tests."""
    return _expm_ss(X, squarings, phi1=True, fused=True)


def _functorch_active() -> bool:
    """True when running under a torch.func transform (vmap/jacrev/grad).

    Used to pick the matrix-exponential implementation: ``torch.matrix_exp``
    is a single fused op with a native backward (fast in plain eager/autograd)
    but has no vmap batching rule, while :func:`_expm_ss` is pure matmul (vmap
    batches it natively) but unrolls ~26 matmuls into the autograd graph.
    """
    try:
        return bool(torch._C._are_functorch_transforms_active())
    except AttributeError:  # very old torch -- keep the always-safe path
        return True


class SparseBilinear(NamedTuple):
    """A bilinear tensor by its entries: ``E`` (``(n_c, m, n, n)``,
    state-input) or ``F`` (``(n_c, m, n, m)``, input-input) as
    ``values[:, e]`` at (input ``k[e]``, row ``i[e]``, column ``j[e]``).  A
    fused block's bilinear terms are a few thousand entries of tens of
    millions (a storey-wide block: ~158 states x 588 inputs); held dense,
    every step broadcasts ``u`` over all of them.  Entries may repeat (they
    add).

    ``plan`` (:func:`bilinear_plan`, :func:`with_plan`): the fixed-order
    layout the effective matrices are summed by, the same bits on every call
    and device; without it they are summed by ``index_add`` (atomic adds on a
    GPU)."""

    k: torch.Tensor  # (nnz,) long: the input
    i: torch.Tensor  # (nnz,) long: the row (a state)
    j: torch.Tensor  # (nnz,) long: the column (a state for E, an input for F)
    values: torch.Tensor  # (n_c, nnz)
    plan: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None  # (slot, inputs, group, flat)


def _as_numpy(t) -> np.ndarray:
    # ``tolist``, not ``numpy()``: a block assembled for the first time inside
    # a functorch transform (an optimizer's jacrev over an object-mode
    # simulation) holds its index tensors wrapped, and a wrapped tensor -- or
    # its unwrapped base while the transform runs -- has no storage numpy can
    # read.
    return np.asarray(t.tolist() if isinstance(t, torch.Tensor) else t, dtype=np.int64).reshape(-1)


def structure_constant(t: torch.Tensor) -> torch.Tensor:
    """``t`` without functorch wrappers: an integer tensor of a model's
    structure (bilinear indices and their plan) built inside a transform,
    which a block keeps and reuses in later calls, inside or outside any
    transform.  Only for constants: nothing differentiates or batches
    through them."""
    while torch._C._functorch.is_functorch_wrapped_tensor(t):
        t = torch._C._functorch.get_unwrapped(t)
    return t


def _table(keys: np.ndarray, pad: int):
    """``(unique keys, table (n_unique, width), row, col, order)``: each key's
    entries (indices into ``keys``) in their order, padded with ``pad``; entry
    ``order[p]`` sits at ``(row[p], col[p])``."""
    order = np.argsort(keys, kind="stable")
    unique, start, counts = np.unique(keys[order], return_index=True, return_counts=True)
    width = int(counts.max()) if len(keys) else 1
    table = np.full((len(unique), width), pad, dtype=np.int64)
    row = np.repeat(np.arange(len(unique)), counts)
    col = np.arange(len(keys)) - np.repeat(start, counts)
    table[row, col] = order
    return unique, table, row, col, order


def bilinear_plan(k, i, j, cols: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fixed-order layout of a :class:`SparseBilinear`'s entries:
    ``(slot, inputs, group, flat)``.

    ``index_add``, and the backward of a gather with repeated indices, add
    repeated positions by atomic adds in whatever order the GPU runs them:
    float32 results differ in their last bits from call to call, and a CUDA
    graph's replay differs from the eager step (its parity check fails).
    With the plan no index repeats where values are summed or gradients flow
    back:

    * ``inputs (n_in,)``: the inputs the entries act on, each once;
      ``slot (n_in, K)``: each input's entries (indices into ``values``),
      padded with ``nnz`` (a zero) -- the inputs enter by broadcasting, not
      by a gather;
    * ``flat (n_out,)``: the matrix positions ``i * cols + j`` the entries
      reach, each once; ``group (n_out, G)``: each position's entries by
      their place in the ``(n_in, K)`` layout flattened, padded with
      ``n_in * K`` (a zero).

    Built from the indices alone (on the host, once per structure)."""
    device = k.device if isinstance(k, torch.Tensor) else None
    k, i, j = _as_numpy(k), _as_numpy(i), _as_numpy(j)
    nnz = len(k)
    inputs, slot, row, col, order = _table(k, nnz)
    width = slot.shape[1]
    place = np.empty(nnz, dtype=np.int64)
    place[order] = row * width + col
    flat, table, _, _, _ = _table(i * int(cols) + j, nnz)
    group = np.append(place, len(inputs) * width)[table]
    return tuple(
        structure_constant(torch.as_tensor(a, dtype=torch.long, device=device))
        for a in (slot, inputs, group, flat)
    )


def with_plan(T: "SparseBilinear", cols: int) -> "SparseBilinear":
    """``T`` with its :func:`bilinear_plan` (``cols``: the matrix's columns:
    ``n`` for ``E``, ``m`` for ``F``)."""
    return T._replace(plan=bilinear_plan(T.k, T.i, T.j, cols))


def bilinear_inputs(T, m: int, device) -> torch.Tensor:
    """``(m,)`` bool: the inputs a bilinear tensor (dense,
    :class:`SparseBilinear` or ``None``) acts on."""
    mask = torch.zeros(m, dtype=torch.bool, device=device)
    if T is None:
        return mask
    if isinstance(T, SparseBilinear):
        live = (T.values.detach().abs() > 0).any(0)
        mask[T.k[live].to(device)] = True
        return mask
    return mask | (T.detach().abs().sum(dim=(-2, -1)) > 0).reshape(-1, m).any(0)


def _add_bilinear(M, T, u):
    """``M + sum_k u_k T_k`` for a :class:`SparseBilinear` ``T``: only its
    entries -- by its ``plan`` in a fixed order (the same bits on every call
    and in a CUDA graph), else added by ``index_add``."""
    rows, cols = M.shape[-2], M.shape[-1]
    if T.plan is not None:
        slot, inputs, group, flat = T.plan
        vals = torch.cat([T.values, T.values.new_zeros(*T.values.shape[:-1], 1)], dim=-1)[..., slot]  # (n_c, n_in, K)
        contrib = vals * u[..., inputs].unsqueeze(-1)  # (..., n_c, n_in, K)
        lead = torch.broadcast_shapes(contrib.shape[:-2], M.shape[:-2])
        contrib = contrib.expand(*lead, *contrib.shape[-2:]).reshape(*lead, -1)
        contrib = torch.cat([contrib, contrib.new_zeros(*lead, 1)], dim=-1)
        summed = contrib[..., group].sum(dim=-1)  # (..., n_out): each position's entries in a fixed order
        dense = contrib.new_zeros(*lead, rows * cols).index_copy(-1, flat, summed)  # every position once
        return M + dense.reshape(*lead, rows, cols)
    contrib = T.values * u[..., T.k]  # (..., n_c, nnz)
    flat = T.i * cols + T.j
    lead = torch.broadcast_shapes(contrib.shape[:-1], M.shape[:-2])
    contrib = contrib.expand(*lead, contrib.shape[-1])
    dense = torch.zeros(*lead, rows * cols, dtype=contrib.dtype, device=contrib.device).index_add(-1, flat, contrib)
    return M + dense.reshape(*lead, rows, cols)


def effective_matrices(A, B, E, F, u):
    """``(A_eff, B_eff)``: the bilinear matrices evaluated at the current ``u``.

    The first step of :func:`_discretize_onestep`, factored out so that a
    caller which only needs *part* of the discretization can form the same
    matrices rather than re-deriving them (see
    :meth:`~twin4build.systems.utils.occupancy_system.OccupancySystem.invert_zoh_occupancy`,
    which drops ``B_eff``'s structurally zero input columns before
    exponentiating).
    """
    # Broadcast-friendly bilinear terms (einsum would require the batch dims of
    # E/F and u to match exactly; the fast single-shooting rollout batches u
    # over periods while the matrices keep their (n_c=1, ...) leading dim).
    # A SparseBilinear term adds its entries only (_add_bilinear).
    u_b = u.unsqueeze(-1).unsqueeze(-1)  # (..., m, 1, 1)
    if E is None:
        A_eff = A
    elif isinstance(E, SparseBilinear):
        A_eff = _add_bilinear(A, E, u)
    else:
        A_eff = A + (E * u_b).sum(dim=-3)
    if F is None:
        B_eff = B
    elif isinstance(F, SparseBilinear):
        B_eff = _add_bilinear(B, F, u)
    else:
        B_eff = B + (F * u_b).sum(dim=-3)
    return A_eff, B_eff


def _discretize_onestep(A, B, E, F, u, step_size, transform_mode=None):
    """Effective-matrix ZOH discretization: ``(Ad, Bd)`` for the current ``u``."""
    A_eff, B_eff = effective_matrices(A, B, E, F, u)

    n = A.shape[-1]
    m = B.shape[-1]
    T = step_size
    if transform_mode is None:
        transform_mode = _traceable()
    # The transform-mode step (compiled, captured, vmapped) discretises as the
    # ``zoh_expm`` and ``expm_checkpoint`` settings say (twin4build.settings).
    if transform_mode and _settings.value("zoh_expm") == "phi1":
        X = A_eff * T
        if _settings.value("expm_checkpoint"):
            expX, phi = torch.utils.checkpoint.checkpoint(_expm_phi1_ss, X, use_reentrant=False)
        else:
            expX, phi = _expm_phi1_ss(X)
        return expX, matmul(phi, B_eff * T)
    # Block matrix M = [[A*T, B*T], [0, 0]]; expm(M) = [[Ad, Bd], [0, I]].
    # Built out-of-place (cat + zero-row pad) so it is ``vmap``-safe -- an
    # in-place ``M[..., :n, :n] = ...`` fails when A_eff/B_eff carry a vmap batch
    # dim that the freshly-allocated M does not.
    top = torch.cat([A_eff * T, B_eff * T], dim=-1)  # (..., n, n+m)
    M = torch.nn.functional.pad(top, (0, 0, 0, m))  # add m zero rows -> (..., n+m, n+m)
    # Callers that know they are part of a transformed/captured pure path pass
    # the mode explicitly.  ``None`` preserves compatibility for direct calls,
    # while the explicit bool keeps the private functorch-state query out of
    # captured production graphs.
    if transform_mode:
        if _settings.value("expm_checkpoint"):
            expM = torch.utils.checkpoint.checkpoint(_expm_ss, M, use_reentrant=False)
        else:
            expM = _expm_ss(M)
    else:
        expM = torch.matrix_exp(M)
    return expM[..., :n, :n], expM[..., :n, n:]


def _onestep_augmented(A, B, C, D, E, F, x, u, step_size):
    """One exact ZOH step from ONE matrix exponential of size ``n + 1``
    (the ``zoh_expm="aug"`` setting).

    The step needs the state's next value, not the discrete matrices: with
    ``X = A_eff T`` and the input folded into a vector ``b = B_eff T u``,
    ``exp([[X, b], [0, 0]]) = [[exp(X), phi1(X) b], [0, 1]]``, so
    ``x_next = exp(X) x + phi1(X) b`` reads off one exponential of an
    ``(n + 1)`` matrix.  The ``phi1`` path computes the pair ``(exp(X),
    phi1(X))`` (twice the products) and then ``Bd = phi1(X) B_eff T``, an
    ``n x m`` product (a storey-wide fused block: ~158 states, ~588 inputs).
    Same series and squarings (:func:`_expm_ss`); on a stiff matrix (``|A T|``
    ~ 1e5) it is the more accurate of the two (~1e-15 of the state against
    ``torch.matrix_exp``, the ``phi1`` recurrence ~1e-6), values and
    gradients."""
    A_eff, B_eff = effective_matrices(A, B, E, F, u)
    T = step_size
    n = A.shape[-1]
    X = A_eff * T
    b = matmul(B_eff * T, u.unsqueeze(-1))  # (..., n, 1)
    lead = torch.broadcast_shapes(X.shape[:-2], b.shape[:-2])
    top = torch.cat([X.expand(*lead, n, n), b.expand(*lead, n, 1)], dim=-1)  # (..., n, n + 1)
    M = torch.nn.functional.pad(top, (0, 0, 0, 1))  # (..., n + 1, n + 1): a zero last row
    if _settings.value("expm_checkpoint"):
        expM = torch.utils.checkpoint.checkpoint(_expm_ss, M, use_reentrant=False)
    else:
        expM = _expm_ss(M)
    x_next = matmul(expM[..., :n, :n], x.unsqueeze(-1)).squeeze(-1) + expM[..., :n, n]
    y = matmul(C, x_next.unsqueeze(-1)).squeeze(-1) + matmul(D, u.unsqueeze(-1)).squeeze(-1)
    return x_next, y


def bilinear_onestep(
    A,
    B,
    C,
    D,
    E,
    F,
    x,
    u,
    step_size,
    disc_cache=None,
    transform_mode=None,
):
    """Pure one ZOH step of a (bilinear) state-space system.

    A functorch-compatible re-expression of :meth:`DiscreteStatespaceSystem.do_step`
    with **no ports, no history, no state mutation** -- the building block of the
    ``forward`` fast path.  Given the continuous matrices and the current
    ``(x, u)`` it forms the input-dependent effective matrices, discretizes them
    via the matrix-exponential block trick, and returns ``(x_next, y)``.

    Shapes (``n_c`` = parallel components, ``n`` = states, ``m`` = inputs)::

        A (n_c, n, n)     B (n_c, n, m)     C (n_c, p, n)     D (n_c, p, m)
        E (n_c, m, n, n) or None            F (n_c, m, n, m) or None
        x (n_c, n)        u (n_c, m)

    ``E`` and ``F`` may also be given by their entries (:class:`SparseBilinear`).

    Returns ``x_next (n_c, n)``, ``y (n_c, p)``.  ``vmap`` maps this over segments.

    ``disc_cache`` (a mutable dict scoped to one theta by the caller) enables
    the same optimization ``do_step`` has had all along: the matrix exponential
    is recomputed only when the *bilinear-relevant* inputs (nonzero E/F slices)
    actually change between steps -- e.g. a closed damper holds the flows
    constant for hours.  Reusing the previous step's ``(Ad, Bd)`` when they
    haven't matches ``do_step``'s semantics exactly (gradients flow to the step
    that computed them).  Ignored under functorch transforms, where the
    data-dependent branch is untraceable, and whenever the bilinear-relevant
    inputs *carry gradient* (e.g. the optimizer's fast rollout, where a flow
    is a decision variable): value-equal inputs at different steps have
    DIFFERENT gradients, so reusing the first step's matrices would misroute
    the discretization sensitivity to that step.  Recomputing per step keeps
    the gradient exact (it is what ``jacrev`` over ``do_step`` does anyway --
    functorch bypasses the cache).
    """
    if transform_mode is None:
        transform_mode = _traceable()
    if transform_mode and _settings.value("zoh_expm") == "aug":
        return _onestep_augmented(A, B, C, D, E, F, x, u, step_size)
    if disc_cache is None or transform_mode:
        Ad, Bd = _discretize_onestep(
            A, B, E, F, u, step_size, transform_mode=transform_mode
        )
    else:
        mask = disc_cache.get("mask")
        if mask is None:
            m_dim = B.shape[-1]
            mask = bilinear_inputs(E, m_dim, u.device) | bilinear_inputs(F, m_dim, u.device)
            disc_cache["mask"] = mask
        u_rel_live = u[..., mask]
        if u_rel_live.requires_grad:
            # Differentiable bilinear inputs: gradient must flow through THIS
            # step's discretization -- no reuse.
            Ad, Bd = _discretize_onestep(
                A, B, E, F, u, step_size, transform_mode=False
            )
        elif _functorch_transform_active():
            # Under vmap/grad transforms the cache is both unusable
            # (``torch.allclose`` has no batching rule) and wrong: one cached
            # discretization cannot stand for a whole batch of inputs.  A
            # value-only batched evaluation lands here -- it has no gradient,
            # so it would otherwise take the caching branch below.
            Ad, Bd = _discretize_onestep(
                A, B, E, F, u, step_size, transform_mode=False
            )
        else:
            u_rel = u_rel_live.detach()
            prev = disc_cache.get("u_rel")
            if (
                prev is not None
                and prev.shape == u_rel.shape
                and torch.allclose(prev, u_rel)
            ):
                Ad, Bd = disc_cache["Ad"], disc_cache["Bd"]
            else:
                Ad, Bd = _discretize_onestep(
                    A, B, E, F, u, step_size, transform_mode=False
                )
                disc_cache.update(u_rel=u_rel.clone(), Ad=Ad, Bd=Bd)

    x_next = matmul(Ad, x.unsqueeze(-1)).squeeze(-1) + matmul(Bd, u.unsqueeze(-1)).squeeze(-1)
    y = matmul(C, x_next.unsqueeze(-1)).squeeze(-1) + matmul(D, u.unsqueeze(-1)).squeeze(-1)
    return x_next, y


# The declared form of DiscreteStatespaceSystem (below); here, next to it, rather than with the imports above.
from twin4build.systems.saref4syst.declarations import Input, Output, Setting, State  # noqa: E402


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else float(value)


def _as_float64(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def _matrix(value):
    """A matrix setting as nested tuples of floats: saved as JSON, and equal matrices batch (compared by value).
    ``None`` stays ``None``."""
    if value is None:
        return None
    array = _as_float64(value)
    if array.dim() < 2:
        raise ValueError(f"DiscreteStatespaceSystem: a matrix has two dimensions or more, got shape {tuple(array.shape)}")
    return _tuples(array.tolist())


def _initial_state(value):
    """``x0``: one number for every state variable, or one per state variable (absent: 0)."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return tuple(float(v) for v in _as_float64(value).reshape(-1).tolist())


#: Why each keyword or attribute of the System written the old way is gone (2.0).
_GONE = {
    "step_size": "a run steps it with the run's step size (forward's step_size)",
    "sample_time": "a run steps it with the run's step size (forward's step_size)",
    "state_names": "the state is x; System.state_names() names its entries",
    "batch_size": "one copy is one System; batch copies with Model.batch_components",
    "sim_batch_size": "one copy is one System; batch copies with Model.batch_components",
    "system_batch_size": "one copy is one System; batch copies with Model.batch_components",
    "discretize_system": "use discretize_onestep(A, B, E, F, u, step_size)",
}


class DiscreteStatespaceSystem(core.System):
    r"""A linear or bilinear state-space System, stepped by zero-order hold.

    .. math::

       \frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \sum_{i=1}^{m} \mathbf{E}_i\mathbf{x}u_i
       + \sum_{i=1}^{m} \mathbf{F}_{i}\mathbf{u} u_i, \qquad \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}

    with :math:`\mathbf{x} \in \mathbb{R}^n` the state, :math:`\mathbf{u} \in \mathbb{R}^m` the input and
    :math:`\mathbf{y} \in \mathbb{R}^p` the output.  ``E`` (state-input coupling, ``(m, n, n)``) and ``F``
    (input-input coupling, ``(m, n, m)``) are optional.  Each step forms the effective matrices
    :math:`\mathbf{A}^* = \mathbf{A} + \sum_i \mathbf{E}_i u_i` and :math:`\mathbf{B}^* = \mathbf{B} + \sum_i
    \mathbf{F}_i u_i` at the step's input, discretises them exactly over the step (the matrix exponential of
    :math:`[[\mathbf{A}^* T, \mathbf{B}^* T], [0, 0]]`, :func:`discretize_onestep`) and publishes the output of
    the end-of-step state, :math:`\mathbf{y}[k] = \mathbf{C}\mathbf{x}[k+1] + \mathbf{D}\mathbf{u}[k]`
    (:func:`bilinear_onestep`, which the thermal, wall and heater Systems step with too).

    Declared:

    - settings ``A`` ``(n, n)``, ``B`` ``(n, m)``, ``C`` ``(p, n)``, ``D`` ``(p, m)`` (zeros when absent),
      ``E`` and ``F`` (absent: linear), and ``x0``, the initial state (one number, or ``n``);
    - the input ``u``, one slot per column of ``B``, as wide as what is wired into it;
    - the output ``y``, one slot per row of ``C``;
    - the state ``x``, ``n`` entries.

    The matrices are constructor data, never estimated, and copies batch when they hold the same matrices.

    Examples:
        >>> import torch
        >>> import twin4build as tb
        >>> A = torch.tensor([[-1e-4, 5e-5], [2e-5, -8e-5]], dtype=torch.float64)
        >>> B = torch.tensor([[1e-4], [5e-5]], dtype=torch.float64)
        >>> C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        >>> system = tb.DiscreteStatespaceSystem(A=A, B=B, C=C, x0=[20.0, 18.0], id="ss")
        >>> # schedule.scheduleValue -> system.u (input_port_index=0); system.y -> sensor
    """

    REMOVED_ATTRIBUTES = _GONE

    A = Setting(convert=_matrix, doc="The state matrix, (n, n)")
    B = Setting(convert=_matrix, doc="The input matrix, (n, m)")
    C = Setting(convert=_matrix, doc="The output matrix, (p, n)")
    D = Setting(None, convert=_matrix, doc="The feedthrough matrix, (p, m); absent: zeros")
    E = Setting(None, convert=_matrix, doc="The state-input coupling, (m, n, n); absent: none")
    F = Setting(None, convert=_matrix, doc="The input-input coupling, (m, n, m); absent: none")
    x0 = Setting(None, convert=_initial_state, doc="The initial state: one number, or one per state entry")
    u = Input(size="wired", doc="The input, one slot per column of B")
    y = Output(size=C, doc="The output, one slot per row of C")
    x = State(size=A, init=x0, doc="The state")

    def derive(self, p, step_size):
        """Nothing depends on parameters; checks that the matrices fit together and the input's width."""
        n, m, rows = len(self.A), len(self.B[0]), len(self.C)
        shapes = {
            "A": ((n, n), self.A),
            "B": ((n, m), self.B),
            "C": ((rows, n), self.C),
            "D": ((rows, m), self.D),
            "E": ((m, n, n), self.E),
            "F": ((m, n, m), self.F),
        }
        wrong = [
            f"{name} is {tuple(_as_float64(value).shape)}, not {shape}"
            for name, (shape, value) in shapes.items()
            if value is not None and tuple(_as_float64(value).shape) != shape
        ]
        if not isinstance(self.x0, float) and len(self.x0) != n:
            wrong.append(f"x0 holds {len(self.x0)} values, not {n}")
        width = self.input["u"]._n_v
        if width is not None and width != m:
            wrong.append(f"u is {width} wide, but B has {m} columns")
        if wrong:
            raise ValueError(f"DiscreteStatespaceSystem '{self.id}': " + "; ".join(wrong))
        return {}

    def forward(self, x, inputs, p, step_size):
        """One zero-order-hold step (:func:`bilinear_onestep`): the next state and its output."""
        u = inputs["u"]

        def matrix(value):
            return None if value is None else torch.as_tensor(value, dtype=u.dtype, device=u.device)

        A, B, C = matrix(self.A), matrix(self.B), matrix(self.C)
        D = matrix(self.D)
        if D is None:
            D = torch.zeros((C.shape[0], B.shape[1]), dtype=u.dtype, device=u.device)
        x_next, y = bilinear_onestep(A, B, C, D, matrix(self.E), matrix(self.F), x, u, step_size)
        return x_next, {"y": y}

    @classmethod
    def from_matrices(cls, A, B, C, D=None, **kwargs):
        """A DiscreteStatespaceSystem of the continuous-time matrices."""
        return cls(A=A, B=B, C=C, D=D, **kwargs)

    @classmethod
    def from_transfer_function(cls, num, den, **kwargs):
        """A DiscreteStatespaceSystem of a continuous-time transfer function (``scipy.signal.tf2ss``)."""
        A, B, C, D = signal.tf2ss(num, den)
        return cls(A=A, B=B, C=C, D=D, **kwargs)


def discretize_onestep(A, B, E, F, u, step_size, transform_mode=None):
    """The exact discretization of a bilinear state-space model over one
    step, for an input held constant over the step (zero-order hold).

    The continuous model is ``dx/dt = A_eff x + B_eff u`` with ``A_eff = A +
    sum_k E[k] u[k]`` and ``B_eff = B + sum_k F[k] u[k]``
    (:func:`effective_matrices`); the step is ``x_next = Ad x + Bd u``.  It
    is the discretization :class:`DiscreteStatespaceSystem` steps with, so
    a caller that needs the one-step map itself (an inverse model, an
    initial guess) gets the very same matrices.

    Args:
        A: The state matrix, ``(..., n, n)``.
        B: The input matrix, ``(..., n, m)``.
        E: The state-input coupling, ``(..., m, n, n)``, or ``None``.
        F: The input-input coupling, ``(..., m, n, m)``, or ``None``.
        u: The input held over the step, ``(..., m)``.
        step_size: The step in seconds.
        transform_mode: Use the matrix exponential that ``torch.func``
            transforms can batch (``True``) or ``torch.matrix_exp``
            (``False``).  ``None`` (default) chooses by whether a transform
            is active.

    Returns:
        ``(Ad, Bd)``, shaped ``(..., n, n)`` and ``(..., n, m)``.
    """
    return _discretize_onestep(A, B, E, F, u, step_size, transform_mode=transform_mode)
