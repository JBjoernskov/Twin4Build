# Benchmarks

Scripts and notebooks that measure twin4build, run from a source checkout.
They are not part of the installed package.

- `run_smoke.py`: runs the canonical simulation, estimation and optimization
  benchmarks as correctness smoke tests (their timings are diagnostic only).
- `*_scaling_benchmark.ipynb`: the scaling benchmarks.

## `small_matmul_crossover.py`

Measures, on a CUDA GPU, the value to give the `small_matmul_max` setting.
While `torch.compile` traces a step, a matrix exponential up to that size
uses broadcast products, which Inductor fuses. A larger one uses `bmm`. Linux
or WSL is needed (Triton).

```
python benchmarks/small_matmul_crossover.py --matrices 16 960 --sizes 8 12 16 24 32 64
```

- `--matrices`: how many matrices are exponentiated per call (blocks times
  shooting windows of the runs you tune for).
- `--sizes`: the matrix sizes `N`. A block of `n` states is `N = n + 1` with
  `zoh_expm="aug"`, and `N = n` with `"phi1"`.
- `--dtype`, `--reps`: the dtype (`float32` or `float64`) and the repetitions
  per timing.

It prints one block per batch. Each row gives the median ms of one
exponential, forward and backward, with fused products and with `bmm`, and
says which is faster. The block ends with the recommended value, the largest
size up to which the fused form is faster, next to the current value and its
source:

```
NVIDIA ..., torch 2.x, float32: ms per exponential (forward + backward)
  960 matrices per call
    N=  12: fused    1.234  bmm    1.567  (fused)
    ...
  -> small_matmul_max = 12 for 960 matrices per call (now 16, from default)
```

Use the value with `T4B_SMALL_MATMUL_MAX=<n>` or
`tb.settings.set(small_matmul_max=<n>)`, before the step is compiled.
