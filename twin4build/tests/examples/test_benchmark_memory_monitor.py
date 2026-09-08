"""The benchmark harness' peak-memory monitor must never call the CUDA runtime
from its sampler thread.

PyTorch captures CUDA Graphs with ``capture_error_mode="global"``: a
potentially unsafe CUDA API call (``cudaMemGetInfo`` included) from *any*
thread while a capture is in flight invalidates the capture.  An earlier
sampler polled ``torch.cuda.mem_get_info`` at 1 Hz and every multi-zone
single-shooting case on an A100 died with an illegal memory access right after
graph capture.  These tests pin the contract with fake CUDA hooks so they run
on CPU-only machines.
"""

from __future__ import annotations

import os
import threading
import time

import pytest
import torch

os.environ.setdefault("T4B_BENCHMARK_MODE", "smoke")
common = pytest.importorskip("benchmarks.common")


@pytest.fixture
def fake_cuda(monkeypatch):
    calls = {"mem_get_info": [], "memory_reserved": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 1_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 2_000)

    def mem_get_info(*a, **k):
        calls["mem_get_info"].append(threading.current_thread().name)
        return (7_000, 10_000)

    def memory_reserved(*a, **k):
        calls["memory_reserved"].append(threading.current_thread().name)
        return 1_500

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.cuda, "memory_reserved", memory_reserved)
    monkeypatch.setattr(common, "MEMORY_SAMPLER_INTERVAL_SECONDS", 0.05)
    return calls


def test_sampler_thread_never_calls_the_cuda_runtime(fake_cuda):
    main = threading.current_thread().name
    _, seconds = common.timed("cuda", lambda: time.sleep(0.6))
    assert seconds >= 0.5
    # Device-wide queries happen on the main thread only: once before, once after.
    assert fake_cuda["mem_get_info"], "device memory was never read"
    assert set(fake_cuda["mem_get_info"]) == {main}
    # The sampler thread did run, and used allocator bookkeeping only.
    threads = set(fake_cuda["memory_reserved"]) - {main}
    assert threads == {"t4b-benchmark-memory"}


def test_memory_stats_carry_baseline_end_and_allocator_peaks(fake_cuda):
    common.timed("cuda", lambda: time.sleep(0.2))
    stats = common.memory_stats_of_last_timed()
    assert stats["cuda_device_used_baseline_bytes"] == 3_000
    assert stats["cuda_device_used_end_bytes"] == 3_000
    assert stats["cuda_device_total_bytes"] == 10_000
    assert stats["torch_cuda_max_reserved_bytes"] == 2_000
    assert stats["torch_cuda_reserved_peak_sampled_bytes"] == 1_500
    assert stats["cuda_oversubscribed"] is False
