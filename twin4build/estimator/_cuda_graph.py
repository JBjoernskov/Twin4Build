"""Small reusable direct-CUDA-Graph callable wrapper."""

import torch


class CudaGraphCallable:
    """Capture and replay a fixed-shape tensor-only callable."""

    def __init__(self, fn, warmup_calls=3):
        self.fn = fn
        self.warmup_calls = int(warmup_calls)
        self.graph = None
        self.static_inputs = None
        self.static_output = None

    def _capture(self, inputs):
        self.static_inputs = tuple(torch.empty_like(value) for value in inputs)
        for target, value in zip(self.static_inputs, inputs):
            target.copy_(value)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.warmup_calls):
                reference_output = self.fn(*self.static_inputs)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.fn(*self.static_inputs)
        self.graph.replay()
        torch.testing.assert_close(
            self.static_output,
            reference_output,
            rtol=1e-9,
            atol=1e-11,
            msg="Direct CUDA Graph replay differs from eager output",
        )
        for index, target in enumerate(self.static_inputs):
            target.add_((index + 1) * 1e-4)
        probe_output = self.fn(*self.static_inputs)
        self.graph.replay()
        torch.testing.assert_close(
            self.static_output,
            probe_output,
            rtol=1e-9,
            atol=1e-11,
            msg="Direct CUDA Graph replay does not track changed inputs",
        )
        for target, value in zip(self.static_inputs, inputs):
            target.copy_(value)
        self.graph.replay()
        return self.static_output

    def __call__(self, *inputs):
        if self.graph is None:
            return self._capture(inputs)
        for target, value in zip(self.static_inputs, inputs):
            target.copy_(value)
        self.graph.replay()
        return self.static_output
