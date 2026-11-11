"""Inference-latency measurement against the IEC 61850 GOOSE transfer-time budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn

__all__ = ["GOOSE_BUDGET_MS", "LatencyStats", "measure_inference_latency"]

# IEC 61850-5 type 1A/P1 transfer-time envelope the paper reports against.
GOOSE_BUDGET_MS: float = 4.0


@dataclass
class LatencyStats:
    """Latency summary in milliseconds; percentiles are the paper's reporting unit."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def measure_inference_latency(
    model: nn.Module, x: Tensor, n: int = 1000, device: str = "cpu", warmup: int = 50
) -> LatencyStats:
    """Wall-clock single-inference latency over ``n`` timed runs after ``warmup``.

    Timing uses ``time.perf_counter_ns`` (monotonic, immune to wall-clock
    adjustments). CUDA kernel launches are asynchronous, so when ``device`` is
    a CUDA device ``torch.cuda.synchronize`` is issued before the run and after
    every timed forward — without it the numbers would measure launch overhead,
    not inference. The forward runs under ``torch.no_grad`` in eval mode (the
    deployment configuration); the model's previous training mode is restored
    afterwards. Latency is per forward call on ``x`` exactly as given — pass a
    batch of 1 for the paper's per-window numbers.
    """
    import time

    import numpy as np
    import torch

    if n < 1:
        raise ValueError(f"need at least one timed run, got n={n}")
    dev = torch.device(device)
    is_cuda = dev.type == "cuda"
    model = model.to(dev)
    x = x.to(dev)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            if is_cuda:
                torch.cuda.synchronize(dev)
            samples_ms = np.empty(n, dtype=np.float64)
            for i in range(n):
                t0 = time.perf_counter_ns()
                model(x)
                if is_cuda:
                    torch.cuda.synchronize(dev)
                samples_ms[i] = (time.perf_counter_ns() - t0) / 1e6
    finally:
        model.train(was_training)
    return LatencyStats(
        mean_ms=float(samples_ms.mean()),
        p50_ms=float(np.percentile(samples_ms, 50)),
        p95_ms=float(np.percentile(samples_ms, 95)),
        p99_ms=float(np.percentile(samples_ms, 99)),
    )
