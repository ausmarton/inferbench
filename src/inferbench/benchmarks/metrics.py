"""Metric computation utilities for benchmark results."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferbench.results.models import BenchmarkResult, BenchmarkRun


@dataclass
class RunSummary:
    """Aggregated metrics for a single benchmark run."""

    prompt_name: str
    output_tokens: int
    ttft_ms: float
    tps: float
    e2e_latency_ms: float
    itl_p50_ms: float
    itl_p99_ms: float
    ram_peak_mb: float

    @classmethod
    def from_run(cls, run: BenchmarkRun) -> RunSummary:
        itls = run.itl_ms
        return cls(
            prompt_name=run.prompt_name,
            output_tokens=run.output_tokens,
            ttft_ms=run.ttft_ms,
            tps=run.tps,
            e2e_latency_ms=run.e2e_latency_ms,
            itl_p50_ms=statistics.median(itls) if itls else 0.0,
            itl_p99_ms=_percentile(itls, 0.99) if itls else 0.0,
            ram_peak_mb=run.resources.ram_peak_mb,
        )


@dataclass
class ResultSummary:
    """Aggregated metrics across all runs in a BenchmarkResult."""

    backend_name: str
    model_id: str
    cold_load_s: float
    warm_load_s: float
    avg_tps: float
    p50_tps: float
    avg_ttft_ms: float
    p50_ttft_ms: float
    avg_e2e_ms: float
    itl_p50_ms: float
    itl_p99_ms: float
    peak_ram_mb: float
    total_runs: int

    @classmethod
    def from_result(cls, result: BenchmarkResult) -> ResultSummary:
        return cls(
            backend_name=result.backend_name,
            model_id=result.model_id,
            cold_load_s=result.cold_load_time_s,
            warm_load_s=result.warm_load_time_s,
            avg_tps=result.avg_tps,
            p50_tps=result.p50_tps,
            avg_ttft_ms=result.avg_ttft_ms,
            p50_ttft_ms=result.p50_ttft_ms,
            avg_e2e_ms=result.avg_e2e_ms,
            itl_p50_ms=result.itl_p50_ms,
            itl_p99_ms=result.itl_p99_ms,
            peak_ram_mb=result.peak_ram_mb,
            total_runs=len(result.runs),
        )


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile from a sorted or unsorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]
