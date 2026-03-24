"""Data models for benchmark results."""

from __future__ import annotations

import statistics
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ResourceSnapshot(BaseModel):
    """Resource usage captured during a benchmark run."""

    ram_used_mb: float = 0.0
    ram_peak_mb: float = 0.0
    cpu_percent: float = 0.0
    gpu_percent: float | None = None
    vram_used_mb: float | None = None
    vram_peak_mb: float | None = None


class BenchmarkRun(BaseModel):
    """A single prompt-response benchmark execution."""

    prompt_name: str
    prompt_text: str
    prompt_tokens: int
    max_output_tokens: int
    temperature: float

    output_text: str
    output_tokens: int

    # Timing (nanoseconds, from time.perf_counter_ns)
    start_ns: int
    first_token_ns: int
    end_ns: int
    token_timestamps_ns: list[int] = []

    # Resources
    resources: ResourceSnapshot = Field(default_factory=ResourceSnapshot)

    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        return (self.first_token_ns - self.start_ns) / 1e6

    @property
    def tps(self) -> float:
        """Output tokens per second (excluding TTFT)."""
        if self.output_tokens <= 1:
            return 0.0
        gen_ns = self.end_ns - self.first_token_ns
        if gen_ns <= 0:
            return 0.0
        return (self.output_tokens - 1) / (gen_ns / 1e9)

    @property
    def e2e_latency_ms(self) -> float:
        """End-to-end latency in milliseconds."""
        return (self.end_ns - self.start_ns) / 1e6

    @property
    def itl_ms(self) -> list[float]:
        """Inter-token latencies in milliseconds."""
        if len(self.token_timestamps_ns) < 2:
            return []
        return [
            (self.token_timestamps_ns[i + 1] - self.token_timestamps_ns[i]) / 1e6
            for i in range(len(self.token_timestamps_ns) - 1)
        ]


class BenchmarkResult(BaseModel):
    """Complete result for one backend + model combination."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # What was benchmarked
    backend_name: str
    backend_version: str
    model_id: str

    # Loading metrics
    cold_load_time_s: float = 0.0
    warm_load_time_s: float = 0.0
    model_memory_mb: float = 0.0

    # Per-prompt results (multiple iterations per prompt)
    runs: list[BenchmarkRun] = []

    @property
    def avg_tps(self) -> float:
        values = [r.tps for r in self.runs if r.tps > 0]
        return statistics.mean(values) if values else 0.0

    @property
    def avg_ttft_ms(self) -> float:
        values = [r.ttft_ms for r in self.runs if r.ttft_ms > 0]
        return statistics.mean(values) if values else 0.0

    @property
    def p50_tps(self) -> float:
        values = sorted(r.tps for r in self.runs if r.tps > 0)
        return statistics.median(values) if values else 0.0

    @property
    def p50_ttft_ms(self) -> float:
        values = sorted(r.ttft_ms for r in self.runs if r.ttft_ms > 0)
        return statistics.median(values) if values else 0.0

    @property
    def avg_e2e_ms(self) -> float:
        values = [r.e2e_latency_ms for r in self.runs if r.e2e_latency_ms > 0]
        return statistics.mean(values) if values else 0.0

    @property
    def itl_p50_ms(self) -> float:
        """Median inter-token latency across all runs."""
        all_itls = []
        for r in self.runs:
            all_itls.extend(r.itl_ms)
        return statistics.median(all_itls) if all_itls else 0.0

    @property
    def itl_p99_ms(self) -> float:
        """99th percentile inter-token latency."""
        all_itls = sorted(r for run in self.runs for r in run.itl_ms)
        if not all_itls:
            return 0.0
        idx = int(len(all_itls) * 0.99)
        return all_itls[min(idx, len(all_itls) - 1)]

    @property
    def peak_ram_mb(self) -> float:
        values = [r.resources.ram_peak_mb for r in self.runs if r.resources.ram_peak_mb > 0]
        return max(values) if values else 0.0


class BenchmarkSession(BaseModel):
    """A full benchmarking session across multiple backends/models."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Hardware context (serialized HardwareProfile)
    hardware: dict = {}

    # Configuration
    warmup_iterations: int = 1
    timed_iterations: int = 3
    prompt_names: list[str] = []

    # Results
    results: list[BenchmarkResult] = []

    metadata: dict = {}
