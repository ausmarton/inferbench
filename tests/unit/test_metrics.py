"""Tests for metrics computation."""

from __future__ import annotations

import pytest

from inferbench.benchmarks.metrics import ResultSummary, RunSummary, _percentile
from inferbench.results.models import BenchmarkResult, BenchmarkRun, ResourceSnapshot


class TestRunSummary:
    def test_from_run(self):
        run = BenchmarkRun(
            prompt_name="short_chat",
            prompt_text="Hi",
            prompt_tokens=3,
            max_output_tokens=256,
            temperature=0.0,
            output_text="Hello there",
            output_tokens=10,
            start_ns=1_000_000_000,
            first_token_ns=1_050_000_000,
            end_ns=1_500_000_000,
            token_timestamps_ns=[1_050_000_000 + i * 50_000_000 for i in range(10)],
            resources=ResourceSnapshot(ram_peak_mb=4096.0),
        )
        summary = RunSummary.from_run(run)
        assert summary.prompt_name == "short_chat"
        assert summary.output_tokens == 10
        assert summary.ttft_ms == pytest.approx(50.0)
        assert summary.tps > 0
        assert summary.ram_peak_mb == 4096.0


class TestResultSummary:
    def test_from_result(self):
        result = BenchmarkResult(
            backend_name="ollama",
            backend_version="0.12.11",
            model_id="test:7b",
            cold_load_time_s=5.0,
            warm_load_time_s=0.5,
            runs=[
                BenchmarkRun(
                    prompt_name="short_chat",
                    prompt_text="Hi",
                    prompt_tokens=3,
                    max_output_tokens=256,
                    temperature=0.0,
                    output_text="Hello",
                    output_tokens=10,
                    start_ns=1_000_000_000,
                    first_token_ns=1_050_000_000,
                    end_ns=1_500_000_000,
                    token_timestamps_ns=[1_050_000_000 + i * 50_000_000 for i in range(10)],
                )
            ],
        )
        summary = ResultSummary.from_result(result)
        assert summary.backend_name == "ollama"
        assert summary.model_id == "test:7b"
        assert summary.cold_load_s == 5.0
        assert summary.avg_tps > 0
        assert summary.total_runs == 1


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 0.5) == 0.0

    def test_single(self):
        assert _percentile([10.0], 0.5) == 10.0

    def test_p99(self):
        values = list(range(100))
        assert _percentile(values, 0.99) == 99

    def test_p50(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 0.5) == 3.0
