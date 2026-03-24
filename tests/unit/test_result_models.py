"""Tests for result data models."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from inferbench.results.models import (
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSession,
    ResourceSnapshot,
)
from inferbench.results.storage import load_session, save_session


class TestBenchmarkRun:
    def _make_run(self, **overrides) -> BenchmarkRun:
        defaults = {
            "prompt_name": "short_chat",
            "prompt_text": "Hello",
            "prompt_tokens": 5,
            "max_output_tokens": 256,
            "temperature": 0.0,
            "output_text": "Hi there",
            "output_tokens": 10,
            "start_ns": 1_000_000_000,
            "first_token_ns": 1_050_000_000,
            "end_ns": 1_500_000_000,
        }
        defaults.update(overrides)
        return BenchmarkRun(**defaults)

    def test_ttft_ms(self):
        run = self._make_run()
        assert run.ttft_ms == pytest.approx(50.0)

    def test_tps(self):
        run = self._make_run(output_tokens=10)
        # 9 tokens in 450ms = 20 TPS
        assert run.tps == pytest.approx(20.0)

    def test_tps_single_token(self):
        run = self._make_run(output_tokens=1)
        assert run.tps == 0.0

    def test_e2e_latency(self):
        run = self._make_run()
        assert run.e2e_latency_ms == pytest.approx(500.0)

    def test_itl_ms(self):
        run = self._make_run(
            token_timestamps_ns=[
                1_050_000_000,
                1_100_000_000,
                1_200_000_000,
            ]
        )
        itls = run.itl_ms
        assert len(itls) == 2
        assert itls[0] == pytest.approx(50.0)
        assert itls[1] == pytest.approx(100.0)

    def test_itl_empty(self):
        run = self._make_run(token_timestamps_ns=[])
        assert run.itl_ms == []


class TestBenchmarkResult:
    def _make_result(self, runs: list[BenchmarkRun] | None = None) -> BenchmarkResult:
        if runs is None:
            runs = [
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
                    resources=ResourceSnapshot(ram_peak_mb=4096.0),
                ),
                BenchmarkRun(
                    prompt_name="short_chat",
                    prompt_text="Hi",
                    prompt_tokens=3,
                    max_output_tokens=256,
                    temperature=0.0,
                    output_text="Hey",
                    output_tokens=8,
                    start_ns=2_000_000_000,
                    first_token_ns=2_060_000_000,
                    end_ns=2_400_000_000,
                    token_timestamps_ns=[2_060_000_000 + i * 48_000_000 for i in range(8)],
                    resources=ResourceSnapshot(ram_peak_mb=4200.0),
                ),
            ]
        return BenchmarkResult(
            backend_name="ollama",
            backend_version="0.12.11",
            model_id="test:7b",
            cold_load_time_s=5.0,
            warm_load_time_s=0.5,
            runs=runs,
        )

    def test_avg_tps(self):
        result = self._make_result()
        assert result.avg_tps > 0

    def test_avg_ttft(self):
        result = self._make_result()
        assert result.avg_ttft_ms > 0

    def test_peak_ram(self):
        result = self._make_result()
        assert result.peak_ram_mb == 4200.0

    def test_itl_p50(self):
        result = self._make_result()
        assert result.itl_p50_ms > 0

    def test_empty_runs(self):
        result = self._make_result(runs=[])
        assert result.avg_tps == 0.0
        assert result.avg_ttft_ms == 0.0
        assert result.peak_ram_mb == 0.0

    def test_serialization_roundtrip(self):
        result = self._make_result()
        data = result.model_dump(mode="json")
        json_str = json.dumps(data, default=str)
        restored = BenchmarkResult.model_validate(json.loads(json_str))
        assert restored.backend_name == "ollama"
        assert len(restored.runs) == 2


class TestBenchmarkSession:
    def test_create(self):
        session = BenchmarkSession()
        assert session.session_id
        assert session.started_at
        assert session.completed_at is None
        assert session.results == []

    def test_save_and_load(self):
        session = BenchmarkSession(
            results=[
                BenchmarkResult(
                    backend_name="test",
                    backend_version="1.0",
                    model_id="test:7b",
                    runs=[],
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_session.json"
            saved_path = save_session(session, path)
            assert saved_path.exists()

            loaded = load_session(saved_path)
            assert loaded.session_id == session.session_id
            assert len(loaded.results) == 1
            assert loaded.results[0].backend_name == "test"


class TestResourceSnapshot:
    def test_defaults(self):
        snap = ResourceSnapshot()
        assert snap.ram_used_mb == 0.0
        assert snap.gpu_percent is None

    def test_with_values(self):
        snap = ResourceSnapshot(
            ram_used_mb=8192.0,
            ram_peak_mb=9000.0,
            cpu_percent=45.0,
            gpu_percent=80.0,
        )
        assert snap.ram_peak_mb == 9000.0
        assert snap.gpu_percent == 80.0
