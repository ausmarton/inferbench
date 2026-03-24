"""Tests for the benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from inferbench.backends.base import GenerationResult, InferenceBackend, LoadedModel, TokenEvent
from inferbench.benchmarks.prompts import ALL_PROMPTS
from inferbench.benchmarks.runner import BenchmarkRunner, _build_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class MockBackend(InferenceBackend):
    """A mock backend for testing the runner."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "Mock"

    def is_available(self) -> bool:
        return True

    def get_version(self) -> str:
        return "1.0.0"

    def supported_model_ids(self, hardware):
        return ["mock-model:7b"]

    async def load_model(self, model_id: str) -> LoadedModel:
        return LoadedModel(
            backend_name=self._name,
            model_id=model_id,
            load_time_s=0.1,
        )

    async def unload_model(self, handle: LoadedModel) -> None:
        pass

    async def generate(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        now = self._now_ns()
        return GenerationResult(
            output_text="Mock response token1 token2 token3",
            token_count=5,
            prompt_token_count=10,
            start_ns=now,
            first_token_ns=now + 10_000_000,  # 10ms TTFT
            end_ns=now + 60_000_000,  # 60ms total
            token_timestamps_ns=[now + 10_000_000 + i * 10_000_000 for i in range(5)],
        )

    async def generate_stream(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[TokenEvent]:
        yield TokenEvent(token="mock", timestamp_ns=self._now_ns())


class TestBenchmarkRunner:
    def test_init_validates_prompts(self):
        with pytest.raises(ValueError, match="Unknown prompt"):
            BenchmarkRunner(
                backends=[MockBackend()],
                model_ids={"mock": ["model:7b"]},
                prompt_names=["nonexistent_prompt"],
            )

    def test_init_accepts_valid_prompts(self):
        runner = BenchmarkRunner(
            backends=[MockBackend()],
            model_ids={"mock": ["model:7b"]},
            prompt_names=["short_chat"],
            warmup_iterations=0,
            timed_iterations=1,
        )
        assert runner.prompt_names == ["short_chat"]
        assert runner.timed_iterations == 1

    def test_run_produces_session(self):
        backend = MockBackend()
        runner = BenchmarkRunner(
            backends=[backend],
            model_ids={"mock": ["model:7b"]},
            prompt_names=["short_chat"],
            warmup_iterations=0,
            timed_iterations=1,
        )
        hardware = MagicMock()
        hardware.model_dump.return_value = {"cpu": {}, "gpus": [], "memory": {}}

        session = runner.run(hardware)

        assert session.completed_at is not None
        assert len(session.results) == 1
        assert session.results[0].backend_name == "mock"
        assert session.results[0].model_id == "model:7b"
        assert len(session.results[0].runs) == 1

    def test_run_multiple_iterations(self):
        backend = MockBackend()
        runner = BenchmarkRunner(
            backends=[backend],
            model_ids={"mock": ["model:7b"]},
            prompt_names=["short_chat"],
            warmup_iterations=1,
            timed_iterations=3,
        )
        hardware = MagicMock()
        hardware.model_dump.return_value = {}

        session = runner.run(hardware)

        assert len(session.results) == 1
        assert len(session.results[0].runs) == 3

    def test_run_multiple_prompts(self):
        backend = MockBackend()
        runner = BenchmarkRunner(
            backends=[backend],
            model_ids={"mock": ["model:7b"]},
            prompt_names=["short_chat", "reasoning"],
            warmup_iterations=0,
            timed_iterations=1,
        )
        hardware = MagicMock()
        hardware.model_dump.return_value = {}

        session = runner.run(hardware)

        assert len(session.results) == 1
        assert len(session.results[0].runs) == 2
        prompt_names = {r.prompt_name for r in session.results[0].runs}
        assert prompt_names == {"short_chat", "reasoning"}

    def test_run_skips_missing_backend(self):
        backend = MockBackend()
        runner = BenchmarkRunner(
            backends=[backend],
            model_ids={"other_backend": ["model:7b"]},  # No models for "mock"
            prompt_names=["short_chat"],
            warmup_iterations=0,
            timed_iterations=1,
        )
        hardware = MagicMock()
        hardware.model_dump.return_value = {}

        session = runner.run(hardware)
        assert len(session.results) == 0


class TestBuildPrompt:
    def test_with_system_prompt(self):
        spec = ALL_PROMPTS["short_chat"]
        result = _build_prompt(spec)
        assert spec.system_prompt in result
        assert spec.user_prompt in result

    def test_all_prompts_build(self):
        for name, spec in ALL_PROMPTS.items():
            result = _build_prompt(spec)
            assert spec.user_prompt in result, f"Failed for {name}"
