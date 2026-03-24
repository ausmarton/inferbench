"""Tests for backend registry and base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from inferbench.backends.base import (
    GenerationResult,
    InferenceBackend,
    LoadedModel,
    TokenEvent,
)
from inferbench.backends.registry import (
    _BACKEND_CLASSES,
    get_all_backends,
    get_backend,
    register,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class TestGenerationResult:
    def test_ttft_ms(self):
        result = GenerationResult(
            output_text="hello",
            token_count=5,
            prompt_token_count=3,
            start_ns=1_000_000_000,
            first_token_ns=1_050_000_000,
            end_ns=1_500_000_000,
        )
        assert result.ttft_ms == pytest.approx(50.0)

    def test_total_time_s(self):
        result = GenerationResult(
            output_text="hello",
            token_count=5,
            prompt_token_count=3,
            start_ns=1_000_000_000,
            first_token_ns=1_050_000_000,
            end_ns=1_500_000_000,
        )
        assert result.total_time_s == pytest.approx(0.5)

    def test_tokens_per_second(self):
        result = GenerationResult(
            output_text="hello world foo bar",
            token_count=4,
            prompt_token_count=3,
            start_ns=1_000_000_000,
            first_token_ns=1_100_000_000,
            end_ns=1_400_000_000,
        )
        # 3 tokens generated in 0.3s after first token = 10 TPS
        assert result.tokens_per_second == pytest.approx(10.0)

    def test_tokens_per_second_single_token(self):
        result = GenerationResult(
            output_text="hi",
            token_count=1,
            prompt_token_count=3,
            start_ns=1_000_000_000,
            first_token_ns=1_050_000_000,
            end_ns=1_050_000_000,
        )
        assert result.tokens_per_second == 0.0

    def test_inter_token_latencies(self):
        result = GenerationResult(
            output_text="hello",
            token_count=3,
            prompt_token_count=2,
            start_ns=1_000_000_000,
            first_token_ns=1_050_000_000,
            end_ns=1_150_000_000,
            token_timestamps_ns=[
                1_050_000_000,
                1_080_000_000,
                1_150_000_000,
            ],
        )
        latencies = result.inter_token_latencies_ms
        assert len(latencies) == 2
        assert latencies[0] == pytest.approx(30.0)
        assert latencies[1] == pytest.approx(70.0)


class TestLoadedModel:
    def test_fields(self):
        handle = LoadedModel(
            backend_name="test",
            model_id="test-model:7b",
            load_time_s=2.5,
            memory_delta_mb=4096.0,
        )
        assert handle.backend_name == "test"
        assert handle.model_id == "test-model:7b"
        assert handle.load_time_s == 2.5


class TestTokenEvent:
    def test_fields(self):
        event = TokenEvent(token="hello", timestamp_ns=123456789)
        assert event.token == "hello"
        assert event.token_id is None


class TestRegistry:
    def setup_method(self):
        """Clear the registry before each test."""
        self._original = dict(_BACKEND_CLASSES)
        _BACKEND_CLASSES.clear()

    def teardown_method(self):
        """Restore the registry after each test."""
        _BACKEND_CLASSES.clear()
        _BACKEND_CLASSES.update(self._original)

    def test_register_decorator(self):
        @register("test_backend")
        class TestBackend(InferenceBackend):
            @property
            def name(self):
                return "test_backend"

            @property
            def display_name(self):
                return "Test"

            def is_available(self):
                return True

            def get_version(self):
                return "1.0"

            def supported_model_ids(self, hardware):
                return []

            async def load_model(self, model_id):
                return LoadedModel("test", model_id, 0.0)

            async def unload_model(self, handle):
                pass

            async def generate(self, handle, prompt, **kwargs):
                return GenerationResult("", 0, 0, 0, 0, 0)

            async def generate_stream(self, handle, prompt, **kwargs) -> AsyncIterator[TokenEvent]:
                yield TokenEvent(token="", timestamp_ns=0)

        assert "test_backend" in _BACKEND_CLASSES
        assert _BACKEND_CLASSES["test_backend"] is TestBackend

    def test_get_all_backends_discovers(self):
        # After clearing, get_all_backends should trigger discovery
        with patch("inferbench.backends.registry._discover_backends") as mock_discover:
            get_all_backends()
            mock_discover.assert_called_once()

    def test_get_backend_unknown(self):
        _BACKEND_CLASSES["dummy"] = type("Dummy", (), {})  # prevent discovery
        assert get_backend("nonexistent") is None
