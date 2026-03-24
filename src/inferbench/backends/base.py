"""Abstract base class for inference backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from inferbench.hardware.models import HardwareProfile

# Re-exported for convenience by backend implementations
__all__ = [
    "GenerationResult",
    "InferenceBackend",
    "LoadedModel",
    "TokenEvent",
]


@dataclass
class TokenEvent:
    """A single token emission event during streaming generation."""

    token: str
    timestamp_ns: int  # time.perf_counter_ns()
    token_id: int | None = None


@dataclass
class LoadedModel:
    """Handle representing a model loaded in a backend."""

    backend_name: str
    model_id: str
    load_time_s: float
    memory_delta_mb: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Complete result from a single generation call."""

    output_text: str
    token_count: int
    prompt_token_count: int

    # Timing
    start_ns: int  # time.perf_counter_ns() at request start
    first_token_ns: int  # time.perf_counter_ns() at first token
    end_ns: int  # time.perf_counter_ns() at completion
    token_timestamps_ns: list[int] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        return (self.first_token_ns - self.start_ns) / 1e6

    @property
    def total_time_s(self) -> float:
        """Total generation time in seconds."""
        return (self.end_ns - self.start_ns) / 1e9

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second (excluding TTFT)."""
        if self.token_count <= 1:
            return 0.0
        gen_time_ns = self.end_ns - self.first_token_ns
        if gen_time_ns <= 0:
            return 0.0
        return (self.token_count - 1) / (gen_time_ns / 1e9)

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        """Latency between consecutive tokens in milliseconds."""
        if len(self.token_timestamps_ns) < 2:
            return []
        return [
            (self.token_timestamps_ns[i + 1] - self.token_timestamps_ns[i]) / 1e6
            for i in range(len(self.token_timestamps_ns) - 1)
        ]


class InferenceBackend(ABC):
    """Abstract base class for all inference backends.

    Each backend must implement the core lifecycle:
    is_available -> load_model -> generate -> unload_model
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g. 'ollama', 'llamacpp')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g. 'Ollama', 'llama.cpp')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's dependencies are installed and functional."""
        ...

    @abstractmethod
    def get_version(self) -> str:
        """Return the backend's version string."""
        ...

    def get_install_hint(self) -> str | None:
        """Return a hint for how to install this backend, or None if built-in."""
        return None

    @abstractmethod
    def supported_model_ids(self, hardware: HardwareProfile) -> list[str]:
        """Return model identifiers this backend can serve on the given hardware.

        For backends that manage their own model registry (like Ollama),
        this queries the backend directly. For others, it returns IDs
        from the catalog that are compatible.
        """
        ...

    @abstractmethod
    async def load_model(self, model_id: str) -> LoadedModel:
        """Load a model and return a handle.

        The model_id is the backend-specific identifier
        (e.g. 'qwen2.5:7b' for Ollama).
        """
        ...

    @abstractmethod
    async def unload_model(self, handle: LoadedModel) -> None:
        """Release model resources."""
        ...

    @abstractmethod
    async def generate(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Run inference and return detailed timing results."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[TokenEvent]:
        """Stream tokens with per-token timing.

        Must yield at least one TokenEvent.
        """
        ...
        # Make this an async generator at the type level
        yield  # type: ignore[misc]  # pragma: no cover

    @asynccontextmanager
    async def loaded(self, model_id: str):
        """Context manager for load/unload lifecycle."""
        handle = await self.load_model(model_id)
        try:
            yield handle
        finally:
            await self.unload_model(handle)

    @staticmethod
    def _now_ns() -> int:
        """High-resolution monotonic timestamp."""
        return time.perf_counter_ns()
