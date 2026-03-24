"""Ollama inference backend — communicates via REST API."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import httpx

from inferbench.backends.base import (
    GenerationResult,
    InferenceBackend,
    LoadedModel,
    TokenEvent,
)
from inferbench.backends.registry import register

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from inferbench.hardware.models import HardwareProfile

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
CONNECT_TIMEOUT = 5.0
GENERATE_TIMEOUT = 300.0


@register("ollama")
class OllamaBackend(InferenceBackend):
    """Backend that communicates with a running Ollama server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._version: str | None = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama"

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            with httpx.Client(timeout=CONNECT_TIMEOUT) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def get_version(self) -> str:
        """Get Ollama server version."""
        if self._version:
            return self._version
        try:
            with httpx.Client(timeout=CONNECT_TIMEOUT) as client:
                resp = client.get(f"{self._base_url}/api/version")
                if resp.status_code == 200:
                    self._version = resp.json().get("version", "unknown")
                    return self._version
        except httpx.HTTPError:
            pass
        return "unknown"

    def get_install_hint(self) -> str | None:
        return "curl -fsSL https://ollama.com/install.sh | sh"

    def supported_model_ids(self, hardware: HardwareProfile) -> list[str]:
        """Query Ollama for locally available models."""
        try:
            with httpx.Client(timeout=CONNECT_TIMEOUT) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                if resp.status_code != 200:
                    return []
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError:
            return []

    async def load_model(self, model_id: str) -> LoadedModel:
        """Load a model in Ollama by triggering a dummy generation.

        Ollama loads models lazily on first request. We send a minimal
        prompt to force the load and measure the time.
        """
        start_ns = self._now_ns()

        async with httpx.AsyncClient(base_url=self._base_url, timeout=GENERATE_TIMEOUT) as client:
            # Use the generate endpoint with keep_alive to force load
            resp = await client.post(
                "/api/generate",
                json={
                    "model": model_id,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
            resp.raise_for_status()

        load_time_s = (self._now_ns() - start_ns) / 1e9

        return LoadedModel(
            backend_name=self.name,
            model_id=model_id,
            load_time_s=load_time_s,
            metadata={"base_url": self._base_url},
        )

    async def unload_model(self, handle: LoadedModel) -> None:
        """Unload a model from Ollama's memory."""
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=CONNECT_TIMEOUT
            ) as client:
                await client.post(
                    "/api/generate",
                    json={
                        "model": handle.model_id,
                        "prompt": "",
                        "keep_alive": 0,
                    },
                )
        except httpx.HTTPError:
            logger.debug("Failed to unload model %s from Ollama", handle.model_id)

    async def generate(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate a response using Ollama's streaming API for token-level timing."""
        token_timestamps: list[int] = []
        tokens: list[str] = []
        prompt_token_count = 0

        start_ns = self._now_ns()
        first_token_ns = 0

        async with (
            httpx.AsyncClient(base_url=self._base_url, timeout=GENERATE_TIMEOUT) as client,
            client.stream(
                "POST",
                "/api/generate",
                json={
                    "model": handle.model_id,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            ) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)

                if chunk.get("response"):
                    now_ns = self._now_ns()
                    if not first_token_ns:
                        first_token_ns = now_ns
                    token_timestamps.append(now_ns)
                    tokens.append(chunk["response"])

                if chunk.get("done"):
                    prompt_token_count = chunk.get("prompt_eval_count", 0)

        end_ns = self._now_ns()

        if not first_token_ns:
            first_token_ns = end_ns

        output_text = "".join(tokens)

        return GenerationResult(
            output_text=output_text,
            token_count=len(tokens),
            prompt_token_count=prompt_token_count,
            start_ns=start_ns,
            first_token_ns=first_token_ns,
            end_ns=end_ns,
            token_timestamps_ns=token_timestamps,
        )

    async def generate_stream(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[TokenEvent]:
        """Stream tokens from Ollama with per-token timing."""
        async with (
            httpx.AsyncClient(base_url=self._base_url, timeout=GENERATE_TIMEOUT) as client,
            client.stream(
                "POST",
                "/api/generate",
                json={
                    "model": handle.model_id,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            ) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)

                if chunk.get("response"):
                    yield TokenEvent(
                        token=chunk["response"],
                        timestamp_ns=time.perf_counter_ns(),
                    )

                if chunk.get("done"):
                    return
