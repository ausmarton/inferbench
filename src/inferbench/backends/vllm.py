"""vLLM inference backend — manages a server subprocess, communicates via OpenAI API."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
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

DEFAULT_PORT = 8100
SERVER_STARTUP_TIMEOUT = 300  # seconds
GENERATE_TIMEOUT = 300.0


@register("vllm")
class VllmBackend(InferenceBackend):
    """Backend that manages a vLLM OpenAI-compatible server subprocess.

    vLLM provides high-throughput GPU inference with PagedAttention.
    Each model load starts a new server process; unload kills it.
    """

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self._port = port
        self._base_url = f"http://localhost:{port}"
        self._server_process: subprocess.Popen | None = None

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def display_name(self) -> str:
        return "vLLM"

    def is_available(self) -> bool:
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def get_version(self) -> str:
        try:
            import vllm

            return vllm.__version__
        except ImportError:
            return "not installed"

    def get_install_hint(self) -> str | None:
        return 'uv sync --extra vllm  # or: pip install "inferbench[vllm]"'

    def supported_model_ids(self, hardware: HardwareProfile) -> list[str]:
        """Return model IDs from the catalog that have vllm backend entries.

        vLLM generally requires a GPU with sufficient VRAM.
        """
        if not hardware.has_gpu:
            return []

        from inferbench.catalog.registry import filter_by_backend, load_catalog

        catalog = load_catalog()
        ids = []
        for model in filter_by_backend(catalog, self.name):
            backend_id = model.backend_ids.get(self.name)
            if backend_id:
                ids.append(backend_id)
        return ids

    async def load_model(self, model_id: str) -> LoadedModel:
        """Start a vLLM server subprocess for the given model."""
        start_ns = self._now_ns()

        # Kill any existing server
        await self._stop_server()

        # Start vLLM server
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_id,
            "--port",
            str(self._port),
            "--dtype",
            "auto",
        ]

        logger.info("Starting vLLM server: %s", " ".join(cmd))
        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        await self._wait_for_server()

        load_time_s = (self._now_ns() - start_ns) / 1e9

        return LoadedModel(
            backend_name=self.name,
            model_id=model_id,
            load_time_s=load_time_s,
            metadata={"port": self._port, "pid": self._server_process.pid},
        )

    async def unload_model(self, handle: LoadedModel) -> None:
        await self._stop_server()

    async def generate(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate via vLLM's OpenAI-compatible streaming API."""
        token_timestamps: list[int] = []
        tokens: list[str] = []
        prompt_token_count = 0

        start_ns = self._now_ns()
        first_token_ns = 0

        async with (
            httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client,
            client.stream(
                "POST",
                f"{self._base_url}/v1/completions",
                json={
                    "model": handle.model_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
            ) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # Strip "data: " prefix
                if data_str.strip() == "[DONE]":
                    break

                chunk = json.loads(data_str)
                text = chunk.get("choices", [{}])[0].get("text", "")
                if text:
                    now_ns = self._now_ns()
                    if not first_token_ns:
                        first_token_ns = now_ns
                    token_timestamps.append(now_ns)
                    tokens.append(text)

                usage = chunk.get("usage")
                if usage:
                    prompt_token_count = usage.get("prompt_tokens", 0)

        end_ns = self._now_ns()
        if not first_token_ns:
            first_token_ns = end_ns

        return GenerationResult(
            output_text="".join(tokens),
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
        """Stream tokens from vLLM's OpenAI-compatible API."""
        async with (
            httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client,
            client.stream(
                "POST",
                f"{self._base_url}/v1/completions",
                json={
                    "model": handle.model_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
            ) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    return

                chunk = json.loads(data_str)
                text = chunk.get("choices", [{}])[0].get("text", "")
                if text:
                    yield TokenEvent(
                        token=text,
                        timestamp_ns=time.perf_counter_ns(),
                    )

    async def _wait_for_server(self) -> None:
        """Wait for the vLLM server to become ready."""
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            # Check if process died
            if self._server_process and self._server_process.poll() is not None:
                stderr = self._server_process.stderr
                error = stderr.read().decode() if stderr else "unknown error"
                rc = self._server_process.returncode
                msg = f"vLLM server exited with code {rc}: {error[:500]}"
                raise RuntimeError(msg)

            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self._base_url}/v1/models")
                    if resp.status_code == 200:
                        logger.info("vLLM server ready")
                        return
            except httpx.HTTPError:
                pass

            await asyncio.sleep(2.0)

        msg = f"vLLM server did not start within {SERVER_STARTUP_TIMEOUT}s"
        raise TimeoutError(msg)

    async def _stop_server(self) -> None:
        """Stop the vLLM server subprocess."""
        if self._server_process is not None:
            logger.info("Stopping vLLM server (pid %d)", self._server_process.pid)
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait(timeout=5)
            self._server_process = None
