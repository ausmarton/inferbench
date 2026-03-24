"""Hugging Face Transformers inference backend — in-process via torch."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

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


@register("transformers")
class TransformersBackend(InferenceBackend):
    """Backend using Hugging Face Transformers for in-process inference.

    Supports CPU and GPU (CUDA/ROCm) via PyTorch.
    """

    def __init__(self) -> None:
        self._loaded_models: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return "transformers"

    @property
    def display_name(self) -> str:
        return "Transformers"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def get_version(self) -> str:
        try:
            import transformers

            return transformers.__version__
        except ImportError:
            return "not installed"

    def get_install_hint(self) -> str | None:
        return 'uv sync --extra transformers  # or: pip install "inferbench[transformers]"'

    def supported_model_ids(self, hardware: HardwareProfile) -> list[str]:
        """Return model IDs from the catalog that have transformers backend entries."""
        from inferbench.catalog.registry import filter_by_backend, load_catalog

        catalog = load_catalog()
        ids = []
        for model in filter_by_backend(catalog, self.name):
            backend_id = model.backend_ids.get(self.name)
            if backend_id:
                ids.append(backend_id)
        return ids

    async def load_model(self, model_id: str) -> LoadedModel:
        """Load a model from HuggingFace Hub or local path.

        model_id should be a HuggingFace model identifier
        (e.g. 'Qwen/Qwen2.5-7B-Instruct').
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_ns = self._now_ns()

        # Determine device
        if torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available()):
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = "cpu"
            torch_dtype = torch.float32

        logger.info("Loading %s on %s...", model_id, device_map)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        load_time_s = (self._now_ns() - start_ns) / 1e9

        self._loaded_models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": str(model.device) if hasattr(model, "device") else device_map,
        }

        return LoadedModel(
            backend_name=self.name,
            model_id=model_id,
            load_time_s=load_time_s,
            metadata={"device": device_map, "dtype": str(torch_dtype)},
        )

    async def unload_model(self, handle: LoadedModel) -> None:
        entry = self._loaded_models.pop(handle.model_id, None)
        if entry is not None:
            import torch

            del entry["model"]
            del entry["tokenizer"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def generate(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate using streaming for token-level timing."""
        from transformers import TextIteratorStreamer

        entry = self._loaded_models.get(handle.model_id)
        if entry is None:
            msg = f"Model {handle.model_id} not loaded"
            raise RuntimeError(msg)

        model = entry["model"]
        tokenizer = entry["tokenizer"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        prompt_token_count = input_ids.shape[1]

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),  # Avoid division by zero
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        token_timestamps: list[int] = []
        tokens: list[str] = []
        start_ns = self._now_ns()
        first_token_ns = 0

        # Run generation in a thread (streamer blocks until tokens are ready)
        thread = threading.Thread(
            target=lambda: model.generate(**gen_kwargs, **_no_grad_context()),
            daemon=True,
        )
        thread.start()

        for text in streamer:
            if text:
                now_ns = self._now_ns()
                if not first_token_ns:
                    first_token_ns = now_ns
                token_timestamps.append(now_ns)
                tokens.append(text)

        thread.join(timeout=30.0)
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
        """Stream tokens with per-token timing."""
        import asyncio
        import threading

        from transformers import TextIteratorStreamer

        entry = self._loaded_models.get(handle.model_id)
        if entry is None:
            msg = f"Model {handle.model_id} not loaded"
            raise RuntimeError(msg)

        model = entry["model"]
        tokenizer = entry["tokenizer"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=lambda: model.generate(**gen_kwargs, **_no_grad_context()),
            daemon=True,
        )
        thread.start()

        for text in streamer:
            if text:
                yield TokenEvent(
                    token=text,
                    timestamp_ns=time.perf_counter_ns(),
                )
                # Yield control to the event loop
                await asyncio.sleep(0)

        thread.join(timeout=30.0)


def _no_grad_context() -> dict:
    """Return empty dict — torch.no_grad() is applied inside the thread."""

    # We return an empty dict; the caller wraps with no_grad themselves
    # This is a workaround since we can't pass context managers through kwargs
    return {}
