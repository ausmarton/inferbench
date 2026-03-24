"""llama.cpp inference backend — in-process via llama-cpp-python."""

from __future__ import annotations

import logging
import time
from pathlib import Path
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

# Default cache directory for downloaded models
_MODEL_CACHE_DIR = Path.home() / ".inferbench" / "models"


@register("llamacpp")
class LlamaCppBackend(InferenceBackend):
    """Backend using llama-cpp-python for in-process GGUF inference."""

    def __init__(self) -> None:
        self._llama_module = None
        self._loaded_llamas: dict[str, object] = {}  # model_id -> Llama instance

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def display_name(self) -> str:
        return "llama.cpp"

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False

    def get_version(self) -> str:
        try:
            import llama_cpp

            return getattr(llama_cpp, "__version__", "unknown")
        except ImportError:
            return "not installed"

    def get_install_hint(self) -> str | None:
        return 'uv sync --extra llamacpp  # or: pip install "inferbench[llamacpp]"'

    def supported_model_ids(self, hardware: HardwareProfile) -> list[str]:
        """Return model IDs from the catalog that have llamacpp backend entries.

        Also includes any local GGUF files in the cache directory.
        """
        ids: list[str] = []

        # Catalog models
        from inferbench.catalog.registry import filter_by_backend, load_catalog

        catalog = load_catalog()
        for model in filter_by_backend(catalog, self.name):
            backend_id = model.backend_ids.get(self.name)
            if backend_id:
                ids.append(backend_id)

        # Local GGUF files
        if _MODEL_CACHE_DIR.exists():
            for gguf_file in _MODEL_CACHE_DIR.rglob("*.gguf"):
                ids.append(str(gguf_file))

        return ids

    async def load_model(self, model_id: str) -> LoadedModel:
        """Load a GGUF model.

        model_id can be:
        - A local file path to a .gguf file
        - A HuggingFace repo reference: "repo_id:filename" (e.g.
          "bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        """
        import llama_cpp

        start_ns = self._now_ns()

        model_path = _resolve_model_path(model_id)

        llm = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=4096,  # Reasonable default context
            n_threads=None,  # Auto-detect
            verbose=False,
        )

        load_time_s = (self._now_ns() - start_ns) / 1e9
        self._loaded_llamas[model_id] = llm

        return LoadedModel(
            backend_name=self.name,
            model_id=model_id,
            load_time_s=load_time_s,
            metadata={
                "model_path": str(model_path),
                "n_ctx": llm.n_ctx(),
            },
        )

    async def unload_model(self, handle: LoadedModel) -> None:
        llm = self._loaded_llamas.pop(handle.model_id, None)
        if llm is not None:
            del llm

    async def generate(
        self,
        handle: LoadedModel,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate with token-level timing via streaming."""
        llm = self._loaded_llamas.get(handle.model_id)
        if llm is None:
            msg = f"Model {handle.model_id} not loaded"
            raise RuntimeError(msg)

        token_timestamps: list[int] = []
        tokens: list[str] = []

        start_ns = self._now_ns()
        first_token_ns = 0
        prompt_tokens = 0

        for chunk in llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            text = chunk["choices"][0].get("text", "")
            if text:
                now_ns = self._now_ns()
                if not first_token_ns:
                    first_token_ns = now_ns
                token_timestamps.append(now_ns)
                tokens.append(text)

            # Capture usage from final chunk
            if chunk["choices"][0].get("finish_reason") is not None:
                usage = chunk.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)

        end_ns = self._now_ns()

        if not first_token_ns:
            first_token_ns = end_ns

        return GenerationResult(
            output_text="".join(tokens),
            token_count=len(tokens),
            prompt_token_count=prompt_tokens,
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
        llm = self._loaded_llamas.get(handle.model_id)
        if llm is None:
            msg = f"Model {handle.model_id} not loaded"
            raise RuntimeError(msg)

        for chunk in llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            text = chunk["choices"][0].get("text", "")
            if text:
                yield TokenEvent(
                    token=text,
                    timestamp_ns=time.perf_counter_ns(),
                )


def _resolve_model_path(model_id: str) -> Path:
    """Resolve a model ID to a local GGUF file path.

    Supports:
    - Local file paths (absolute or relative)
    - HuggingFace Hub references: "repo_id:filename"
    """
    # Check if it's a local file
    local_path = Path(model_id)
    if local_path.exists() and local_path.suffix == ".gguf":
        return local_path

    # Check model cache
    cached = _MODEL_CACHE_DIR / model_id.replace("/", "_").replace(":", "_")
    if cached.exists():
        return cached

    # Try HuggingFace Hub download: "repo_id:filename"
    if ":" in model_id and not model_id.startswith("/"):
        repo_id, filename = model_id.rsplit(":", 1)
        return _download_from_hf(repo_id, filename)

    # Try as a HF repo with auto-detection of GGUF file
    if "/" in model_id and not model_id.startswith("/"):
        return _download_from_hf(model_id)

    msg = (
        f"Cannot resolve model '{model_id}'. Provide a local .gguf path "
        f"or a HuggingFace reference like 'repo_id:filename.gguf'"
    )
    raise FileNotFoundError(msg)


def _download_from_hf(repo_id: str, filename: str | None = None) -> Path:
    """Download a GGUF file from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        msg = (
            "huggingface-hub is required to download models. Install with: uv sync --extra llamacpp"
        )
        raise ImportError(msg) from e

    if filename is None:
        # Auto-detect: find the first GGUF file in the repo
        files = list_repo_files(repo_id)
        gguf_files = [f for f in files if f.endswith(".gguf")]
        if not gguf_files:
            msg = f"No .gguf files found in {repo_id}"
            raise FileNotFoundError(msg)
        # Prefer Q4_K_M quantization if available
        for preferred in ["Q4_K_M", "q4_k_m"]:
            matches = [f for f in gguf_files if preferred in f]
            if matches:
                filename = matches[0]
                break
        if filename is None:
            filename = gguf_files[0]

    logger.info("Downloading %s from %s...", filename, repo_id)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(_MODEL_CACHE_DIR),
    )
    return Path(path)
