"""Data models for the model catalog."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ModelFamily(StrEnum):
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    PHI = "phi"
    GEMMA = "gemma"
    STARCODER = "starcoder"
    BERT = "bert"
    OTHER = "other"


class QuantizationType(StrEnum):
    NONE = "none"
    Q4_0 = "Q4_0"
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q8_0 = "Q8_0"
    GPTQ_4BIT = "gptq-4bit"
    AWQ_4BIT = "awq-4bit"
    EXL2 = "exl2"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class ModelSpec(BaseModel):
    """Backend-agnostic model specification.

    Maps a canonical model identity to backend-specific identifiers
    and hardware requirements.
    """

    canonical_name: str  # e.g. "qwen2.5-7b-q4km"
    family: ModelFamily
    parameter_count_b: float  # in billions (e.g. 7.6)
    context_length: int  # max tokens
    quantization: QuantizationType = QuantizationType.NONE
    estimated_ram_mb: int = 0  # RAM needed for CPU inference
    estimated_vram_mb: int = 0  # VRAM needed for GPU inference
    tags: list[str] = []  # e.g. ["chat", "instruct", "code"]

    # Per-backend model identifiers
    backend_ids: dict[str, str] = {}  # {"ollama": "qwen2.5:7b", "transformers": "Qwen/..."}
