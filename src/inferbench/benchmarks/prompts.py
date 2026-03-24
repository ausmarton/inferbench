"""Standardized prompt sets for benchmarking.

Each prompt set defines a consistent workload that can be run
across all backends for fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """A single benchmark prompt definition."""

    name: str
    description: str
    system_prompt: str
    user_prompt: str
    max_output_tokens: int
    temperature: float = 0.0


# ── Short prompt → short output (chat-style) ──────────────────────────
SHORT_CHAT = PromptSpec(
    name="short_chat",
    description="Short prompt, short output (typical chat interaction)",
    system_prompt="You are a helpful assistant. Be concise.",
    user_prompt="What are three practical uses of Python's asyncio library? Give a brief answer.",
    max_output_tokens=256,
)

# ── Long prompt → short output (summarization-style) ──────────────────
_LONG_ARTICLE = (
    "The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' "
    "by Vaswani et al., has fundamentally changed the landscape of natural language processing "
    "and, increasingly, other domains of machine learning. At its core, the transformer relies "
    "on a mechanism called self-attention, which allows the model to weigh the importance of "
    "different parts of the input sequence when producing each element of the output. Unlike "
    "recurrent neural networks (RNNs), which process sequences step by step, transformers can "
    "process all positions in parallel, leading to significant speedups during training.\n\n"
    "The architecture consists of an encoder and a decoder, each composed of multiple identical "
    "layers. Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a "
    "position-wise fully connected feed-forward network. The decoder has an additional sub-layer "
    "that performs multi-head attention over the encoder's output. Residual connections and layer "
    "normalization are applied around each sub-layer.\n\n"
    "Since transformers lack recurrence, they use positional encodings to inject information about "
    "the position of tokens in the sequence. The original paper used sinusoidal positional "
    "encodings, though later work introduced learned positional embeddings and more sophisticated "
    "methods like rotary positional embeddings (RoPE).\n\n"
    "The multi-head attention mechanism is particularly important. It allows the model to attend "
    "to information from different representation subspaces at different positions. Each head "
    "computes attention independently using queries, keys, and values derived from the input, "
    "and the outputs are concatenated and linearly transformed. This enables the model to capture "
    "diverse relationships within the data.\n\n"
    "Transformers have been scaled to enormous sizes. Models like GPT-4, Claude, Llama, and "
    "PaLM contain tens or hundreds of billions of parameters and are trained on trillions of "
    "tokens of text data. This scaling has led to emergent capabilities including in-context "
    "learning, chain-of-thought reasoning, and the ability to follow complex instructions.\n\n"
    "The impact extends beyond NLP. Vision Transformers (ViT) apply the same architecture to "
    "image patches, achieving competitive results on image classification. Multimodal models "
    "like GPT-4V and Gemini process both text and images. In protein science, AlphaFold uses "
    "transformer-like attention mechanisms to predict protein structures "
    "with remarkable accuracy.\n\n"
    "Despite their success, transformers face challenges. The self-attention mechanism has "
    "quadratic complexity with respect to sequence length, making very long sequences expensive "
    "to process. Various approaches address this, including sparse attention patterns, linear "
    "attention approximations, and the use of sliding window attention. The memory requirements "
    "for large models are substantial, driving research into quantization, distillation, and "
    "efficient inference techniques."
)

LONG_SUMMARIZE = PromptSpec(
    name="long_summarize",
    description="Long prompt (~500 tokens), short output (summarization task)",
    system_prompt="You are a helpful assistant. Be concise and precise.",
    user_prompt=f"{_LONG_ARTICLE}\n\nSummarize the above article in exactly 3 bullet points.",
    max_output_tokens=256,
)

# ── Short prompt → long output (code generation) ─────────────────────
CODE_GENERATE = PromptSpec(
    name="code_generate",
    description="Short prompt, long output (code generation task)",
    system_prompt="You are an expert Python developer. Write clean, well-documented code.",
    user_prompt=(
        "Write a Python implementation of a binary search tree (BST) with the following methods:\n"
        "- insert(value): Insert a new value\n"
        "- search(value): Return True if value exists\n"
        "- delete(value): Remove a value\n"
        "- inorder(): Return values in sorted order\n"
        "- height(): Return the height of the tree\n"
        "Include type hints and docstrings."
    ),
    max_output_tokens=2048,
)

# ── Reasoning task ────────────────────────────────────────────────────
REASONING = PromptSpec(
    name="reasoning",
    description="Moderate prompt, moderate output (logical reasoning task)",
    system_prompt="You are a helpful assistant. Think step by step.",
    user_prompt=(
        "A farmer has 100 meters of fencing and wants to enclose a rectangular area "
        "along a straight river. The river serves as one side, so fencing is needed "
        "for only three sides. What dimensions maximize the enclosed area? "
        "Show your work step by step."
    ),
    max_output_tokens=512,
)

# ── All standard prompts ─────────────────────────────────────────────
ALL_PROMPTS: dict[str, PromptSpec] = {
    p.name: p for p in [SHORT_CHAT, LONG_SUMMARIZE, CODE_GENERATE, REASONING]
}

DEFAULT_PROMPT_NAMES: list[str] = ["short_chat", "code_generate"]
