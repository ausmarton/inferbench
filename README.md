# InferBench

**Local LLM inference benchmarking across multiple backends.**

InferBench detects your hardware capabilities and benchmarks LLM inference across multiple runtimes (Ollama, llama.cpp, vLLM, Transformers, ONNX Runtime) to help you pick the best model and backend for your use case.

## Features

- **Hardware detection** — CPU features (AVX-512, AMX, NEON), GPU (NVIDIA/AMD/Intel), memory (unified/discrete)
- **Multiple backends** — Ollama, llama.cpp, vLLM, Hugging Face Transformers, ONNX Runtime
- **Standardized benchmarks** — consistent prompt sets across all backends for fair comparison
- **Detailed metrics** — Time to First Token (TTFT), Tokens Per Second (TPS), Inter-Token Latency, memory usage, load times
- **Comparison reports** — side-by-side results across backends and models

## Quick Start

```bash
# Install with uv (recommended)
uv pip install inferbench

# Detect your hardware
inferbench detect

# List available backends
inferbench backends

# Run benchmarks (coming soon)
inferbench run --backend ollama --iterations 3
```

## Development

```bash
# Clone and set up
git clone https://github.com/ausmarton/inferbench.git
cd inferbench
uv sync --extra dev

# Run
uv run inferbench detect

# Test
uv run pytest tests/unit/ -v

# Lint
uv run ruff check src/ tests/
```

## Architecture

InferBench is built around a **backend abstraction** — each inference runtime implements a common interface for model loading, generation, and metric collection. This allows fair, apples-to-apples comparison across runtimes.

```
inferbench detect        → Hardware capabilities
inferbench backends      → Available runtimes
inferbench models        → Compatible models for your hardware
inferbench run           → Execute benchmarks
inferbench report        → View results
inferbench compare       → Cross-backend comparison
```

## Dependency Isolation

InferBench uses `uv` with a project-local virtual environment. Heavy backend dependencies (PyTorch, vLLM, etc.) are optional groups — install only what you need:

```bash
uv sync                          # Core CLI only (~20MB)
uv sync --extra ollama           # + Ollama support (uses REST API, no extra deps)
uv sync --extra llamacpp         # + llama.cpp Python bindings
uv sync --extra transformers     # + PyTorch + Transformers
uv sync --extra all              # Everything
```

## License

Apache-2.0
