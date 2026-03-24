# InferBench

**Local LLM inference benchmarking across multiple backends.**

InferBench detects your hardware capabilities and benchmarks LLM inference across multiple runtimes to help you pick the best model and backend for your use case.

## Features

- **Hardware detection** — CPU features (AVX-512, AMX, NEON), GPU (NVIDIA/AMD/Intel), memory (unified/discrete)
- **Multiple backends** — Ollama, llama.cpp, vLLM, Hugging Face Transformers
- **Standardized benchmarks** — consistent prompt sets across all backends for fair comparison
- **Detailed metrics** — Time to First Token (TTFT), Tokens Per Second (TPS), Inter-Token Latency (ITL), memory usage, load times
- **Comparison reports** — side-by-side results across backends and models with best-value highlighting

## Quick Start

```bash
# Clone and set up
git clone https://github.com/ausmarton/inferbench.git
cd inferbench
uv sync --extra dev

# Detect your hardware
uv run inferbench detect

# List available backends and their status
uv run inferbench backends

# List models compatible with your hardware
uv run inferbench models

# Benchmark a model on Ollama (requires Ollama running)
uv run inferbench run qwen2.5:0.5b --backend ollama --iterations 3

# Benchmark across multiple backends
uv run inferbench run qwen2.5:0.5b \
  "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  --backend ollama --backend llamacpp --iterations 2

# View saved results
uv run inferbench report ~/.inferbench/results/<session-id>.json

# Compare results side-by-side
uv run inferbench compare ~/.inferbench/results/<session-id>.json
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `inferbench detect` | Detect and display hardware capabilities |
| `inferbench backends` | List backends with availability status and install hints |
| `inferbench models` | List models from the catalog, filtered by hardware |
| `inferbench run` | Execute benchmarks across backends and models |
| `inferbench report` | Display results from a saved session |
| `inferbench compare` | Side-by-side comparison across backends |

All commands support `--json` for machine-readable output.

## Metrics Collected

| Metric | Description |
|--------|-------------|
| **Cold Load** | Time to load model from disk (not cached) |
| **Warm Load** | Time to load model when cached |
| **TPS** | Tokens per second (generation throughput) |
| **TTFT** | Time to first token (latency) |
| **ITL p50/p99** | Inter-token latency percentiles |
| **E2E Latency** | End-to-end request time |
| **Peak RAM** | Peak memory usage during generation |
| **GPU %** | GPU utilization (AMD via sysfs) |

## Backends

| Backend | Type | Install |
|---------|------|---------|
| **Ollama** | REST API to running server | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **llama.cpp** | In-process GGUF inference | `uv sync --extra llamacpp` |
| **Transformers** | In-process via PyTorch | `uv sync --extra transformers` |
| **vLLM** | Managed server subprocess | `uv sync --extra vllm` |

## Dependency Isolation

All Python dependencies live in a project-local `.venv/` — zero impact on system Python. Heavy backend deps are optional groups:

```bash
uv sync                          # Core CLI only (~20MB)
uv sync --extra ollama           # + Ollama support (uses REST API, no extra deps)
uv sync --extra llamacpp         # + llama.cpp Python bindings + HuggingFace Hub
uv sync --extra transformers     # + PyTorch + Transformers + Accelerate
uv sync --extra all              # Everything
```

## Development

```bash
# Set up dev environment
uv sync --extra dev

# Run
uv run inferbench --help

# Test (135 unit tests)
uv run pytest tests/unit/ -v

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## License

Apache-2.0
