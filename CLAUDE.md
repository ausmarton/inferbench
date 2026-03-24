# CLAUDE.md — InferBench Development Guide

## Project Overview
InferBench is a CLI tool for benchmarking local LLM inference across multiple backends.
It detects hardware capabilities, runs standardized benchmarks, and produces comparison reports.

## Build & Run
- Package manager: uv (project-local .venv/, no global installs)
- Install dev: `uv sync --extra dev`
- Install with backends: `uv sync --extra dev --extra llamacpp`
- Run: `uv run inferbench --help`
- Test: `uv run pytest tests/unit/ -v`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`

## Architecture
- `src/inferbench/` — main package (src layout)
- `cli/app.py` — Typer CLI, all commands: detect, backends, models, run, report, compare
- `cli/output.py` — Rich table formatting for all display functions
- `hardware/` — hardware detection (CPU, GPU, RAM); models in `hardware/models.py`
- `backends/` — inference backend implementations (each is optional):
  - `base.py` — `InferenceBackend` ABC, `LoadedModel`, `TokenEvent`, `GenerationResult`
  - `registry.py` — `@register` decorator, lazy discovery
  - `ollama.py` — REST API via httpx
  - `llamacpp.py` — in-process GGUF via llama-cpp-python
  - `transformers.py` — in-process via HF transformers + PyTorch
  - `vllm.py` — managed server subprocess + OpenAI API
- `catalog/` — YAML model registry (`builtin.yaml`) mapping canonical names to backend IDs
- `benchmarks/` — prompt sets (`prompts.py`), runner orchestration (`runner.py`), resource monitoring (`sampler.py`)
- `results/` — data models (`models.py`), JSON storage (`storage.py`), Rich reports (`report.py`)

## Conventions
- All data models use Pydantic v2 BaseModel
- All backends extend `InferenceBackend` ABC in `backends/base.py`
- Async throughout the generation path (sync wrapper at CLI boundary via `asyncio.run`)
- Type hints on all public functions; TYPE_CHECKING guards for import-heavy types
- Tests in tests/unit/ (mocked, 135 tests) and tests/integration/ (requires real backends)
- `from __future__ import annotations` in every module

## Backend Development
To add a new backend:
1. Create `src/inferbench/backends/newbackend.py`
2. Implement `InferenceBackend` ABC (name, display_name, is_available, get_version, supported_model_ids, load_model, unload_model, generate, generate_stream)
3. Decorate class with `@register("newbackend")`
4. Add optional dependency group in `pyproject.toml`
5. Add backend module to `_KNOWN_BACKENDS` list in `registry.py`
6. Add backend IDs to relevant models in `catalog/builtin.yaml`
7. Add unit tests in `tests/unit/test_newbackend.py`

## Hardware
Dev machine: AMD Ryzen AI MAX+ PRO 395, 128GB unified RAM, Radeon 8060S (ROCm KFD).
No NVIDIA GPU. Ollama + llama.cpp available for integration testing.
