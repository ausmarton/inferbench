# CLAUDE.md — InferBench Development Guide

## Project Overview
InferBench is a CLI tool for benchmarking local LLM inference across multiple backends.
It detects hardware capabilities, runs standardized benchmarks, and produces comparison reports.

## Build & Run
- Package manager: uv (project-local .venv/, no global installs)
- Install dev: `uv sync --extra dev`
- Install with backend: `uv sync --extra dev --extra ollama --extra llamacpp`
- Run: `uv run inferbench --help`
- Test: `uv run pytest tests/unit/ -v`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`

## Architecture
- `src/inferbench/` — main package (src layout)
- `cli/` — Typer CLI commands
- `hardware/` — hardware detection (CPU, GPU, RAM)
- `backends/` — inference backend implementations (each is optional)
- `catalog/` — model registry mapping canonical names to backend IDs
- `benchmarks/` — prompt sets, runner orchestration, metric collection
- `results/` — data models, storage, and report generation

## Conventions
- All data models use Pydantic v2 BaseModel
- All backends extend `InferenceBackend` ABC in `backends/base.py`
- Async throughout the generation path (sync wrappers at CLI boundary)
- Type hints on all public functions
- Tests in tests/unit/ (mocked) and tests/integration/ (requires real backends)

## Backend Development
To add a new backend:
1. Create `src/inferbench/backends/newbackend.py`
2. Implement `InferenceBackend` ABC
3. Decorate class with `@register("newbackend")`
4. Add optional dependency group in `pyproject.toml`
5. Add backend IDs to relevant models in `catalog/builtin.yaml`
6. Add unit tests in `tests/unit/`
