"""Benchmark runner — orchestrates execution across backends, models, and prompts."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from inferbench.benchmarks.prompts import ALL_PROMPTS, DEFAULT_PROMPT_NAMES
from inferbench.benchmarks.sampler import ResourceMonitor
from inferbench.results.models import BenchmarkResult, BenchmarkRun, BenchmarkSession

if TYPE_CHECKING:
    from inferbench.backends.base import InferenceBackend
    from inferbench.benchmarks.prompts import PromptSpec
    from inferbench.hardware.models import HardwareProfile

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark execution across a matrix of backends x models x prompts."""

    def __init__(
        self,
        backends: list[InferenceBackend],
        model_ids: dict[str, list[str]],  # {backend_name: [model_id, ...]}
        prompt_names: list[str] | None = None,
        warmup_iterations: int = 1,
        timed_iterations: int = 3,
    ) -> None:
        self.backends = backends
        self.model_ids = model_ids
        self.prompt_names = prompt_names or DEFAULT_PROMPT_NAMES
        self.warmup_iterations = warmup_iterations
        self.timed_iterations = timed_iterations

        # Validate prompts
        for name in self.prompt_names:
            if name not in ALL_PROMPTS:
                available = ", ".join(ALL_PROMPTS.keys())
                msg = f"Unknown prompt '{name}'. Available: {available}"
                raise ValueError(msg)

    def run(self, hardware: HardwareProfile) -> BenchmarkSession:
        """Run the full benchmark suite (sync wrapper)."""
        return asyncio.run(self.run_async(hardware))

    async def run_async(self, hardware: HardwareProfile) -> BenchmarkSession:
        """Run the full benchmark suite."""
        session = BenchmarkSession(
            hardware=hardware.model_dump(),
            warmup_iterations=self.warmup_iterations,
            timed_iterations=self.timed_iterations,
            prompt_names=self.prompt_names,
        )

        prompts = [ALL_PROMPTS[name] for name in self.prompt_names]

        # Count total work items for progress bar
        total_items = sum(
            len(self.model_ids.get(b.name, [])) * len(prompts) * self.timed_iterations
            for b in self.backends
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Benchmarking...", total=total_items)

            for backend in self.backends:
                models_for_backend = self.model_ids.get(backend.name, [])
                for model_id in models_for_backend:
                    result = await self._benchmark_model(backend, model_id, prompts, progress, task)
                    session.results.append(result)

        session.completed_at = datetime.now(UTC)
        return session

    async def _benchmark_model(
        self,
        backend: InferenceBackend,
        model_id: str,
        prompts: list[PromptSpec],
        progress: Progress,
        task_id,
    ) -> BenchmarkResult:
        """Benchmark a single model on a single backend."""
        progress.update(
            task_id,
            description=(
                f"[cyan]{backend.display_name}[/cyan] / [bold]{model_id}[/bold]: loading..."
            ),
        )

        # Cold load
        cold_load_time = 0.0
        try:
            # Ensure model is unloaded first for a true cold start
            from inferbench.backends.base import LoadedModel

            dummy_handle = LoadedModel(
                backend_name=backend.name, model_id=model_id, load_time_s=0.0
            )
            await backend.unload_model(dummy_handle)
            await asyncio.sleep(1.0)  # Brief pause to let memory free

            handle = await backend.load_model(model_id)
            cold_load_time = handle.load_time_s
            await backend.unload_model(handle)
        except Exception:
            logger.warning("Cold load measurement failed for %s on %s", model_id, backend.name)

        # Warm load (model may be cached in memory/disk cache)
        warm_load_time = 0.0
        try:
            handle = await backend.load_model(model_id)
            warm_load_time = handle.load_time_s
        except Exception:
            logger.exception("Failed to load %s on %s", model_id, backend.name)
            return BenchmarkResult(
                backend_name=backend.name,
                backend_version=backend.get_version(),
                model_id=model_id,
                cold_load_time_s=cold_load_time,
            )

        runs: list[BenchmarkRun] = []

        try:
            for prompt_spec in prompts:
                # Warmup iterations (not recorded)
                for _ in range(self.warmup_iterations):
                    progress.update(
                        task_id,
                        description=(
                            f"[cyan]{backend.display_name}[/cyan] / "
                            f"[bold]{model_id}[/bold]: "
                            f"warmup ({prompt_spec.name})"
                        ),
                    )
                    try:
                        await backend.generate(
                            handle,
                            _build_prompt(prompt_spec),
                            max_tokens=prompt_spec.max_output_tokens,
                            temperature=prompt_spec.temperature,
                        )
                    except Exception:
                        logger.warning(
                            "Warmup failed for %s on %s/%s",
                            prompt_spec.name,
                            backend.name,
                            model_id,
                        )

                # Timed iterations
                for i in range(self.timed_iterations):
                    progress.update(
                        task_id,
                        description=(
                            f"[cyan]{backend.display_name}[/cyan] / "
                            f"[bold]{model_id}[/bold]: "
                            f"{prompt_spec.name} [{i + 1}/{self.timed_iterations}]"
                        ),
                    )

                    monitor = ResourceMonitor(interval_ms=100)
                    monitor.start()

                    try:
                        gen_result = await backend.generate(
                            handle,
                            _build_prompt(prompt_spec),
                            max_tokens=prompt_spec.max_output_tokens,
                            temperature=prompt_spec.temperature,
                        )
                        resources = monitor.stop()

                        run = BenchmarkRun(
                            prompt_name=prompt_spec.name,
                            prompt_text=prompt_spec.user_prompt[:200],  # Truncate for storage
                            prompt_tokens=gen_result.prompt_token_count,
                            max_output_tokens=prompt_spec.max_output_tokens,
                            temperature=prompt_spec.temperature,
                            output_text=gen_result.output_text[:500],  # Truncate for storage
                            output_tokens=gen_result.token_count,
                            start_ns=gen_result.start_ns,
                            first_token_ns=gen_result.first_token_ns,
                            end_ns=gen_result.end_ns,
                            token_timestamps_ns=gen_result.token_timestamps_ns,
                            resources=resources,
                        )
                        runs.append(run)
                    except Exception:
                        logger.warning(
                            "Run failed: %s on %s/%s iteration %d",
                            prompt_spec.name,
                            backend.name,
                            model_id,
                            i + 1,
                        )
                        monitor.stop()

                    progress.advance(task_id)

        finally:
            await backend.unload_model(handle)

        return BenchmarkResult(
            backend_name=backend.name,
            backend_version=backend.get_version(),
            model_id=model_id,
            cold_load_time_s=cold_load_time,
            warm_load_time_s=warm_load_time,
            runs=runs,
        )


def _build_prompt(spec: PromptSpec) -> str:
    """Build a complete prompt string from a PromptSpec.

    For now, simple concatenation. Backends that support chat format
    can parse this or we can extend later.
    """
    if spec.system_prompt:
        return f"{spec.system_prompt}\n\n{spec.user_prompt}"
    return spec.user_prompt
