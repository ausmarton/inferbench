"""InferBench CLI — main Typer application."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from inferbench import __version__

app = typer.Typer(
    name="inferbench",
    help="Local LLM inference benchmarking across multiple backends.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"inferbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-v", help="Show version and exit.", callback=version_callback),
    ] = None,
) -> None:
    """InferBench — benchmark local LLM inference across multiple backends."""


@app.command()
def detect(
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON instead of formatted tables."),
    ] = False,
) -> None:
    """Detect and display hardware capabilities."""
    from inferbench.cli.output import console, print_hardware_profile
    from inferbench.hardware.detector import detect_hardware

    with console.status("Detecting hardware..."):
        profile = detect_hardware()

    if json_output:
        typer.echo(json.dumps(profile.model_dump(), indent=2))
    else:
        console.print()
        console.rule("[bold]Hardware Profile[/bold]")
        console.print()
        print_hardware_profile(profile)


@app.command()
def backends(
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON."),
    ] = False,
) -> None:
    """List inference backends and their availability."""
    from inferbench.backends.registry import get_all_backends
    from inferbench.cli.output import console, print_backends

    all_backends = get_all_backends()

    if json_output:
        data = []
        for b in all_backends:
            entry = {
                "name": b.name,
                "display_name": b.display_name,
                "available": b.is_available(),
            }
            if b.is_available():
                entry["version"] = b.get_version()
            hint = b.get_install_hint()
            if hint:
                entry["install_hint"] = hint
            data.append(entry)
        typer.echo(json.dumps(data, indent=2))
    else:
        console.print()
        console.rule("[bold]Inference Backends[/bold]")
        console.print()
        print_backends(all_backends)


@app.command()
def models(
    backend: Annotated[
        str | None,
        typer.Option("--backend", "-b", help="Filter to a specific backend."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON."),
    ] = False,
) -> None:
    """List models from the catalog, filtered by hardware compatibility."""
    from inferbench.backends.registry import get_available_backends, get_backend
    from inferbench.catalog.registry import filter_by_backend, filter_by_hardware, load_catalog
    from inferbench.cli.output import console, print_models
    from inferbench.hardware.detector import detect_hardware

    with console.status("Detecting hardware..."):
        hardware = detect_hardware()

    catalog = load_catalog()
    compatible = filter_by_hardware(catalog, hardware)

    if backend:
        compatible = filter_by_backend(compatible, backend)
        # Also show models the backend has locally but aren't in our catalog
        b = get_backend(backend)
        if b and b.is_available():
            local_ids = set(b.supported_model_ids(hardware))
            catalog_ids = {m.backend_ids.get(backend) for m in compatible}
            extra_ids = local_ids - catalog_ids - {None}
            if extra_ids and not json_output:
                console.print(
                    f"\n[dim]Note: {len(extra_ids)} additional model(s) available in "
                    f"{b.display_name} not in the catalog: "
                    f"{', '.join(sorted(extra_ids)[:5])}"
                    f"{'...' if len(extra_ids) > 5 else ''}[/dim]"
                )
    else:
        # Show which available backends each model supports
        available_backends = get_available_backends()
        _ = available_backends  # used implicitly via backend_ids display

    if json_output:
        typer.echo(json.dumps([m.model_dump() for m in compatible], indent=2))
    else:
        console.print()
        console.rule("[bold]Compatible Models[/bold]")
        console.print()
        print_models(compatible, backend_filter=backend)


@app.command(name="run")
def run_benchmarks(
    model_ids: Annotated[
        list[str] | None,
        typer.Argument(help="Model IDs to benchmark (backend-specific, e.g. 'qwen2.5:7b')."),
    ] = None,
    backend: Annotated[
        list[str] | None,
        typer.Option("--backend", "-b", help="Backend(s) to use. Can be repeated."),
    ] = None,
    prompts: Annotated[
        list[str] | None,
        typer.Option("--prompts", "-p", help="Prompt set(s) to run. Can be repeated."),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-n", help="Number of timed iterations per prompt."),
    ] = 3,
    warmup: Annotated[
        int,
        typer.Option("--warmup", help="Number of warmup iterations (discarded)."),
    ] = 1,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output file path for results JSON."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output results as JSON to stdout."),
    ] = False,
) -> None:
    """Run inference benchmarks."""
    from pathlib import Path

    from inferbench.backends.registry import get_available_backends, get_backend
    from inferbench.benchmarks.runner import BenchmarkRunner
    from inferbench.cli.output import console
    from inferbench.hardware.detector import detect_hardware
    from inferbench.results.report import print_session_summary
    from inferbench.results.storage import save_session

    with console.status("Detecting hardware..."):
        hardware = detect_hardware()

    # Resolve backends
    if backend:
        backends = []
        for name in backend:
            b = get_backend(name)
            if b is None or not b.is_available():
                console.print(f"[red]Backend '{name}' is not available.[/red]")
                raise typer.Exit(1)
            backends.append(b)
    else:
        backends = get_available_backends()
        if not backends:
            console.print("[red]No backends available. Install one first.[/red]")
            console.print("Run [bold]inferbench backends[/bold] to see install instructions.")
            raise typer.Exit(1)

    # Resolve models per backend
    models_per_backend: dict[str, list[str]] = {}
    for b in backends:
        if model_ids:
            models_per_backend[b.name] = list(model_ids)
        else:
            # Use all locally available models for this backend
            available = b.supported_model_ids(hardware)
            if not available:
                console.print(
                    f"[yellow]No models available for {b.display_name}. Skipping.[/yellow]"
                )
                continue
            models_per_backend[b.name] = available

    if not models_per_backend:
        console.print("[red]No models to benchmark.[/red]")
        raise typer.Exit(1)

    # Count total models
    total_models = sum(len(v) for v in models_per_backend.values())
    backend_names = ", ".join(b.display_name for b in backends)
    console.print(
        f"\nBenchmarking [bold]{total_models}[/bold] model(s) across [cyan]{backend_names}[/cyan]"
    )
    console.print(
        f"Iterations: {iterations} timed + {warmup} warmup | "
        f"Prompts: {', '.join(prompts or ['default'])}\n"
    )

    runner = BenchmarkRunner(
        backends=backends,
        model_ids=models_per_backend,
        prompt_names=prompts,
        warmup_iterations=warmup,
        timed_iterations=iterations,
    )

    session = runner.run(hardware)

    # Save results
    output_path = Path(output) if output else None
    saved_path = save_session(session, output_path)

    if json_output:
        typer.echo(json.dumps(session.model_dump(mode="json"), indent=2, default=str))
    else:
        console.print()
        print_session_summary(session)
        console.print(f"\n[dim]Results saved to {saved_path}[/dim]")


@app.command()
def report(
    path: Annotated[
        str,
        typer.Argument(help="Path to a results JSON file."),
    ],
    detailed: Annotated[
        bool,
        typer.Option("--detailed", "-d", help="Show per-run details."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON."),
    ] = False,
) -> None:
    """Display results from a saved benchmark session."""
    from pathlib import Path

    from inferbench.cli.output import console
    from inferbench.results.report import print_detailed_results, print_session_summary
    from inferbench.results.storage import load_session

    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    session = load_session(file_path)

    if json_output:
        typer.echo(json.dumps(session.model_dump(mode="json"), indent=2, default=str))
    else:
        console.print()
        console.rule("[bold]Benchmark Report[/bold]")
        console.print(
            f"Session: {session.session_id} | "
            f"Started: {session.started_at} | "
            f"Results: {len(session.results)}"
        )
        console.print()
        print_session_summary(session)
        if detailed:
            print_detailed_results(session)
