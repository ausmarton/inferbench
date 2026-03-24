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
