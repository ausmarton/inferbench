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
