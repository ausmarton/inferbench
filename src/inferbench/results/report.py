"""Report generation from benchmark results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from inferbench.benchmarks.metrics import ResultSummary

if TYPE_CHECKING:
    from inferbench.results.models import BenchmarkResult, BenchmarkSession

console = Console()


def print_session_summary(session: BenchmarkSession) -> None:
    """Print a summary table of all results in a session."""
    if not session.results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(title="Benchmark Results", title_style="bold cyan")
    table.add_column("Backend", style="bold")
    table.add_column("Model", style="bold")
    table.add_column("Cold Load", justify="right")
    table.add_column("Warm Load", justify="right")
    table.add_column("Avg TPS", justify="right", style="green")
    table.add_column("Avg TTFT", justify="right", style="yellow")
    table.add_column("Avg E2E", justify="right")
    table.add_column("ITL p50", justify="right")
    table.add_column("ITL p99", justify="right")
    table.add_column("Peak RAM", justify="right")
    table.add_column("Runs", justify="right")

    for result in session.results:
        summary = ResultSummary.from_result(result)
        table.add_row(
            summary.backend_name,
            _truncate(summary.model_id, 25),
            f"{summary.cold_load_s:.1f}s",
            f"{summary.warm_load_s:.1f}s",
            f"{summary.avg_tps:.1f}",
            f"{summary.avg_ttft_ms:.0f}ms",
            f"{summary.avg_e2e_ms:.0f}ms",
            f"{summary.itl_p50_ms:.1f}ms",
            f"{summary.itl_p99_ms:.1f}ms",
            f"{summary.peak_ram_mb:.0f}MB" if summary.peak_ram_mb > 0 else "-",
            str(summary.total_runs),
        )

    console.print(table)


def print_detailed_results(session: BenchmarkSession) -> None:
    """Print detailed per-prompt results for each backend/model."""
    for result in session.results:
        console.print()
        console.rule(f"[bold]{result.backend_name}[/bold] / [cyan]{result.model_id}[/cyan]")
        console.print(
            f"  Cold load: {result.cold_load_time_s:.2f}s | "
            f"Warm load: {result.warm_load_time_s:.2f}s"
        )
        console.print()

        if not result.runs:
            console.print("  [yellow]No runs completed.[/yellow]")
            continue

        table = Table(show_header=True, min_width=80)
        table.add_column("Prompt", style="bold")
        table.add_column("Tokens", justify="right")
        table.add_column("TTFT", justify="right", style="yellow")
        table.add_column("TPS", justify="right", style="green")
        table.add_column("E2E", justify="right")
        table.add_column("ITL p50", justify="right")

        for run in result.runs:
            itls = run.itl_ms
            itl_p50 = f"{_median(itls):.1f}ms" if itls else "-"

            table.add_row(
                run.prompt_name,
                str(run.output_tokens),
                f"{run.ttft_ms:.0f}ms",
                f"{run.tps:.1f}",
                f"{run.e2e_latency_ms:.0f}ms",
                itl_p50,
            )

        console.print(table)


def print_result_comparison(results: list[BenchmarkResult]) -> None:
    """Print a side-by-side comparison of multiple results."""
    if not results:
        console.print("[yellow]No results to compare.[/yellow]")
        return

    table = Table(title="Comparison", title_style="bold cyan")
    table.add_column("Metric", style="bold")

    for r in results:
        table.add_column(f"{r.backend_name}\n{_truncate(r.model_id, 20)}", justify="right")

    summaries = [ResultSummary.from_result(r) for r in results]

    # Highlight the best value in each row
    metrics = [
        ("Cold Load (s)", [s.cold_load_s for s in summaries], "{:.2f}", "min"),
        ("Warm Load (s)", [s.warm_load_s for s in summaries], "{:.2f}", "min"),
        ("Avg TPS", [s.avg_tps for s in summaries], "{:.1f}", "max"),
        ("Avg TTFT (ms)", [s.avg_ttft_ms for s in summaries], "{:.0f}", "min"),
        ("Avg E2E (ms)", [s.avg_e2e_ms for s in summaries], "{:.0f}", "min"),
        ("ITL p50 (ms)", [s.itl_p50_ms for s in summaries], "{:.1f}", "min"),
        ("ITL p99 (ms)", [s.itl_p99_ms for s in summaries], "{:.1f}", "min"),
        ("Peak RAM (MB)", [s.peak_ram_mb for s in summaries], "{:.0f}", "min"),
        ("Total Runs", [float(s.total_runs) for s in summaries], "{:.0f}", None),
    ]

    for label, values, fmt, best_fn in metrics:
        if best_fn and any(v > 0 for v in values):
            if best_fn == "min":
                positive_vals = [v for v in values if v > 0]
                best_val = min(positive_vals) if positive_vals else 0
            else:
                best_val = max(values)
        else:
            best_val = None

        row = [label]
        for v in values:
            formatted = fmt.format(v) if v > 0 else "-"
            if best_val is not None and v == best_val and v > 0:
                formatted = f"[bold green]{formatted}[/bold green]"
            row.append(formatted)

        table.add_row(*row)

    console.print(table)


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "\u2026"


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]
