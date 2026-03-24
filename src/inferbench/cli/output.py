"""Rich console formatting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from inferbench.hardware.models import GpuDriver, HardwareProfile

if TYPE_CHECKING:
    from inferbench.backends.base import InferenceBackend
    from inferbench.catalog.models import ModelSpec

console = Console()


def print_hardware_profile(profile: HardwareProfile) -> None:
    """Print a hardware profile as Rich tables."""
    _print_cpu_info(profile)
    console.print()
    _print_memory_info(profile)
    console.print()
    _print_gpu_info(profile)
    console.print()
    _print_system_info(profile)


def _print_cpu_info(profile: HardwareProfile) -> None:
    """Print CPU information table."""
    cpu = profile.cpu

    table = Table(title="CPU", show_header=False, title_style="bold cyan", min_width=60)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Model", cpu.model_name)
    table.add_row("Vendor", cpu.vendor)
    table.add_row("Architecture", cpu.architecture)
    table.add_row("Cores", f"{cpu.cores_physical} physical / {cpu.cores_logical} logical")

    if cpu.freq_max_mhz:
        freq_ghz = cpu.freq_max_mhz / 1000
        table.add_row("Max Frequency", f"{freq_ghz:.2f} GHz")

    if cpu.cache_l3_kb:
        if cpu.cache_l3_kb >= 1024:
            cache_str = f"{cpu.cache_l3_kb / 1024:.0f} MB"
        else:
            cache_str = f"{cpu.cache_l3_kb} KB"
        table.add_row("L3 Cache", cache_str)

    # Instruction set features
    features = []
    if cpu.has_avx:
        features.append("AVX")
    if cpu.has_avx2:
        features.append("AVX2")
    if cpu.has_avx512:
        features.append("AVX-512")
    if cpu.has_amx:
        features.append("AMX")
    if cpu.has_neon:
        features.append("NEON")

    if features:
        table.add_row("SIMD Features", ", ".join(features))

    console.print(table)


def _print_memory_info(profile: HardwareProfile) -> None:
    """Print memory information table."""
    mem = profile.memory

    table = Table(title="Memory", show_header=False, title_style="bold cyan", min_width=60)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    total_gb = mem.total_mb / 1024
    avail_gb = mem.available_mb / 1024
    table.add_row("Total RAM", f"{total_gb:.1f} GB")
    table.add_row("Available RAM", f"{avail_gb:.1f} GB")
    table.add_row("Memory Type", "Unified (shared with GPU)" if mem.is_unified else "Discrete")

    console.print(table)


def _print_gpu_info(profile: HardwareProfile) -> None:
    """Print GPU information table."""
    if not profile.gpus:
        console.print(
            Panel(
                Text("No GPUs detected", style="yellow"),
                title="GPU",
                title_align="left",
                border_style="cyan",
            )
        )
        return

    for i, gpu in enumerate(profile.gpus):
        title = f"GPU {i}" if len(profile.gpus) > 1 else "GPU"
        table = Table(title=title, show_header=False, title_style="bold cyan", min_width=60)
        table.add_column("Property", style="bold")
        table.add_column("Value")

        table.add_row("Name", gpu.name)
        table.add_row("Vendor", gpu.vendor.value.upper())
        table.add_row("Type", "Integrated (APU)" if gpu.is_integrated else "Discrete")

        if gpu.vram_total_mb > 0:
            vram_gb = gpu.vram_total_mb / 1024
            table.add_row("VRAM Total", f"{vram_gb:.1f} GB")

        if gpu.vram_available_mb is not None and gpu.vram_available_mb > 0:
            vram_avail_gb = gpu.vram_available_mb / 1024
            table.add_row("VRAM Available", f"{vram_avail_gb:.1f} GB")

        driver_label = _driver_label(gpu.driver)
        if gpu.driver_version:
            driver_label += f" ({gpu.driver_version})"
        table.add_row("Driver", driver_label)

        if gpu.compute_capability:
            table.add_row("Compute Capability", gpu.compute_capability)

        if gpu.rocm_arch:
            table.add_row("ROCm Architecture", gpu.rocm_arch)

        console.print(table)


def _print_system_info(profile: HardwareProfile) -> None:
    """Print system information table."""
    table = Table(title="System", show_header=False, title_style="bold cyan", min_width=60)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("OS", f"{profile.os_name} {profile.os_version}")
    table.add_row("Kernel", profile.kernel_version)
    table.add_row("Python", profile.python_version)

    console.print(table)


def print_backends(backends: list[InferenceBackend]) -> None:
    """Print backend availability table."""
    table = Table(title="Backends", title_style="bold cyan")
    table.add_column("Backend", style="bold")
    table.add_column("Status")
    table.add_column("Version")
    table.add_column("Install / Notes")

    for b in backends:
        available = b.is_available()
        status = Text("available", style="green") if available else Text("missing", style="red")
        version = b.get_version() if available else "-"
        hint = b.get_install_hint() or ""

        table.add_row(b.display_name, status, version, hint)

    console.print(table)


def print_models(models: list[ModelSpec], *, backend_filter: str | None = None) -> None:
    """Print model catalog table."""

    table = Table(title="Models", title_style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Family")
    table.add_column("Params", justify="right")
    table.add_column("Quant")
    table.add_column("Context", justify="right")
    table.add_column("Est. RAM", justify="right")
    table.add_column("Tags")
    if not backend_filter:
        table.add_column("Backends")

    for m in models:
        params_str = f"{m.parameter_count_b:.1f}B"
        ctx_str = f"{m.context_length // 1024}K"
        ram_str = f"{m.estimated_ram_mb / 1024:.1f} GB" if m.estimated_ram_mb else "-"
        tags_str = ", ".join(m.tags) if m.tags else "-"

        row = [
            m.canonical_name,
            m.family.value,
            params_str,
            m.quantization.value,
            ctx_str,
            ram_str,
            tags_str,
        ]

        if not backend_filter:
            backends_str = ", ".join(sorted(m.backend_ids.keys()))
            row.append(backends_str)

        table.add_row(*row)

    if not models:
        console.print("[yellow]No compatible models found.[/yellow]")
    else:
        console.print(table)
        console.print(f"\n[dim]{len(models)} model(s)[/dim]")


def _driver_label(driver: GpuDriver) -> str:
    """Human-readable driver name."""
    return {
        GpuDriver.CUDA: "CUDA",
        GpuDriver.ROCM: "ROCm",
        GpuDriver.VULKAN: "Vulkan",
        GpuDriver.METAL: "Metal",
        GpuDriver.NONE: "None",
    }.get(driver, driver.value)
