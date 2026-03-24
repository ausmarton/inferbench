"""CPU detection — features, cores, frequency."""

from __future__ import annotations

import contextlib
import platform
import subprocess

import psutil

from inferbench.hardware.models import CpuFeatures


def detect_cpu() -> CpuFeatures:
    """Detect CPU capabilities from system information."""
    arch = platform.machine()

    if platform.system() == "Linux":
        return _detect_cpu_linux(arch)
    elif platform.system() == "Darwin":
        return _detect_cpu_darwin(arch)
    else:
        return _detect_cpu_fallback(arch)


def _detect_cpu_linux(arch: str) -> CpuFeatures:
    """Detect CPU on Linux via /proc/cpuinfo and lscpu."""
    vendor = ""
    model_name = ""
    flags: set[str] = set()
    cache_l3_kb: int | None = None
    freq_max: float | None = None
    freq_min: float | None = None

    # Parse /proc/cpuinfo for the first processor entry
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("vendor_id"):
                    vendor = line.split(":", 1)[1].strip()
                elif line.startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                elif line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].strip().split())
                    break  # flags is usually last field we need; stop after first CPU
    except OSError:
        pass

    # Parse lscpu for frequency and cache info
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            key, _, value = line.partition(":")
            value = value.strip()
            if key.strip() == "CPU max MHz" and value:
                with contextlib.suppress(ValueError):
                    freq_max = float(value)
            elif key.strip() == "CPU min MHz" and value:
                with contextlib.suppress(ValueError):
                    freq_min = float(value)
            elif key.strip() == "L3 cache" and value:
                cache_l3_kb = _parse_cache_size(value)
    except (OSError, subprocess.TimeoutExpired):
        pass

    return CpuFeatures(
        vendor=vendor,
        model_name=model_name,
        architecture=arch,
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count(logical=True) or 1,
        has_avx="avx" in flags,
        has_avx2="avx2" in flags,
        has_avx512="avx512f" in flags,
        has_amx="amx_tile" in flags,
        has_neon=False,  # NEON is ARM only
        cache_l3_kb=cache_l3_kb,
        freq_max_mhz=freq_max,
        freq_min_mhz=freq_min,
    )


def _detect_cpu_darwin(arch: str) -> CpuFeatures:
    """Detect CPU on macOS via sysctl."""
    vendor = ""
    model_name = ""

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        model_name = result.stdout.strip()
        vendor = "Apple" if "Apple" in model_name else "GenuineIntel"
    except (OSError, subprocess.TimeoutExpired):
        pass

    is_arm = arch in ("arm64", "aarch64")

    return CpuFeatures(
        vendor=vendor,
        model_name=model_name,
        architecture=arch,
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count(logical=True) or 1,
        has_avx=not is_arm,  # x86 Macs have AVX
        has_avx2=not is_arm,
        has_avx512=False,
        has_amx=False,
        has_neon=is_arm,
        cache_l3_kb=None,
        freq_max_mhz=None,
        freq_min_mhz=None,
    )


def _detect_cpu_fallback(arch: str) -> CpuFeatures:
    """Fallback CPU detection using only psutil and platform."""
    return CpuFeatures(
        vendor="unknown",
        model_name=platform.processor() or "unknown",
        architecture=arch,
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count(logical=True) or 1,
    )


def _parse_cache_size(value: str) -> int | None:
    """Parse a cache size string like '32 MiB' or '32768 KiB' into KB."""
    parts = value.strip().split()
    if len(parts) < 1:
        return None

    # Sometimes lscpu reports per-instance with instance count, e.g. "32 MiB (1 instance)"
    try:
        size = float(parts[0])
    except ValueError:
        return None

    if len(parts) >= 2:
        unit = parts[1].lower().rstrip("b")
        if unit in ("mi", "mib", "m"):
            return int(size * 1024)
        elif unit in ("gi", "gib", "g"):
            return int(size * 1024 * 1024)
        elif unit in ("ki", "kib", "k"):
            return int(size)

    return int(size)  # Assume KB if no unit
