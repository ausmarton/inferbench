"""System memory detection."""

from __future__ import annotations

import psutil

from inferbench.hardware.models import MemoryInfo


def detect_memory() -> MemoryInfo:
    """Detect system memory information."""
    vm = psutil.virtual_memory()
    return MemoryInfo(
        total_mb=vm.total // (1024 * 1024),
        available_mb=vm.available // (1024 * 1024),
    )
