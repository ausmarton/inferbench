"""Main hardware detection orchestrator."""

from __future__ import annotations

import platform
import sys

from inferbench.hardware.cpu import detect_cpu
from inferbench.hardware.gpu import detect_gpus
from inferbench.hardware.memory import detect_memory
from inferbench.hardware.models import HardwareProfile, MemoryInfo


def detect_hardware() -> HardwareProfile:
    """Run full hardware detection and return a complete profile."""
    cpu = detect_cpu()
    gpus = detect_gpus()
    memory = detect_memory()

    # Check for unified memory (APU)
    has_integrated_gpu = any(g.is_integrated for g in gpus)
    if has_integrated_gpu:
        memory = MemoryInfo(
            total_mb=memory.total_mb,
            available_mb=memory.available_mb,
            is_unified=True,
        )

    # OS info
    os_name = platform.system()
    os_version = platform.release()
    kernel_version = platform.version()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return HardwareProfile(
        cpu=cpu,
        gpus=gpus,
        memory=memory,
        os_name=os_name,
        os_version=os_version,
        kernel_version=kernel_version,
        python_version=python_version,
    )
