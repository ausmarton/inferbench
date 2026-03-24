"""Shared test fixtures."""

from __future__ import annotations

import pytest

from inferbench.hardware.models import (
    CpuFeatures,
    GpuDriver,
    GpuInfo,
    GpuVendor,
    HardwareProfile,
    MemoryInfo,
)


@pytest.fixture
def amd_apu_profile() -> HardwareProfile:
    """Hardware profile for an AMD APU with unified memory."""
    return HardwareProfile(
        cpu=CpuFeatures(
            vendor="AuthenticAMD",
            model_name="AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S",
            architecture="x86_64",
            cores_physical=16,
            cores_logical=32,
            has_avx=True,
            has_avx2=True,
            has_avx512=True,
            cache_l3_kb=65536,
            freq_max_mhz=5187.5,
            freq_min_mhz=625.0,
        ),
        gpus=[
            GpuInfo(
                vendor=GpuVendor.AMD,
                name="Radeon 8060S",
                vram_total_mb=512,
                vram_available_mb=400,
                driver=GpuDriver.ROCM,
                driver_version="6.2.0",
                rocm_arch="gfx1151",
                is_integrated=True,
            ),
        ],
        memory=MemoryInfo(
            total_mb=128000,
            available_mb=60000,
            is_unified=True,
        ),
        os_name="Linux",
        os_version="6.18.16",
        kernel_version="#1 SMP PREEMPT_DYNAMIC",
        python_version="3.14.3",
    )


@pytest.fixture
def nvidia_desktop_profile() -> HardwareProfile:
    """Hardware profile for a desktop with NVIDIA GPU."""
    return HardwareProfile(
        cpu=CpuFeatures(
            vendor="GenuineIntel",
            model_name="Intel Core i9-14900K",
            architecture="x86_64",
            cores_physical=24,
            cores_logical=32,
            has_avx=True,
            has_avx2=True,
            has_avx512=False,
            cache_l3_kb=36864,
            freq_max_mhz=6000.0,
        ),
        gpus=[
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="NVIDIA RTX 4090",
                vram_total_mb=24576,
                vram_available_mb=22000,
                driver=GpuDriver.CUDA,
                driver_version="550.54.14",
                compute_capability="8.9",
            ),
        ],
        memory=MemoryInfo(
            total_mb=65536,
            available_mb=55000,
        ),
        os_name="Linux",
        os_version="6.8.0",
        kernel_version="#1 SMP",
        python_version="3.12.4",
    )


@pytest.fixture
def no_gpu_profile() -> HardwareProfile:
    """Hardware profile for a CPU-only system."""
    return HardwareProfile(
        cpu=CpuFeatures(
            vendor="GenuineIntel",
            model_name="Intel Xeon E5-2680 v4",
            architecture="x86_64",
            cores_physical=14,
            cores_logical=28,
            has_avx=True,
            has_avx2=True,
        ),
        gpus=[],
        memory=MemoryInfo(
            total_mb=131072,
            available_mb=120000,
        ),
        os_name="Linux",
        os_version="5.15.0",
        kernel_version="#1 SMP",
        python_version="3.11.9",
    )
