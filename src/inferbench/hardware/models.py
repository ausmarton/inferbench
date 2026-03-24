"""Data models for hardware detection results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class GpuVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    NONE = "none"


class GpuDriver(StrEnum):
    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    METAL = "metal"
    NONE = "none"


class CpuFeatures(BaseModel):
    """CPU capabilities relevant to inference performance."""

    vendor: str
    model_name: str
    architecture: str
    cores_physical: int
    cores_logical: int
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_amx: bool = False
    has_neon: bool = False
    cache_l3_kb: int | None = None
    freq_max_mhz: float | None = None
    freq_min_mhz: float | None = None


class GpuInfo(BaseModel):
    """GPU capabilities relevant to inference performance."""

    vendor: GpuVendor
    name: str
    vram_total_mb: int
    vram_available_mb: int | None = None
    driver: GpuDriver = GpuDriver.NONE
    driver_version: str = ""
    compute_capability: str | None = None  # NVIDIA only (e.g. "8.9")
    rocm_arch: str | None = None  # AMD only (e.g. "gfx1151")
    is_integrated: bool = False


class MemoryInfo(BaseModel):
    """System memory information."""

    total_mb: int
    available_mb: int
    is_unified: bool = False  # True for APUs where GPU shares system RAM


class HardwareProfile(BaseModel):
    """Complete hardware profile for a system."""

    cpu: CpuFeatures
    gpus: list[GpuInfo]
    memory: MemoryInfo
    os_name: str
    os_version: str
    kernel_version: str
    python_version: str

    @property
    def has_gpu(self) -> bool:
        return any(g.vendor != GpuVendor.NONE for g in self.gpus)

    @property
    def has_cuda(self) -> bool:
        return any(g.driver == GpuDriver.CUDA for g in self.gpus)

    @property
    def has_rocm(self) -> bool:
        return any(g.driver == GpuDriver.ROCM for g in self.gpus)

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_total_mb for g in self.gpus)

    @property
    def best_gpu(self) -> GpuInfo | None:
        """Return the GPU with the most VRAM, or None."""
        gpu_list = [g for g in self.gpus if g.vendor != GpuVendor.NONE]
        return max(gpu_list, key=lambda g: g.vram_total_mb) if gpu_list else None
