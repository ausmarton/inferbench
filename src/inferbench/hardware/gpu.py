"""GPU detection — NVIDIA, AMD, Intel, Apple."""

from __future__ import annotations

import subprocess
from pathlib import Path

from inferbench.hardware.models import GpuDriver, GpuInfo, GpuVendor

# PCI vendor IDs
_VENDOR_IDS = {
    "0x10de": GpuVendor.NVIDIA,
    "0x1002": GpuVendor.AMD,
    "0x8086": GpuVendor.INTEL,
}


def detect_gpus() -> list[GpuInfo]:
    """Detect all GPUs in the system."""
    gpus: list[GpuInfo] = []

    # Try NVIDIA first (most common for ML)
    gpus.extend(_detect_nvidia())

    # Try AMD via sysfs + ROCm
    gpus.extend(_detect_amd())

    # Try Intel via sysfs
    gpus.extend(_detect_intel())

    return gpus


def _detect_nvidia() -> list[GpuInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    GpuInfo(
                        vendor=GpuVendor.NVIDIA,
                        name=parts[0],
                        vram_total_mb=int(float(parts[1])),
                        vram_available_mb=int(float(parts[2])),
                        driver=GpuDriver.CUDA,
                        driver_version=parts[3],
                        compute_capability=parts[4],
                    )
                )
        return gpus
    except (OSError, subprocess.TimeoutExpired):
        return []


def _detect_amd() -> list[GpuInfo]:
    """Detect AMD GPUs via sysfs and ROCm."""
    gpus: list[GpuInfo] = []

    # Check if ROCm kernel driver is present
    has_rocm_kfd = Path("/dev/kfd").exists()

    # Scan DRM devices in sysfs
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return gpus

    seen_devices: set[str] = set()

    for card_dir in sorted(drm_path.iterdir()):
        if not card_dir.name.startswith("card") or "-" in card_dir.name:
            continue

        device_dir = card_dir / "device"
        if not device_dir.exists():
            continue

        # Check vendor
        vendor_id = _read_sysfs(device_dir / "vendor")
        if vendor_id != "0x1002":
            continue

        # Avoid duplicate entries for same PCI device
        device_path = str(device_dir.resolve())
        if device_path in seen_devices:
            continue
        seen_devices.add(device_path)

        name = _get_amd_gpu_name(device_dir)
        vram_total_mb = _get_amd_vram_total(device_dir)
        vram_available_mb = _get_amd_vram_available(device_dir)

        # Check if this is an integrated GPU (APU)
        is_integrated = _is_amd_integrated(device_dir)

        # Determine driver type
        driver = GpuDriver.ROCM if has_rocm_kfd else GpuDriver.NONE

        # Try to get ROCm version
        driver_version = _get_rocm_version() if has_rocm_kfd else ""

        # Try to get gfx arch from rocm-smi or amdgpu info
        rocm_arch_str = _get_amd_gfx_arch(device_dir)

        gpus.append(
            GpuInfo(
                vendor=GpuVendor.AMD,
                name=name,
                vram_total_mb=vram_total_mb,
                vram_available_mb=vram_available_mb,
                driver=driver,
                driver_version=driver_version,
                rocm_arch=rocm_arch_str,
                is_integrated=is_integrated,
            )
        )

    return gpus


def _detect_intel() -> list[GpuInfo]:
    """Detect Intel GPUs via sysfs."""
    gpus: list[GpuInfo] = []
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return gpus

    seen_devices: set[str] = set()

    for card_dir in sorted(drm_path.iterdir()):
        if not card_dir.name.startswith("card") or "-" in card_dir.name:
            continue

        device_dir = card_dir / "device"
        if not device_dir.exists():
            continue

        vendor_id = _read_sysfs(device_dir / "vendor")
        if vendor_id != "0x8086":
            continue

        device_path = str(device_dir.resolve())
        if device_path in seen_devices:
            continue
        seen_devices.add(device_path)

        # Intel iGPU detection is limited without specialized tools
        gpus.append(
            GpuInfo(
                vendor=GpuVendor.INTEL,
                name=_get_pci_device_name(device_dir) or "Intel GPU",
                vram_total_mb=0,
                driver=GpuDriver.NONE,
                is_integrated=True,
            )
        )

    return gpus


def _read_sysfs(path: Path) -> str:
    """Read a sysfs file, returning empty string on failure."""
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _get_amd_gpu_name(device_dir: Path) -> str:
    """Get AMD GPU name from various sysfs sources."""
    # Try product name
    name = _read_sysfs(device_dir / "product_name")
    if name:
        return name

    # Try lspci for the PCI slot
    slot = _read_sysfs(device_dir / "uevent")
    for line in slot.splitlines():
        if line.startswith("PCI_SLOT_NAME="):
            pci_slot = line.split("=", 1)[1]
            return _lspci_device_name(pci_slot) or "AMD GPU"

    return "AMD GPU"


def _get_amd_vram_total(device_dir: Path) -> int:
    """Get total VRAM in MB from sysfs."""
    # Try mem_info_vram_total (bytes)
    vram_str = _read_sysfs(device_dir / "mem_info_vram_total")
    if vram_str:
        try:
            return int(vram_str) // (1024 * 1024)
        except ValueError:
            pass

    return 0


def _get_amd_vram_available(device_dir: Path) -> int | None:
    """Get available VRAM in MB from sysfs."""
    vram_str = _read_sysfs(device_dir / "mem_info_vram_used")
    total_str = _read_sysfs(device_dir / "mem_info_vram_total")
    if vram_str and total_str:
        try:
            total = int(total_str) // (1024 * 1024)
            used = int(vram_str) // (1024 * 1024)
            return total - used
        except ValueError:
            pass

    return None


def _is_amd_integrated(device_dir: Path) -> bool:
    """Check if an AMD GPU is integrated (APU)."""
    # If VRAM total equals 0 or is very small relative to what we'd expect,
    # it might be integrated. Also check the device class.
    device_class = _read_sysfs(device_dir / "class")
    # 0x038000 = Display controller (often used for APU display engines)
    # 0x030000 = VGA compatible controller
    if device_class.startswith("0x03800"):
        return True

    # Check if mem_info_vram_total is suspiciously close to system RAM
    # (unified memory APUs report a portion of system RAM as VRAM)
    vram_str = _read_sysfs(device_dir / "mem_info_vram_total")
    if vram_str:
        try:
            vram_bytes = int(vram_str)
            # APUs typically carve out a large chunk (e.g. 96GB out of 128GB)
            # If VRAM > 16GB and there's no discrete GPU marker, likely an APU
            if vram_bytes > 16 * 1024 * 1024 * 1024:
                return True
        except ValueError:
            pass

    return False


def _get_amd_gfx_arch(device_dir: Path) -> str | None:
    """Get the GFX architecture string (e.g. gfx1151) for an AMD GPU."""
    # Try rocm-smi first
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Parse output for architecture info
        for line in result.stdout.splitlines():
            if "gfx" in line.lower():
                # Extract gfxNNNN pattern
                for word in line.split():
                    if word.lower().startswith("gfx"):
                        return word.lower()
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Try reading from amdgpu driver sysfs
    fw_ver = _read_sysfs(device_dir / "gpu_id")
    if fw_ver:
        return f"gfx_id:{fw_ver}"

    return None


def _get_rocm_version() -> str:
    """Get the installed ROCm version."""
    # Check /opt/rocm/.info/version
    version_path = Path("/opt/rocm/.info/version")
    if version_path.exists():
        return _read_sysfs(version_path)

    # Try rocm-smi --version
    try:
        result = subprocess.run(
            ["rocm-smi", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip().split()[-1] if result.stdout.strip() else ""
    except (OSError, subprocess.TimeoutExpired):
        pass

    return "kfd-present"


def _get_pci_device_name(device_dir: Path) -> str | None:
    """Get a PCI device name via lspci."""
    uevent = _read_sysfs(device_dir / "uevent")
    for line in uevent.splitlines():
        if line.startswith("PCI_SLOT_NAME="):
            return _lspci_device_name(line.split("=", 1)[1])
    return None


def _lspci_device_name(slot: str) -> str | None:
    """Get device name for a PCI slot from lspci."""
    try:
        result = subprocess.run(
            ["lspci", "-s", slot],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            # Format: "03:00.0 Display controller: AMD/ATI Strix Halo ..."
            parts = result.stdout.strip().split(": ", 1)
            if len(parts) >= 2:
                return parts[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None
