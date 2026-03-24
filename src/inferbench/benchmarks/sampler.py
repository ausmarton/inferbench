"""Resource monitoring during benchmark runs."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from inferbench.results.models import ResourceSnapshot


@dataclass
class _Sample:
    timestamp_ns: int
    ram_used_mb: float
    cpu_percent: float
    gpu_percent: float | None = None
    vram_used_mb: float | None = None


class ResourceMonitor:
    """Monitors CPU, RAM, and GPU usage in a background thread.

    Usage:
        monitor = ResourceMonitor(interval_ms=100)
        monitor.start()
        # ... do work ...
        snapshot = monitor.stop()
    """

    def __init__(self, interval_ms: int = 100, pid: int | None = None) -> None:
        self._interval_s = interval_ms / 1000.0
        self._pid = pid or psutil.Process().pid
        self._samples: list[_Sample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_sysfs_path = _find_amd_gpu_busy_path()

    def start(self) -> None:
        """Start background monitoring."""
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSnapshot:
        """Stop monitoring and return aggregated snapshot."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self._aggregate()

    def _sample_loop(self) -> None:
        """Collect samples at regular intervals."""
        try:
            process = psutil.Process(self._pid)
        except psutil.NoSuchProcess:
            return

        while not self._stop_event.is_set():
            try:
                mem_info = process.memory_info()
                cpu_pct = process.cpu_percent(interval=None)

                sample = _Sample(
                    timestamp_ns=time.perf_counter_ns(),
                    ram_used_mb=mem_info.rss / (1024 * 1024),
                    cpu_percent=cpu_pct,
                )

                # Try to read AMD GPU utilization
                if self._gpu_sysfs_path:
                    sample.gpu_percent = _read_amd_gpu_busy(self._gpu_sysfs_path)

                self._samples.append(sample)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            self._stop_event.wait(self._interval_s)

    def _aggregate(self) -> ResourceSnapshot:
        """Aggregate samples into a single snapshot."""
        if not self._samples:
            return ResourceSnapshot()

        ram_values = [s.ram_used_mb for s in self._samples]
        cpu_values = [s.cpu_percent for s in self._samples]
        gpu_values = [s.gpu_percent for s in self._samples if s.gpu_percent is not None]

        return ResourceSnapshot(
            ram_used_mb=sum(ram_values) / len(ram_values),
            ram_peak_mb=max(ram_values),
            cpu_percent=sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            gpu_percent=sum(gpu_values) / len(gpu_values) if gpu_values else None,
        )

    def __enter__(self) -> ResourceMonitor:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


def _find_amd_gpu_busy_path() -> Path | None:
    """Find the sysfs path for AMD GPU utilization."""
    drm = Path("/sys/class/drm")
    if not drm.exists():
        return None
    for card in sorted(drm.iterdir()):
        if not card.name.startswith("card") or "-" in card.name:
            continue
        busy_path = card / "device" / "gpu_busy_percent"
        if busy_path.exists():
            return busy_path
    return None


def _read_amd_gpu_busy(path: Path) -> float | None:
    """Read AMD GPU busy percentage from sysfs."""
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None
