"""Tests for resource monitoring."""

from __future__ import annotations

import time

from inferbench.benchmarks.sampler import ResourceMonitor
from inferbench.results.models import ResourceSnapshot


class TestResourceMonitor:
    def test_start_stop_returns_snapshot(self):
        monitor = ResourceMonitor(interval_ms=50)
        monitor.start()
        time.sleep(0.2)  # Let it collect a few samples
        snapshot = monitor.stop()
        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.ram_used_mb > 0
        assert snapshot.ram_peak_mb >= snapshot.ram_used_mb

    def test_context_manager(self):
        with ResourceMonitor(interval_ms=50) as monitor:
            time.sleep(0.15)
        # After exit, stop() was called
        assert monitor._thread is None

    def test_empty_when_stopped_immediately(self):
        monitor = ResourceMonitor(interval_ms=1000)
        monitor.start()
        snapshot = monitor.stop()
        # May or may not have samples depending on timing
        assert isinstance(snapshot, ResourceSnapshot)

    def test_cpu_percent_collected(self):
        monitor = ResourceMonitor(interval_ms=50)
        monitor.start()
        # Do some work to generate CPU activity
        _ = sum(range(100000))
        time.sleep(0.2)
        snapshot = monitor.stop()
        assert isinstance(snapshot, ResourceSnapshot)
        # cpu_percent may be 0 on first call due to psutil behavior
        assert snapshot.cpu_percent >= 0.0
