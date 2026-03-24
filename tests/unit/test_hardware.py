"""Tests for hardware detection modules."""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

from inferbench.hardware.cpu import _parse_cache_size, detect_cpu
from inferbench.hardware.gpu import _detect_nvidia
from inferbench.hardware.memory import detect_memory
from inferbench.hardware.models import (
    CpuFeatures,
    GpuDriver,
    GpuInfo,
    GpuVendor,
    HardwareProfile,
    MemoryInfo,
)


class TestCpuFeatures:
    def test_defaults(self):
        cpu = CpuFeatures(
            vendor="test",
            model_name="test",
            architecture="x86_64",
            cores_physical=4,
            cores_logical=8,
        )
        assert cpu.has_avx is False
        assert cpu.has_avx2 is False
        assert cpu.has_avx512 is False
        assert cpu.has_amx is False
        assert cpu.has_neon is False
        assert cpu.cache_l3_kb is None
        assert cpu.freq_max_mhz is None

    def test_all_features(self):
        cpu = CpuFeatures(
            vendor="AuthenticAMD",
            model_name="AMD Ryzen",
            architecture="x86_64",
            cores_physical=16,
            cores_logical=32,
            has_avx=True,
            has_avx2=True,
            has_avx512=True,
            cache_l3_kb=65536,
            freq_max_mhz=5000.0,
        )
        assert cpu.has_avx512 is True
        assert cpu.cache_l3_kb == 65536


class TestGpuInfo:
    def test_nvidia_gpu(self):
        gpu = GpuInfo(
            vendor=GpuVendor.NVIDIA,
            name="RTX 4090",
            vram_total_mb=24576,
            driver=GpuDriver.CUDA,
            driver_version="550.54.14",
            compute_capability="8.9",
        )
        assert gpu.vendor == GpuVendor.NVIDIA
        assert gpu.driver == GpuDriver.CUDA
        assert gpu.is_integrated is False

    def test_amd_apu(self):
        gpu = GpuInfo(
            vendor=GpuVendor.AMD,
            name="Radeon 8060S",
            vram_total_mb=512,
            driver=GpuDriver.ROCM,
            is_integrated=True,
        )
        assert gpu.is_integrated is True
        assert gpu.driver == GpuDriver.ROCM


class TestMemoryInfo:
    def test_discrete(self):
        mem = MemoryInfo(total_mb=65536, available_mb=55000)
        assert mem.is_unified is False

    def test_unified(self):
        mem = MemoryInfo(total_mb=128000, available_mb=60000, is_unified=True)
        assert mem.is_unified is True


class TestHardwareProfile:
    def test_has_gpu(self, amd_apu_profile: HardwareProfile):
        assert amd_apu_profile.has_gpu is True

    def test_no_gpu(self, no_gpu_profile: HardwareProfile):
        assert no_gpu_profile.has_gpu is False

    def test_has_cuda(self, nvidia_desktop_profile: HardwareProfile):
        assert nvidia_desktop_profile.has_cuda is True

    def test_has_rocm(self, amd_apu_profile: HardwareProfile):
        assert amd_apu_profile.has_rocm is True

    def test_total_vram(self, nvidia_desktop_profile: HardwareProfile):
        assert nvidia_desktop_profile.total_vram_mb == 24576

    def test_best_gpu(self, nvidia_desktop_profile: HardwareProfile):
        best = nvidia_desktop_profile.best_gpu
        assert best is not None
        assert best.name == "NVIDIA RTX 4090"

    def test_best_gpu_none(self, no_gpu_profile: HardwareProfile):
        assert no_gpu_profile.best_gpu is None

    def test_serialization_roundtrip(self, amd_apu_profile: HardwareProfile):
        """Test that a profile survives JSON serialization/deserialization."""
        json_data = amd_apu_profile.model_dump()
        restored = HardwareProfile.model_validate(json_data)
        assert restored.cpu.model_name == amd_apu_profile.cpu.model_name
        assert restored.gpus[0].vendor == GpuVendor.AMD
        assert restored.memory.is_unified is True


class TestParseCacheSize:
    def test_mib(self):
        assert _parse_cache_size("64 MiB") == 65536

    def test_kib(self):
        assert _parse_cache_size("512 KiB") == 512

    def test_gib(self):
        assert _parse_cache_size("1 GiB") == 1048576

    def test_no_unit(self):
        assert _parse_cache_size("1024") == 1024

    def test_empty(self):
        assert _parse_cache_size("") is None

    def test_with_instance_info(self):
        # lscpu sometimes reports "32 MiB (1 instance)"
        assert _parse_cache_size("32 MiB (1 instance)") == 32768


class TestDetectCpuLinux:
    PROC_CPUINFO = """\
processor	: 0
vendor_id	: AuthenticAMD
model name	: AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S
flags		: fpu avx avx2 avx512f sse sse2
"""

    LSCPU_OUTPUT = """\
Architecture:          x86_64
CPU(s):                32
CPU max MHz:           5187.5000
CPU min MHz:           625.0000
L3 cache:              64 MiB
"""

    @patch("inferbench.hardware.cpu.platform")
    @patch("inferbench.hardware.cpu.psutil")
    @patch("inferbench.hardware.cpu.subprocess.run")
    @patch("builtins.open", mock_open(read_data=PROC_CPUINFO))
    def test_linux_detection(self, mock_subprocess, mock_psutil, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        mock_psutil.cpu_count.side_effect = lambda logical: 32 if logical else 16

        mock_result = MagicMock()
        mock_result.stdout = self.LSCPU_OUTPUT
        mock_subprocess.return_value = mock_result

        cpu = detect_cpu()

        assert cpu.vendor == "AuthenticAMD"
        assert cpu.model_name == "AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S"
        assert cpu.has_avx is True
        assert cpu.has_avx2 is True
        assert cpu.has_avx512 is True
        assert cpu.cores_physical == 16
        assert cpu.cores_logical == 32
        assert cpu.freq_max_mhz == 5187.5
        assert cpu.cache_l3_kb == 65536


class TestDetectNvidia:
    NVIDIA_SMI_OUTPUT = "NVIDIA RTX 4090, 24576, 22000, 550.54.14, 8.9\n"

    @patch("inferbench.hardware.gpu.subprocess.run")
    def test_nvidia_detected(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = self.NVIDIA_SMI_OUTPUT
        mock_run.return_value = mock_result

        gpus = _detect_nvidia()
        assert len(gpus) == 1
        assert gpus[0].vendor == GpuVendor.NVIDIA
        assert gpus[0].name == "NVIDIA RTX 4090"
        assert gpus[0].vram_total_mb == 24576
        assert gpus[0].driver == GpuDriver.CUDA
        assert gpus[0].compute_capability == "8.9"

    @patch("inferbench.hardware.gpu.subprocess.run")
    def test_nvidia_not_present(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert _detect_nvidia() == []


class TestDetectMemory:
    @patch("inferbench.hardware.memory.psutil.virtual_memory")
    def test_detect(self, mock_vm):
        mock_vm.return_value = MagicMock(
            total=128 * 1024 * 1024 * 1024,  # 128 GB
            available=60 * 1024 * 1024 * 1024,  # 60 GB
        )
        mem = detect_memory()
        assert mem.total_mb == 131072
        assert mem.available_mb == 61440


class TestDetectCli:
    def test_detect_command_runs(self):
        """Integration-style test: the detect command should not crash."""
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["detect", "--json"])
        assert result.exit_code == 0
        # Should be valid JSON
        import json

        data = json.loads(result.stdout)
        assert "cpu" in data
        assert "gpus" in data
        assert "memory" in data

    def test_version(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "inferbench" in result.stdout
