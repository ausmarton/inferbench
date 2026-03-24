"""Tests for vLLM backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from inferbench.backends.vllm import VllmBackend


class TestVllmAvailability:
    def test_available_when_imported(self):
        mock_vllm = MagicMock()
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            backend = VllmBackend()
            assert backend.is_available() is True

    def test_not_available_when_missing(self):
        with patch.dict("sys.modules", {"vllm": None}):
            backend = VllmBackend()
            assert backend.is_available() is False

    def test_name_and_display(self):
        backend = VllmBackend()
        assert backend.name == "vllm"
        assert backend.display_name == "vLLM"

    def test_install_hint(self):
        backend = VllmBackend()
        hint = backend.get_install_hint()
        assert hint is not None
        assert "vllm" in hint


class TestVllmVersion:
    def test_get_version(self):
        mock_vllm = MagicMock()
        mock_vllm.__version__ = "0.6.2"
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            backend = VllmBackend()
            assert backend.get_version() == "0.6.2"

    def test_get_version_not_installed(self):
        with patch.dict("sys.modules", {"vllm": None}):
            backend = VllmBackend()
            assert backend.get_version() == "not installed"


class TestVllmModels:
    def test_no_models_without_gpu(self):
        backend = VllmBackend()
        hardware = MagicMock()
        hardware.has_gpu = False
        ids = backend.supported_model_ids(hardware)
        assert ids == []

    def test_custom_port(self):
        backend = VllmBackend(port=9000)
        assert backend._base_url == "http://localhost:9000"
