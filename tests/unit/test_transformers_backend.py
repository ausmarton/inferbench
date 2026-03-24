"""Tests for Transformers backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from inferbench.backends.transformers import TransformersBackend


class TestTransformersAvailability:
    def test_available_when_imported(self):
        mock_torch = MagicMock()
        mock_transformers = MagicMock()
        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            backend = TransformersBackend()
            assert backend.is_available() is True

    def test_not_available_when_torch_missing(self):
        with patch.dict("sys.modules", {"torch": None}):
            backend = TransformersBackend()
            assert backend.is_available() is False

    def test_name_and_display(self):
        backend = TransformersBackend()
        assert backend.name == "transformers"
        assert backend.display_name == "Transformers"

    def test_install_hint(self):
        backend = TransformersBackend()
        hint = backend.get_install_hint()
        assert hint is not None
        assert "transformers" in hint


class TestTransformersVersion:
    def test_get_version(self):
        mock_transformers = MagicMock()
        mock_transformers.__version__ = "4.45.0"
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            backend = TransformersBackend()
            assert backend.get_version() == "4.45.0"

    def test_get_version_not_installed(self):
        with patch.dict("sys.modules", {"transformers": None}):
            backend = TransformersBackend()
            assert backend.get_version() == "not installed"


class TestTransformersModels:
    def test_supported_model_ids(self):
        backend = TransformersBackend()
        hardware = MagicMock()
        ids = backend.supported_model_ids(hardware)
        # Should return models from catalog with transformers backend IDs
        assert isinstance(ids, list)
        # Our catalog has transformers entries for qwen2.5-7b, llama3.1-8b, etc.
        assert any("Qwen" in mid or "llama" in mid.lower() for mid in ids)
