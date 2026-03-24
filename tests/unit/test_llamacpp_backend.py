"""Tests for llama.cpp backend."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferbench.backends.llamacpp import LlamaCppBackend, _resolve_model_path


class TestLlamaCppAvailability:
    def test_available_when_imported(self):
        with patch.dict("sys.modules", {"llama_cpp": MagicMock()}):
            backend = LlamaCppBackend()
            assert backend.is_available() is True

    def test_not_available_when_missing(self):
        with patch.dict("sys.modules", {"llama_cpp": None}):
            backend = LlamaCppBackend()
            # Import will raise since module is None
            assert backend.is_available() is False

    def test_name_and_display(self):
        backend = LlamaCppBackend()
        assert backend.name == "llamacpp"
        assert backend.display_name == "llama.cpp"

    def test_install_hint(self):
        backend = LlamaCppBackend()
        hint = backend.get_install_hint()
        assert hint is not None
        assert "llamacpp" in hint


class TestLlamaCppVersion:
    def test_get_version(self):
        mock_module = MagicMock()
        mock_module.__version__ = "0.3.18"
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            backend = LlamaCppBackend()
            assert backend.get_version() == "0.3.18"

    def test_get_version_not_installed(self):
        with patch.dict("sys.modules", {"llama_cpp": None}):
            backend = LlamaCppBackend()
            assert backend.get_version() == "not installed"


class TestResolveModelPath:
    def test_local_gguf_file(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            path = Path(f.name)
            try:
                resolved = _resolve_model_path(str(path))
                assert resolved == path
            finally:
                path.unlink()

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _resolve_model_path("/nonexistent/model.gguf")

    def test_hf_reference_format(self):
        """Test that HF repo:filename format triggers download attempt."""
        with (
            patch("inferbench.backends.llamacpp._download_from_hf") as mock_dl,
            patch("inferbench.backends.llamacpp._MODEL_CACHE_DIR", Path("/nonexistent")),
        ):
            mock_dl.return_value = Path("/tmp/model.gguf")
            result = _resolve_model_path("org/repo:model.gguf")
            mock_dl.assert_called_once_with("org/repo", "model.gguf")
            assert result == Path("/tmp/model.gguf")


class TestLlamaCppCli:
    def test_backends_includes_llamacpp(self):
        """llama.cpp should show up in the backends list."""
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["backends"])
        assert result.exit_code == 0
        assert "llama.cpp" in result.stdout

    def test_models_with_llamacpp_filter(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["models", "--backend", "llamacpp"])
        assert result.exit_code == 0

    def test_compare_command_help(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "Compare" in result.stdout
