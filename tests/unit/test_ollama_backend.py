"""Tests for Ollama backend with mocked HTTP responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from inferbench.backends.ollama import OllamaBackend


class TestOllamaAvailability:
    def test_available_when_server_responds(self):
        with patch("inferbench.backends.ollama.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            backend = OllamaBackend()
            assert backend.is_available() is True

    def test_not_available_when_connection_fails(self):
        with patch("inferbench.backends.ollama.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client_cls.return_value = mock_client

            backend = OllamaBackend()
            assert backend.is_available() is False

    def test_name_and_display_name(self):
        backend = OllamaBackend()
        assert backend.name == "ollama"
        assert backend.display_name == "Ollama"

    def test_install_hint(self):
        backend = OllamaBackend()
        hint = backend.get_install_hint()
        assert hint is not None
        assert "ollama" in hint.lower()


class TestOllamaVersion:
    def test_get_version(self):
        with patch("inferbench.backends.ollama.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"version": "0.12.11"}
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            backend = OllamaBackend()
            assert backend.get_version() == "0.12.11"


class TestOllamaModels:
    def test_supported_model_ids(self):
        with patch("inferbench.backends.ollama.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "models": [
                    {"name": "qwen2.5:7b"},
                    {"name": "llama3.1:8b"},
                ]
            }
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            backend = OllamaBackend()
            ids = backend.supported_model_ids(hardware=MagicMock())
            assert ids == ["qwen2.5:7b", "llama3.1:8b"]

    def test_supported_model_ids_connection_error(self):
        with patch("inferbench.backends.ollama.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client_cls.return_value = mock_client

            backend = OllamaBackend()
            assert backend.supported_model_ids(hardware=MagicMock()) == []


class TestOllamaCli:
    def test_backends_command_runs(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["backends"])
        assert result.exit_code == 0
        assert "Ollama" in result.stdout

    def test_backends_json(self):
        import json

        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["backends", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert any(b["name"] == "ollama" for b in data)

    def test_models_command_runs(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
