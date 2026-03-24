"""Tests for report generation and storage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from inferbench.results.models import BenchmarkResult, BenchmarkRun, BenchmarkSession
from inferbench.results.storage import list_sessions, load_session, save_session


def _make_session(num_results: int = 2) -> BenchmarkSession:
    """Create a test session with sample results."""
    results = []
    for i in range(num_results):
        runs = [
            BenchmarkRun(
                prompt_name="short_chat",
                prompt_text="Hello",
                prompt_tokens=5,
                max_output_tokens=256,
                temperature=0.0,
                output_text=f"Response {i}",
                output_tokens=10,
                start_ns=1_000_000_000,
                first_token_ns=1_050_000_000,
                end_ns=1_500_000_000,
                token_timestamps_ns=[1_050_000_000 + j * 50_000_000 for j in range(10)],
            )
        ]
        results.append(
            BenchmarkResult(
                backend_name=f"backend_{i}",
                backend_version="1.0",
                model_id=f"model_{i}:7b",
                cold_load_time_s=5.0 + i,
                warm_load_time_s=0.5 + i * 0.1,
                runs=runs,
            )
        )
    return BenchmarkSession(results=results, prompt_names=["short_chat"])


class TestStorage:
    def test_save_creates_file(self):
        session = _make_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            saved = save_session(session, path)
            assert saved.exists()
            assert saved == path

    def test_save_default_path(self):
        session = _make_session()
        with tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory() as results_dir:
            path = Path(results_dir) / f"{session.session_id}.json"
            saved = save_session(session, path)
            assert saved.exists()

    def test_load_roundtrip(self):
        session = _make_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_session(session, path)
            loaded = load_session(path)

            assert loaded.session_id == session.session_id
            assert len(loaded.results) == 2
            assert loaded.results[0].backend_name == "backend_0"
            assert len(loaded.results[0].runs) == 1

    def test_load_preserves_metrics(self):
        session = _make_session(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_session(session, path)
            loaded = load_session(path)

            original_run = session.results[0].runs[0]
            loaded_run = loaded.results[0].runs[0]
            assert loaded_run.ttft_ms == original_run.ttft_ms
            assert loaded_run.tps == original_run.tps

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            # Create some session files
            for i in range(3):
                (d / f"session_{i}.json").write_text("{}")

            sessions = list_sessions(d)
            assert len(sessions) == 3

    def test_list_sessions_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = list_sessions(Path(tmpdir))
            assert sessions == []

    def test_list_sessions_nonexistent_dir(self):
        sessions = list_sessions(Path("/nonexistent/path"))
        assert sessions == []


class TestReportOutput:
    def test_print_session_summary_no_crash(self):
        """Report printing should not crash with valid data."""
        from io import StringIO

        from rich.console import Console

        from inferbench.results.report import print_session_summary

        session = _make_session()
        # Redirect console output to a string
        console = Console(file=StringIO())
        import inferbench.results.report as report_mod

        original_console = report_mod.console
        report_mod.console = console
        try:
            print_session_summary(session)
        finally:
            report_mod.console = original_console

    def test_print_session_summary_empty(self):
        """Report printing should handle empty sessions gracefully."""
        from io import StringIO

        from rich.console import Console

        from inferbench.results.report import print_session_summary

        session = BenchmarkSession()
        console = Console(file=StringIO())
        import inferbench.results.report as report_mod

        original_console = report_mod.console
        report_mod.console = console
        try:
            print_session_summary(session)
        finally:
            report_mod.console = original_console

    def test_print_comparison_no_crash(self):
        from io import StringIO

        from rich.console import Console

        from inferbench.results.report import print_result_comparison

        session = _make_session()
        console = Console(file=StringIO())
        import inferbench.results.report as report_mod

        original_console = report_mod.console
        report_mod.console = console
        try:
            print_result_comparison(session.results)
        finally:
            report_mod.console = original_console


class TestCliReportCommand:
    def test_report_command_with_file(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        session = _make_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_session(session, path)

            runner = CliRunner()
            result = runner.invoke(app, ["report", str(path)])
            assert result.exit_code == 0
            assert "Benchmark" in result.stdout

    def test_report_json_output(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        session = _make_session(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_session(session, path)

            runner = CliRunner()
            result = runner.invoke(app, ["report", str(path), "--json"])
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "results" in data

    def test_report_nonexistent_file(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["report", "/nonexistent/file.json"])
        assert result.exit_code == 1

    def test_compare_command(self):
        from typer.testing import CliRunner

        from inferbench.cli.app import app

        session = _make_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_session(session, path)

            runner = CliRunner()
            result = runner.invoke(app, ["compare", str(path)])
            assert result.exit_code == 0
