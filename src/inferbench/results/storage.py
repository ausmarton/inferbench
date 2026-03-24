"""Persistence for benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from inferbench.results.models import BenchmarkSession

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path.home() / ".inferbench" / "results"


def save_session(session: BenchmarkSession, path: Path | None = None) -> Path:
    """Save a benchmark session to JSON.

    If no path is provided, saves to ~/.inferbench/results/<session_id>.json.
    Returns the path the file was written to.
    """
    if path is None:
        path = DEFAULT_RESULTS_DIR / f"{session.session_id}.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    data = session.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, default=str))

    logger.info("Saved session %s to %s", session.session_id, path)
    return path


def load_session(path: Path) -> BenchmarkSession:
    """Load a benchmark session from a JSON file."""
    data = json.loads(path.read_text())
    return BenchmarkSession.model_validate(data)


def list_sessions(results_dir: Path | None = None) -> list[Path]:
    """List all saved session files."""
    d = results_dir or DEFAULT_RESULTS_DIR
    if not d.exists():
        return []
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
