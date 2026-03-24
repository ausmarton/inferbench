"""Backend discovery and registration."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferbench.backends.base import InferenceBackend

logger = logging.getLogger(__name__)

# Registry of backend classes keyed by their short name
_BACKEND_CLASSES: dict[str, type[InferenceBackend]] = {}

# Known backend modules to attempt importing
_KNOWN_BACKENDS = [
    "inferbench.backends.ollama",
    "inferbench.backends.llamacpp",
    "inferbench.backends.vllm",
    "inferbench.backends.transformers",
]


def register(name: str):
    """Decorator to register a backend class.

    Usage:
        @register("ollama")
        class OllamaBackend(InferenceBackend):
            ...
    """

    def decorator(cls: type[InferenceBackend]) -> type[InferenceBackend]:
        _BACKEND_CLASSES[name] = cls
        return cls

    return decorator


def _discover_backends() -> None:
    """Attempt to import all known backend modules.

    Each module registers itself via the @register decorator on import.
    Import failures (missing deps) are silently ignored.
    """
    for module_name in _KNOWN_BACKENDS:
        try:
            importlib.import_module(module_name)
        except ImportError:
            logger.debug("Backend module %s not importable (missing deps)", module_name)
        except Exception:
            logger.warning("Error importing backend module %s", module_name, exc_info=True)


def get_all_backends() -> list[InferenceBackend]:
    """Return instances of all registered backends (available or not).

    Triggers discovery of backend modules on first call.
    """
    if not _BACKEND_CLASSES:
        _discover_backends()

    backends = []
    for name, cls in _BACKEND_CLASSES.items():
        try:
            backends.append(cls())
        except Exception:
            logger.warning("Failed to instantiate backend %s", name, exc_info=True)
    return backends


def get_available_backends() -> list[InferenceBackend]:
    """Return instances of backends whose dependencies are installed and functional."""
    return [b for b in get_all_backends() if b.is_available()]


def get_backend(name: str) -> InferenceBackend | None:
    """Get a specific backend by name, or None if not registered."""
    if not _BACKEND_CLASSES:
        _discover_backends()

    cls = _BACKEND_CLASSES.get(name)
    if cls is None:
        return None
    try:
        return cls()
    except Exception:
        logger.warning("Failed to instantiate backend %s", name, exc_info=True)
        return None
