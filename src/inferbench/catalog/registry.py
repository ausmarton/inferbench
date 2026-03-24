"""Model catalog loading and filtering."""

from __future__ import annotations

import logging
from importlib.resources import files
from typing import TYPE_CHECKING

import yaml

from inferbench.catalog.models import ModelSpec

if TYPE_CHECKING:
    from inferbench.hardware.models import HardwareProfile

logger = logging.getLogger(__name__)

_catalog: list[ModelSpec] | None = None


def load_catalog() -> list[ModelSpec]:
    """Load the built-in model catalog from builtin.yaml."""
    global _catalog
    if _catalog is not None:
        return _catalog

    catalog_file = files("inferbench.catalog").joinpath("builtin.yaml")
    raw = yaml.safe_load(catalog_file.read_text())

    _catalog = [ModelSpec.model_validate(entry) for entry in raw.get("models", [])]
    logger.debug("Loaded %d models from catalog", len(_catalog))
    return _catalog


def filter_by_hardware(
    models: list[ModelSpec],
    hardware: HardwareProfile,
) -> list[ModelSpec]:
    """Filter models to those that can plausibly run on the given hardware.

    Uses a simple heuristic: the model's estimated RAM requirement must fit
    in available system memory (with headroom for the OS and runtime).
    """
    available_mb = hardware.memory.available_mb
    # Reserve ~2GB for OS and runtime overhead
    usable_mb = max(available_mb - 2048, 0)

    compatible = []
    for model in models:
        required_mb = model.estimated_ram_mb or model.estimated_vram_mb
        if required_mb == 0 or required_mb <= usable_mb:
            compatible.append(model)

    return compatible


def filter_by_backend(
    models: list[ModelSpec],
    backend_name: str,
) -> list[ModelSpec]:
    """Filter models to those that have an identifier for the given backend."""
    return [m for m in models if backend_name in m.backend_ids]


def get_model_for_backend(
    canonical_name: str,
    backend_name: str,
) -> tuple[ModelSpec, str] | None:
    """Look up a model by canonical name and return (spec, backend_id).

    Returns None if the model isn't in the catalog or doesn't support
    the given backend.
    """
    catalog = load_catalog()
    for model in catalog:
        if model.canonical_name == canonical_name:
            backend_id = model.backend_ids.get(backend_name)
            if backend_id:
                return model, backend_id
            return None
    return None
