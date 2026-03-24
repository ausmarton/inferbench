"""Tests for model catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from inferbench.catalog.models import ModelFamily, ModelSpec, QuantizationType
from inferbench.catalog.registry import (
    filter_by_backend,
    filter_by_hardware,
    load_catalog,
)

if TYPE_CHECKING:
    from inferbench.hardware.models import HardwareProfile


class TestModelSpec:
    def test_create(self):
        spec = ModelSpec(
            canonical_name="test-7b-q4km",
            family=ModelFamily.QWEN,
            parameter_count_b=7.6,
            context_length=131072,
            quantization=QuantizationType.Q4_K_M,
            estimated_ram_mb=5500,
            backend_ids={"ollama": "test:7b"},
            tags=["chat"],
        )
        assert spec.canonical_name == "test-7b-q4km"
        assert spec.family == ModelFamily.QWEN
        assert "ollama" in spec.backend_ids

    def test_serialization_roundtrip(self):
        spec = ModelSpec(
            canonical_name="test-7b",
            family=ModelFamily.LLAMA,
            parameter_count_b=7.0,
            context_length=8192,
            backend_ids={"ollama": "test:7b", "transformers": "test/Test-7B"},
        )
        data = spec.model_dump()
        restored = ModelSpec.model_validate(data)
        assert restored.canonical_name == spec.canonical_name
        assert restored.backend_ids == spec.backend_ids


class TestLoadCatalog:
    def test_loads_models(self):
        catalog = load_catalog()
        assert len(catalog) > 0
        # Every entry should be a valid ModelSpec
        for model in catalog:
            assert isinstance(model, ModelSpec)
            assert model.canonical_name
            assert model.parameter_count_b > 0

    def test_has_ollama_ids(self):
        catalog = load_catalog()
        ollama_models = [m for m in catalog if "ollama" in m.backend_ids]
        assert len(ollama_models) > 0

    def test_known_model_present(self):
        catalog = load_catalog()
        names = {m.canonical_name for m in catalog}
        assert "qwen2.5-7b-q4km" in names
        assert "llama3.1-8b-q4km" in names


class TestFilterByHardware:
    def test_filters_large_models(self, no_gpu_profile: HardwareProfile):
        models = [
            ModelSpec(
                canonical_name="small",
                family=ModelFamily.QWEN,
                parameter_count_b=0.5,
                context_length=8192,
                estimated_ram_mb=500,
            ),
            ModelSpec(
                canonical_name="huge",
                family=ModelFamily.LLAMA,
                parameter_count_b=405.0,
                context_length=8192,
                estimated_ram_mb=500_000,  # 500 GB — won't fit
            ),
        ]
        compatible = filter_by_hardware(models, no_gpu_profile)
        assert len(compatible) == 1
        assert compatible[0].canonical_name == "small"

    def test_all_fit_on_large_system(self, amd_apu_profile: HardwareProfile):
        models = [
            ModelSpec(
                canonical_name="model-a",
                family=ModelFamily.QWEN,
                parameter_count_b=7.6,
                context_length=8192,
                estimated_ram_mb=5500,
            ),
            ModelSpec(
                canonical_name="model-b",
                family=ModelFamily.LLAMA,
                parameter_count_b=32.0,
                context_length=8192,
                estimated_ram_mb=20000,
            ),
        ]
        compatible = filter_by_hardware(models, amd_apu_profile)
        assert len(compatible) == 2


class TestFilterByBackend:
    def test_filters_by_backend(self):
        models = [
            ModelSpec(
                canonical_name="ollama-only",
                family=ModelFamily.QWEN,
                parameter_count_b=7.0,
                context_length=8192,
                backend_ids={"ollama": "model:7b"},
            ),
            ModelSpec(
                canonical_name="transformers-only",
                family=ModelFamily.LLAMA,
                parameter_count_b=7.0,
                context_length=8192,
                backend_ids={"transformers": "org/model"},
            ),
            ModelSpec(
                canonical_name="both",
                family=ModelFamily.MISTRAL,
                parameter_count_b=7.0,
                context_length=8192,
                backend_ids={"ollama": "mistral:7b", "transformers": "mistralai/Mistral-7B"},
            ),
        ]
        ollama_models = filter_by_backend(models, "ollama")
        assert len(ollama_models) == 2
        assert {m.canonical_name for m in ollama_models} == {"ollama-only", "both"}

        transformers_models = filter_by_backend(models, "transformers")
        assert len(transformers_models) == 2
