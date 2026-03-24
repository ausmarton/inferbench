"""Tests for benchmark prompt definitions."""

from __future__ import annotations

from inferbench.benchmarks.prompts import ALL_PROMPTS, DEFAULT_PROMPT_NAMES, PromptSpec


class TestPromptSpec:
    def test_all_prompts_are_promptspec(self):
        for name, prompt in ALL_PROMPTS.items():
            assert isinstance(prompt, PromptSpec), f"{name} is not a PromptSpec"

    def test_all_prompts_have_required_fields(self):
        for name, prompt in ALL_PROMPTS.items():
            assert prompt.name == name
            assert prompt.description, f"{name} missing description"
            assert prompt.user_prompt, f"{name} missing user_prompt"
            assert prompt.max_output_tokens > 0, f"{name} has invalid max_output_tokens"

    def test_default_prompts_exist(self):
        for name in DEFAULT_PROMPT_NAMES:
            assert name in ALL_PROMPTS, f"Default prompt '{name}' not in ALL_PROMPTS"

    def test_prompt_names(self):
        expected = {"short_chat", "long_summarize", "code_generate", "reasoning"}
        assert set(ALL_PROMPTS.keys()) == expected

    def test_long_summarize_has_substantial_prompt(self):
        prompt = ALL_PROMPTS["long_summarize"]
        # The long article should be at least 200 words
        word_count = len(prompt.user_prompt.split())
        assert word_count > 200

    def test_prompts_are_frozen(self):
        prompt = ALL_PROMPTS["short_chat"]
        try:
            prompt.name = "modified"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass
