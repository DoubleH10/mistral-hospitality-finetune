"""Tests for formatting functions in src/inference.py."""

from src.inference import format_dialog_prompt, format_faq_prompt


class TestFormatDialogPrompt:
    def test_basic_formatting(self):
        context = "User: Hello\nAssistant: Hi!"
        result = format_dialog_prompt(context)
        assert result.startswith("[INST]")
        assert result.endswith("[/INST]")
        assert context in result

    def test_includes_instruction(self):
        result = format_dialog_prompt("Some context")
        assert "hotel booking conversation" in result
        assert "helpful assistant" in result

    def test_includes_context_label(self):
        result = format_dialog_prompt("User: Test")
        assert "Context:" in result


class TestFormatFaqPrompt:
    def test_basic_question(self):
        result = format_faq_prompt("What time is checkout?")
        assert result == "[INST] What time is checkout? [/INST]"

    def test_with_category_and_intent(self):
        result = format_faq_prompt(
            "Cancel my booking",
            category="booking",
            intent="cancel_booking",
        )
        assert "[Category: booking | Intent: cancel_booking]" in result
        assert "Cancel my booking" in result
        assert result.startswith("[INST]")
        assert result.endswith("[/INST]")

    def test_without_metadata(self):
        result = format_faq_prompt("Is parking free?")
        assert "[Category:" not in result
        assert result == "[INST] Is parking free? [/INST]"

    def test_partial_metadata_ignored(self):
        # Only category, no intent — should not add metadata prefix
        result = format_faq_prompt("Test", category="booking", intent=None)
        assert "[Category:" not in result

    def test_empty_metadata_ignored(self):
        result = format_faq_prompt("Test", category="", intent="")
        assert "[Category:" not in result
