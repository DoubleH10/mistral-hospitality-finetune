"""Tests for data processing functions in src/train.py."""

from src.train import clean_context, process_sgd_hotels, process_bitext_hospitality


# ============================================================
# clean_context
# ============================================================
class TestCleanContext:
    def test_replaces_sep_with_newlines(self):
        text = "User: Hello<SEP>Assistant: Hi there"
        result = clean_context(text)
        assert "<SEP>" not in result
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result

    def test_empty_string_returns_empty(self):
        assert clean_context("") == ""

    def test_none_returns_empty(self):
        assert clean_context(None) == ""

    def test_strips_blank_lines(self):
        text = "Line 1\n\n\nLine 2\n\n"
        result = clean_context(text)
        assert "\n\n" not in result

    def test_truncates_long_context(self):
        # Create a context longer than max_chars
        text = "Line\n" * 500  # ~2500 chars
        result = clean_context(text, max_chars=100)
        assert len(result) <= 100

    def test_truncation_preserves_line_boundary(self):
        # After truncation, should start at a clean line boundary
        lines = [f"Turn {i}: Some dialog text here" for i in range(100)]
        text = "\n".join(lines)
        result = clean_context(text, max_chars=200)
        # Should not start mid-sentence (no partial "Turn" prefix)
        assert result.startswith("Turn")

    def test_short_context_unchanged(self):
        text = "Short context"
        result = clean_context(text, max_chars=1200)
        assert result == text


# ============================================================
# process_sgd_hotels
# ============================================================
class TestProcessSgdHotels:
    def test_valid_assistant_turn(self, sgd_assistant_example):
        result = process_sgd_hotels(sgd_assistant_example)
        assert result["text"] is not None
        assert "[INST]" in result["text"]
        assert "[/INST]" in result["text"]
        assert result["text"].endswith("</s>")
        assert "hotel booking conversation" in result["text"]
        assert sgd_assistant_example["response"] in result["text"]

    def test_filters_user_turn(self, sgd_user_example):
        result = process_sgd_hotels(sgd_user_example)
        assert result["text"] is None

    def test_filters_empty_response(self, sgd_empty_response):
        result = process_sgd_hotels(sgd_empty_response)
        assert result["text"] is None

    def test_filters_missing_context(self):
        example = {"speaker": 1, "context": "", "response": "Hello"}
        result = process_sgd_hotels(example)
        assert result["text"] is None

    def test_context_cleaning_applied(self, sgd_assistant_example):
        result = process_sgd_hotels(sgd_assistant_example)
        # <SEP> should be cleaned out by clean_context
        assert "<SEP>" not in result["text"]

    def test_mistral_template_format(self, sgd_assistant_example):
        result = process_sgd_hotels(sgd_assistant_example)
        text = result["text"]
        # Should follow [INST] ... [/INST]response</s> format
        assert text.startswith("[INST]")
        inst_end = text.index("[/INST]")
        assert inst_end > 0
        after_inst = text[inst_end + len("[/INST]"):]
        assert after_inst.endswith("</s>")


# ============================================================
# process_bitext_hospitality
# ============================================================
class TestProcessBitextHospitality:
    def test_valid_example_with_metadata(self, bitext_example):
        result = process_bitext_hospitality(bitext_example)
        assert result["text"] is not None
        assert "[Category: booking | Intent: cancel_booking]" in result["text"]
        assert "[INST]" in result["text"]
        assert "[/INST]" in result["text"]
        assert result["text"].endswith("</s>")

    def test_example_without_metadata(self, bitext_no_metadata):
        result = process_bitext_hospitality(bitext_no_metadata)
        assert result["text"] is not None
        # Should not have category/intent prefix
        assert "[Category:" not in result["text"]
        assert "What time is breakfast?" in result["text"]

    def test_filters_empty_instruction(self, bitext_empty_instruction):
        result = process_bitext_hospitality(bitext_empty_instruction)
        assert result["text"] is None

    def test_filters_empty_response(self):
        example = {"instruction": "Hello", "response": "", "intent": "", "category": ""}
        result = process_bitext_hospitality(example)
        assert result["text"] is None

    def test_response_after_inst_marker(self, bitext_example):
        result = process_bitext_hospitality(bitext_example)
        text = result["text"]
        response_part = text.split("[/INST]")[1].replace("</s>", "")
        assert response_part.strip() == bitext_example["response"]
