"""Shared fixtures for the test suite. All tests run on CPU with no GPU required."""

import pytest


@pytest.fixture
def sgd_assistant_example():
    """A valid SGD Hotels example (assistant turn, speaker=1)."""
    return {
        "speaker": 1,
        "context": "User: I need a hotel in SF<SEP>Assistant: Sure, when?<SEP>User: Next Friday",
        "response": "I found several options in San Francisco for next Friday.",
    }


@pytest.fixture
def sgd_user_example():
    """An SGD Hotels example that should be filtered out (user turn, speaker=0)."""
    return {
        "speaker": 0,
        "context": "Assistant: How can I help you?",
        "response": "I need a room for two nights.",
    }


@pytest.fixture
def sgd_empty_response():
    """An SGD Hotels example with empty response (should be filtered)."""
    return {
        "speaker": 1,
        "context": "User: Hello",
        "response": "",
    }


@pytest.fixture
def bitext_example():
    """A valid Bitext hospitality example with intent and category."""
    return {
        "instruction": "I want to cancel my reservation",
        "response": "I'd be happy to help you cancel your reservation. Could you provide your booking reference?",
        "intent": "cancel_booking",
        "category": "booking",
    }


@pytest.fixture
def bitext_no_metadata():
    """A Bitext example without intent/category metadata."""
    return {
        "instruction": "What time is breakfast?",
        "response": "Breakfast is served from 7:00 AM to 10:30 AM in the main dining area.",
        "intent": "",
        "category": "",
    }


@pytest.fixture
def bitext_empty_instruction():
    """A Bitext example with empty instruction (should be filtered)."""
    return {
        "instruction": "",
        "response": "Some response",
        "intent": "test",
        "category": "test",
    }
