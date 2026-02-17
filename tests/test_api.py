"""Tests for FastAPI endpoints in api/main.py. Uses mocked model — no GPU required."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_state():
    """Patch model loading so the API starts without a real model."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

    with patch("api.main.load_model", return_value=(mock_model, mock_tokenizer)):
        from api.main import app, _state

        _state["model"] = mock_model
        _state["tokenizer"] = mock_tokenizer
        _state["ready"] = True
        yield app, _state
        _state.clear()


@pytest.fixture
def client(mock_state):
    """Create a test client with mocked model."""
    app, _ = mock_state
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "device" in data
        assert "gpu_memory" in data

    def test_ready_status(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ready"


class TestGenerateEndpoint:
    def test_returns_200(self, client):
        with patch("api.main.generate", return_value="Checkout is at 11 AM."):
            response = client.post(
                "/generate",
                json={"prompt": "What time is checkout?"},
            )
            assert response.status_code == 200

    def test_response_fields(self, client):
        with patch("api.main.generate", return_value="Check-in is at 3 PM."):
            response = client.post(
                "/generate",
                json={"prompt": "What time is check-in?"},
            )
            data = response.json()
            assert "response" in data
            assert "prompt" in data
            assert "tokens_generated" in data

    def test_echoes_prompt(self, client):
        prompt = "Do you have parking?"
        with patch("api.main.generate", return_value="Yes, we do."):
            response = client.post("/generate", json={"prompt": prompt})
            assert response.json()["prompt"] == prompt

    def test_rejects_empty_prompt(self, client):
        response = client.post("/generate", json={"prompt": ""})
        assert response.status_code == 422

    def test_model_not_ready(self, mock_state):
        app, state = mock_state
        state["ready"] = False
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/generate", json={"prompt": "hello"})
        assert response.status_code == 503
