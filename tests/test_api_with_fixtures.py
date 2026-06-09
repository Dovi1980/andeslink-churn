"""
test_api_with_fixtures.py
-------------------------
Tests de la API usando los payloads de sample_payloads.py.
Estos tests funcionan en local (con el modelo real) y en CI
(con el modelo mockeado).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from tests.fixtures.sample_payloads import (
    PAYLOAD_CHURN,
    PAYLOAD_NO_CHURN,
    INVALID_PAYLOAD,
    ALL_VALID_PAYLOADS,
)

client = TestClient(app)

# ── Mock del modelo para CI (no necesita model.pkl) ───────────────────────
def make_mock_model(prob_class1: float):
    """Crea un pipeline mock que devuelve la probabilidad indicada."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - prob_class1, prob_class1]])
    return mock


# ── Tests basicos ─────────────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_invalid_payload():
    """Payload incompleto debe retornar 422 Unprocessable Entity."""
    r = client.post("/predict", json=INVALID_PAYLOAD)
    assert r.status_code == 422


# ── Tests con mock (funcionan en CI sin model.pkl) ────────────────────────
def test_predict_churn_mocked():
    """Cliente de alto riesgo: modelo mockeado devuelve prob=0.85."""
    with patch("src.api.model_loader.get_model", return_value=make_mock_model(0.85)):
        r = client.post("/predict", json=PAYLOAD_CHURN)
    assert r.status_code == 200
    data = r.json()
    assert "churn_probability" in data
    assert "churn_label" in data
    assert data["churn_label"] == 1
    assert data["churn_probability"] == pytest.approx(0.85, abs=0.01)


def test_predict_no_churn_mocked():
    """Cliente fidelizado: modelo mockeado devuelve prob=0.12."""
    with patch("src.api.model_loader.get_model", return_value=make_mock_model(0.12)):
        r = client.post("/predict", json=PAYLOAD_NO_CHURN)
    assert r.status_code == 200
    data = r.json()
    assert data["churn_label"] == 0
    assert data["churn_probability"] == pytest.approx(0.12, abs=0.01)


def test_response_schema():
    """Verifica que la respuesta tiene exactamente las claves esperadas."""
    with patch("src.api.model_loader.get_model", return_value=make_mock_model(0.5)):
        r = client.post("/predict", json=PAYLOAD_CHURN)
    keys = set(r.json().keys())
    assert keys == {"churn_probability", "churn_label"}


# ── Test parametrizado: todos los payloads deben retornar 200 ─────────────
@pytest.mark.parametrize("payload", ALL_VALID_PAYLOADS)
def test_all_payloads_return_200(payload):
    with patch("src.api.model_loader.get_model", return_value=make_mock_model(0.5)):
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
