"""
create_test_data.py
-------------------
Genera dos artefactos para el proyecto AndesLink Churn:

1. tests/fixtures/sample_payloads.py
   Payloads Python listos para usar en pytest (sin CSV, sin dependencias externas).

2. data/test/churn_test_sample.csv
   Mini-dataset de 20 filas (10 churn=1, 10 churn=0) que replica la estructura
   exacta de churn_sintetico.csv. Se puede commitear en Git para CI/CD.

Uso:
    python scripts/create_test_data.py

No requiere el dataset original. Todos los valores son representativos de los
rangos y categorias reales del dataset de AndesLink.
"""

import csv
import os
import random
import textwrap

random.seed(42)

# ── Categorias y rangos del dataset AndesLink ──────────────────────────────
CONTRACT_TYPES    = ["mensual", "anual", "bianual"]
PAYMENT_METHODS   = ["credito", "debito", "efectivo", "transferencia"]
INTERNET_SERVICES = ["cable", "fibra", "movil", "ninguno"]
REGIONS           = ["centro", "norte", "sur", "oeste"]

# Perfiles de cliente con ALTA probabilidad de churn (churn=1)
# Patron tipico: poco tenure, contrato mensual, muchos tickets y pagos tardios
CHURN_PROFILES = [
    {"tenure_months": 3,  "monthly_charge": 75.50, "total_charges": 226.50,
     "support_tickets": 5, "late_payments": 3, "avg_monthly_usage_gb": 45.0,
     "contract_type": "mensual", "payment_method": "transferencia",
     "internet_service": "cable", "has_streaming": 0, "has_security_pack": 0,
     "num_products": 1, "region": "norte", "customer_age": 28, "is_promo": 0},

    {"tenure_months": 5,  "monthly_charge": 89.90, "total_charges": 449.50,
     "support_tickets": 4, "late_payments": 2, "avg_monthly_usage_gb": 30.0,
     "contract_type": "mensual", "payment_method": "efectivo",
     "internet_service": "cable", "has_streaming": 0, "has_security_pack": 0,
     "num_products": 1, "region": "sur",   "customer_age": 22, "is_promo": 1},

    {"tenure_months": 2,  "monthly_charge": 99.00, "total_charges": 198.00,
     "support_tickets": 6, "late_payments": 2, "avg_monthly_usage_gb": 20.0,
     "contract_type": "mensual", "payment_method": "transferencia",
     "internet_service": "movil", "has_streaming": 1, "has_security_pack": 0,
     "num_products": 2, "region": "oeste", "customer_age": 35, "is_promo": 0},

    {"tenure_months": 7,  "monthly_charge": 58.23, "total_charges": 326.50,
     "support_tickets": 2, "late_payments": 1, "avg_monthly_usage_gb": 81.83,
     "contract_type": "mensual", "payment_method": "transferencia",
     "internet_service": "cable", "has_streaming": 0, "has_security_pack": 1,
     "num_products": 3, "region": "centro","customer_age": 53, "is_promo": 1},

    {"tenure_months": 4,  "monthly_charge": 115.00,"total_charges": 460.00,
     "support_tickets": 7, "late_payments": 4, "avg_monthly_usage_gb": 18.5,
     "contract_type": "mensual", "payment_method": "credito",
     "internet_service": "cable", "has_streaming": 0, "has_security_pack": 0,
     "num_products": 1, "region": "norte", "customer_age": 19, "is_promo": 1},

    {"tenure_months": 6,  "monthly_charge": 67.00, "total_charges": 402.00,
     "support_tickets": 3, "late_payments": 2, "avg_monthly_usage_gb": 55.0,
     "contract_type": "mensual", "payment_method": "efectivo",
     "internet_service": "movil", "has_streaming": 1, "has_security_pack": 0,
     "num_products": 2, "region": "sur",   "customer_age": 31, "is_promo": 0},

    {"tenure_months": 1,  "monthly_charge": 120.00,"total_charges": 120.00,
     "support_tickets": 8, "late_payments": 1, "avg_monthly_usage_gb": 10.0,
     "contract_type": "mensual", "payment_method": "transferencia",
     "internet_service": "cable", "has_streaming": 0, "has_security_pack": 0,
     "num_products": 1, "region": "centro","customer_age": 44, "is_promo": 0},

    {"tenure_months": 9,  "monthly_charge": 82.00, "total_charges": 738.00,
     "support_tickets": 4, "late_payments": 3, "avg_monthly_usage_gb": 70.0,
     "contract_type": "mensual", "payment_method": "debito",
     "internet_service": "fibra", "has_streaming": 0, "has_security_pack": 0,
     "num_products": 2, "region": "norte", "customer_age": 61, "is_promo": 1},

    {"tenure_months": 3,  "monthly_charge": 55.00, "total_charges": 165.00,
     "support_tickets": 5, "late_payments": 2, "avg_monthly_usage_gb": 40.0,
     "contract_type": "mensual", "payment_method": "credito",
     "internet_service": "movil", "has_streaming": 1, "has_security_pack": 0,
     "num_products": 1, "region": "oeste", "customer_age": 25, "is_promo": 0},

    {"tenure_months": 8,  "monthly_charge": 95.00, "total_charges": 760.00,
     "support_tickets": 3, "late_payments": 2, "avg_monthly_usage_gb": 25.0,
     "contract_type": "mensual", "payment_method": "efectivo",
     "internet_service": "cable", "has_streaming": 0, "has_security_pack": 0,
     "num_products": 1, "region": "sur",   "customer_age": 38, "is_promo": 1},
]

# Perfiles de cliente con BAJA probabilidad de churn (churn=0)
# Patron tipico: alto tenure, contrato anual/bianual, pocos tickets
NO_CHURN_PROFILES = [
    {"tenure_months": 56, "monthly_charge": 56.75, "total_charges": 3154.21,
     "support_tickets": 0, "late_payments": 2, "avg_monthly_usage_gb": 96.52,
     "contract_type": "anual",   "payment_method": "debito",
     "internet_service": "fibra", "has_streaming": 0, "has_security_pack": 0,
     "num_products": 4, "region": "centro","customer_age": 53, "is_promo": 0},

    {"tenure_months": 48, "monthly_charge": 45.00, "total_charges": 2160.00,
     "support_tickets": 1, "late_payments": 0, "avg_monthly_usage_gb": 150.0,
     "contract_type": "bianual", "payment_method": "credito",
     "internet_service": "fibra", "has_streaming": 1, "has_security_pack": 1,
     "num_products": 3, "region": "sur",   "customer_age": 41, "is_promo": 0},

    {"tenure_months": 60, "monthly_charge": 70.00, "total_charges": 4200.00,
     "support_tickets": 0, "late_payments": 0, "avg_monthly_usage_gb": 200.0,
     "contract_type": "bianual", "payment_method": "debito",
     "internet_service": "fibra", "has_streaming": 1, "has_security_pack": 1,
     "num_products": 4, "region": "norte", "customer_age": 55, "is_promo": 0},

    {"tenure_months": 36, "monthly_charge": 55.00, "total_charges": 1980.00,
     "support_tickets": 1, "late_payments": 1, "avg_monthly_usage_gb": 130.0,
     "contract_type": "anual",   "payment_method": "credito",
     "internet_service": "cable", "has_streaming": 0, "has_security_pack": 1,
     "num_products": 3, "region": "oeste", "customer_age": 48, "is_promo": 1},

    {"tenure_months": 72, "monthly_charge": 80.00, "total_charges": 5760.00,
     "support_tickets": 0, "late_payments": 0, "avg_monthly_usage_gb": 180.0,
     "contract_type": "bianual", "payment_method": "debito",
     "internet_service": "fibra", "has_streaming": 1, "has_security_pack": 1,
     "num_products": 4, "region": "centro","customer_age": 63, "is_promo": 0},

    {"tenure_months": 24, "monthly_charge": 60.00, "total_charges": 1440.00,
     "support_tickets": 2, "late_payments": 0, "avg_monthly_usage_gb": 110.0,
     "contract_type": "anual",   "payment_method": "transferencia",
     "internet_service": "cable", "has_streaming": 1, "has_security_pack": 0,
     "num_products": 2, "region": "sur",   "customer_age": 35, "is_promo": 1},

    {"tenure_months": 42, "monthly_charge": 50.00, "total_charges": 2100.00,
     "support_tickets": 0, "late_payments": 1, "avg_monthly_usage_gb": 95.0,
     "contract_type": "anual",   "payment_method": "debito",
     "internet_service": "movil", "has_streaming": 0, "has_security_pack": 1,
     "num_products": 3, "region": "norte", "customer_age": 52, "is_promo": 0},

    {"tenure_months": 30, "monthly_charge": 75.00, "total_charges": 2250.00,
     "support_tickets": 1, "late_payments": 0, "avg_monthly_usage_gb": 160.0,
     "contract_type": "bianual", "payment_method": "credito",
     "internet_service": "fibra", "has_streaming": 1, "has_security_pack": 1,
     "num_products": 4, "region": "oeste", "customer_age": 29, "is_promo": 0},

    {"tenure_months": 54, "monthly_charge": 65.00, "total_charges": 3510.00,
     "support_tickets": 0, "late_payments": 0, "avg_monthly_usage_gb": 220.0,
     "contract_type": "anual",   "payment_method": "debito",
     "internet_service": "fibra", "has_streaming": 1, "has_security_pack": 0,
     "num_products": 3, "region": "centro","customer_age": 71, "is_promo": 0},

    {"tenure_months": 18, "monthly_charge": 40.00, "total_charges": 720.00,
     "support_tickets": 1, "late_payments": 1, "avg_monthly_usage_gb": 85.0,
     "contract_type": "anual",   "payment_method": "credito",
     "internet_service": "cable", "has_streaming": 0, "has_security_pack": 0,
     "num_products": 2, "region": "sur",   "customer_age": 22, "is_promo": 1},
]

FIELDNAMES = [
    "tenure_months", "monthly_charge", "total_charges", "support_tickets",
    "late_payments", "avg_monthly_usage_gb", "contract_type", "payment_method",
    "internet_service", "has_streaming", "has_security_pack", "num_products",
    "region", "customer_age", "is_promo", "churn"
]


# ── 1. Generar CSV de prueba ───────────────────────────────────────────────
def write_test_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for p in CHURN_PROFILES:
        rows.append({**p, "churn": 1})
    for p in NO_CHURN_PROFILES:
        rows.append({**p, "churn": 0})

    random.shuffle(rows)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] CSV generado: {path}  ({len(rows)} filas)")


# ── 2. Generar fixtures Python para pytest ────────────────────────────────
def write_pytest_fixtures(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Usamos el primer perfil de cada clase como ejemplo canónico
    ex_churn    = {**CHURN_PROFILES[0]}
    ex_no_churn = {**NO_CHURN_PROFILES[0]}

    # Payload para la API (sin columna 'churn')
    api_churn    = {k: v for k, v in ex_churn.items()}
    api_no_churn = {k: v for k, v in ex_no_churn.items()}

    content = textwrap.dedent(f"""\
    \"\"\"
    fixtures/sample_payloads.py
    ---------------------------
    Payloads listos para usar en los tests de pytest de la API AndesLink.
    Generado automaticamente por scripts/create_test_data.py

    Uso:
        from tests.fixtures.sample_payloads import PAYLOAD_CHURN, PAYLOAD_NO_CHURN, INVALID_PAYLOAD

    Descripcion de los perfiles:
    - PAYLOAD_CHURN:    cliente de alto riesgo (tenure corto, contrato mensual,
                        muchos tickets). El modelo deberia predecir churn_label=1.
    - PAYLOAD_NO_CHURN: cliente fidelizado (tenure alto, contrato bianual, sin
                        incidencias). El modelo deberia predecir churn_label=0.
    - INVALID_PAYLOAD:  payload incompleto para testear validacion (HTTP 422).
    \"\"\"

    # ── Cliente de alto riesgo de churn ───────────────────────────────────────
    PAYLOAD_CHURN = {{
        "tenure_months":        {api_churn['tenure_months']},
        "monthly_charge":       {api_churn['monthly_charge']},
        "total_charges":        {api_churn['total_charges']},
        "support_tickets":      {api_churn['support_tickets']},
        "late_payments":        {api_churn['late_payments']},
        "avg_monthly_usage_gb": {api_churn['avg_monthly_usage_gb']},
        "contract_type":        "{api_churn['contract_type']}",
        "payment_method":       "{api_churn['payment_method']}",
        "internet_service":     "{api_churn['internet_service']}",
        "has_streaming":        {api_churn['has_streaming']},
        "has_security_pack":    {api_churn['has_security_pack']},
        "num_products":         {api_churn['num_products']},
        "region":               "{api_churn['region']}",
        "customer_age":         {api_churn['customer_age']},
        "is_promo":             {api_churn['is_promo']},
    }}

    # ── Cliente fidelizado, bajo riesgo de churn ─────────────────────────────
    PAYLOAD_NO_CHURN = {{
        "tenure_months":        {api_no_churn['tenure_months']},
        "monthly_charge":       {api_no_churn['monthly_charge']},
        "total_charges":        {api_no_churn['total_charges']},
        "support_tickets":      {api_no_churn['support_tickets']},
        "late_payments":        {api_no_churn['late_payments']},
        "avg_monthly_usage_gb": {api_no_churn['avg_monthly_usage_gb']},
        "contract_type":        "{api_no_churn['contract_type']}",
        "payment_method":       "{api_no_churn['payment_method']}",
        "internet_service":     "{api_no_churn['internet_service']}",
        "has_streaming":        {api_no_churn['has_streaming']},
        "has_security_pack":    {api_no_churn['has_security_pack']},
        "num_products":         {api_no_churn['num_products']},
        "region":               "{api_no_churn['region']}",
        "customer_age":         {api_no_churn['customer_age']},
        "is_promo":             {api_no_churn['is_promo']},
    }}

    # ── Payload invalido (falta tenure_months) ────────────────────────────────
    INVALID_PAYLOAD = {{
        "monthly_charge":   75.50,
        "total_charges":    226.50,
        "contract_type":    "mensual",
    }}

    # ── Lista de todos los payloads de prueba (util para tests parametrizados) ─
    ALL_VALID_PAYLOADS = [
    """)

    for i, p in enumerate(CHURN_PROFILES + NO_CHURN_PROFILES):
        label = "churn=1" if i < len(CHURN_PROFILES) else "churn=0"
        content += f"    # {label}\n    {{"
        items = [f'"{k}": {repr(v)}' for k, v in p.items()]
        content += ", ".join(items)
        content += "},\n"

    content += "]\n"

    with open(path, "w") as f:
        f.write(content)

    print(f"[OK] Fixtures generados: {path}")


# ── 3. Generar test de ejemplo que usa las fixtures ───────────────────────
def write_example_test(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = textwrap.dedent("""\
    \"\"\"
    test_api_with_fixtures.py
    -------------------------
    Tests de la API usando los payloads de sample_payloads.py.
    Estos tests funcionan en local (con el modelo real) y en CI
    (con el modelo mockeado).
    \"\"\"

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
        \"\"\"Crea un pipeline mock que devuelve la probabilidad indicada.\"\"\"
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[1 - prob_class1, prob_class1]])
        return mock


    # ── Tests basicos ─────────────────────────────────────────────────────────
    def test_health():
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


    def test_predict_invalid_payload():
        \"\"\"Payload incompleto debe retornar 422 Unprocessable Entity.\"\"\"
        r = client.post("/predict", json=INVALID_PAYLOAD)
        assert r.status_code == 422


    # ── Tests con mock (funcionan en CI sin model.pkl) ────────────────────────
    def test_predict_churn_mocked():
        \"\"\"Cliente de alto riesgo: modelo mockeado devuelve prob=0.85.\"\"\"
        with patch("src.api.model_loader.get_model", return_value=make_mock_model(0.85)):
            r = client.post("/predict", json=PAYLOAD_CHURN)
        assert r.status_code == 200
        data = r.json()
        assert "churn_probability" in data
        assert "churn_label" in data
        assert data["churn_label"] == 1
        assert data["churn_probability"] == pytest.approx(0.85, abs=0.01)


    def test_predict_no_churn_mocked():
        \"\"\"Cliente fidelizado: modelo mockeado devuelve prob=0.12.\"\"\"
        with patch("src.api.model_loader.get_model", return_value=make_mock_model(0.12)):
            r = client.post("/predict", json=PAYLOAD_NO_CHURN)
        assert r.status_code == 200
        data = r.json()
        assert data["churn_label"] == 0
        assert data["churn_probability"] == pytest.approx(0.12, abs=0.01)


    def test_response_schema():
        \"\"\"Verifica que la respuesta tiene exactamente las claves esperadas.\"\"\"
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
    """)

    with open(path, "w") as f:
        f.write(content)

    print(f"[OK] Test de ejemplo generado: {path}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    write_test_csv(
        os.path.join(base, "data", "test", "churn_test_sample.csv")
    )
    write_pytest_fixtures(
        os.path.join(base, "tests", "fixtures", "sample_payloads.py")
    )
    write_example_test(
        os.path.join(base, "tests", "test_api_with_fixtures.py")
    )

    print("\nArchivos generados:")
    print("  data/test/churn_test_sample.csv       <- dataset ficticio para CI")
    print("  tests/fixtures/sample_payloads.py     <- payloads para pytest")
    print("  tests/test_api_with_fixtures.py       <- tests listos para usar")
