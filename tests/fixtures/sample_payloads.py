"""
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
"""

# ── Cliente de alto riesgo de churn ───────────────────────────────────────
PAYLOAD_CHURN = {
    "tenure_months":        3,
    "monthly_charge":       75.5,
    "total_charges":        226.5,
    "support_tickets":      5,
    "late_payments":        3,
    "avg_monthly_usage_gb": 45.0,
    "contract_type":        "mensual",
    "payment_method":       "transferencia",
    "internet_service":     "cable",
    "has_streaming":        0,
    "has_security_pack":    0,
    "num_products":         1,
    "region":               "norte",
    "customer_age":         28,
    "is_promo":             0,
}

# ── Cliente fidelizado, bajo riesgo de churn ─────────────────────────────
PAYLOAD_NO_CHURN = {
    "tenure_months":        56,
    "monthly_charge":       56.75,
    "total_charges":        3154.21,
    "support_tickets":      0,
    "late_payments":        2,
    "avg_monthly_usage_gb": 96.52,
    "contract_type":        "anual",
    "payment_method":       "debito",
    "internet_service":     "fibra",
    "has_streaming":        0,
    "has_security_pack":    0,
    "num_products":         4,
    "region":               "centro",
    "customer_age":         53,
    "is_promo":             0,
}

# ── Payload invalido (falta tenure_months) ────────────────────────────────
INVALID_PAYLOAD = {
    "monthly_charge":   75.50,
    "total_charges":    226.50,
    "contract_type":    "mensual",
}

# ── Lista de todos los payloads de prueba (util para tests parametrizados) ─
ALL_VALID_PAYLOADS = [
    # churn=1
    {"tenure_months": 3, "monthly_charge": 75.5, "total_charges": 226.5, "support_tickets": 5, "late_payments": 3, "avg_monthly_usage_gb": 45.0, "contract_type": 'mensual', "payment_method": 'transferencia', "internet_service": 'cable', "has_streaming": 0, "has_security_pack": 0, "num_products": 1, "region": 'norte', "customer_age": 28, "is_promo": 0},
    # churn=1
    {"tenure_months": 5, "monthly_charge": 89.9, "total_charges": 449.5, "support_tickets": 4, "late_payments": 2, "avg_monthly_usage_gb": 30.0, "contract_type": 'mensual', "payment_method": 'efectivo', "internet_service": 'cable', "has_streaming": 0, "has_security_pack": 0, "num_products": 1, "region": 'sur', "customer_age": 22, "is_promo": 1},
    # churn=1
    {"tenure_months": 2, "monthly_charge": 99.0, "total_charges": 198.0, "support_tickets": 6, "late_payments": 2, "avg_monthly_usage_gb": 20.0, "contract_type": 'mensual', "payment_method": 'transferencia', "internet_service": 'movil', "has_streaming": 1, "has_security_pack": 0, "num_products": 2, "region": 'oeste', "customer_age": 35, "is_promo": 0},
    # churn=1
    {"tenure_months": 7, "monthly_charge": 58.23, "total_charges": 326.5, "support_tickets": 2, "late_payments": 1, "avg_monthly_usage_gb": 81.83, "contract_type": 'mensual', "payment_method": 'transferencia', "internet_service": 'cable', "has_streaming": 0, "has_security_pack": 1, "num_products": 3, "region": 'centro', "customer_age": 53, "is_promo": 1},
    # churn=1
    {"tenure_months": 4, "monthly_charge": 115.0, "total_charges": 460.0, "support_tickets": 7, "late_payments": 4, "avg_monthly_usage_gb": 18.5, "contract_type": 'mensual', "payment_method": 'credito', "internet_service": 'cable', "has_streaming": 0, "has_security_pack": 0, "num_products": 1, "region": 'norte', "customer_age": 19, "is_promo": 1},
    # churn=1
    {"tenure_months": 6, "monthly_charge": 67.0, "total_charges": 402.0, "support_tickets": 3, "late_payments": 2, "avg_monthly_usage_gb": 55.0, "contract_type": 'mensual', "payment_method": 'efectivo', "internet_service": 'movil', "has_streaming": 1, "has_security_pack": 0, "num_products": 2, "region": 'sur', "customer_age": 31, "is_promo": 0},
    # churn=1
    {"tenure_months": 1, "monthly_charge": 120.0, "total_charges": 120.0, "support_tickets": 8, "late_payments": 1, "avg_monthly_usage_gb": 10.0, "contract_type": 'mensual', "payment_method": 'transferencia', "internet_service": 'cable', "has_streaming": 0, "has_security_pack": 0, "num_products": 1, "region": 'centro', "customer_age": 44, "is_promo": 0},
    # churn=1
    {"tenure_months": 9, "monthly_charge": 82.0, "total_charges": 738.0, "support_tickets": 4, "late_payments": 3, "avg_monthly_usage_gb": 70.0, "contract_type": 'mensual', "payment_method": 'debito', "internet_service": 'fibra', "has_streaming": 0, "has_security_pack": 0, "num_products": 2, "region": 'norte', "customer_age": 61, "is_promo": 1},
    # churn=1
    {"tenure_months": 3, "monthly_charge": 55.0, "total_charges": 165.0, "support_tickets": 5, "late_payments": 2, "avg_monthly_usage_gb": 40.0, "contract_type": 'mensual', "payment_method": 'credito', "internet_service": 'movil', "has_streaming": 1, "has_security_pack": 0, "num_products": 1, "region": 'oeste', "customer_age": 25, "is_promo": 0},
    # churn=1
    {"tenure_months": 8, "monthly_charge": 95.0, "total_charges": 760.0, "support_tickets": 3, "late_payments": 2, "avg_monthly_usage_gb": 25.0, "contract_type": 'mensual', "payment_method": 'efectivo', "internet_service": 'cable', "has_streaming": 0, "has_security_pack": 0, "num_products": 1, "region": 'sur', "customer_age": 38, "is_promo": 1},
    # churn=0
    {"tenure_months": 56, "monthly_charge": 56.75, "total_charges": 3154.21, "support_tickets": 0, "late_payments": 2, "avg_monthly_usage_gb": 96.52, "contract_type": 'anual', "payment_method": 'debito', "internet_service": 'fibra', "has_streaming": 0, "has_security_pack": 0, "num_products": 4, "region": 'centro', "customer_age": 53, "is_promo": 0},
    # churn=0
    {"tenure_months": 48, "monthly_charge": 45.0, "total_charges": 2160.0, "support_tickets": 1, "late_payments": 0, "avg_monthly_usage_gb": 150.0, "contract_type": 'bianual', "payment_method": 'credito', "internet_service": 'fibra', "has_streaming": 1, "has_security_pack": 1, "num_products": 3, "region": 'sur', "customer_age": 41, "is_promo": 0},
    # churn=0
    {"tenure_months": 60, "monthly_charge": 70.0, "total_charges": 4200.0, "support_tickets": 0, "late_payments": 0, "avg_monthly_usage_gb": 200.0, "contract_type": 'bianual', "payment_method": 'debito', "internet_service": 'fibra', "has_streaming": 1, "has_security_pack": 1, "num_products": 4, "region": 'norte', "customer_age": 55, "is_promo": 0},
    # churn=0
    {"tenure_months": 36, "monthly_charge": 55.0, "total_charges": 1980.0, "support_tickets": 1, "late_payments": 1, "avg_monthly_usage_gb": 130.0, "contract_type": 'anual', "payment_method": 'credito', "internet_service": 'cable', "has_streaming": 0, "has_security_pack": 1, "num_products": 3, "region": 'oeste', "customer_age": 48, "is_promo": 1},
    # churn=0
    {"tenure_months": 72, "monthly_charge": 80.0, "total_charges": 5760.0, "support_tickets": 0, "late_payments": 0, "avg_monthly_usage_gb": 180.0, "contract_type": 'bianual', "payment_method": 'debito', "internet_service": 'fibra', "has_streaming": 1, "has_security_pack": 1, "num_products": 4, "region": 'centro', "customer_age": 63, "is_promo": 0},
    # churn=0
    {"tenure_months": 24, "monthly_charge": 60.0, "total_charges": 1440.0, "support_tickets": 2, "late_payments": 0, "avg_monthly_usage_gb": 110.0, "contract_type": 'anual', "payment_method": 'transferencia', "internet_service": 'cable', "has_streaming": 1, "has_security_pack": 0, "num_products": 2, "region": 'sur', "customer_age": 35, "is_promo": 1},
    # churn=0
    {"tenure_months": 42, "monthly_charge": 50.0, "total_charges": 2100.0, "support_tickets": 0, "late_payments": 1, "avg_monthly_usage_gb": 95.0, "contract_type": 'anual', "payment_method": 'debito', "internet_service": 'movil', "has_streaming": 0, "has_security_pack": 1, "num_products": 3, "region": 'norte', "customer_age": 52, "is_promo": 0},
    # churn=0
    {"tenure_months": 30, "monthly_charge": 75.0, "total_charges": 2250.0, "support_tickets": 1, "late_payments": 0, "avg_monthly_usage_gb": 160.0, "contract_type": 'bianual', "payment_method": 'credito', "internet_service": 'fibra', "has_streaming": 1, "has_security_pack": 1, "num_products": 4, "region": 'oeste', "customer_age": 29, "is_promo": 0},
    # churn=0
    {"tenure_months": 54, "monthly_charge": 65.0, "total_charges": 3510.0, "support_tickets": 0, "late_payments": 0, "avg_monthly_usage_gb": 220.0, "contract_type": 'anual', "payment_method": 'debito', "internet_service": 'fibra', "has_streaming": 1, "has_security_pack": 0, "num_products": 3, "region": 'centro', "customer_age": 71, "is_promo": 0},
    # churn=0
    {"tenure_months": 18, "monthly_charge": 40.0, "total_charges": 720.0, "support_tickets": 1, "late_payments": 1, "avg_monthly_usage_gb": 85.0, "contract_type": 'anual', "payment_method": 'credito', "internet_service": 'cable', "has_streaming": 0, "has_security_pack": 0, "num_products": 2, "region": 'sur', "customer_age": 22, "is_promo": 1},
]
