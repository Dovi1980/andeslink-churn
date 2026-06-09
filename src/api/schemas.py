from pydantic import BaseModel, Field

class ChurnRequest(BaseModel):
    tenure_months: int = Field(..., description="Meses como cliente activo", ge=1, le=120)
    monthly_charge: float = Field(..., description="Cargo mensual en pesos/USD", ge=10.0, le=200.0)
    total_charges: float = Field(..., description="Acumulado histórico en pesos/USD", ge=10.0, le=15000.0)
    support_tickets: int = Field(..., description="Tickets abiertos en el periodo", ge=0, le=20)
    late_payments: int = Field(..., description="Pagos con demora", ge=0, le=10)
    avg_monthly_usage_gb: float = Field(..., description="Consumo promedio mensual de internet en GB", ge=0.0, le=1000.0)
    contract_type: str = Field(..., description="Tipo de contrato vigente (mensual, anual, bianual)")
    payment_method: str = Field(..., description="Medio de pago (credito, debito, efectivo, transferencia)")
    internet_service: str = Field(..., description="Tipo de servicio de internet (cable, fibra, movil, ninguno)")
    has_streaming: int = Field(..., description="1 si tiene servicio de streaming, 0 si no", ge=0, le=1)
    has_security_pack: int = Field(..., description="1 si tiene pack de seguridad, 0 si no", ge=0, le=1)
    num_products: int = Field(..., description="Cantidad de productos contratados", ge=1, le=10)
    region: str = Field(..., description="Región geográfica del cliente (centro, norte, sur, oeste)")
    customer_age: int = Field(..., description="Edad del cliente en años", ge=18, le=100)
    is_promo: int = Field(..., description="1 si está en tarifa promocional, 0 si no", ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 3,
                "monthly_charge": 75.50,
                "total_charges": 226.50,
                "support_tickets": 5,
                "late_payments": 3,
                "avg_monthly_usage_gb": 45.0,
                "contract_type": "mensual",
                "payment_method": "transferencia",
                "internet_service": "cable",
                "has_streaming": 0,
                "has_security_pack": 0,
                "num_products": 1,
                "region": "norte",
                "customer_age": 28,
                "is_promo": 0
            }
        }
    }

class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., description="Probabilidad de abandono del cliente (0 a 1)")
    churn_label: int = Field(..., description="Predicción de abandono (1 = abandona, 0 = permanece)")
