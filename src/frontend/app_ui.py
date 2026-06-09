import streamlit as st
import requests
import os
import pandas as pd

# Configuración de página
st.set_page_config(
    page_title="AndesLink Churn Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos CSS personalizados para lograr un diseño "Premium" y moderno
st.markdown("""
    <style>
    /* Importar tipografía moderna */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Configuración global de fuentes */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Degradado en el título principal */
    .title-gradient {
        background: linear-gradient(135deg, #2563EB 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Tarjetas de resultados personalizadas */
    .card-result-churn {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #FCA5A5;
        border-radius: 12px;
        padding: 24px;
        color: #991B1B;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.1);
    }
    
    .card-result-loyal {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #6EE7B7;
        border-radius: 12px;
        padding: 24px;
        color: #065F46;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.1);
    }

    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    /* Botón de predicción premium */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Ajustes generales de contenedores */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# URL de la API de predicción (configurable mediante variable de entorno)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Título y encabezado principal
st.markdown("<h1 class='title-gradient'>AndesLink Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Analizador de comportamiento y retención de clientes en tiempo real.</p>", unsafe_allow_html=True)

# Inicializar estado por defecto en session_state si está vacío
defaults = {
    "tenure_months": 12, "monthly_charge": 50.0, "total_charges": 600.0,
    "support_tickets": 0, "late_payments": 0, "avg_monthly_usage_gb": 80.0,
    "contract_type": "mensual", "payment_method": "debito", "internet_service": "cable",
    "has_streaming": "No", "has_security_pack": "No", "num_products": 2,
    "region": "centro", "customer_age": 35, "is_promo": "No"
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Barra lateral informativa y de perfiles rápidos
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/network-cable.png", width=120)
    st.markdown("### Configuración de Inferencia")
    st.info(f"Conectado a la API en:\n`{API_URL}`")
    
    st.markdown("---")
    st.markdown("### Perfiles rápidos de prueba")
    st.write("Usa estos perfiles precargados para verificar el funcionamiento rápido:")
    
    if st.button("Cargar Perfil A (Alto riesgo - Churn)"):
        st.session_state.tenure_months = 3
        st.session_state.monthly_charge = 75.50
        st.session_state.total_charges = 226.50
        st.session_state.support_tickets = 5
        st.session_state.late_payments = 3
        st.session_state.avg_monthly_usage_gb = 45.0
        st.session_state.contract_type = "mensual"
        st.session_state.payment_method = "transferencia"
        st.session_state.internet_service = "cable"
        st.session_state.has_streaming = "No"
        st.session_state.has_security_pack = "No"
        st.session_state.num_products = 1
        st.session_state.region = "norte"
        st.session_state.customer_age = 28
        st.session_state.is_promo = "No"
        st.success("Cargado perfil de alto riesgo.")

    if st.button("Cargar Perfil B (Bajo riesgo - Fiel)"):
        st.session_state.tenure_months = 56
        st.session_state.monthly_charge = 56.75
        st.session_state.total_charges = 3154.21
        st.session_state.support_tickets = 0
        st.session_state.late_payments = 2
        st.session_state.avg_monthly_usage_gb = 96.52
        st.session_state.contract_type = "anual"
        st.session_state.payment_method = "debito"
        st.session_state.internet_service = "fibra"
        st.session_state.has_streaming = "No"
        st.session_state.has_security_pack = "No"
        st.session_state.num_products = 4
        st.session_state.region = "centro"
        st.session_state.customer_age = 53
        st.session_state.is_promo = "No"
        st.success("Cargado perfil de bajo riesgo.")

# Formulario principal de ingreso estructurado en Tabs con bindings de key directos a session_state
tab1, tab2, tab3 = st.tabs([
    "👤 Información del Cliente",
    "💼 Detalles del Contrato",
    "💻 Uso del Servicio e Incidencias"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        customer_age = st.number_input(
            "Edad del cliente", 
            min_value=18, max_value=100, 
            key="customer_age"
        )
    with col2:
        region = st.selectbox(
            "Región geográfica", 
            options=["centro", "norte", "sur", "oeste"],
            key="region"
        )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        contract_type = st.selectbox(
            "Tipo de contrato", 
            options=["mensual", "anual", "bianual"],
            key="contract_type"
        )
        payment_method = st.selectbox(
            "Método de pago", 
            options=["credito", "debito", "efectivo", "transferencia"],
            key="payment_method"
        )
        is_promo = st.selectbox(
            "¿Tiene tarifa promocional?", 
            options=["No", "Sí"],
            key="is_promo"
        )
    with col2:
        tenure_months = st.slider(
            "Antigüedad (Meses como cliente)", 
            min_value=1, max_value=120, 
            key="tenure_months"
        )
        monthly_charge = st.number_input(
            "Cargo mensual ($/USD)", 
            min_value=10.0, max_value=200.0, step=0.5, 
            key="monthly_charge"
        )
        total_charges = st.number_input(
            "Facturación histórica total ($/USD)", 
            min_value=10.0, max_value=15000.0, step=10.0, 
            key="total_charges"
        )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        internet_service = st.selectbox(
            "Servicio de Internet", 
            options=["cable", "fibra", "movil", "ninguno"],
            key="internet_service"
        )
        has_streaming = st.selectbox(
            "¿Tiene contratado Streaming?", 
            options=["No", "Sí"],
            key="has_streaming"
        )
        has_security_pack = st.selectbox(
            "¿Tiene Pack de Seguridad?", 
            options=["No", "Sí"],
            key="has_security_pack"
        )
        num_products = st.slider(
            "Cantidad de productos contratados", 
            min_value=1, max_value=10, 
            key="num_products"
        )
    with col2:
        avg_monthly_usage_gb = st.number_input(
            "Consumo promedio mensual (GB)", 
            min_value=0.0, max_value=1000.0, step=5.0, 
            key="avg_monthly_usage_gb"
        )
        support_tickets = st.number_input(
            "Tickets de soporte abiertos en el periodo", 
            min_value=0, max_value=20, step=1, 
            key="support_tickets"
        )
        late_payments = st.number_input(
            "Pagos con retraso en el periodo", 
            min_value=0, max_value=10, step=1, 
            key="late_payments"
        )

st.write("")
st.write("")

# Armado del payload a enviar a la API (mapeando variables de interfaz a modelo)
payload = {
    "tenure_months": int(tenure_months),
    "monthly_charge": float(monthly_charge),
    "total_charges": float(total_charges),
    "support_tickets": int(support_tickets),
    "late_payments": int(late_payments),
    "avg_monthly_usage_gb": float(avg_monthly_usage_gb),
    "contract_type": str(contract_type),
    "payment_method": str(payment_method),
    "internet_service": str(internet_service),
    "has_streaming": 1 if has_streaming == "Sí" else 0,
    "has_security_pack": 1 if has_security_pack == "Sí" else 0,
    "num_products": int(num_products),
    "region": str(region),
    "customer_age": int(customer_age),
    "is_promo": 1 if is_promo == "Sí" else 0
}

# Botón y procesamiento de predicción
if st.button("Predecir Probabilidad de Abandono"):
    try:
        with st.spinner("Conectando con el motor de inferencia..."):
            r = requests.post(API_URL, json=payload, timeout=5)
            
        if r.status_code == 200:
            result = r.json()
            prob = result["churn_probability"]
            label = result["churn_label"]
            
            # Mostrar resultado estilizado según el riesgo de Churn
            if label == 1:
                st.markdown(f"""
                    <div class="card-result-churn">
                        <div class="card-title">⚠️ ALTO RIESGO DE ABANDONO (CHURN)</div>
                        <p>El modelo estima que el cliente tiene una probabilidad de abandonar la compañía del <b>{prob:.1%}</b>.</p>
                        <p><b>Acción sugerida:</b> Activar inmediatamente campaña de retención personalizada, ofrecer promociones o resolver incidencias de soporte técnicas.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="card-result-loyal">
                        <div class="card-title">✅ CLIENTE FIDELIZADO (BAJO RIESGO)</div>
                        <p>El cliente posee un comportamiento saludable con una probabilidad de abandono de solo el <b>{prob:.1%}</b>.</p>
                        <p><b>Acción sugerida:</b> Mantener la calidad del servicio. Buen candidato para venta cruzada (cross-selling) de servicios adicionales.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Componente de progreso nativo para visualización
            st.write("")
            st.progress(prob)
            
        else:
            st.error(f"Error devuelto por la API (HTTP {r.status_code}): {r.text}")
            
    except requests.exceptions.ConnectionError:
        st.error(
            f"No se pudo establecer conexión con la API en '{API_URL}'. "
            "Por favor, verifica que el servicio de FastAPI esté levantado."
        )
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar la respuesta: {str(e)}")
