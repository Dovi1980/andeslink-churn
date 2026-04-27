# AndesLink Churn Prediction — MLOps Local

> Sistema end-to-end de predicción de abandono de clientes, construido con prácticas de MLOps para la empresa simulada **AndesLink Servicios Digitales S.A.**
>
> Proyecto académico · ISTEA · Laboratorio de Minería de Datos · Prof. Diego Mosquera

---

## Tabla de contenidos

- [Contexto de negocio](#contexto-de-negocio)
- [Objetivo analítico](#objetivo-analítico)
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Dataset](#dataset)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y configuración](#instalación-y-configuración)
- [Ejecución del pipeline](#ejecución-del-pipeline)
- [Servicios y puertos](#servicios-y-puertos)
- [Métricas del modelo](#métricas-del-modelo)
- [Monitoreo](#monitoreo)
- [Equipo](#equipo)

---

## Contexto de negocio

**AndesLink Servicios Digitales S.A.** comercializa planes de suscripción para servicios digitales orientados a consumidores finales. Durante los últimos trimestres, la empresa detectó una tasa creciente de cancelación voluntaria de clientes (_churn_).

La pérdida de clientes impacta directamente en los ingresos recurrentes, el costo de adquisición de nuevos usuarios y la eficiencia de las campañas comerciales. El área de negocio requiere una solución que permita **anticipar el abandono** y activar campañas de retención antes de que el cliente cancele.

> En el sector de suscripciones, retener a un cliente existente cuesta entre 5 y 7 veces menos que adquirir uno nuevo. Detectar a tiempo a los clientes en riesgo es directamente rentable.

---

## Objetivo analítico

Construir un sistema de Machine Learning que estime la **probabilidad de churn** de cada cliente a partir de variables de comportamiento, antigüedad, facturación y relación con el servicio.

El sistema debe:

- Predecir correctamente a los clientes que van a cancelar (**maximizar Recall**)
- Estar disponible como servicio web con respuesta en tiempo real (**API REST**)
- Ser reproducible, versionado y monitoreable (**MLOps**)
- Poder desplegarse localmente con un solo comando (**Docker Compose**)

**Variable objetivo:** `churn` — binaria (1 = cancela, 0 = se queda)

**Métrica principal:** ROC-AUC · Métrica secundaria: Recall de la clase churn

---

## Arquitectura del sistema

El proyecto está organizado en tres hitos que se apilan progresivamente:

```
┌─────────────────────────────────────────────────────────────────┐
│  HITO 1 · Entrenamiento                                         │
│  EDA · Preprocesamiento · Entrenamiento · Evaluación            │
│  MLflow (tracking) · DVC (versionado de datos y pipeline)       │
├─────────────────────────────────────────────────────────────────┤
│  HITO 2 · Despliegue                                            │
│  FastAPI (API de inferencia) · Streamlit (GUI)                  │
│  Docker Compose (orquestación de servicios)                     │
├─────────────────────────────────────────────────────────────────┤
│  HITO 3 · Monitoreo                                             │
│  Prometheus → Grafana (métricas técnicas)                       │
│  Evidently (data drift · calidad de datos)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Flujo de datos

```
churn_sintetico.csv
       │
       ▼ dvc add
data/raw/  ──► src/data/prepare.py ──► data/processed/train.csv
                                                      test.csv
                                                         │
                                                         ▼
                                       src/models/train.py
                                                         │
                                                         ▼
                                              models/model.pkl
                                                         │
                                              ┌──────────┴──────────┐
                                              ▼                     ▼
                                         FastAPI                Streamlit
                                        /predict                  GUI
                                              │
                                              ▼
                                         Prometheus
                                              │
                                              ▼
                                           Grafana
```

---

## Dataset

| Atributo                | Detalle                                               |
| ----------------------- | ----------------------------------------------------- |
| **Nombre**              | `churn_sintetico.csv`                                 |
| **Origen**              | Dataset sintético generado para el proyecto AndesLink |
| **Filas**               | 5.000 registros                                       |
| **Columnas**            | 16 variables + 1 target                               |
| **Valores nulos**       | Ninguno                                               |
| **Distribución target** | 66% no-churn · 34% churn                              |

### Variables

| Variable               | Tipo       | Descripción                                  |
| ---------------------- | ---------- | -------------------------------------------- |
| `tenure_months`        | Numérica   | Meses como cliente                           |
| `monthly_charge`       | Numérica   | Cargo mensual en USD                         |
| `total_charges`        | Numérica   | Facturación total acumulada                  |
| `support_tickets`      | Numérica   | Tickets de soporte generados                 |
| `late_payments`        | Numérica   | Pagos realizados con atraso                  |
| `avg_monthly_usage_gb` | Numérica   | Consumo promedio mensual en GB               |
| `customer_age`         | Numérica   | Edad del cliente en años                     |
| `num_products`         | Numérica   | Cantidad de productos contratados            |
| `has_streaming`        | Binaria    | Tiene servicio de streaming (0/1)            |
| `has_security_pack`    | Binaria    | Tiene paquete de seguridad (0/1)             |
| `is_promo`             | Binaria    | Ingresó con promoción (0/1)                  |
| `contract_type`        | Categórica | Tipo de contrato (mensual / anual / bianual) |
| `payment_method`       | Categórica | Método de pago                               |
| `internet_service`     | Categórica | Tipo de servicio de internet                 |
| `region`               | Categórica | Región geográfica del cliente                |
| `churn`                | **Target** | 1 = canceló · 0 = se quedó                   |

> `total_charges` presenta alta multicolinealidad con `tenure_months × monthly_charge` (correlación 0.997 · VIF = 15.37). Se elimina del feature set antes del entrenamiento para evitar inestabilidad en la Regresión Logística.

### Señales más predictivas detectadas en el EDA

- **`contract_type`**: contratos mensuales tienen 47.5% de churn vs 11.6% en contratos bianuales
- **`internet_service`**: servicio móvil registra 50.9% de churn vs 26.8% en fibra óptica
- **`tenure_months`**: correlación negativa de -0.17 con churn — más antigüedad, menor riesgo
- **`support_tickets`** y **`late_payments`**: correlación positiva — más problemas, más riesgo

---

## Stack tecnológico

| Capa                  | Herramienta                       | Uso                                    |
| --------------------- | --------------------------------- | -------------------------------------- |
| Lenguaje              | Python 3.11                       | Base del proyecto                      |
| Entorno               | conda                             | Gestión de dependencias reproducible   |
| Editor                | VSCode                            | Desarrollo                             |
| Exploración           | JupyterLab                        | EDA y análisis inicial                 |
| Datos                 | pandas · NumPy                    | Manipulación y transformación          |
| ML                    | scikit-learn · XGBoost · LightGBM | Entrenamiento y evaluación             |
| Versionado código     | Git / GitHub                      | Control de versiones                   |
| Versionado datos      | DVC                               | Versionado de dataset y pipeline       |
| Tracking experimentos | MLflow                            | Registro de métricas e hiperparámetros |
| Serialización         | joblib                            | Guardado del pipeline completo         |
| API                   | FastAPI + Uvicorn                 | Servicio de inferencia                 |
| GUI                   | Streamlit                         | Interfaz gráfica para el usuario       |
| Contenedores          | Docker + Docker Compose           | Despliegue reproducible                |
| Métricas técnicas     | Prometheus                        | Recolección de métricas del servicio   |
| Dashboards            | Grafana                           | Visualización de métricas              |
| Monitoreo de datos    | Evidently                         | Detección de data drift                |
| Testing               | pytest                            | Pruebas de la API                      |

---

## Estructura del repositorio

```
andeslink-churn/
│
├── data/
│   ├── raw/                        # Dataset original (gestionado por DVC)
│   │   ├── churn_sintetico.csv.dvc # Puntero DVC — va en Git
│   │   └── .gitignore
│   └── processed/                  # Datos procesados (gestionados por DVC)
│       ├── train.csv.dvc
│       └── test.csv.dvc
│
├── models/                         # Artefactos serializados (gestionados por DVC)
│   └── model.pkl                   # Pipeline completo: preprocesador + modelo
│
├── notebooks/
│   └── 01_eda.ipynb                # Análisis exploratorio de datos
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare.py              # Limpieza y split train/test
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py       # ColumnTransformer del proyecto
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                # Entrenamiento + MLflow tracking
│   │   └── evaluate.py             # Evaluación en test set
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # Aplicación FastAPI
│       └── schemas.py              # Modelos Pydantic
│
├── tests/
│   ├── __init__.py
│   └── test_api.py                 # Pruebas de los endpoints
│
├── monitoring/
│   ├── prometheus.yml              # Configuración de Prometheus
│   ├── grafana/                    # Dashboards de Grafana
│   └── evidently_report.py         # Script de reporte de drift
│
├── reports/
│   └── figures/                    # Gráficos generados por el EDA
│
├── scripts/
│   └── check_model.py              # Verificación rápida del modelo serializado
│
├── .dvc/                           # Configuración de DVC
├── .dvcignore
├── dvc.yaml                        # Definición del pipeline
├── dvc.lock                        # Estado exacto del pipeline (va en Git)
├── params.yaml                     # Hiperparámetros del pipeline
├── docker-compose.yml              # Orquestación de servicios
├── Dockerfile                      # Imagen de la API
├── Dockerfile.gui                  # Imagen de la GUI
├── environment.yml                 # Entorno conda reproducible
├── .gitignore
└── README.md
```

---

## Instalación y configuración

### Requisitos previos

Verificar que estén instalados antes de continuar:

```bash
python --version     # 3.10 o 3.11
conda --version      # 23.x o superior
git --version        # 2.x o superior
docker --version     # 24.x o superior
```

### 1. Clonar el repositorio

```bash
git clone https://github.com/Dovi1980/andeslink-churn.git
cd andeslink-churn
```

### 2. Crear y activar el entorno conda

```bash
conda env create -f environment.yml
conda activate andeslink-churn
```

> La primera vez puede tardar entre 5 y 15 minutos. Las ejecuciones siguientes solo requieren `conda activate andeslink-churn`.

### 3. Restaurar los datos con DVC

```bash
dvc checkout
```

> Si es la primera vez y no hay remote DVC configurado, copiar manualmente `churn_sintetico.csv` a `data/raw/` y ejecutar `dvc repro`.

### 4. Verificar la instalación

```bash
python -c "import pandas, sklearn, mlflow, fastapi, dvc; print('OK')"
```

---

## Ejecución del pipeline

### Ejecutar el pipeline completo con DVC

```bash
dvc repro
```

DVC detecta automáticamente qué etapas cambiaron y solo re-ejecuta las necesarias. La primera ejecución completa corre:

1. `prepare` — limpieza y split del dataset
2. `featurize` — construcción de features
3. `train` — entrenamiento y serialización del modelo
4. `evaluate` — evaluación en test set

### Ejecutar etapas individualmente

```bash
# Solo preparación de datos
python src/data/prepare.py

# Solo entrenamiento
python src/models/train.py

# Solo evaluación
python src/models/evaluate.py
```

### Ver experimentos en MLflow

```bash
mlflow ui --port 5000
# Abrir: http://localhost:5000
```

### Verificar el modelo serializado

```bash
python -c "
import joblib, pandas as pd
pipeline = joblib.load('models/model.pkl')
df = pd.read_csv('data/processed/test.csv').drop(columns=['churn'])
proba = pipeline.predict_proba(df.head(5))[:, 1]
print('Predicciones OK:', proba.round(3))
"
```

---

## Servicios y puertos

### Levantar todos los servicios (Hito 2 y 3)

```bash
docker compose up -d --build
```

| Servicio              | URL                          | Descripción               |
| --------------------- | ---------------------------- | ------------------------- |
| **API FastAPI**       | http://localhost:8000        | Servicio de inferencia    |
| **Documentación API** | http://localhost:8000/docs   | Swagger UI interactiva    |
| **Health check**      | http://localhost:8000/health | Estado del servicio       |
| **GUI Streamlit**     | http://localhost:8501        | Interfaz gráfica          |
| **MLflow**            | http://localhost:5000        | Tracking de experimentos  |
| **Prometheus**        | http://localhost:9090        | Métricas raw del servicio |
| **Grafana**           | http://localhost:3000        | Dashboards de monitoreo   |

### Ejemplo de uso de la API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 6,
    "monthly_charge": 85.5,
    "support_tickets": 3,
    "late_payments": 2,
    "avg_monthly_usage_gb": 12.4,
    "customer_age": 34,
    "num_products": 1,
    "has_streaming": 0,
    "has_security_pack": 0,
    "is_promo": 1,
    "contract_type": "mensual",
    "payment_method": "tarjeta",
    "internet_service": "fibra",
    "region": "norte"
  }'
```

Respuesta esperada:

```json
{
  "churn": 1,
  "probabilidad_churn": 0.7823,
  "riesgo": "alto"
}
```

### Detener los servicios

```bash
docker compose down
```

---

## Métricas del modelo

Las métricas se registran automáticamente en MLflow con cada ejecución de `train.py`. La evaluación usa validación cruzada estratificada con K=5 para garantizar resultados robustos con el dataset de 5.000 filas.

| Métrica       | Descripción                                      | Por qué se usa                                                    |
| ------------- | ------------------------------------------------ | ----------------------------------------------------------------- |
| **ROC-AUC**   | Capacidad discriminativa en todos los umbrales   | Métrica principal para comparar modelos con clases desbalanceadas |
| **Recall**    | % de churners reales detectados                  | La métrica de negocio más importante — minimizar falsos negativos |
| **F1-Score**  | Balance entre Precision y Recall                 | Métrica complementaria para reportes                              |
| **Precision** | % de alertas correctas sobre el total de alertas | Control del costo de campañas de retención innecesarias           |

> **Umbral de decisión:** el umbral por defecto de 0.5 se ajusta según el análisis de la curva Precision-Recall para maximizar el Recall sin generar un exceso de falsas alarmas. Ver `src/models/evaluate.py` para el procedimiento de ajuste.

---

## Monitoreo

### Métricas técnicas del servicio (Prometheus + Grafana)

Prometheus recolecta métricas de la API cada 15 segundos. Los paneles de Grafana muestran en tiempo real:

- Requests por minuto
- Latencia de respuesta (p50, p95, p99)
- Tasa de errores HTTP
- Disponibilidad del servicio

### Monitoreo de datos y modelo (Evidently)

El script `monitoring/evidently_report.py` compara los datos de producción con el dataset de referencia (train) y genera un reporte HTML:

```bash
python monitoring/evidently_report.py
# Genera: reports/drift_report.html
```

El reporte incluye análisis de drift por variable, calidad de datos y degradación de performance del modelo.

**Señales que justifican reentrenamiento:**

- Data drift detectado en `contract_type`, `tenure_months` o `monthly_charge`
- Caída de ROC-AUC mayor a 0.05 respecto a la línea base
- Cambio en la distribución del target (proporción de churn)

---

## Flujo de trabajo diario del equipo

```bash
# Al empezar
conda activate andeslink-churn
git pull origin main
dvc checkout

# Al terminar
git add <archivos modificados>
git commit -m "tipo: descripción del cambio"
git push origin main
```

**Convención de commits:**

| Prefijo     | Uso                                         |
| ----------- | ------------------------------------------- |
| `feat:`     | Nueva funcionalidad                         |
| `fix:`      | Corrección de error                         |
| `data:`     | Cambios en datos o pipeline DVC             |
| `docs:`     | Documentación                               |
| `refactor:` | Reorganización sin cambio de comportamiento |
| `test:`     | Pruebas pytest                              |
| `chore:`    | Configuración y mantenimiento               |

---

## Equipo

Proyecto desarrollado como trabajo integrador de la materia **Laboratorio de Minería de Datos II** — ISTEA · 3er año · Data Science and AI.

Grupo integrado por:

- Diaz, Ariana:[@diazariana](https://github.com/diazariana)
- Lopez, Maria:[@MariaLopez1999](https://github.com/MariaLopez1999)
- Riveros, David:[@driveros-cpu](https://github.com/driveros-cpu)
- Vdovichenko, Walter:[@Dovi1980](https://github.com/Dovi1980)

---
