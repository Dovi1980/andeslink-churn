# AndesLink Churn Prediction — MLOps Local

> Sistema de predicción de abandono de clientes basado en prácticas de MLOps para la empresa simulada **AndesLink Servicios Digitales S.A.**
>
> Proyecto académico · ISTEA · Laboratorio de Minería de Datos · Prof. Diego Mosquera

---

## Tabla de contenidos

- [Contexto de negocio](#contexto-de-negocio)
- [Objetivo analítico](#objetivo-analítico)
- [Arquitectura (Hito 1)](#arquitectura-hito-1)
- [Dataset](#dataset)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y configuración](#instalación-y-configuración)
- [Ejecución del pipeline](#ejecución-del-pipeline)
- [Métricas del modelo](#métricas-del-modelo)
- [Próximos Pasos (Hitos 2 y 3)](#próximos-pasos-hitos-2-y-3)
- [Equipo](#equipo)

---

## Contexto de negocio

**AndesLink Servicios Digitales S.A.** busca reducir la tasa de cancelación voluntaria de clientes (_churn_). La pérdida de suscriptores impacta directamente en los ingresos recurrentes y en la rentabilidad de la compañía. Se requiere una solución proactiva que identifique a los clientes en riesgo para activar campañas de retención oportunas.

---

## Objetivo analítico

Construir un modelo de Machine Learning capaz de estimar la **probabilidad de churn** a partir de variables de comportamiento, antigüedad y facturación.

- **Variable objetivo:** `churn` (1 = cancela, 0 = permanece)
- **Métrica primaria:** **Recall** (Minimizar los falsos negativos para capturar la mayor cantidad de fugas posibles).

---

## Arquitectura (Hito 1)

En esta primera etapa, el foco está en la **reproducibilidad del entrenamiento** y la **modularización del código**.

### Flujo de datos Hito 1

```
churn_sintetico.csv (Raw)
       │
       ▼ (DVC repro)
src/data/prepare.py  ──► data/processed/ (train.csv / test.csv)
       │
       ▼
src/models/train.py  ──► models/model.pkl (Pipeline completo)
       │                 └── Tracking en MLflow (Métricas/Parámetros)
```

---

## Dataset

| Atributo          | Detalle                                               |
| ----------------- | ----------------------------------------------------- |
| **Nombre**        | `churn_sintetico.csv`                                 |
| **Filas**         | 5.000 registros                                       |
| **Columnas**      | 16 variables + 1 target                               |
| **Distribución**  | 66% no-churn · 34% churn                              |
| **Observaciones** | Se eliminó `total_charges` por alta multicolinealidad. |

---

## Stack tecnológico

| Capa                  | Herramienta        | Uso                                    |
| --------------------- | ------------------ | -------------------------------------- |
| Lenguaje              | Python 3.11        | Base del proyecto                      |
| Entorno               | conda              | Gestión de dependencias                |
| Exploración           | JupyterLab         | EDA y análisis inicial                 |
| Procesamiento         | pandas · sklearn   | Transformación de datos                |
| Tracking experimentos | MLflow             | Registro de métricas y modelos         |
| Versionado datos      | DVC                | Orquestación del pipeline              |
| Serialización         | joblib             | Guardado del modelo final              |

---

## Estructura del repositorio

```
andeslink-churn/
│
├── data/
│   ├── raw/                # Dataset original
│   └── processed/          # Datos divididos para entrenamiento/prueba
│
├── models/                 # Pipeline serializado (preprocesador + modelo)
│
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb
│   └── 02_train_test_mlflow.ipynb
│
├── src/
│   ├── data/
│   │   └── prepare.py      # Limpieza y split de datos
│   ├── features/
│   │   └── build_features.py # Definición del ColumnTransformer
│   ├── models/
│   │   └── train.py        # Entrenamiento y tracking con MLflow
│   └── api/                # (En desarrollo - Hito 2)
│
├── reports/
│   ├── figures/            # Gráficos del EDA
│   └── informe_tecnico_hito1.md # Resumen ejecutivo del parcial
│
├── dvc.yaml                # Definición de etapas (prepare, train)
├── params.yaml             # Parámetros de configuración
├── environment.yml         # Entorno conda reproducible
└── README.md
```

---

## Instalación y configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/Dovi1980/andeslink-churn.git
cd andeslink-churn
```

### 2. Entorno conda
```bash
conda env create -f environment.yml
conda activate andeslink-churn
```

### 3. Recuperar datos
```bash
dvc pull
# O en su defecto, colocar churn_sintetico.csv en data/raw/
```

---

## Ejecución del pipeline

El pipeline está automatizado con DVC. Para ejecutarlo completo:

```bash
dvc repro
```

Esto ejecutará secuencialmente:
1. **prepare**: Limpieza de datos y split estratificado.
2. **train**: Entrenamiento del RandomForest, registro en MLflow y exportación del modelo.

### Ver experimentos
```bash
mlflow ui
# Abrir: http://localhost:5000
```

---

## Métricas del modelo

| Métrica   | Objetivo | Justificación                                          |
| --------- | -------- | ------------------------------------------------------ |
| **Recall**| **Max**  | Evitar perder clientes reales (Falsos Negativos).      |
| Precision | Medir    | Controlar el costo de campañas de retención.          |
| ROC-AUC   | Medir    | Evaluar la capacidad de discriminación del modelo.     |

---

## Próximos Pasos (Hitos 2 y 3)

- [x] **Despliegue (Hito 2):** Creación de API REST con FastAPI y contenedorización con Docker.
- [x] **Interfaz (Hito 2):** Desarrollo de una GUI en Streamlit para consumo del modelo.
- [ ] **Monitoreo (Hito 3):** Implementación de Prometheus, Grafana y monitoreo de Data Drift con Evidently.

---

## Hito 2 — Guía de Operación y Despliegue

Esta sección detalla cómo levantar y verificar toda la infraestructura del Hito 2 (API de inferencia + Interfaz Gráfica).

### Requisitos del Sistema
* **Docker Desktop** (con soporte para Linux Containers y Compose v2 instalado y ejecutándose).
* **Python 3.11** (en caso de querer ejecutar pruebas locales fuera del contenedor).

### Arquitectura de la Solución Local
El siguiente diagrama detalla la interacción entre servicios de la red de Docker:

```mermaid
graph LR
    Usuario([Usuario / Navegador]) -->|HTTP :8501| GUI[Servicio GUI - Streamlit]
    GUI -->|HTTP POST| API[Servicio API - FastAPI]
    API -->|Carga local| Model[models/model.pkl]
    
    subgraph Red Interna Docker (docker-compose)
        GUI
        API
    end
```

### 1. Iniciar la Solución (Docker Compose)
Para levantar ambos servicios en segundo plano con un único comando:

```bash
docker compose up --build -d
```

Este comando:
1. Construye las imágenes basadas en [Dockerfile](file:///C:/andeslink-churn/Dockerfile) (API) y [Dockerfile.frontend](file:///C:/andeslink-churn/Dockerfile.frontend) (GUI).
2. Orquesta las dependencias mediante [docker-compose.yml](file:///C:/andeslink-churn/docker-compose.yml), esperando a que la API esté saludable (`healthcheck`) para levantar la GUI.
3. Monta la carpeta de modelos local `./models` en modo lectura (`ro`) dentro del contenedor.

### 2. URLs de Acceso
Una vez levantado el entorno:
* **Interfaz de Usuario (Streamlit):** [http://localhost:8501](http://localhost:8501)
* **Documentación Interactiva de la API (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Endpoint de Salud de la API:** [http://localhost:8000/health](http://localhost:8000/health)

### 3. Ejecución de Pruebas Unitarias e Integración
Puedes correr la suite de pruebas localmente para verificar el correcto funcionamiento (incluyendo mocks del modelo para CI/CD):

```bash
pytest tests/ -v
```

Para correr las pruebas dentro del contenedor de la API una vez levantado:

```bash
docker compose exec api pytest tests/ -v
```

---

## Equipo

- Diaz, Ariana
- Lopez, Maria
- Riveros, David
- Vdovichenko, Walter

