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

- [ ] **Despliegue:** Creación de API REST con FastAPI y contenedorización con Docker.
- [ ] **Interfaz:** Desarrollo de una GUI en Streamlit para consumo del modelo.
- [ ] **Monitoreo:** Implementación de Prometheus, Grafana y monitoreo de Data Drift con Evidently.

---

## Equipo

- Diaz, Ariana
- Lopez, Maria
- Riveros, David
- Vdovichenko, Walter
