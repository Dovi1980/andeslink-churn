# Informe Técnico Hito 1 — AndesLink Churn Prediction

**Curso:** Laboratorio de Minería de Datos II
**Institución:** ISTEA (Data Science and AI, 3er año)
**Profesor:** Diego Mosquera
**Equipo de Trabajo:** Diaz Ariana, Lopez Maria, Riveros David, Vdovichenko Walter

---

## 1. Introducción y Contexto de Negocio

**AndesLink Servicios Digitales S.A.** busca abordar y mitigar de forma proactiva la pérdida voluntaria de clientes (_churn_). Dado que retener a un cliente existente tiene un costo sustancialmente menor que la adquisición de uno nuevo, el desarrollo de un modelo predictivo robusto se vuelve una necesidad estratégica.

El objetivo principal de este desarrollo es estimar con precisión la probabilidad de que un cliente se dé de baja en el período subsiguiente. A nivel de métrica de Machine Learning, el negocio prioriza el **Recall (Sensibilidad)**:
$$\text{Recall} = \frac{VP}{VP + FN}$$
Esto se debe a que la empresa prefiere incurrir en el costo menor de ofrecer campañas de fidelización a clientes estables (Falsos Positivos) en lugar de perder clientes reales debido a una no detección (Falsos Negativos).

---

## 2. Análisis Exploratorio de Datos (EDA) y Hallazgos Clave

El dataset provisto contiene **5,000 registros** y **16 variables** (15 características explicativas y 1 variable objetivo). No se detectaron valores nulos, lo que facilitó un procesamiento directo y limpio sin necesidad de imputación.

### A. Distribución del Target

- **Permanencia (Clase 0):** 66% (3,300 registros)
- **Churn (Clase 1):** 34% (1,700 registros)
  Existe un leve desbalanceo de clases que requiere el uso de parámetros como `class_weight='balanced'` en el modelado para compensar el sesgo hacia la clase mayoritaria.

### B. Análisis Bivariado y Comportamiento del Cliente

El análisis exploratorio reveló relaciones críticas entre el comportamiento del usuario y la fuga:

1. **Impacto de la Antigüedad (`tenure_months`):**
   - Presenta una correlación negativa de **-0.170** con el churn.
   - La propensión al abandono es sumamente alta durante los primeros 3 a 6 meses de servicio (_onboarding_). A medida que el usuario supera los 12 meses, la fidelización se consolida y el churn disminuye drásticamente.

2. **Tipos de Contrato y Estructura Comercial:**
   - **Contrato Mensual:** Registra una tasa de churn crítica del **47.5%**. Prácticamente la mitad de los usuarios en esta modalidad rescinden el servicio en el corto plazo.
   - **Contrato Anual:** Registra una tasa de churn del **21.1%**.
   - **Contrato Bianual:** Muestra una tasa de churn de tan solo **11.6%**, consolidándose como la opción más estable y efectiva para la retención.

3. **Uso y Tipo de Servicio de Internet:**
   - **Móvil:** Registra una tasa de churn récord del **50.9%**, indicando posibles deficiencias de cobertura o alta volatilidad del mercado prepago/postpago móvil.
   - **Cable:** Registra un **34.8%** de churn.
   - **Fibra Óptica:** Registra un **26.8%** de churn, confirmando que una mejor tecnología de conectividad (más estable y de mayor ancho de banda) actúa como factor de retención.

4. **Variables de Fricción Operativa y Pago:**
   - **Tickets de Soporte Técnico (`support_tickets`):** Muestra una correlación positiva de **+0.102** con la fuga.
   - **Pagos Atrasados (`late_payments`):** Muestra una correlación positiva de **+0.081** con la fuga.
     Ambas variables representan fricciones del cliente que, al acumularse, detonan el abandono voluntario.

---

## 3. Decisiones de Ingeniería de Datos y Preprocesamiento

### A. Exclusión de Multicolinealidad Estructural

Durante el análisis de correlación numérica, se observó una colinealidad del **0.99** entre la característica `total_charges` (cargos acumulados) y la combinación lineal de `tenure_months` (meses de antigüedad) y `monthly_charge` (cargo mensual). Para evitar inestabilidades en los coeficientes del modelo y redundancia de información, se decidió **eliminar la variable `total_charges`** del dataset final.

### B. Preprocesamiento Seguro (Pipeline)

Para evitar el **Data Leakage** (fuga de información del conjunto de test al de entrenamiento), el procesamiento de las características se estructuró a través de un `ColumnTransformer`:

- **Numéricas (`StandardScaler`):** Normaliza y escala las variables continuas como cargos mensuales y consumo de GB.
- **Categóricas (`OneHotEncoder`):** Codifica las variables cualitativas sin orden jerárquico (`contract_type`, `payment_method`, `internet_service`, `region`) evitando la imposición de una ordinalidad inexistente.

---

## 4. Comparación de Modelos y Evaluación de Métricas

Para la primera entrega del hito, se evaluaron tres modelos bajo un umbral de decisión estándar de **0.5**:

| Modelo                      | Recall (Métrica Primaria) | Precision  |  F1-Score  |  ROC-AUC   |
| :-------------------------- | :-----------------------: | :--------: | :--------: | :--------: |
| **Random Forest** (Ganador) |        **75.88%**         |   47.78%   |   58.64%   |   74.03%   |
| **Logistic Regression**     |          72.94%           | **51.35%** | **60.27%** | **75.75%** |
| **XGBoost**                 |          51.47%           |   50.58%   |   51.02%   |   69.92%   |

### Análisis de la Comparativa:

1. **Random Forest:** Fue seleccionado como el modelo campeón debido a que alcanza el mayor **Recall (75.88%)**. Logra capturar la mayor cantidad de clientes propensos a la baja. Su precisión de 47.78% es aceptable considerando el bajo coste de las campañas de fidelización asociadas.
2. **Regresión Logística:** Muestra un rendimiento muy equilibrado con un F1-Score superior y mejor ROC-AUC. Sin embargo, su Recall es menor (72.94%) en comparación con Random Forest.
3. **XGBoost:** Con los hiperparámetros por defecto, mostró un bajo Recall (51.47%) debido a que no maneja de forma óptima el desbalanceo sin un pesado tuning previo.

---

## 5. Trazabilidad MLOps (DVC y MLflow)

El flujo de trabajo implementa principios de MLOps para garantizar la reproducibilidad y auditoría:

- **DVC:** Orquesta la separación de etapas (`prepare.py` y `train.py`) versionando los datasets intermedios y el modelo.
- **MLflow:** Almacena de manera estructurada el histórico de experimentos, registrando las métricas del modelo final y su firma de datos (_signature_) para el despliegue automático del Hito 2.
