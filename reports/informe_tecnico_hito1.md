# Informe Técnico: Hito 1 - AndesLink Churn Prediction

## 1. Comprensión del Negocio
AndesLink enfrenta una tasa de cancelación creciente. El objetivo es identificar clientes en riesgo antes de la baja. Se define como problema de clasificación binaria donde la variable objetivo es `churn`.

## 2. El Dataset
- **Fuente:** Dataset sintético estructurado.
- **Volumen:** 5,000 registros, 16 características.
- **Calidad:** 0% de valores nulos detectados.
- **Distribución:** 66% Permanencia / 34% Churn (Leve desbalanceo).

## 3. Decisiones de Ingeniería de Datos
- **Eliminación de Multicolinealidad:** Se descartó `total_charges` por tener una correlación de 0.99 con `tenure_months` y `monthly_charge`, evitando ruido en modelos lineales.
- **Preprocesamiento:** Uso de `OneHotEncoder` para variables categóricas y `StandardScaler` para numéricas, integrados en un `Pipeline` para evitar Data Leakage.

## 4. Estrategia de Evaluación
Se seleccionó **Recall** como métrica principal. 
*Razón:* Para AndesLink es preferible contactar a un cliente que no se iba a ir (Falso Positivo) que perder a un cliente real por no detectarlo (Falso Negativo).

## 5. Resultados del Modelo Ganador
El modelo **Random Forest** mostró la mejor estabilidad y capacidad de detección.
- **Recall:** ~75% - 78% (en validación y prueba)
- **Artefacto:** Serializado como `models/churn_model.pkl` incluyendo el preprocesador.

## 6. Reproducibilidad
El flujo está automatizado mediante **DVC** y los experimentos trazados en **MLflow**, cumpliendo con los estándares de MLOps requeridos.
