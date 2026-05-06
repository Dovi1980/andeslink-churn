import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, f1_score
from mlflow.models.signature import infer_signature
from src.features.build_features import get_preprocessor
import os

def train_model():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Cargar datos
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    target = params['prepare']['target_column']
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    
    # Definir features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # MLflow tracking
    mlflow.set_experiment("AndesLink_Churn_Prediction")
    
    with mlflow.start_run():
        # Obtener preprocesador
        preprocessor = get_preprocessor(numeric_features, categorical_features)
        
        # Crear Pipeline con hiperparámetros del yaml
        rf_params = params['train']
        model = RandomForestClassifier(**rf_params)
        
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Entrenar
        clf.fit(X_train, y_train)
        
        # Predecir y Evaluar
        y_pred = clf.predict(X_test)
        metrics = {
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        
        # Logear parámetros y métricas
        mlflow.log_params(rf_params)
        mlflow.log_metrics(metrics)
        
        # Signature y Log del Modelo
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(clf, "model", signature=signature)
        
        # Guardar localmente
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/model.pkl")
        
        print(f"Model trained. Metrics: {metrics}")

if __name__ == "__main__":
    train_model()
