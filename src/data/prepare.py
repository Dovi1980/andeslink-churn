import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    # Cargar parámetros
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Leer datos raw
    df = pd.read_csv("data/raw/churn_sintetico.csv")
    
    # Limpieza: Eliminar total_charges por multicolinealidad
    if 'total_charges' in df.columns:
        df = df.drop(columns=['total_charges'])
    
    # Split Estratificado
    target = params['prepare']['target_column']
    test_size = params['prepare']['test_size']
    seed = params['prepare']['random_state']
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target], 
        random_state=seed
    )
    
    # Guardar procesados
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("Data preparation complete: train.csv and test.csv saved.")

if __name__ == "__main__":
    prepare_data()
