import os
import joblib

_model = None

def get_model():
    """
    Carga el modelo serializado (pipeline de preprocesamiento + clasificador) una sola vez (Singleton).
    Soporta rutas relativas a la ejecución y rutas absolutas mediante variables de entorno.
    """
    global _model
    if _model is None:
        model_path = os.getenv("MODEL_PATH", "models/model.pkl")
        
        # Robustez: si el archivo no existe en el CWD actual, intentamos encontrarlo
        # relativo a la ubicación de este script
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            fallback_path = os.path.join(base_dir, "models", "model.pkl")
            if os.path.exists(fallback_path):
                model_path = fallback_path
                
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se pudo encontrar el modelo en {model_path} ni en el fallback {fallback_path}")
            
        print(f"Cargando modelo desde: {model_path}")
        _model = joblib.load(model_path)
    return _model
