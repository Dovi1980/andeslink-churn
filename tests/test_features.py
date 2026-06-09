from src.features.build_features import get_preprocessor
from sklearn.compose import ColumnTransformer

def test_get_preprocessor():
    numeric_features = ['tenure_months', 'monthly_charge', 'support_tickets']
    categorical_features = ['contract_type', 'payment_method', 'internet_service']
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    assert isinstance(preprocessor, ColumnTransformer)
