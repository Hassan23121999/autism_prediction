import pandas as pd
from src.utils import preprocess_input

def predict_autism(input_data: pd.DataFrame, best_model, label_encoders, x_columns):
    """Generate predictions for input data."""
    processed_data = preprocess_input(input_data, label_encoders, x_columns)
    return best_model.predict(processed_data)
