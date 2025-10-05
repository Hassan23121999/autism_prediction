import pickle

def load_model(model_path: str):
    """Load trained ML model from pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_label_encoders(encoders_path: str):
    """Load label encoders dictionary."""
    with open(encoders_path, 'rb') as f:
        return pickle.load(f)
