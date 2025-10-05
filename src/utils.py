import numpy as np
import pandas as pd

def preprocess_input(input_data: pd.DataFrame, label_encoders: dict, feature_columns: list):
    """Clean and encode input data to match model training."""
    input_data = input_data.drop(columns=['ID', 'age_desc'], errors='ignore')
    input_data = input_data.replace('?', np.nan)

    # Fill missing values
    if 'ethnicity' in input_data.columns:
        input_data['ethnicity'] = input_data['ethnicity'].fillna('Others')
    if 'relation' in input_data.columns:
        input_data['relation'] = input_data['relation'].fillna('Others')

    # Convert age
    if not pd.api.types.is_integer_dtype(input_data['age']):
        input_data['age'] = input_data['age'].astype(int)

    # Clean country names
    if 'contry_of_res' in input_data.columns:
        input_data['contry_of_res'] = input_data['contry_of_res'].replace({
            'Viet Nam': 'Vietnam',
            'Hong Kong': 'China',
            'AmericanSamoa': 'United States'
        })

    # Apply label encoders
    for col in label_encoders:
        if col in input_data.columns:
            le = label_encoders[col]
            try:
                import numpy as _np
                if _np.issubdtype(le.classes_.dtype, _np.number):
                    mapping = {str(c): i for i, c in enumerate(le.classes_)}
                    input_data[col] = input_data[col].astype(str).map(mapping).fillna(-1).astype(int)
                else:
                    input_data[col] = le.transform(input_data[col].astype(str))
            except Exception:
                input_data[col] = le.transform(input_data[col].astype(str))

    # Ensure consistent column order
    input_data = input_data[feature_columns]

    return input_data
