# Autism Prediction

This repository contains a Jupyter notebook and supporting artifacts used to train and run a model that predicts an autism-related label from a questionnaire dataset.

Contents
- `Autism_prediction.ipynb` — main notebook: data loading, EDA, preprocessing, model training, saving models and encoders, and a prediction function for inference.
- `data/` — CSV files used by the notebook (`train.csv`, `test.csv`, `sample_submission.csv`).
- `label_encoders/label_encoders.pkl` — pickled dictionary of fitted LabelEncoder objects.
- `models/` — saved model artifacts (for example, `best_model.pkl`, `random_forestmodel.pkl`).

Quick start
1. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Open and run the notebook:

```powershell
# In VS Code or Jupyter Lab/Notebook open Autism_prediction.ipynb and run cells in order.
``` 

How to use the prediction helper
- The notebook provides a `predict_autism(input_data)` helper that reads a DataFrame and returns model predictions.
- Inference behavior notes:
	- The notebook saves label encoders during preprocessing. If those encoders were fit on numeric labels (int dtype), transforming raw string inputs at inference (for example, `'m'` for gender) can raise a ValueError because the encoder tries to cast inputs to the dtype of `classes_`.
	- The notebook now includes a robust mapping approach in the prediction function that maps incoming string values to the numeric indices expected by saved encoders. This prevents a "invalid literal for int()" error when an encoder's `classes_` are numeric.

Recommended long-term fix
- Refit and re-save the label encoders using string values from the original training data so `le.classes_` have string dtype. This keeps training and inference transforms consistent:

```python
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in object_columns:
		le = LabelEncoder()
		le.fit(df_train[col].astype(str))
		label_encoders[col] = le
		# persist with pickle
```

Notes and known issues
- The notebook currently saves models using pickle. When deploying, consider using joblib or model export formats appropriate for your serving environment.
- Consider building a scikit-learn Pipeline / ColumnTransformer for preprocessing + model so that the exact same preprocessing is applied during training and inference.

Next steps
- Add a small unit test verifying `predict_autism` on a sample of `data/test.csv`.
- Re-fit encoders on training strings and replace the mapping-based fallback.

If you'd like, I can refit the encoders now and update `label_encoders/label_encoders.pkl`, or add a test cell to the notebook that verifies `predict_autism` against `data/test.csv`.

