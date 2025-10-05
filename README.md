# Autism Prediction

This repository contains a Jupyter notebook and supporting artifacts used to train and run a model that predicts an autism-related label from a questionnaire dataset.

Contents

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

---


### Project purpose
This project demonstrates a full notebook-based workflow: load and clean questionnaire data, perform exploratory data analysis (EDA), preprocess categorical and numerical features, train several classifiers (Decision Tree, Random Forest, XGBoost), select and save the best model, and provide a simple inference helper to run predictions on new CSV input.

### Repository layout
- `Autism_prediction.ipynb` — main notebook containing the full workflow.
- `data/` — dataset files used by the notebook (`train.csv`, `test.csv`, `sample_submission.csv`).
- `label_encoders/label_encoders.pkl` — pickled dict of fitted LabelEncoder objects created during preprocessing.
- `models/` — saved model artifacts (for example, `best_model.pkl`, `random_forestmodel.pkl`).
- `requirements.txt` — Python dependencies used by the notebook.

### Installation
1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Launch the notebook in Jupyter or VS Code and run cells in order:

```powershell
# from the project root
jupyter lab   # or jupyter notebook
# open Autism_prediction.ipynb
```

### Running the notebook
- Run the notebook cells top-to-bottom to reproduce preprocessing, model training, and saving artifacts. Large training steps (hyperparameter search) may take minutes depending on your machine.
- The notebook saves encoders and models; later cells assume those saved artifacts are present.

### Inference example
The notebook exposes a helper `predict_autism(input_data)` that accepts a pandas DataFrame and returns model predictions. Example usage within the notebook:

```python
import pandas as pd
input_df = pd.read_csv(r"data\test.csv")
preds = predict_autism(input_df)
print(preds)
```

Notes: the helper does basic cleaning (replaces `'?'` with NaN, fills some missing categorical values, converts `age` to int) and applies label encoding before predicting.

### Troubleshooting: LabelEncoder dtype mismatch (common issue)
Symptom: ValueError: invalid literal for int() with base 10: 'm'

Cause: A saved LabelEncoder was fit on numeric labels (its `classes_` have an integer dtype). At inference time you pass string inputs (for example `'m'`/`'f'`) and `LabelEncoder.transform` attempts to cast those strings to the encoder's dtype, causing the ValueError.

Mitigations included in this repo:
- The notebook's `predict_autism` was updated to detect numeric `classes_` in saved encoders and map incoming string values to the numeric indices expected by the encoder. This prevents the error at runtime.

Recommended permanent fix (preferred):
1. Refit encoders on the original training columns using string values so `le.classes_` are strings:

```python
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in object_columns:
	le = LabelEncoder()
	le.fit(df_train[col].astype(str))
	label_encoders[col] = le
	# save with pickle
```

2. Save `label_encoders/label_encoders.pkl` and use `le.transform(input_col.astype(str))` at inference.

### Reproducibility
- The notebook uses `pickle` to save models and encoders. For production use prefer `joblib` or model export formats supported by your serving stack.
- Consider wrapping preprocessing and model into a scikit-learn `Pipeline` or `ColumnTransformer` so training and inference use identical transformations.

### Next steps I can do for you
- Refit encoders on `data/train.csv` and overwrite `label_encoders/label_encoders.pkl` (then restore direct `transform` calls in `predict_autism`).
- Add a test cell that asserts `predict_autism(pd.read_csv('data/test.csv'))` runs and returns an array of expected length.
- Add a small CLI script `predict.py` that loads `models/best_model.pkl` and `label_encoders/label_encoders.pkl` and reads a CSV path to output predictions.

If you'd like any of the above, tell me which and I'll implement it.
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

---

### `src/` package and `app.py` (Streamlit)

This repo includes a small `src/` package with helper modules used by the notebook and the Streamlit app.

- `src/model_loader.py` — utilities to load saved artifacts (models and encoders). Functions:
	- `load_model(model_path: str)` — loads a pickled model.
	- `load_label_encoders(encoders_path: str)` — loads the saved label encoders dict.

- `src/utils.py` — preprocessing helper used at inference time (`preprocess_input`). It:
	- cleans the DataFrame (drops `ID`, `age_desc`, replaces `'?'`, fills `ethnicity` and `relation`),
	- converts `age` to integer,
	- normalizes some country names (`'Viet Nam' -> 'Vietnam'`, `'Hong Kong' -> 'China'`, `'AmericanSamoa' -> 'United States'`),
	- applies saved label encoders safely (includes a mapping fallback for encoders whose `classes_` are numeric),
	- reorders columns to match the model's training features.

- `src/predictor.py` — small wrapper that calls `preprocess_input(...)` and then `best_model.predict(...)` to produce predictions from a DataFrame.

`app.py` is a Streamlit frontend that:
- Loads `models/best_model.pkl` and `label_encoders/label_encoders.pkl` using `src.model_loader`.
- Provides two ways to run inference:
	1. Upload a CSV with the same feature columns and click "Predict from CSV".
	2. Fill a web form (manual input) and submit to predict a single row.
- Aligns and renames form fields (for example `A1`→`A1_Score`) to match model feature names, fills missing columns, and predicts.

How to run the Streamlit app

1. Ensure dependencies are installed (see `requirements.txt`).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
streamlit run app.py
```

3. The app will open in a browser where you can either upload `data/test.csv` or use the manual form.

Notes and tips
- `app.py` expects `models/best_model.pkl` and `label_encoders/label_encoders.pkl` to exist in their respective folders. If you retrain or re-save different artifacts, update the paths at the top of `app.py`.
- The Streamlit form uses friendly names for options (e.g., `male`/`female`) — these must match the encoders or be handled by the `preprocess_input` mapping; otherwise you will see encoding errors.
- If you change preprocessing or encoder strategy, rebuild and re-save `label_encoders/label_encoders.pkl` and `models/best_model.pkl`.

Deployment
- For simple deployments you can containerize this app (Docker) and run it on any host that supports Streamlit. For production consider decoupling the model into an API (FastAPI/Flask) and using a scalable front-end.

If you'd like, I can add a small `predict.py` CLI or a `Dockerfile` to demonstrate containerized deployment.

