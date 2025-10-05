import streamlit as st
import pandas as pd
from src.model_loader import load_model, load_label_encoders
from src.predictor import predict_autism
import numpy as np

# Load model and encoders once
MODEL_PATH = r"models\best_model.pkl"
ENCODER_PATH = r"label_encoders\label_encoders.pkl"

best_model = load_model(MODEL_PATH)
label_encoders = load_label_encoders(ENCODER_PATH)

# Get training feature names
try:
    x_columns = list(best_model.feature_names_in_)
except AttributeError:
    # Fallback if feature_names_in_ not available
    x_columns = [
        'age', 'gender', 'jaundice', 'austim', 'ethnicity', 'contry_of_res', 'relation',
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'used_app_before', 'result'
    ]

# ====================================================
# Streamlit Page Setup
# ====================================================
st.set_page_config(page_title="Autism Prediction System", layout="wide")
st.title("🧠 Autism Prediction System")
st.markdown("Predict autism likelihood using trained ML model. You can either upload a CSV or fill the form below.")

# ====================================================
# Option 1: Upload CSV
# ====================================================
st.subheader("📂 Option 1: Upload CSV File")

uploaded_file = st.file_uploader("Upload a CSV file with the same feature columns:", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(input_data.head())

    if st.button("Predict from CSV"):
        predictions = predict_autism(input_data, best_model, label_encoders, x_columns)
        st.write("### ✅ Predictions")
        st.dataframe(pd.DataFrame(predictions, columns=["Predicted_Label"]))
else:
    st.info("Or use the manual form below 👇")

# ====================================================
# Option 2: Manual Input Form
# ====================================================
st.subheader("📝 Option 2: Manual Input Form")

with st.form("autism_form"):
    st.write("Enter the individual's details:")

    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["male", "female"])
    jaundice = st.selectbox("Jaundice (born with jaundice?)", ["yes", "no"])
    austim = st.selectbox("Family history of autism?", ["yes", "no"])
    ethnicity = st.selectbox("Ethnicity", [
        "Asian", "White-European", "Black", "Middle Eastern", "Hispanic", "Others"
    ])
    country = st.selectbox("Country of residence", [
        "United States", "United Kingdom", "India", "China", "Vietnam", "Germany", "Others"
    ])
    relation = st.selectbox("Who is completing the test?", [
        "Parent", "Self", "Health care professional", "Relative", "Others"
    ])

    # Example behavioral questions
    st.write("### Behavior Assessment Questions (1 = Yes, 0 = No)")
    a1 = st.selectbox("Q1: Does the person make eye contact?", [0, 1])
    a2 = st.selectbox("Q2: Does the person enjoy social interaction?", [0, 1])
    a3 = st.selectbox("Q3: Does the person use gestures?", [0, 1])
    a4 = st.selectbox("Q4: Does the person engage in pretend play?", [0, 1])
    a5 = st.selectbox("Q5: Does the person react to name being called?", [0, 1])
    a6 = st.selectbox("Q6: Does the person point to objects?", [0, 1])
    a7 = st.selectbox("Q7: Does the person smile in response to others?", [0, 1])
    a8 = st.selectbox("Q8: Does the person respond to emotions of others?", [0, 1])
    a9 = st.selectbox("Q9: Does the person understand simple instructions?", [0, 1])
    a10 = st.selectbox("Q10: Does the person show repetitive behaviors?", [0, 1])

    used_app_before = st.selectbox("Used app before?", [0, 1])
    result = 0  # placeholder (not used for prediction)

    submitted = st.form_submit_button("Predict from Form")

    if submitted:
        # Create DataFrame for prediction
        form_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "jaundice": jaundice,
            "austim": austim,
            "ethnicity": ethnicity,
            "contry_of_res": country,
            "relation": relation,
            "A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5,
            "A6": a6, "A7": a7, "A8": a8, "A9": a9, "A10": a10,
            "used_app_before": used_app_before,
            "result": result
        }])

        # ==== Fix: Map A1→A1_Score etc. ====
        rename_map = {
            "A1": "A1_Score", "A2": "A2_Score", "A3": "A3_Score", "A4": "A4_Score",
            "A5": "A5_Score", "A6": "A6_Score", "A7": "A7_Score", "A8": "A8_Score",
            "A9": "A9_Score", "A10": "A10_Score"
        }
        form_data.rename(columns=rename_map, inplace=True)

        # ==== Fill any missing columns expected by model ====
        for col in x_columns:
            if col not in form_data.columns:
                form_data[col] = 0

        # Align columns
        form_data = form_data[x_columns]

        # Predict
        predictions = predict_autism(form_data, best_model, label_encoders, x_columns)
        st.success(f"🎯 Prediction Result: {predictions[0]}")

